import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (FLOOR SKULL CRUSHER - SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_LOCKOUT_MIN_ELBOW_ANGLE = 154.0 # Fully locked vertical arms over chest
SETUP_UPPER_ARM_MIN_ANGLE = 75.0      # Lower bound for vertical upper arm angle
SETUP_UPPER_ARM_MAX_ANGLE = 115.0     # Upper bound for vertical upper arm angle
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.42  # Side profile ratio
SETUP_MIN_READY_FRAMES = 8            # Consecutive frames holding lockout to confirm READY
UNLOCK_RESET_FRAMES = 15              # Frames held at lockout to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.5        # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds
MIN_UPPER_ARM_VERTICAL_ANGLE = 68.0   # Drifting forward toward knees
ECCENTRIC_DROP_SPEED_MAX = 24.0       # Forehead drop speed cutoff (deg/frame)
EARLY_REVERSAL_DELTA = 8.0            # Early upward extension before reaching depth

# 3. Extension Depth & Range of Motion
FULL_DEPTH_MAX_ELBOW_ANGLE = 75.0     # Forehead/floor level triceps stretch
HALF_DEPTH_MAX_ELBOW_ANGLE = 95.0     # Flexing past 95° but short of 75° awards half rep
LOWERING_START_ELBOW_ANGLE = 145.0    # Initiating descent
PRESSING_START_ELBOW_ANGLE = 85.0     # Reversing upwards toward lockout
REP_COMPLETE_ELBOW_ANGLE = 154.0      # Full lockout evaluating rep completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_ARM_DRIFT = 4
DEBOUNCE_PRESS_CHEAT = 4
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class FloorSkullCrusherTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.full_depth_reached = False
        self.half_depth_reached = False
        self.min_elbow_angle = 180.0

        # Posture Tripwires (SIDE VIEW - LYING DOWN)
        self.is_side_view = True
        self.upper_arm_vertical = True
        self.shoulders_stable = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.arm_drift_frames = 0
        self.press_cheat_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.setup_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Lie flat sideways, arms straight up above chest"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_floor_skull_crusher.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception:
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            lm = landmarks[idx]
            coords.append([lm.x, lm.y, lm.z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.full_depth_reached = False
        self.half_depth_reached = False
        self.min_elbow_angle = 180.0
        self.reversal_frames = 0
        self.prev_elbow_angle = None

        self.is_side_view = True
        self.upper_arm_vertical = True
        self.shoulders_stable = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.arm_drift_frames = 0
        self.press_cheat_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Floor Skull Crusher", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Show full side profile lying down",
                "elbow_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < SETUP_MAX_SHOULDER_SPAN_RATIO

        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        is_lying_horizontal = dx_torso > (dy_torso * 0.85)

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_at_top = (
            (elbow_angle >= SETUP_LOCKOUT_MIN_ELBOW_ANGLE)
            and (SETUP_UPPER_ARM_MIN_ANGLE <= upper_arm_angle <= SETUP_UPPER_ARM_MAX_ANGLE)
            and is_side
            and is_lying_horizontal
        )

        if is_at_top:
            self.setup_frames += 1
            if self.setup_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.setup_frames = max(0, self.setup_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Lie flat on floor, lock arms straight over chest"
            return {
                "exercise": "Floor Skull Crusher", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER (SOFTMAX + DEBOUNCE)
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_started:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["TOP", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "BOTTOM", "PRESSING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            if upper_arm_angle < MIN_UPPER_ARM_VERTICAL_ANGLE:
                self.arm_drift_frames += 1
                if self.arm_drift_frames >= DEBOUNCE_ARM_DRIFT:
                    self.upper_arm_vertical = False
            else:
                self.arm_drift_frames = max(0, self.arm_drift_frames - 1)

            if active_elbow.y > active_shoulder.y:
                self.press_cheat_frames += 1
                if self.press_cheat_frames >= DEBOUNCE_PRESS_CHEAT:
                    self.shoulders_stable = False
            else:
                self.press_cheat_frames = max(0, self.press_cheat_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_at_top:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "TOP"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Bend elbows, lower weight toward forehead."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Floor Skull Crusher", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Bend elbows, lower weight toward forehead"

        elif self.state in ["TOP", "PRESSING"] and elbow_angle < LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            self.rep_started = True
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower to 90 degrees or deeper..."

        # Full Depth: Elbow reaches <= 75°
        elif self.state == "LOWERING" and elbow_angle <= FULL_DEPTH_MAX_ELBOW_ANGLE:
            self.full_depth_reached = True
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full depth! Extend straight up."

        # Early upward reversal (Half rep branch)
        elif self.state == "LOWERING" and not self.full_depth_reached and (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_elbow_angle <= HALF_DEPTH_MAX_ELBOW_ANGLE:
                    self.half_depth_reached = True
                self.state = "PRESSING"
                self.current_feedback = "Incomplete range! Returning weights early..."

        elif self.state == "BOTTOM" and elbow_angle > PRESSING_START_ELBOW_ANGLE:
            self.state = "PRESSING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock out at the top..."

        if self.state == "LOWERING" and not ((elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop detector
        if self.rep_started and self.state in ["LOWERING", "PRESSING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["PRESSING", "BOTTOM", "LOWERING"] and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Lie sideways to camera! Return to start to reset."
                is_bad_rep = True
            elif not self.upper_arm_vertical:
                self.current_feedback = "Bad rep: Keep upper arms fixed! Do not swing elbows forward."
                is_bad_rep = True
            elif not self.shoulders_stable:
                self.current_feedback = "Bad rep: Do not bench press! Rotate only at elbow joint."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Do not drop weights toward face."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor stability/kinetic breakdown."
                is_bad_rep = True
            elif not self.full_depth_reached:
                if self.min_elbow_angle <= HALF_DEPTH_MAX_ELBOW_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Half rep: Incomplete range! Lower weight fully to forehead level."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Movement was too shallow! Bend elbows deeper."
                    is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.half_reps += 1
                self.state = "TOP"
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean skull crusher rep!"
                self.state = "TOP"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 75) * 0.5))) if self.state != "TOP" else 100
        if not self.upper_arm_vertical: form_score -= 35
        if not self.shoulders_stable: form_score -= 30
        if not self.controlled_descent: form_score -= 20
        if not self.full_depth_reached and self.state == "PRESSING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Floor Skull Crusher", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "depth_reached": self.full_depth_reached
        }