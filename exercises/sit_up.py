import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIT-UP - SAGITTAL / SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_HIP_MIN_ANGLE = 95.0           # Lying flat on floor
SETUP_KNEE_MAX_ANGLE = 145.0         # Knees bent naturally
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.48 # Side profile tolerance ratio
SETUP_MIN_READY_FRAMES = 5           # Frames lying down to confirm READY
UNLOCK_RESET_FRAMES = 12             # Frames lying flat to clear LOCKED_FAULT (~0.4s)
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility for profile

# 2. Arm Placement & Floor Support Guard (Hand-assist cheat prevention)
# Smaller Y = higher on screen (toward ceiling / away from floor).
MAX_ALLOWED_ELBOW_Y_DRIFT = 0.04     # Elbow must stay near or above ear/head level
DEBOUNCE_ARM_SUPPORT = 3             # Debounce filter for floor push detection

# 3. Kinematic & Form Thresholds
MIN_NECK_NEUTRAL_ANGLE = 115.0       # Neck yanking cutoff
MAX_FOOT_LIFT_DELTA = 0.085          # Foot lift off floor tolerance
ECCENTRIC_DROP_SPEED_MAX = 34.0      # Drop speed cutoff (deg/frame)
EARLY_REVERSAL_DELTA = 10.0          # Early downward reversal delta

# 4. Compression Depth & Range of Motion
FULL_COMPRESSION_MAX_HIP_ANGLE = 75.0  # Torso must compress near thighs (<= 75°)
HALF_COMPRESSION_MAX_HIP_ANGLE = 90.0  # Shallow range flagged strictly as bad rep
ASCENDING_START_HIP_ANGLE = 92.0       # Curling off floor initiates ASCENDING
DESCENDING_START_HIP_ANGLE = 80.0      # Lowering back initiates DESCENDING
REP_COMPLETE_HIP_ANGLE = 98.0          # Return to floor evaluates completion

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_NECK = 4
DEBOUNCE_FEET = 4
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 6

# 6. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.78

# 7. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 35

# =====================================================================


class SitUpTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility

        self.rep_started = False
        self.depth_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.arms_overhead = True          # Catches pushing off floor with arms/hands
        self.neck_neutral = True
        self.feet_planted = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.arm_support_frames = 0
        self.neck_frames = 0
        self.feet_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.lying_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_ankle_y = None
        self.prev_hip_angle = None
        self.current_feedback = "Lie flat with knees bent, hands behind head"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_sit_up.tflite")
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
        self.depth_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0
        self.reversal_frames = 0
        self.prev_hip_angle = None

        self.is_side_view = True
        self.arms_overhead = True
        self.neck_neutral = True
        self.feet_planted = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.arm_support_frames = 0
        self.neck_frames = 0
        self.feet_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_ear.visibility, l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_ear.visibility, r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Sit-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Full side profile not visible",
                "hip_angle": 0.0, "neck_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Elbow anchor check:
        # Elbow stays elevated near or above ear line (allowing bent hands behind head)
        is_elbow_in_position = active_elbow.y < (active_ear.y + MAX_ALLOWED_ELBOW_Y_DRIFT)

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_lying = (
            (hip_angle > SETUP_HIP_MIN_ANGLE)
            and (knee_angle < SETUP_KNEE_MAX_ANGLE)
            and is_elbow_in_position
            and is_side
        )

        if is_lying:
            self.lying_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y
            if self.lying_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.lying_frames = max(0, self.lying_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif not is_elbow_in_position:
                feedback_msg = "Keep elbows up near your head (do not rest arms on floor)"
            else:
                feedback_msg = "Lie flat on floor with knees bent to begin"
            return {
                "exercise": "Sit-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["BOTTOM", "WAITING", "LOCKED_FAULT"]:
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
        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Floor Pushing Cheat Detection:
            # 1. Elbow drops below shoulder line toward floor (bracing arm to push off)
            elbow_pushing_floor = active_elbow.y > (active_shoulder.y + 0.05)

            # 2. Wrist check ONLY if clearly visible (handles occluded hands behind head)
            wrist_pushing_floor = False
            if active_wrist.visibility >= 0.5:
                # Wrist dropping below shoulder/hip line toward the floor
                wrist_pushing_floor = active_wrist.y > (active_shoulder.y + 0.06)

            if elbow_pushing_floor or wrist_pushing_floor:
                self.arm_support_frames += 1
                if self.arm_support_frames >= DEBOUNCE_ARM_SUPPORT:
                    self.arms_overhead = False
            else:
                self.arm_support_frames = max(0, self.arm_support_frames - 1)

            if neck_angle < MIN_NECK_NEUTRAL_ANGLE:
                self.neck_frames += 1
                if self.neck_frames >= DEBOUNCE_NECK:
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > MAX_FOOT_LIFT_DELTA:
                self.feet_frames += 1
                if self.feet_frames >= DEBOUNCE_FEET:
                    self.feet_planted = False
            else:
                self.feet_frames = max(0, self.feet_frames - 1)

            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_lying:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "BOTTOM"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep hands bent behind head."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Sit-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Curl torso up toward knees without using arms!"

        elif self.state in ["BOTTOM", "DESCENDING"] and hip_angle < ASCENDING_START_HIP_ANGLE:
            self.state = "ASCENDING"
            self.rep_started = True
            self.min_hip_angle = hip_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Squeeze abs, sit up... do not push off floor!"

        # Full ROM: Torso reaches near knees (<= 75°)
        elif self.state == "ASCENDING" and hip_angle <= FULL_COMPRESSION_MAX_HIP_ANGLE:
            self.depth_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full compression! Lower back down with control."

        # Early downward reversal (Flagged as Bad Rep)
        elif self.state == "ASCENDING" and not self.depth_reached and (hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.was_half_rep = True
                self.state = "DESCENDING"
                self.current_feedback = "Incomplete sit-up! Lowering back early..."

        elif self.state == "TOP" and hip_angle > DESCENDING_START_HIP_ANGLE:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent back to floor..."

        if self.state == "ASCENDING" and not ((hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop detector
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_hip_angle is not None:
                drop_speed = hip_angle - self.prev_hip_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_hip_angle = hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "TOP", "ASCENDING"] and hip_angle >= REP_COMPLETE_HIP_ANGLE:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Lie sideways to camera! Return to start to reset."
                is_bad_rep = True
            elif not self.arms_overhead:
                self.current_feedback = "Bad rep: Do not push off your self with hands or elbows!"
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "Bad rep: Stop violently yanking your neck! Lead with chest and abs."
                is_bad_rep = True
            elif not self.feet_planted:
                self.current_feedback = "Bad rep: Keep feet down! Do not kick legs high for momentum."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Do not drop heavily to floor."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged body twist or broken kinetic chain."
                is_bad_rep = True
            elif self.was_half_rep or not self.depth_reached:
                if self.min_hip_angle <= HALF_COMPRESSION_MAX_HIP_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Compress all the way up near your thighs."
                else:
                    self.current_feedback = "Bad rep: Sit-up was too shallow! Compress all the way up."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Great sit-up! Clean rep."
                self.state = "BOTTOM"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((115 - hip_angle) / 50 * 100))) if self.state != "BOTTOM" else 100
        if not self.arms_overhead: form_score -= 40
        if not self.neck_neutral: form_score -= 30
        if not self.feet_planted: form_score -= 30
        if not self.controlled_descent: form_score -= 20
        if (self.was_half_rep or not self.depth_reached) and self.state == "DESCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Sit-Up", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": self.depth_reached
        }