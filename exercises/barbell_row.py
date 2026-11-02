import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (ADJUST PARAMETERS HERE)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_TORSO_MIN_ANGLE = 10.0      # Minimum forward hinge angle to count as setup
SETUP_TORSO_MAX_ANGLE = 70.0      # Maximum forward hinge angle to count as setup
SETUP_ELBOW_MIN_ANGLE = 115.0     # Arms must hang extended (> 115°)
SETUP_MIN_READY_FRAMES = 5        # Frames required in setup to transition to READY
UNLOCK_RESET_FRAMES = 15          # Frames in setup position required to clear LOCKED_FAULT (~0.5s)

# 2. Kinematic Angle Windows & Form Thresholds
TORSO_MIN_ANGLE = 10.0            # Minimum forward hinge angle during rep
TORSO_MAX_ANGLE = 72.0            # Maximum torso upright limit during rep
SHIN_MIN_ANGLE = 80.0             # Minimum shin angle to horizontal
SHIN_MAX_ANGLE = 100.0            # Maximum shin forward tilt angle
THIGH_MIN_ANGLE = 48.0            # Minimum knee-to-butt bend angle
THIGH_MAX_ANGLE = 88.0            # Maximum knee-to-butt bend angle
KNEE_LOCKOUT_MAX = 168.0          # Knees locked straight cutoff threshold
SPINE_MIN_ANGLE = 112.0           # Angle below which back rounding is flagged
FOREARM_MAX_TILT = 50.0           # Forearm tilt angle limit before flagging curling
FOOT_ALIGN_MAX_OFFSET = 0.10      # Screen-width ankle offset limit before flagging split stance

# 3. Pull Depth, Descent & Range of Motion
FULL_PULL_ELBOW_ANGLE = 92.0      # Elbow angle required to count full row depth
ARM_HORIZONTAL_TILT_MAX = 22.0    # Upper arm tilt angle relative to horizontal
ARM_ELBOW_SHOULDER_Y_MARGIN = 0.035  # Max vertical offset of elbow below shoulder
PULL_START_ELBOW_ANGLE = 125.0    # Transition threshold from EXTENDED/DESCENDING to ASCENDING
RETURN_START_ELBOW_ANGLE = 100.0  # Transition threshold from TOP to DESCENDING
EARLY_REVERSAL_DELTA = 10.0       # Reversal angle delta from peak before full ROM triggers half-rep
ECCENTRIC_DROP_SPEED_MAX = 35.0   # Angular speed threshold for uncontrolled free-fall drop
REP_COMPLETE_ELBOW_ANGLE = 128.0  # Arm extension threshold to evaluate rep completion

# 4. Debounce Frame Thresholds (Higher = Slower/More Deliberate detection)
DEBOUNCE_FOOT_ALIGN = 8           # Frames before flagging foot misalignment
DEBOUNCE_SHIN = 8                 # Frames before flagging shin drift
DEBOUNCE_KNEE = 8                 # Frames before flagging straight/locked legs
DEBOUNCE_TORSO = 12               # Frames before flagging improper torso hinge
DEBOUNCE_SPINE = 8                # Frames before flagging rounded back
DEBOUNCE_FOREARM = 8              # Frames before flagging bicep curl cheat
DEBOUNCE_DROP = 3                 # Frames before flagging eccentric free-fall
DEBOUNCE_HALF_REP = 8             # Frames before flagging premature reversal / incomplete row
DEBOUNCE_ML = 14                  # Frames before AI model flags bad spinal mechanics

# 5. Model & Buffer Settings
ML_CONFIDENCE_THRESHOLD = 0.90    # Minimum model probability to flag bad posture
SEQUENCE_LENGTH = 30              # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40     # Duration feedback remains visible post-rep

# =====================================================================


class BarbellRowTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.pull_reached = False
        self.min_elbow_angle = 180.0

        # Biomechanical Tripwires
        self.feet_aligned = True
        self.shin_angle_proper = True
        self.knee_bend_proper = True
        self.torso_angle_proper = True
        self.neutral_spine = True
        self.forearm_path_correct = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.prev_elbow_angle = None
        self.drop_frames = 0

        # Debounce counters
        self.foot_align_frames = 0
        self.shin_frames = 0
        self.knee_frames = 0
        self.torso_frames = 0
        self.spine_frames = 0
        self.forearm_frames = 0
        self.half_rep_frames = 0
        self.ml_bad_frames = 0
        self.post_rep_timer = 0

        self.ready_frames = 0
        self.unlock_frames = 0
        self.is_ready = False
        self.current_feedback = "Hinge at hips, keep knees softly bent (~50°-85°), shins at ~90°"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_barbell_row.tflite")
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
        self.pull_reached = False
        self.min_elbow_angle = 180.0
        self.prev_elbow_angle = None

        self.feet_aligned = True
        self.shin_angle_proper = True
        self.knee_bend_proper = True
        self.torso_angle_proper = True
        self.neutral_spine = True
        self.forearm_path_correct = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.foot_align_frames = 0
        self.shin_frames = 0
        self.knee_frames = 0
        self.torso_frames = 0
        self.spine_frames = 0
        self.forearm_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_bad_frames = 0

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

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_knee.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 1.6:
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Step back to show full body profile",
                "elbow_angle": 0.0,
                "knee_angle": 0.0,
                "hip_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Joint angle calculations
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        raw_knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        spine_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Knee-to-Butt (Femur/Thigh) Angle from Horizontal Floor Plane
        dx_thigh = abs(active_hip.x - active_knee.x)
        dy_thigh = abs(active_hip.y - active_knee.y)
        thigh_angle_horizontal = np.degrees(np.arctan2(dy_thigh, dx_thigh)) if dx_thigh > 0 else 90.0

        is_knee_locked_straight = raw_knee_angle > KNEE_LOCKOUT_MAX
        is_thigh_in_bend_window = (THIGH_MIN_ANGLE <= thigh_angle_horizontal <= THIGH_MAX_ANGLE)

        # 1. Torso Angle from Horizontal (Hips to Shoulder plane)
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        torso_angle_horizontal = np.degrees(np.arctan2(dy_torso, dx_torso)) if dx_torso > 0 else 90.0

        # 2. Shin Angle to Horizontal Floor (Ankle to Knee vector)
        dx_shin = abs(active_knee.x - active_ankle.x)
        dy_shin = abs(active_knee.y - active_ankle.y)
        shin_angle_horizontal = np.degrees(np.arctan2(dy_shin, dx_shin)) if dx_shin > 0 else 90.0

        # 3. Foot Alignment Offset
        ankle_x_offset = abs(l_ankle.x - r_ankle.x)

        # 4. Upper Arm Horizontal Tilt Check
        dx_arm = abs(active_elbow.x - active_shoulder.x)
        dy_arm = abs(active_elbow.y - active_shoulder.y)
        arm_tilt_from_horizontal = np.degrees(np.arctan2(dy_arm, dx_arm)) if dx_arm > 0 else 90.0
        is_upper_arm_horizontal = (arm_tilt_from_horizontal <= ARM_HORIZONTAL_TILT_MAX) or (active_elbow.y <= (active_shoulder.y + ARM_ELBOW_SHOULDER_Y_MARGIN))

        # 5. Forearm Angle
        dx_forearm = abs(active_wrist.x - active_elbow.x)
        dy_forearm = abs(active_wrist.y - active_elbow.y)
        forearm_tilt = np.degrees(np.arctan2(dx_forearm, dy_forearm)) if dy_forearm > 0 else 90.0

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_in_hinge = (SETUP_TORSO_MIN_ANGLE <= torso_angle_horizontal <= SETUP_TORSO_MAX_ANGLE) and (elbow_angle > SETUP_ELBOW_MIN_ANGLE) and (not is_knee_locked_straight)

        if is_in_hinge:
            self.ready_frames += 1
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if self.ready_frames > SETUP_MIN_READY_FRAMES:
            self.is_ready = True

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Bend knees slightly, avoid locking legs" if is_knee_locked_straight else "Hinge at hips, keep knees soft, let arms hang down"
            return {
                "exercise": "Barbell Row",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "knee_angle": round(raw_knee_angle, 1),
                "hip_angle": round(torso_angle_horizontal, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN BUFFER & HIGH-CONFIDENCE FILTER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_logits = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                exp_vals = np.exp(output_logits - np.max(output_logits))
                probs = exp_vals / np.sum(exp_vals)

                if probs[0] > ML_CONFIDENCE_THRESHOLD and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= DEBOUNCE_ML:
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["EXTENDED", "ASCENDING", "TOP", "DESCENDING"]:
            # A. Foot Alignment Check
            if ankle_x_offset > FOOT_ALIGN_MAX_OFFSET:
                self.foot_align_frames += 1
                if self.foot_align_frames >= DEBOUNCE_FOOT_ALIGN:
                    self.feet_aligned = False
            else:
                self.foot_align_frames = max(0, self.foot_align_frames - 1)

            # B. Shin Angle Scale
            if (shin_angle_horizontal < SHIN_MIN_ANGLE) or (shin_angle_horizontal > SHIN_MAX_ANGLE):
                self.shin_frames += 1
                if self.shin_frames >= DEBOUNCE_SHIN:
                    self.shin_angle_proper = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # C. Knee Straightening Check
            if is_knee_locked_straight or (not is_thigh_in_bend_window and raw_knee_angle > 164.0):
                self.knee_frames += 1
                if self.knee_frames >= DEBOUNCE_KNEE:
                    self.knee_bend_proper = False
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

            # D. Torso Inclination Check
            if (torso_angle_horizontal < TORSO_MIN_ANGLE) or (torso_angle_horizontal > TORSO_MAX_ANGLE):
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    self.torso_angle_proper = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # E. Spinal Rounding Check
            if spine_angle < SPINE_MIN_ANGLE:
                self.spine_frames += 1
                if self.spine_frames >= DEBOUNCE_SPINE:
                    self.neutral_spine = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

            # F. Forearm Path
            if forearm_tilt >= FOREARM_MAX_TILT:
                self.forearm_frames += 1
                if self.forearm_frames >= DEBOUNCE_FOREARM:
                    self.forearm_path_correct = False
            else:
                self.forearm_frames = max(0, self.forearm_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_in_hinge:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "EXTENDED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Pull bar toward your lower chest/belly."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Barbell Row",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "knee_angle": round(raw_knee_angle, 1),
                "hip_angle": round(torso_angle_horizontal, 1),
                "shin_angle": round(shin_angle_horizontal, 1),
                "thigh_angle": round(thigh_angle_horizontal, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull bar toward your lower chest/belly!"

        elif self.state in ["EXTENDED", "DESCENDING"] and elbow_angle < PULL_START_ELBOW_ANGLE:
            self.state = "ASCENDING"
            self.min_elbow_angle = elbow_angle
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive elbows up past your torso..."

        # Full Rep Depth Hit
        elif self.state == "ASCENDING" and (elbow_angle <= FULL_PULL_ELBOW_ANGLE or is_upper_arm_horizontal):
            self.pull_reached = True
            self.half_rep_frames = 0
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full row depth hit! Lower under control."

        elif self.state == "TOP" and elbow_angle > RETURN_START_ELBOW_ANGLE:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the bar down..."

        # Incomplete rep: Premature descent check with debounce
        elif self.state == "ASCENDING" and (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                self.state = "DESCENDING"
                self.current_feedback = "Perform a full rep! Pull elbows up level with your torso."
        else:
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Eccentric Free-Fall Monitor
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
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
        # 5. REP COMPLETION & EVALUATION (ONLY ON DESCENDING OR TOP)
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "TOP"] and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False

            if not self.knee_bend_proper:
                self.current_feedback = "Bad rep: Do not lock knees! Soften knees and return to start to reset."
                is_bad_rep = True
            elif not self.feet_aligned:
                self.current_feedback = "Bad rep: Align feet parallel! Return to start to reset."
                is_bad_rep = True
            elif not self.shin_angle_proper:
                self.current_feedback = "Bad rep: Keep shins near vertical! Return to start to reset."
                is_bad_rep = True
            elif not self.torso_angle_proper:
                self.current_feedback = "Bad rep: Keep torso hinged low! Return to start to reset."
                is_bad_rep = True
            elif not self.neutral_spine:
                self.current_feedback = "Bad rep: Back rounded! Keep spine flat and return to start to reset."
                is_bad_rep = True
            elif not self.forearm_path_correct:
                self.current_feedback = "Bad rep: Do not curl the bar! Drive elbows and return to start."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Do not drop the bar."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor spinal mechanics! Return to start."
                is_bad_rep = True
            elif not self.pull_reached:
                self.current_feedback = "Bad rep: Incomplete row! Pull elbows level with torso to reset."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Great row! Clean rep."
                self.state = "EXTENDED"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        return {
            "exercise": "Barbell Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "knee_angle": round(raw_knee_angle, 1),
            "hip_angle": round(torso_angle_horizontal, 1),
            "shin_angle": round(shin_angle_horizontal, 1),
            "thigh_angle": round(thigh_angle_horizontal, 1)
        }