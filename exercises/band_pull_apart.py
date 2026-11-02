import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (ADJUST PARAMETERS HERE)
# =====================================================================

# 1. Setup Guard Parameters
SETUP_MAX_WRIST_RATIO = 1.20        # Hands must start narrow (< 1.2x shoulder width)
SETUP_MAX_WRIST_Y_DIFF = 0.08       # Maximum vertical height difference between wrists at setup
SETUP_WRIST_HIP_MARGIN = 0.06       # Hands must stay comfortably above waistline
SETUP_WRIST_SHOULDER_MARGIN = 0.16  # Hands must stay below shoulders at chest level
SETUP_MIN_READY_FRAMES = 6          # Frames held in setup to transition to READY
UNLOCK_RESET_FRAMES = 15            # Frames held steadily in setup to clear LOCKED_FAULT (~0.5s)
BASELINE_EXPONENTIAL_ALPHA = 0.85   # Moving average weight for neck baseline
BASELINE_NEW_SAMPLE_WEIGHT = 0.15   # Sample update weight for neck baseline

# 2. Strict Kinematic & Form Thresholds
MIN_ELBOW_ANGLE_THRESHOLD = 120.0   # Elbow angle threshold (below this = bent elbow cheat)
SHRUG_NECK_DROP_RATIO = 0.85        # Neck ratio drop limit before flagging shrug
ARM_HEIGHT_TOLERANCE = 0.20         # Strict vertical corridor around chest/shoulder height
MAX_PULL_WRIST_Y_DIFF = 0.20        # Max vertical difference between hands during the pull (strict horizontal path)
MIN_SHOULDER_WIDTH = 0.25           # Fallback minimum shoulder width in normalized units
DEFAULT_TORSO_HEIGHT = 0.40         # Fallback torso height

# 3. Pull Depth & Range of Motion
PULL_START_RATIO = 1.30             # Must expand past this to enter PULLING state
FULL_ROM_PULL_RATIO = 1.70          # Must expand past this for rep completion credit
RETURN_START_RATIO = 1.45           # Moving inward below this transitions to RETURNING
EARLY_RETURN_DELTA = 0.15           # Inward drift allowed before full pull flags incomplete rep

# 4. Debounce Frame Thresholds
DEBOUNCE_ELBOW_BEND = 3             # Frames before flagging bent elbows
DEBOUNCE_SHRUG = 3                  # Frames before flagging shrugging
DEBOUNCE_HEIGHT_FAULT = 5           # Frames before flagging hands too high/low
DEBOUNCE_TILT_FAULT = 5             # Frames before flagging uneven/tilted hands

# 5. Model & Buffer Settings
SEQUENCE_LENGTH = 30                # ST-GCN temporal buffer frame capacity

# =====================================================================


class BandPullTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0
        self.pull_reached = False
        self.max_wrist_dist = 0.0

        # Tripwires
        self.arms_straight = True
        self.shoulders_shrugged = False
        self.arm_height_valid = True
        self.hands_horizontal = True
        self.ml_posture_maintained = True

        self.is_ready = False
        self.ready_frames = 0
        self.unlock_frames = 0

        # Debounce counters
        self.elbow_bend_frames = 0
        self.shrug_frames = 0
        self.height_fault_frames = 0
        self.tilt_fault_frames = 0
        self.baseline_neck_ratio = None

        self.current_feedback = "Bring hands together in front of your chest"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_band_pull_apart.tflite")
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
        self.max_wrist_dist = 0.0
        self.arms_straight = True
        self.shoulders_shrugged = False
        self.arm_height_valid = True
        self.hands_horizontal = True
        self.ml_posture_maintained = True
        self.elbow_bend_frames = 0
        self.shrug_frames = 0
        self.height_fault_frames = 0
        self.tilt_fault_frames = 0
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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        # Elbow straightness
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        wrist_distance = abs(l_wrist.x - r_wrist.x)
        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), MIN_SHOULDER_WIDTH)

        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        avg_hip_y = (l_hip.y + r_hip.y) / 2.0
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2.0

        # Torso-normalized neck metric
        torso_height = abs(avg_shoulder_y - avg_hip_y)
        if torso_height == 0:
            torso_height = DEFAULT_TORSO_HEIGHT

        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        hands_inside_body = wrist_distance < (shoulder_width * SETUP_MAX_WRIST_RATIO)
        hands_aligned = abs(l_wrist.y - r_wrist.y) < SETUP_MAX_WRIST_Y_DIFF
        hands_at_chest_level = (avg_wrist_y < (avg_hip_y - SETUP_WRIST_HIP_MARGIN)) and (avg_wrist_y > (avg_shoulder_y - SETUP_WRIST_SHOULDER_MARGIN))

        if hands_inside_body and hands_aligned and hands_at_chest_level:
            self.ready_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            elif self.ready_frames < 10:
                self.baseline_neck_ratio = BASELINE_EXPONENTIAL_ALPHA * self.baseline_neck_ratio + BASELINE_NEW_SAMPLE_WEIGHT * neck_ratio
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if self.ready_frames > SETUP_MIN_READY_FRAMES:
            self.is_ready = True

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Band Pull Apart", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hold hands together in front of your chest",
                "knee_angle": round(min_elbow_angle, 1), "hip_angle": 0.0
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(output_data, axis=1)[0] == 0 and self.state not in ["WAITING", "LOCKED_FAULT"]:
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["PULLING", "TOP", "RETURNING"]:
            # A. Elbow Bend Check
            if min_elbow_angle < MIN_ELBOW_ANGLE_THRESHOLD:
                self.elbow_bend_frames += 1
                if self.elbow_bend_frames >= DEBOUNCE_ELBOW_BEND:
                    self.arms_straight = False
            else:
                self.elbow_bend_frames = max(0, self.elbow_bend_frames - 1)

            # B. Responsive Shrug Check
            if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_DROP_RATIO):
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    self.shoulders_shrugged = True
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # C. Arm Height Corridor (Elevation)
            too_low = avg_wrist_y > (avg_shoulder_y + ARM_HEIGHT_TOLERANCE)
            too_high = avg_wrist_y < (avg_shoulder_y - ARM_HEIGHT_TOLERANCE)

            if too_low or too_high:
                self.height_fault_frames += 1
                if self.height_fault_frames >= DEBOUNCE_HEIGHT_FAULT:
                    self.arm_height_valid = False
                    if too_low:
                        self.current_feedback = "Hands too low! Raise arms higher"
                    else:
                        self.current_feedback = "Hands too high! Lower hands slightly"
            else:
                self.height_fault_frames = max(0, self.height_fault_frames - 1)

            # D. Hand Horizontal Plane Check (Tilt)
            wrist_y_gap = abs(l_wrist.y - r_wrist.y)
            if wrist_y_gap > MAX_PULL_WRIST_Y_DIFF:
                self.tilt_fault_frames += 1
                if self.tilt_fault_frames >= DEBOUNCE_TILT_FAULT:
                    self.hands_horizontal = False
            else:
                self.tilt_fault_frames = max(0, self.tilt_fault_frames - 1)

            self.max_wrist_dist = max(self.max_wrist_dist, wrist_distance)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if hands_inside_body and hands_aligned and hands_at_chest_level:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Pull band apart outward."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Band Pull Apart",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "knee_angle": round(min_elbow_angle, 1),
                "hip_angle": 0.0
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Pull band apart outward!"

        # Pull initiated
        elif self.state in ["RESTING", "RETURNING"] and wrist_distance > (shoulder_width * PULL_START_RATIO):
            self.state = "PULLING"
            self.max_wrist_dist = wrist_distance
            if self.arm_height_valid and self.hands_horizontal:
                self.current_feedback = "Pull wide, keep arms extended..."

        # Full ROM: Extension reached
        elif self.state == "PULLING" and wrist_distance >= (shoulder_width * FULL_ROM_PULL_RATIO):
            self.pull_reached = True
            self.state = "TOP"
            if self.arm_height_valid and self.hands_horizontal:
                self.current_feedback = "Squeeze upper back! Return under control."

        # Returning to starting position
        elif self.state == "TOP" and wrist_distance < (shoulder_width * RETURN_START_RATIO):
            self.state = "RETURNING"
            if self.arm_height_valid and self.hands_horizontal:
                self.current_feedback = "Control the band back to center..."

        # Incomplete rep: Early reversal
        elif self.state == "PULLING" and (self.max_wrist_dist - wrist_distance) > (shoulder_width * EARLY_RETURN_DELTA):
            self.state = "RETURNING"
            if self.arm_height_valid and self.hands_horizontal:
                self.current_feedback = "Perform a full rep! Pull wide until band touches chest."

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["RETURNING", "TOP"] and hands_inside_body and hands_aligned:
            is_bad_rep = False

            if not self.arm_height_valid:
                self.current_feedback = "Bad rep: Keep arms level with chest! Bring hands to start to reset."
                is_bad_rep = True
            elif not self.hands_horizontal:
                self.current_feedback = "Bad rep: Keep hands level! Do not tilt or pull unevenly."
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "Bad rep: Elbows bent! Keep arms extended. Bring hands to start to reset."
                is_bad_rep = True
            elif self.shoulders_shrugged:
                self.current_feedback = "Bad rep: Shoulders shrugged! Relax traps and return to start."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: Poor posture flagged by AI! Return to start to reset."
                is_bad_rep = True
            elif not self.pull_reached:
                self.current_feedback = "Bad rep: Incomplete pull! Return to start to reset."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Excellent rep! Perfect form."
                self.state = "RESTING"
                self._reset_rep_tracking()

        # Score Calculation
        form_score = 100
        if not self.arm_height_valid: form_score -= 25
        if not self.hands_horizontal: form_score -= 20
        if not self.pull_reached: form_score -= 25
        if not self.arms_straight: form_score -= 30
        if self.shoulders_shrugged: form_score -= 30
        if not self.ml_posture_maintained: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Band Pull Apart",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0
        }