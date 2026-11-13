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
SETUP_RACKED_MAX_ELBOW_ANGLE = 120.0 # Elbows bent below this to count as racked stance
SETUP_WRIST_HIP_MARGIN = 0.06        # Wrists must be held above hips by at least this normalized margin
SETUP_MIN_READY_FRAMES = 5           # Frames required holding racked stance to transition to READY
UNLOCK_RESET_FRAMES = 15             # Frames held in racked stance to clear LOCKED_FAULT (~0.5s)

# 2. Kinematic & Form Thresholds
STRAIGHT_ARM_SWING_ANGLE = 135.0     # Angle cutoff flagging arms extended too low below chest
SWING_WRIST_SHOULDER_MARGIN = 0.05   # Hands below shoulder level margin flagging straight-arm swing
MAX_ELBOW_FLARE_RATIO = 0.45         # Max horizontal elbow flare displacement / shoulder width
MAX_TORSO_LATERAL_OFFSET = 0.09      # Mid-shoulder to mid-hip lateral difference limit (spine lean)
MAX_WRIST_ASYMMETRY = 0.14           # Max vertical difference between left and right wrists
MIN_KEYPOINT_VISIBILITY = 0.35       # Minimum landmark confidence for front view
MIN_FRONT_SHOULDER_SPAN_RATIO = 0.30 # Shoulder span / torso height below this means turned sideways (side view)

# 3. Press Depth, Extension & Range of Motion
FULL_LOCKOUT_MIN_ANGLE = 155.0       # Minimum elbow angle required for full overhead lockout credit
HALF_PRESS_THRESHOLD_ANGLE = 155.0   # Reversing before this angle flags incomplete/half press
PRESS_START_ELBOW_ANGLE = 110.0      # Transition threshold from RACKED to PRESSING
LOWERING_START_ELBOW_ANGLE = 140.0   # Moving below this angle from TOP transitions to LOWERING
EARLY_REVERSAL_DELTA = 8.0           # Drop in elbow angle from peak during ascent to catch early descent
REP_COMPLETE_ELBOW_ANGLE = 120.0     # Return angle to racked stance evaluating rep completion

# 4. Debounce Frame Thresholds (Higher = Slower/More Deliberate detection)
DEBOUNCE_FLARE = 3                   # Frames before flagging elbow flare
DEBOUNCE_TORSO = 3                   # Frames before flagging lateral lean
DEBOUNCE_EVEN = 3                    # Frames before flagging asymmetric press
DEBOUNCE_HALF_PRESS = 2              # Frames before confirming premature reversal
DEBOUNCE_VIEW = 3                    # Frames before flagging a side view (not facing camera)

# 5. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size

# =====================================================================


class DumbbellOverheadShoulderPressTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.lockout_reached = False
        self.half_press_reached = False
        self.started_from_racked = False
        self.straight_arm_swing = False
        self.max_reach_angle = 0.0
        self.reversal_frames = 0

        # Front View Form Tripwires
        self.arm_path_aligned = True
        self.torso_upright = True
        self.arms_even = True
        self.ml_posture_maintained = True
        self.is_front_view = True

        # Debounce counters
        self.flare_frames = 0
        self.torso_frames = 0
        self.even_frames = 0
        self.view_frames = 0

        self.is_ready = False
        self.racked_frames = 0
        self.unlock_frames = 0

        self.current_feedback = "Bring dumbbells to shoulder level"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_overhead_shoulder_press.tflite")
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

    def _reset_rep_tracking(self, is_proper_racked_stance=False):
        self.lockout_reached = False
        self.half_press_reached = False
        self.started_from_racked = is_proper_racked_stance
        self.straight_arm_swing = False
        self.max_reach_angle = 0.0
        self.reversal_frames = 0

        self.flare_frames = 0
        self.torso_frames = 0
        self.even_frames = 0
        self.view_frames = 0

        self.arm_path_aligned = True
        self.torso_upright = True
        self.arms_even = True
        self.ml_posture_maintained = True
        self.is_front_view = True

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

        vis_all = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < MIN_KEYPOINT_VISIBILITY:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face camera directly: show full torso and arms",
                "elbow_angle": 0.0, "hip_angle": 0.0
            }

        # Elbow extension angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)
        max_elbow_angle = max(l_elbow_angle, r_elbow_angle)

        shoulder_width = abs(l_shoulder.x - r_shoulder.x) or 0.30
        current_hip_y = (l_hip.y + r_hip.y) / 2.0
        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0

        # ---------------------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------------------
        elbows_bent = max_elbow_angle <= SETUP_RACKED_MAX_ELBOW_ANGLE
        wrists_above_waist = (l_wrist.y < (current_hip_y - SETUP_WRIST_HIP_MARGIN)) and (r_wrist.y < (current_hip_y - SETUP_WRIST_HIP_MARGIN))
        is_proper_racked_stance = elbows_bent and wrists_above_waist

        if is_proper_racked_stance:
            self.racked_frames += 1
            if self.racked_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.racked_frames = max(0, self.racked_frames - 1)

        if not self.is_ready and self.state not in ["RACKED", "PRESSING", "TOP", "LOWERING", "LOCKED_FAULT"]:
            self.state = "WAITING"
            self.frame_buffer.clear()
            self.started_from_racked = False
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Bring dumbbells to shoulder level to begin",
                "elbow_angle": round(min_elbow_angle, 1), "hip_angle": 0.0
            }

        # ---------------------------------------------------------------------
        # 2. ST-GCN BUFFER
        # ---------------------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # ---------------------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------------------
        if self.state in ["RACKED", "PRESSING", "TOP", "LOWERING"]:
            # 0. Side View Check (turned sideways to the camera)
            torso_height = abs(avg_shoulder_y - current_hip_y) or 0.40
            shoulder_span_ratio = abs(l_shoulder.x - r_shoulder.x) / torso_height
            if shoulder_span_ratio < MIN_FRONT_SHOULDER_SPAN_RATIO:
                self.view_frames += 1
                if self.view_frames >= DEBOUNCE_VIEW:
                    self.is_front_view = False
            else:
                self.view_frames = max(0, self.view_frames - 1)

            # A. Catch Straight-Arm Swing
            hands_low = (l_wrist.y > (avg_shoulder_y + SWING_WRIST_SHOULDER_MARGIN)) or (r_wrist.y > (avg_shoulder_y + SWING_WRIST_SHOULDER_MARGIN))
            if hands_low and (min_elbow_angle > STRAIGHT_ARM_SWING_ANGLE):
                self.straight_arm_swing = True

            # B. Excessive Elbow Flare
            l_elbow_flare = abs(l_elbow.x - l_shoulder.x) / shoulder_width
            r_elbow_flare = abs(r_elbow.x - r_shoulder.x) / shoulder_width
            if (l_elbow_flare > MAX_ELBOW_FLARE_RATIO) or (r_elbow_flare > MAX_ELBOW_FLARE_RATIO):
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_FLARE:
                    self.arm_path_aligned = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # C. Lateral Spine Sway
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            if abs(mid_shoulder_x - mid_hip_x) > MAX_TORSO_LATERAL_OFFSET:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # D. Asymmetric Pressing
            if abs(l_wrist.y - r_wrist.y) > MAX_WRIST_ASYMMETRY:
                self.even_frames += 1
                if self.even_frames >= DEBOUNCE_EVEN:
                    self.arms_even = False
            else:
                self.even_frames = max(0, self.even_frames - 1)

            self.max_reach_angle = max(self.max_reach_angle, min_elbow_angle)

        # ---------------------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_proper_racked_stance:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RACKED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking(is_proper_racked_stance=True)
                    self.current_feedback = "Reset complete! Press dumbbells overhead."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Dumbbell Overhead Press",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(min_elbow_angle, 1),
                "hip_angle": 0.0
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RACKED"
            self.started_from_racked = True
            self.current_feedback = "Press dumbbells overhead!"

        elif self.state == "RACKED":
            if is_proper_racked_stance:
                self.started_from_racked = True
            if min_elbow_angle > PRESS_START_ELBOW_ANGLE:
                self.state = "PRESSING"
                self.max_reach_angle = min_elbow_angle
                self.reversal_frames = 0
                self.current_feedback = "Drive weights straight up..."

        # Full Lockout (>= 155 deg overhead)
        elif self.state == "PRESSING" and min_elbow_angle >= FULL_LOCKOUT_MIN_ANGLE:
            if (l_wrist.y < avg_shoulder_y) and (r_wrist.y < avg_shoulder_y):
                self.state = "TOP"
                self.lockout_reached = True
                self.current_feedback = "Full lockout! Lower under control."

        # Premature descent check
        elif self.state == "PRESSING" and (self.max_reach_angle < HALF_PRESS_THRESHOLD_ANGLE) and (self.max_reach_angle - min_elbow_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_PRESS:
                self.half_press_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete rep: Lowering arms early..."

        elif self.state == "TOP" and max_elbow_angle < LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            self.current_feedback = "Lower weights down..."

        # Reversal frame decay
        if self.state == "PRESSING" and not ((self.max_reach_angle < HALF_PRESS_THRESHOLD_ANGLE) and (self.max_reach_angle - min_elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # ---------------------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "PRESSING"] and (self.lockout_reached or self.half_press_reached) and max_elbow_angle <= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera! Do not turn sideways."
                is_bad_rep = True
            elif self.straight_arm_swing or not self.started_from_racked:
                self.current_feedback = "Bad rep: Do not swing arms from waist! Press from shoulder level."
                is_bad_rep = True
            elif not self.arm_path_aligned:
                self.current_feedback = "Bad rep: Keep elbows in! Do not flare arms wide outward."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Do not lean sideways! Keep spine upright."
                is_bad_rep = True
            elif not self.arms_even:
                self.current_feedback = "Bad rep: Press evenly! One arm lagged behind."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected posture breakdown."
                is_bad_rep = True
            elif not self.lockout_reached:
                if self.half_press_reached:
                    self.current_feedback = "Bad rep: Half rep! Press all the way to full lockout."
                else:
                    self.current_feedback = "Bad rep: Incomplete range! Press all the way to full lockout."
                is_bad_rep = True

            # Rep Evaluation
            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking(is_proper_racked_stance=is_proper_racked_stance)
            else:
                self.good_reps += 1
                self.current_feedback = "Clean press! Full lockout."
                self.state = "RACKED"
                self._reset_rep_tracking(is_proper_racked_stance=is_proper_racked_stance)

        # Dynamic Scoring
        form_score = max(0, min(100, int((min_elbow_angle / 160) * 100)))
        if not self.is_front_view: form_score -= 35
        if not self.arm_path_aligned: form_score -= 30
        if not self.torso_upright: form_score -= 20
        if not self.arms_even: form_score -= 20
        if self.straight_arm_swing: form_score -= 35
        if not self.lockout_reached and (self.state in ["PRESSING", "LOWERING", "TOP"]): form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Overhead Press",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0
        }