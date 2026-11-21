import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (MULE KICK - TWO-FOOT BILATERAL LIFT)
# =====================================================================

# 1. Setup & Starting Guard (Side View, Hands Planted Under Shoulders, Both Feet on Floor)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.40 # Strict side profile view (16:9 webcam calibrated)
SETUP_MAX_TORSO_DROP = 0.32          # Quadruped / bear-crouch torso angle
SETUP_MAX_HIP_ANGLE = 122.0          # Knees bent in loaded quadruped/crouch base
SETUP_MIN_SUPPORT_ARM_ANGLE = 135.0  # Arms extended supporting body on hands (not elbows)
SETUP_MAX_WRIST_SHOULDER_DRIFT = 0.35# Wrists stacked roughly under shoulders
SETUP_MIN_READY_FRAMES = 6           # Frames holding steady base to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held in base to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility

# 2. Bilateral Two-Foot Lift & Form Thresholds
MIN_FOOT_LIFT_RATIO = 0.14           # Minimum vertical lift of ankles off floor (normalized by torso_len)
FULL_KICK_LIFT_RATIO = 0.38          # Both heels driven high (ankles rise >= 38% of torso_len off floor or above hips)
HALF_KICK_LIFT_RATIO = 0.22          # Partial hop (< 38% lift) flags Half Rep (counted as Bad Rep)
MAX_BILATERAL_ANKLE_SPLIT = 0.16     # Max Y-difference between left & right ankles (catches 1-foot donkey kick!)
MAX_BILATERAL_KNEE_SPLIT = 0.16      # Both knees must lift together off the floor
MIN_KICK_KNEE_BEND_ANGLE = 45.0      # Knees stay bent as heels kick toward glutes/ceiling
MAX_KICK_KNEE_BEND_ANGLE = 145.0     # Rejects piking with locked straight legs
MAX_SPINE_COLLAPSE_DROP = 0.10       # Prevents collapsing chest/hips onto the floor
EARLY_REVERSAL_RATIO_DELTA = 0.08    # Dropping feet down before reaching full kick height

# 3. State Machine Return Threshold
RETURNING_START_RATIO_DROP = 0.10    # Dropping 10% below peak height transitions TOP -> RETURNING
REP_COMPLETE_LIFT_RATIO = 0.10       # Both feet returned to the floor evaluates rep completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 3
DEBOUNCE_SINGLE_FOOT = 3             # Frames before flagging kicking with only 1 foot instead of 2
DEBOUNCE_KNEE_ANGLE = 4
DEBOUNCE_ARMS = 3                    # Frames before flagging collapsed support arms
DEBOUNCE_HALF_REP = 3                # Frames before confirming premature downward return
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.90

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class MuleKickTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.target_height_reached = False
        self.half_height_reached = False
        self.max_lift_ratio = 0.0
        self.max_kick_angle = 0.0

        # Instructor Tripwires (SIDE VIEW - TWO-FOOT MULE KICK)
        self.is_side_view = True
        self.both_feet_lifted = True         # Enforces BOTH feet lifting together (rejects 1-leg kick)
        self.knee_bent_90 = True             # Knees bent driving heels up
        self.spine_neutral = True            # Core braced, no collapsing
        self.arms_supported = True           # Wrists planted under shoulders, arms straight
        self.ml_posture_maintained = True

        self.baseline_ankle_y = None
        self.baseline_knee_y = None
        self.baseline_hip_y = None
        self.baseline_wrist_y = None

        # Debounce counters
        self.profile_frames = 0
        self.single_foot_frames = 0
        self.knee_bend_frames = 0
        self.arch_frames = 0
        self.arm_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.tabletop_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Get on all fours sideways to camera: hands under shoulders, ready to kick BOTH feet up"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_mule_kick.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[MULE KICK ERROR] __init__ model_path={model_path} | error={exc}",
                    flush=True,
                )
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        """Maps MediaPipe 33 landmarks to the NTU-25 skeleton layout."""
        ntu = np.zeros((25, 3), dtype=np.float32)

        def xyz(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        ntu[3] = xyz(0)    # Nose
        ntu[4] = xyz(11)   # Left Shoulder
        ntu[5] = xyz(13)   # Left Elbow
        ntu[6] = xyz(15)   # Left Wrist
        ntu[7] = xyz(19)   # Left Hand/Index
        ntu[8] = xyz(12)   # Right Shoulder
        ntu[9] = xyz(14)   # Right Elbow
        ntu[10] = xyz(16)  # Right Wrist
        ntu[11] = xyz(20)  # Right Hand/Index
        ntu[12] = xyz(23)  # Left Hip
        ntu[13] = xyz(25)  # Left Knee
        ntu[14] = xyz(27)  # Left Ankle
        ntu[15] = xyz(31)  # Left Foot
        ntu[16] = xyz(24)  # Right Hip
        ntu[17] = xyz(26)  # Right Knee
        ntu[18] = xyz(28)  # Right Ankle
        ntu[19] = xyz(32)  # Right Foot

        ntu[0] = (ntu[12] + ntu[16]) / 2.0  # Base Spine (Mid-Hip)
        ntu[2] = (ntu[4] + ntu[8]) / 2.0    # Spine Shoulder / Neck (Mid-Shoulder)
        ntu[1] = (ntu[0] + ntu[2]) / 2.0    # Mid Spine
        ntu[20] = ntu[2]                    # Chest

        ntu[21] = xyz(17)  # Left Pinky
        ntu[22] = xyz(21)  # Left Thumb
        ntu[23] = xyz(18)  # Right Pinky
        ntu[24] = xyz(22)  # Right Thumb
        return ntu

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.target_height_reached = False
        self.half_height_reached = False
        self.max_lift_ratio = 0.0
        self.max_kick_angle = 0.0
        self.reversal_frames = 0

        self.is_side_view = True
        self.both_feet_lifted = True
        self.knee_bent_90 = True
        self.spine_neutral = True
        self.arms_supported = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.single_foot_frames = 0
        self.knee_bend_frames = 0
        self.arch_frames = 0
        self.arm_frames = 0
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Mule Kick", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: show full side profile on the floor",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Primary camera-facing side for upper-body quadruped alignment
        if r_vis >= l_vis:
            active_shoulder, active_elbow, active_wrist, active_hip = r_shoulder, r_elbow, r_wrist, r_hip
        else:
            active_shoulder, active_elbow, active_wrist, active_hip = l_shoulder, l_elbow, l_wrist, l_hip

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # 1. Quadruped / Bear-Crouch Upper-Body Support (Wrists under shoulders, arms straight)
        support_arm_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        wrist_shoulder_drift = abs(active_wrist.x - active_shoulder.x) / torso_len
        arms_properly_stacked = (
            (support_arm_angle >= SETUP_MIN_SUPPORT_ARM_ANGLE)
            and (wrist_shoulder_drift <= SETUP_MAX_WRIST_SHOULDER_DRIFT)
        )

        # 2. Bilateral Hip & Knee Angles
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        both_ankles_vis = (l_ankle.visibility > 0.40 and r_ankle.visibility > 0.40)
        both_knees_vis = (l_knee.visibility > 0.40 and r_knee.visibility > 0.40)

        if both_ankles_vis:
            avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0
            avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
            # Use max(l_ankle.y, r_ankle.y) — the LOWER foot on screen — so BOTH feet must rise to count!
            lowest_ankle_y = max(l_ankle.y, r_ankle.y)
            highest_ankle_y = min(l_ankle.y, r_ankle.y)
            ankle_split_ratio = abs(l_ankle.y - r_ankle.y) / torso_len
        else:
            avg_hip_angle = r_hip_angle if r_vis >= l_vis else l_hip_angle
            avg_knee_angle = r_knee_angle if r_vis >= l_vis else l_knee_angle
            lowest_ankle_y = r_ankle.y if r_vis >= l_vis else l_ankle.y
            highest_ankle_y = lowest_ankle_y
            ankle_split_ratio = 0.0

        knee_split_ratio = (abs(l_knee.y - r_knee.y) / torso_len) if both_knees_vis else 0.0

        # 3. Two-Foot Vertical Lift Calculation (Relative to Floor Baseline)
        floor_ankle_y = self.baseline_ankle_y if self.baseline_ankle_y is not None else lowest_ankle_y
        # lift_ratio measures how high the LOWER foot has risen off the floor (requires BOTH feet to lift!)
        both_feet_lift_ratio = max(0.0, (floor_ankle_y - lowest_ankle_y) / torso_len)
        single_foot_lift_ratio = max(0.0, (floor_ankle_y - highest_ankle_y) / torso_len)

        # Check if heels kicked up to or above hip height
        both_heels_near_hip_height = lowest_ankle_y <= (active_hip.y + 0.08)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Hands Under Shoulders, Both Feet on Floor)
        # ---------------------------------------------------------
        vertical_drop = abs(active_shoulder.y - active_hip.y)
        is_horizontal = vertical_drop < SETUP_MAX_TORSO_DROP
        knees_loaded = (l_hip_angle < SETUP_MAX_HIP_ANGLE) or (r_hip_angle < SETUP_MAX_HIP_ANGLE)
        feet_on_floor = both_feet_lift_ratio <= REP_COMPLETE_LIFT_RATIO and single_foot_lift_ratio <= MIN_FOOT_LIFT_RATIO

        is_tabletop = is_side and is_horizontal and knees_loaded and arms_properly_stacked and feet_on_floor

        if is_tabletop:
            self.tabletop_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = lowest_ankle_y
                self.baseline_knee_y = max(l_knee.y, r_knee.y)
                self.baseline_hip_y = active_hip.y
                self.baseline_wrist_y = active_wrist.y
            else:
                if not self.rep_started:
                    self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * lowest_ankle_y
                    self.baseline_knee_y = 0.90 * self.baseline_knee_y + 0.10 * max(l_knee.y, r_knee.y)
                    self.baseline_hip_y = 0.90 * self.baseline_hip_y + 0.10 * active_hip.y
                    self.baseline_wrist_y = 0.90 * self.baseline_wrist_y + 0.10 * active_wrist.y

            if self.tabletop_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.tabletop_frames = max(0, self.tabletop_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera on all fours"
            elif not is_horizontal:
                feedback_msg = "Place hands on floor and keep back flat"
            elif not arms_properly_stacked:
                feedback_msg = "Stack wrists directly under shoulders with straight arms"
            else:
                feedback_msg = "Bend knees with both feet on the floor to begin"

            return {
                "exercise": "Mule Kick", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "knee_angle": round(avg_knee_angle, 1), "hip_angle": round(avg_hip_angle, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN BUFFER (SOFTMAX + DEBOUNCE)
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_started:
            try:
                buf_tvc = np.array(self.frame_buffer, dtype=np.float32)
                expected_shape = self.input_details[0]['shape']
                if len(expected_shape) == 5:
                    seq = np.transpose(buf_tvc, (2, 0, 1))[None, :, :, :, None]
                elif len(expected_shape) == 4 and tuple(expected_shape[1:]) == (buf_tvc.shape[2], buf_tvc.shape[0], buf_tvc.shape[1]):
                    seq = np.transpose(buf_tvc, (2, 0, 1))[None, :, :, :]
                else:
                    seq = np.expand_dims(buf_tvc, axis=0)

                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                bad_posture_prob = float(probs[0])

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["TABLETOP", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[MULE KICK FAULT] AI ST-GCN flagged posture breakdown: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[MULE KICK ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (TWO-FOOT LIFT ENFORCEMENT)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["KICKING", "TOP", "RETURNING"]:
            self.max_lift_ratio = max(self.max_lift_ratio, both_feet_lift_ratio)
            self.max_kick_angle = max(self.max_kick_angle, avg_hip_angle)

            # A. Strict Side View Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[MULE KICK FAULT] Lost side view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Two-Foot Simultaneous Lift Check (Catches kicking with only 1 foot while the other stays down!)
            one_foot_left_on_floor = (
                both_ankles_vis
                and (single_foot_lift_ratio > 0.22)
                and (
                    (both_feet_lift_ratio < 0.08)
                    or (ankle_split_ratio > MAX_BILATERAL_ANKLE_SPLIT)
                    or (knee_split_ratio > MAX_BILATERAL_KNEE_SPLIT)
                )
            )
            if one_foot_left_on_floor:
                self.single_foot_frames += 1
                if self.single_foot_frames >= DEBOUNCE_SINGLE_FOOT:
                    if self.both_feet_lifted:
                        print(
                            f"[MULE KICK FAULT] Only 1 foot lifted! ankle_split={ankle_split_ratio:.2f}, "
                            f"low_foot_lift={both_feet_lift_ratio:.2f}, high_foot_lift={single_foot_lift_ratio:.2f}",
                            flush=True,
                        )
                    self.both_feet_lifted = False
            else:
                self.single_foot_frames = max(0, self.single_foot_frames - 1)

            # C. Bent Knees Check (Heels kick toward glutes/ceiling, not straight-leg pike)
            if both_feet_lift_ratio > 0.20:
                is_knee_bent = (MIN_KICK_KNEE_BEND_ANGLE <= avg_knee_angle <= MAX_KICK_KNEE_BEND_ANGLE)
                if not is_knee_bent:
                    self.knee_bend_frames += 1
                    if self.knee_bend_frames >= DEBOUNCE_KNEE_ANGLE:
                        if self.knee_bent_90:
                            print(
                                f"[MULE KICK FAULT] Knees not bent during kick: avg_knee={avg_knee_angle:.1f}°",
                                flush=True,
                            )
                        self.knee_bent_90 = False
                else:
                    self.knee_bend_frames = max(0, self.knee_bend_frames - 1)

            # D. Support Arms Locked & Hands Planted on Floor
            if support_arm_angle < (SETUP_MIN_SUPPORT_ARM_ANGLE - 12.0):
                self.arm_frames += 1
                if self.arm_frames >= DEBOUNCE_ARMS:
                    if self.arms_supported:
                        print(f"[MULE KICK FAULT] Support arms buckled: angle={support_arm_angle:.1f}°", flush=True)
                    self.arms_supported = False
            else:
                self.arm_frames = max(0, self.arm_frames - 1)

            # E. Spine / Hip Collapse Check
            hip_collapse = (active_hip.y - self.baseline_hip_y) if self.baseline_hip_y is not None else 0.0
            if hip_collapse > MAX_SPINE_COLLAPSE_DROP:
                self.arch_frames += 1
                if self.arch_frames >= 4:
                    if self.spine_neutral:
                        print(f"[MULE KICK FAULT] Hips/spine sagged downward: drop={hip_collapse:.3f}", flush=True)
                    self.spine_neutral = False
            else:
                self.arch_frames = max(0, self.arch_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (TABLETOP -> KICKING -> TOP -> RETURNING)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_tabletop:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "TABLETOP"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Press through hands and kick BOTH feet up."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Mule Kick", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "knee_angle": round(avg_knee_angle, 1), "hip_angle": round(avg_hip_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "TABLETOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Press into hands and kick BOTH feet up toward the ceiling!"

        # Initiate Kick (Triggers when feet leave the floor from TABLETOP ONLY)
        elif self.state == "TABLETOP" and single_foot_lift_ratio >= MIN_FOOT_LIFT_RATIO:
            self.state = "KICKING"
            self.rep_started = True
            self.max_lift_ratio = both_feet_lift_ratio
            self.max_kick_angle = avg_hip_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive BOTH heels high into the air, keep arms locked..."

        # Full Kick Height Reached: BOTH feet lifted high (>= FULL_KICK_LIFT_RATIO or both heels near/above hip level)
        elif self.state == "KICKING" and self.both_feet_lifted and (
            both_feet_lift_ratio >= FULL_KICK_LIFT_RATIO
            or (both_heels_near_hip_height and both_feet_lift_ratio >= 0.26)
        ):
            self.state = "TOP"
            self.target_height_reached = True
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Great two-foot kick height! Land softly on the balls of your feet."

        # Early downward reversal (Half Rep Branch: Dropping feet before reaching full kick height)
        elif self.state == "KICKING" and not self.target_height_reached and (
            (self.max_lift_ratio - both_feet_lift_ratio) > EARLY_REVERSAL_RATIO_DELTA
            or (single_foot_lift_ratio < MIN_FOOT_LIFT_RATIO and self.max_lift_ratio > 0.05)
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_lift_ratio >= HALF_KICK_LIFT_RATIO:
                    self.half_height_reached = True
                self.state = "RETURNING"
                self.current_feedback = "Incomplete kick height! Feet dropping early..."
                print(
                    f"[MULE KICK FAULT] Early reversal before full two-foot kick height: "
                    f"max_lift_ratio={self.max_lift_ratio:.2f} (required>={FULL_KICK_LIFT_RATIO})",
                    flush=True,
                )

        # Controlled Return from TOP
        elif self.state == "TOP" and (self.max_lift_ratio - both_feet_lift_ratio) >= RETURNING_START_RATIO_DROP:
            self.state = "RETURNING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Land softly on both feet with knees bent..."

        if self.state == "KICKING" and not ((self.max_lift_ratio - both_feet_lift_ratio) > EARLY_REVERSAL_RATIO_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (WHEN BOTH FEET LAND BACK ON FLOOR)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["RETURNING", "TOP", "KICKING"] and single_foot_lift_ratio <= REP_COMPLETE_LIFT_RATIO:
            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to camera! Return to start to reset."
                is_bad_rep = True
            elif not self.both_feet_lifted:
                self.current_feedback = "Bad rep: Kick with BOTH feet together! Do not leave one foot on the floor."
                is_bad_rep = True
            elif not self.arms_supported:
                self.current_feedback = "Bad rep: Keep your arms locked straight with hands under shoulders!"
                is_bad_rep = True
            elif not self.knee_bent_90:
                self.current_feedback = "Bad rep: Keep your knees bent as you kick both heels up!"
                is_bad_rep = True
            elif not self.spine_neutral:
                self.current_feedback = "Bad rep: Keep your core braced! Do not collapse your hips."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged posture breakdown during the kick."
                is_bad_rep = True
            elif not self.target_height_reached:
                if self.max_lift_ratio >= HALF_KICK_LIFT_RATIO or self.half_height_reached:
                    self.current_feedback = "Bad rep: Half rep! Kick both heels higher up toward the ceiling."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Kick was too low! Drive both feet higher off the floor."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep or is_half_rep:
                self.bad_reps += 1
                print(
                    f"[MULE KICK REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_lift={self.max_lift_ratio:.2f}, knee={avg_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Two-Foot Mule Kick! Great bilateral lift and soft landing."
                print(
                    f"[MULE KICK REP #{rep_num} -> GOOD REP] "
                    f"max_lift={self.max_lift_ratio:.2f}, knee={avg_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "TABLETOP"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Form Score
        form_score = 100
        if not self.is_side_view: form_score -= 35
        if not self.both_feet_lifted: form_score -= 45
        if not self.knee_bent_90: form_score -= 30
        if not self.spine_neutral: form_score -= 30
        if not self.arms_supported: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 35
        if not self.target_height_reached and self.state == "RETURNING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Mule Kick", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": round(avg_knee_angle, 1), "hip_angle": round(avg_hip_angle, 1)
        }