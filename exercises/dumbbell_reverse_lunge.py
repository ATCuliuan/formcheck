import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SAGITTAL / SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1: Stand Tall, Feet Together, Arms at Sides)
SETUP_KNEE_MIN_ANGLE = 160.0         # Knees extended to confirm tall standing posture
SETUP_HIP_MIN_ANGLE = 155.0          # Hips extended to confirm tall standing posture
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.45 # Max shoulder width / torso length for profile view
SETUP_MAX_ANKLE_SPREAD_RATIO = 0.35  # Max horizontal ankle distance / torso_len when standing tall
MIN_LUNGE_STRIDE_RATIO = 0.45        # Minimum horizontal ankle separation required during Step 2 (Step Backward)
MIN_ARM_HANG_ELBOW_ANGLE = 125.0     # Dumbbells held down at sides (prevents curling weights up)
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames standing upright to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames standing tall to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum landmark visibility for profile detection

# 2. Kinematic & Form Thresholds
MAX_TORSO_FORWARD_LEAN_DEG = 28.0    # Max torso forward tilt from vertical (replaces buggy hip-knee angle)
MAX_FRONT_SHIN_DISPLACEMENT = 0.32   # Max horizontal drift between front knee and ankle (vertical shin check)
ECCENTRIC_DROP_SPEED_MAX = 25.0      # Maximum descent speed (deg/frame) before flagging free-fall

# 3. Lunge Depth & Range of Motion (Step 3: Both Knees ~90°)
FULL_LUNGE_BACK_KNEE_ANGLE = 105.0   # Back knee bending to <= 105° confirms deep lunge
FULL_LUNGE_FRONT_KNEE_ANGLE = 110.0  # Front knee bending to <= 110° confirms both knees bent ~90°
HALF_LUNGE_BACK_KNEE_ANGLE = 130.0   # Reaching between 105° and 130° flags half rep (counted as bad rep)
DESCENT_START_KNEE_ANGLE = 145.0     # Dropping below this initiates DESCENDING
ASCENT_START_KNEE_ANGLE = 110.0      # Rising from bottom initiates ASCENDING
EARLY_REVERSAL_DELTA = 8.0           # Early upward bounce delta before reaching depth
REP_COMPLETE_KNEE_ANGLE = 158.0      # Standing extension lockout to complete rep

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4                 # Frames before flagging lost side profile
DEBOUNCE_TORSO = 4                   # Frames before flagging torso lean
DEBOUNCE_SHIN = 4                    # Frames before flagging forward front knee drift
DEBOUNCE_ARMS = 4                    # Frames before flagging curled dumbbells
DEBOUNCE_DROP = 2                    # Frames before flagging uncontrolled drop
DEBOUNCE_HALF_REP = 3                # Frames before confirming premature reversal
DEBOUNCE_ML_FAULT = 5                # Consecutive frames required to trip ML fault

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 # Softmax probability threshold

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40        # Feedback display duration post-rep

# =====================================================================


class DumbbellReverseLungeTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_depth_reached = False
        self.stride_achieved = False
        self.min_back_knee_angle = 180.0
        self.min_front_knee_angle = 180.0
        self.locked_back_leg = None          # Locks 'left' or 'right' during a rep to prevent mid-rep jitter swaps

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.torso_upright = True
        self.front_shin_vertical = True
        self.arms_at_sides = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.torso_frames = 0
        self.shin_frames = 0
        self.arm_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_knee_angle = None
        self.current_feedback = "Stand sideways to camera with feet together and dumbbells at sides"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_reverse_lunge.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[REVERSE LUNGE ERROR] __init__ model_path={model_path} | error={exc}",
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
        self.depth_reached = False
        self.half_depth_reached = False
        self.stride_achieved = False
        self.min_back_knee_angle = 180.0
        self.min_front_knee_angle = 180.0
        self.locked_back_leg = None
        self.reversal_frames = 0
        self.prev_knee_angle = None

        self.is_side_view = True
        self.torso_upright = True
        self.front_shin_vertical = True
        self.arms_at_sides = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.torso_frames = 0
        self.shin_frames = 0
        self.arm_frames = 0
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full lower body not visible",
                "front_knee": 0.0,
                "back_knee": 0.0,
                "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        # Side profile & stride metrics
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Horizontal foot separation (Step 1 feet together vs Step 2 backward lunge stride)
        ankle_spread_ratio = abs(l_ankle.x - r_ankle.x) / torso_len
        feet_together = ankle_spread_ratio <= SETUP_MAX_ANKLE_SPREAD_RATIO
        has_lunge_stride = ankle_spread_ratio >= MIN_LUNGE_STRIDE_RATIO

        # Arm hang check (Step 1: Hold dumbbells at sides)
        active_elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        is_arm_hanging = (active_elbow_angle >= MIN_ARM_HANG_ELBOW_ANGLE) or (active_wrist.y >= active_hip.y - 0.05)

        # True vertical torso lean angle (0° = dead vertical upright)
        torso_dx = abs(active_shoulder.x - active_hip.x)
        torso_dy = max(abs(active_hip.y - active_shoulder.y), 0.10)
        torso_lean_deg = np.degrees(np.arctan2(torso_dx, torso_dy))
        is_torso_tall = torso_lean_deg <= MAX_TORSO_FORWARD_LEAN_DEG

        # Bilateral knee angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Lock active back leg once descending so mid-rep jitter doesn't swap legs
        if self.rep_started and self.locked_back_leg is not None:
            use_left_as_back = (self.locked_back_leg == "left")
        else:
            use_left_as_back = (l_knee.y > r_knee.y)

        if use_left_as_back:
            back_knee_angle = l_knee_angle
            front_knee_angle = r_knee_angle
            front_knee = r_knee
            front_ankle = r_ankle
        else:
            back_knee_angle = r_knee_angle
            front_knee_angle = l_knee_angle
            front_knee = l_knee
            front_ankle = l_ankle

        # Front shin verticality check (prevents front knee shooting over toe)
        front_shin_len = np.hypot(front_knee.x - front_ankle.x, front_knee.y - front_ankle.y) or 0.35
        front_shin_drift = abs(front_knee.x - front_ankle.x) / front_shin_len
        is_shin_vertical = front_shin_drift <= MAX_FRONT_SHIN_DISPLACEMENT

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1: Stand Tall, Feet Together, Dumbbells at Sides)
        # ---------------------------------------------------------
        is_standing = (
            (front_knee_angle > SETUP_KNEE_MIN_ANGLE)
            and (back_knee_angle > SETUP_KNEE_MIN_ANGLE)
            and feet_together
            and is_torso_tall
            and is_arm_hanging
            and is_side
        )

        if is_standing:
            self.standing_frames += 1
            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif not feet_together:
                feedback_msg = "Stand tall with feet hip-width apart (aligned side-by-side)"
            elif not is_arm_hanging:
                feedback_msg = "Hold dumbbells down at your sides"
            else:
                feedback_msg = "Stand tall with chest up to begin reverse lunge"
            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "front_knee": round(front_knee_angle, 1),
                "back_knee": round(back_knee_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER (SOFTMAX + DEBOUNCE)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[REVERSE LUNGE FAULT] AI ST-GCN flagged bad posture: prob={bad_posture_prob:.2f} "
                                f"(threshold={ML_FAULT_CONFIDENCE_THRESHOLD})",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[REVERSE LUNGE ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            if has_lunge_stride:
                self.stride_achieved = True

            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[REVERSE LUNGE FAULT] Lost side view: shoulder_span_ratio={shoulder_span_ratio:.2f} "
                            f"(max={SETUP_MAX_SHOULDER_SPAN_RATIO})",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Torso collapse forward check (measured against true vertical)
            if not is_torso_tall:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_upright:
                        print(
                            f"[REVERSE LUNGE FAULT] Torso leaned forward: torso_lean_deg={torso_lean_deg:.1f}° "
                            f"(max={MAX_TORSO_FORWARD_LEAN_DEG}°)",
                            flush=True,
                        )
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # Front shin verticality check
            if not is_shin_vertical:
                self.shin_frames += 1
                if self.shin_frames >= DEBOUNCE_SHIN:
                    if self.front_shin_vertical:
                        print(
                            f"[REVERSE LUNGE FAULT] Front shin drifted past ankle: drift={front_shin_drift:.2f} "
                            f"(max={MAX_FRONT_SHIN_DISPLACEMENT})",
                            flush=True,
                        )
                    self.front_shin_vertical = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # Arms at sides check
            if not is_arm_hanging:
                self.arm_frames += 1
                if self.arm_frames >= DEBOUNCE_ARMS:
                    if self.arms_at_sides:
                        print(
                            f"[REVERSE LUNGE FAULT] Dumbbells raised/curled: elbow_angle={active_elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.arms_at_sides = False
            else:
                self.arm_frames = max(0, self.arm_frames - 1)

            self.min_back_knee_angle = min(self.min_back_knee_angle, back_knee_angle)
            self.min_front_knee_angle = min(self.min_front_knee_angle, front_knee_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Step backward into a lunge."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "front_knee": round(front_knee_angle, 1),
                "back_knee": round(back_knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Step backward and lower both knees to ~90°"

        # Step 2: Step backward & lower body initiates
        elif self.state in ["STANDING", "ASCENDING"] and back_knee_angle < DESCENT_START_KNEE_ANGLE:
            self.state = "DESCENDING"
            self.rep_started = True
            self.locked_back_leg = "left" if (l_knee.y > r_knee.y) else "right"
            self.stride_achieved = has_lunge_stride
            self.min_back_knee_angle = back_knee_angle
            self.min_front_knee_angle = front_knee_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower straight down, keep torso upright..."

        # Step 3: Full Lunge depth (Both knees bent ~90°: back knee <= 105° and front knee <= 110°)
        elif self.state == "DESCENDING" and back_knee_angle <= FULL_LUNGE_BACK_KNEE_ANGLE and front_knee_angle <= FULL_LUNGE_FRONT_KNEE_ANGLE:
            self.depth_reached = True
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Good 90° depth! Drive through front heel and step forward."

        # Premature upward reversal (Half rep branch)
        elif self.state == "DESCENDING" and not self.depth_reached and (back_knee_angle - self.min_back_knee_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_back_knee_angle <= HALF_LUNGE_BACK_KNEE_ANGLE:
                    self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Incomplete depth! Rising early..."
                print(
                    f"[REVERSE LUNGE FAULT] Early reversal before 90° depth: "
                    f"min_back_knee={self.min_back_knee_angle:.1f}°, min_front_knee={self.min_front_knee_angle:.1f}°",
                    flush=True,
                )

        # Step 4: Driving upward from bottom
        elif self.state == "BOTTOM" and back_knee_angle > ASCENT_START_KNEE_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Push through front heel, step back foot forward to start"

        # Decay reversal debounce
        if self.state == "DESCENDING" and not ((back_knee_angle - self.min_back_knee_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop monitor
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_knee_angle is not None:
                drop_speed = self.prev_knee_angle - back_knee_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[REVERSE LUNGE FAULT] Uncontrolled descent speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_knee_angle = back_knee_angle
        else:
            self.prev_knee_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if (
            self.rep_started
            and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"]
            and front_knee_angle >= REP_COMPLETE_KNEE_ANGLE
            and back_knee_angle >= REP_COMPLETE_KNEE_ANGLE
            and feet_together
        ):
            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to camera! Stand tall to reset."
                is_bad_rep = True
            elif not self.stride_achieved:
                self.current_feedback = "Bad rep: Take a full step backward into a lunge! Stand tall to reset."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Torso leaned too far forward! Keep chest upright. Stand tall to reset."
                is_bad_rep = True
            elif not self.front_shin_vertical:
                self.current_feedback = "Bad rep: Front knee drifted forward past ankle! Keep front shin vertical."
                is_bad_rep = True
            elif not self.arms_at_sides:
                self.current_feedback = "Bad rep: Keep dumbbells hanging at your sides! Stand tall to reset."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control your descent! Do not crash your knee down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor lunge posture. Stand tall to reset."
                is_bad_rep = True
            elif not self.depth_reached:
                if self.min_back_knee_angle <= HALF_LUNGE_BACK_KNEE_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Bad rep: Half rep! Bend both knees to ~90°. Stand tall to reset."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Lunge was too shallow! Bend both knees to ~90°. Stand tall to reset."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[REVERSE LUNGE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_back={self.min_back_knee_angle:.1f}°, min_front={self.min_front_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.bad_reps += 1  # Flag half rep as bad rep and lock state
                print(
                    f"[REVERSE LUNGE REP #{rep_num} -> BAD REP (HALF REP)] "
                    f"Reason: {self.current_feedback} | min_back={self.min_back_knee_angle:.1f}°, min_front={self.min_front_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean reverse lunge! Great 90° depth and drive."
                print(
                    f"[REVERSE LUNGE REP #{rep_num} -> GOOD REP] "
                    f"min_back={self.min_back_knee_angle:.1f}°, min_front={self.min_front_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((160 - back_knee_angle) / 60 * 100))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 35
        if not self.torso_upright: form_score -= 30
        if not self.front_shin_vertical: form_score -= 30
        if not self.arms_at_sides: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Reverse Lunge",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "front_knee": round(front_knee_angle, 1),
            "back_knee": round(back_knee_angle, 1),
            "depth_reached": self.depth_reached
        }