import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (STANDING STRAIGHT LEG RAISE - SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Stand Tall, Strict Side View, Soft Standing Knee)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.28 # Strict side profile (16:9 webcam calibrated)
SETUP_MAX_HIP_SPAN_RATIO = 0.22      # Hips aligned sideways to camera
SETUP_HIP_MIN_ANGLE = 156.0          # Standing tall at start (allows toe pointed out ready to lift)
SETUP_KNEE_MIN_ANGLE = 148.0         # Step 1: Allows slight soft bend in standing knee
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames standing upright
UNLOCK_RESET_FRAMES = 12             # Frames held standing tall to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds (Step 1, 2, 3, 4, 6)
MAX_BACKWARD_SWING_RATIO = 0.12      # Catches swinging leg backward behind support ankle for momentum
MAX_TORSO_LEAN_DELTA = 14.0          # Step 4: Max degrees upper body can lean backward/forward from baseline
MIN_LIFT_KNEE_ANGLE = 150.0          # Step 3: Working leg must stay locked out straight (no knee bend)
MIN_SUPPORT_KNEE_ANGLE = 140.0       # Step 1: Allows soft bend in standing knee, catches deep knee buckling
ECCENTRIC_DROP_SPEED_MAX = 22.0      # Step 6: Lower over a controlled 2-second count (no heavy dropping)
EARLY_REVERSAL_DELTA = 8.0           # Early downward drop before reaching target height

# 3. Raise Height & Range of Motion (Step 4, 5, 6)
LIFTING_START_HIP_ANGLE = 150.0      # Raising leg in front initiates LIFTING
MIN_VALID_REP_FLEXION_ANGLE = 142.0  # Ignore minor standing posture shifts (> 142°)
HALF_RAISE_MAX_HIP_ANGLE = 135.0     # Lifting past 135° without reaching full height flags Half Rep (Bad Rep)
FULL_RAISE_MAX_HIP_ANGLE = 112.0     # Step 4 & 5: Leg raised high in front to hip/waist level (<= 112°)
LOWERING_START_HIP_ANGLE = 120.0     # Step 6: Lowering leg down from peak
REP_COMPLETE_HIP_ANGLE = 155.0       # Step 6: Touching toe lightly to floor completes the rep

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 3                 # Frames before flagging lost side view
DEBOUNCE_SWING = 3                   # Frames before flagging backward pendulum swing
DEBOUNCE_TORSO = 3                   # Frames before flagging backward torso lean
DEBOUNCE_LIFT_KNEE = 3               # Frames before flagging bent working knee
DEBOUNCE_SUPPORT_KNEE = 4            # Frames before flagging buckled support knee
DEBOUNCE_DROP = 2                    # Frames before flagging uncontrolled leg drop
DEBOUNCE_HALF_REP = 3                # Frames before confirming early reversal
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingActiveStraightLegRaiseTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility

        self.rep_started = False
        self.target_height_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0
        self.locked_active_leg = None      # Locks 'LEFT' or 'RIGHT' working leg during the rep

        # Posture Tripwires (STRICT SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.leg_swung_backward = False    # Catches pendulum momentum swing behind support leg
        self.lifted_knee_straight = True   # Step 3: Lock working knee out completely
        self.support_knee_straight = True  # Step 1: Soft bend allowed, no buckling
        self.torso_upright = True          # Step 2 & 4: Chest tall, do not lean backward
        self.controlled_descent = True     # Step 6: Controlled 2-second lowering
        self.ml_posture_maintained = True

        # Debounce counters
        self.facing_camera_frames = 0
        self.backward_swing_frames = 0
        self.lift_knee_frames = 0
        self.support_knee_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_torso_angle = None
        self.prev_hip_angle = None
        self.current_feedback = "Stand sideways to the camera: chest tall, hands on hips"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_active_straight_leg_raise.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[STANDING LEG RAISE ERROR] __init__ model_path={model_path} | error={exc}",
                    flush=True,
                )
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        ntu = np.zeros((25, 3), dtype=np.float32)

        def xyz(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        ntu[3] = xyz(0)
        ntu[4] = xyz(11)
        ntu[5] = xyz(13)
        ntu[6] = xyz(15)
        ntu[7] = xyz(19)
        ntu[8] = xyz(12)
        ntu[9] = xyz(14)
        ntu[10] = xyz(16)
        ntu[11] = xyz(20)
        ntu[12] = xyz(23)
        ntu[13] = xyz(25)
        ntu[14] = xyz(27)
        ntu[15] = xyz(31)
        ntu[16] = xyz(24)
        ntu[17] = xyz(26)
        ntu[18] = xyz(28)
        ntu[19] = xyz(32)

        ntu[0] = (ntu[12] + ntu[16]) / 2.0
        ntu[2] = (ntu[4] + ntu[8]) / 2.0
        ntu[1] = (ntu[0] + ntu[2]) / 2.0
        ntu[20] = ntu[2]

        ntu[21] = xyz(17)
        ntu[22] = xyz(21)
        ntu[23] = xyz(18)
        ntu[24] = xyz(22)
        return ntu

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.target_height_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0
        self.locked_active_leg = None
        self.reversal_frames = 0
        self.prev_hip_angle = None

        self.is_side_view = True
        self.leg_swung_backward = False
        self.lifted_knee_straight = True
        self.support_knee_straight = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.facing_camera_frames = 0
        self.backward_swing_frames = 0
        self.lift_knee_frames = 0
        self.support_knee_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_toe = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        r_toe = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Leg Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Show full side profile",
                "hip_angle": 0.0, "knee_angle": 0.0, "depth_reached": False
            }

        # 1. Strict Side Profile Verification (16:9 Webcam Calibrated)
        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0)) or 0.42
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        hip_span_x = abs(l_hip.x - r_hip.x)
        shoulder_aspect = shoulder_span_x / torso_height
        hip_aspect = hip_span_x / torso_height
        is_turned_sideways = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        facing_right = (l_toe.x > l_ankle.x) if l_vis > r_vis else (r_toe.x > r_ankle.x)

        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # 2. Locked Working Leg Selection (Prevents mid-rep leg swapping)
        if self.rep_started and self.locked_active_leg is not None:
            use_left_active = (self.locked_active_leg == "LEFT")
        else:
            if l_hip_angle < r_hip_angle - 6.0:
                use_left_active = True
            elif r_hip_angle < l_hip_angle - 6.0:
                use_left_active = False
            else:
                use_left_active = (l_hip_angle <= r_hip_angle)

        if use_left_active:
            active_hip_angle = l_hip_angle
            active_knee_angle = l_knee_angle
            support_knee_angle = r_knee_angle
            active_ankle, support_ankle = l_ankle, r_ankle
            active_knee_pt, support_knee_pt = l_knee, r_knee
            active_shoulder, active_hip = l_shoulder, l_hip
            support_leg_vis = (r_knee.visibility > 0.45 and r_ankle.visibility > 0.45)
        else:
            active_hip_angle = r_hip_angle
            active_knee_angle = r_knee_angle
            support_knee_angle = l_knee_angle
            active_ankle, support_ankle = r_ankle, l_ankle
            active_knee_pt, support_knee_pt = r_knee, l_knee
            active_shoulder, active_hip = r_shoulder, r_hip
            support_leg_vis = (l_knee.visibility > 0.45 and l_ankle.visibility > 0.45)

        # 3. Torso Inclination Relative to Vertical
        primary_shoulder = r_shoulder if r_vis >= l_vis else l_shoulder
        primary_hip = r_hip if r_vis >= l_vis else l_hip
        dx_spine = primary_shoulder.x - primary_hip.x
        dy_spine = max(abs(primary_shoulder.y - primary_hip.y), 0.10)
        spine_tilt_deg = np.degrees(np.arctan2(dx_spine, dy_spine))

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1 & 2: Stand Tall in Strict Side View)
        # ---------------------------------------------------------
        is_standing_tall = (
            is_turned_sideways
            and (l_hip_angle > SETUP_HIP_MIN_ANGLE)
            and (r_hip_angle > SETUP_HIP_MIN_ANGLE)
            and (l_knee_angle > SETUP_KNEE_MIN_ANGLE)
            and (r_knee_angle > SETUP_KNEE_MIN_ANGLE)
        )

        if is_standing_tall:
            self.standing_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = spine_tilt_deg
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * spine_tilt_deg

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_turned_sideways:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            else:
                feedback_msg = "Stand tall sideways with hands on hips to begin"
            return {
                "exercise": "Standing Leg Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(min(l_hip_angle, r_hip_angle), 1),
                "knee_angle": round(min(l_knee_angle, r_knee_angle), 1),
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
                                f"[STANDING LEG RAISE FAULT] AI ST-GCN flagged posture breakdown: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[STANDING LEG RAISE ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES (ACTIVE ACROSS LIFTING, PEAK, AND LOWERING)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LIFTING", "PEAK", "LOWERING"]:
            self.min_hip_angle = min(self.min_hip_angle, active_hip_angle)

            # A. Strict Side Profile Tripwire
            if not is_turned_sideways:
                self.facing_camera_frames += 1
                if self.facing_camera_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[STANDING LEG RAISE FAULT] Lost side view: shoulder_aspect={shoulder_aspect:.2f}, "
                            f"hip_aspect={hip_aspect:.2f}",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.facing_camera_frames = max(0, self.facing_camera_frames - 1)

            # B. Directional Backward Pendulum Swing Tripwire (Only when support leg is visible)
            if support_leg_vis:
                if facing_right:
                    ankle_backward_drift = support_ankle.x - active_ankle.x
                    knee_backward_drift = support_knee_pt.x - active_knee_pt.x
                else:
                    ankle_backward_drift = active_ankle.x - support_ankle.x
                    knee_backward_drift = active_knee_pt.x - support_knee_pt.x

                is_swinging_back = (
                    (ankle_backward_drift > MAX_BACKWARD_SWING_RATIO * torso_height)
                    or (knee_backward_drift > MAX_BACKWARD_SWING_RATIO * torso_height)
                )
                if is_swinging_back:
                    self.backward_swing_frames += 1
                    if self.backward_swing_frames >= DEBOUNCE_SWING:
                        if not self.leg_swung_backward:
                            print(
                                f"[STANDING LEG RAISE FAULT] Leg swung backward behind support leg: "
                                f"drift={ankle_backward_drift:.2f}",
                                flush=True,
                            )
                        self.leg_swung_backward = True
                else:
                    self.backward_swing_frames = max(0, self.backward_swing_frames - 1)

            # C. Step 2 & 4: Upright Torso Check (Catches leaning backward to heave leg up)
            if self.baseline_torso_angle is not None:
                spine_delta = abs(spine_tilt_deg - self.baseline_torso_angle)
                if spine_delta > MAX_TORSO_LEAN_DELTA:
                    self.torso_frames += 1
                    if self.torso_frames >= DEBOUNCE_TORSO:
                        if self.torso_upright:
                            print(
                                f"[STANDING LEG RAISE FAULT] Upper body leaned backward/forward: "
                                f"delta={spine_delta:.1f}° (max={MAX_TORSO_LEAN_DELTA}°)",
                                flush=True,
                            )
                        self.torso_upright = False
                else:
                    self.torso_frames = max(0, self.torso_frames - 1)

            # D. Step 3: Working Leg Locked Straight (No bend in moving knee)
            if active_knee_angle < MIN_LIFT_KNEE_ANGLE:
                self.lift_knee_frames += 1
                if self.lift_knee_frames >= DEBOUNCE_LIFT_KNEE:
                    if self.lifted_knee_straight:
                        print(
                            f"[STANDING LEG RAISE FAULT] Working knee bent: angle={active_knee_angle:.1f}° "
                            f"(min={MIN_LIFT_KNEE_ANGLE}°)",
                            flush=True,
                        )
                    self.lifted_knee_straight = False
            else:
                self.lift_knee_frames = max(0, self.lift_knee_frames - 1)

            # E. Step 1: Standing Leg Stability (Soft bend allowed, catches deep knee buckling)
            if support_leg_vis and support_knee_angle < MIN_SUPPORT_KNEE_ANGLE:
                self.support_knee_frames += 1
                if self.support_knee_frames >= DEBOUNCE_SUPPORT_KNEE:
                    if self.support_knee_straight:
                        print(
                            f"[STANDING LEG RAISE FAULT] Standing leg buckled: support_knee={support_knee_angle:.1f}°",
                            flush=True,
                        )
                    self.support_knee_straight = False
            else:
                self.support_knee_frames = max(0, self.support_knee_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_tall:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Lock working quad and lift leg straight out in front."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Standing Leg Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "hip_angle": round(active_hip_angle, 1), "knee_angle": round(active_knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock working quad, flex foot, and raise leg straight out in front!"

        # Step 4: Initiate Controlled Lift
        elif self.state in ["STANDING", "LOWERING"] and min(l_hip_angle, r_hip_angle) <= LIFTING_START_HIP_ANGLE:
            self.state = "LIFTING"
            self.rep_started = True
            self.locked_active_leg = "LEFT" if (l_hip_angle <= r_hip_angle) else "RIGHT"
            self.min_hip_angle = active_hip_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lift straight leg up without leaning your upper body back..."

        # Step 5: Peak Reached & Squeeze (<= 112°)
        elif self.state == "LIFTING" and active_hip_angle <= FULL_RAISE_MAX_HIP_ANGLE:
            self.target_height_reached = True
            self.state = "PEAK"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Squeeze hip & quad at peak! Slowly lower over 2 seconds."

        # Early downward reversal (Half Rep Branch -> Routed to Bad Rep)
        elif self.state == "LIFTING" and not self.target_height_reached and (active_hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_hip_angle <= HALF_RAISE_MAX_HIP_ANGLE:
                    self.was_half_rep = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete raise! Lowering leg early..."
                print(
                    f"[STANDING LEG RAISE FAULT] Early reversal before hip level: min_hip={self.min_hip_angle:.1f}° "
                    f"(required<={FULL_RAISE_MAX_HIP_ANGLE}°)",
                    flush=True,
                )

        # Step 6: Lower Under Tension from Peak
        elif self.state == "PEAK" and active_hip_angle > LOWERING_START_HIP_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower leg slowly under tension (do not drop heavily)..."

        if self.state == "LIFTING" and not ((active_hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 6: Eccentric drop speed detector
        if self.rep_started and self.state in ["LOWERING", "LIFTING"]:
            if self.prev_hip_angle is not None:
                drop_speed = active_hip_angle - self.prev_hip_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[STANDING LEG RAISE FAULT] Leg dropped too fast: speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_hip_angle = active_hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and active_hip_angle >= REP_COMPLETE_HIP_ANGLE and self.state in ["LOWERING", "PEAK", "LIFTING"]:
            # Ignore minor standing shifts (> 142°)
            if self.min_hip_angle > MIN_VALID_REP_FLEXION_ANGLE and self.is_side_view and self.lifted_knee_straight:
                self._reset_rep_tracking()
                self.state = "STANDING"
                return {
                    "exercise": "Standing Leg Raise", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "hip_angle": round(active_hip_angle, 1), "knee_angle": round(active_knee_angle, 1),
                    "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Do not face front."
                is_bad_rep = True
            elif self.leg_swung_backward:
                self.current_feedback = "Bad rep: Do not swing leg backward for momentum! Lift straight from the front."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Do not lean your upper body backward! Keep chest tall."
                is_bad_rep = True
            elif not self.lifted_knee_straight:
                self.current_feedback = "Bad rep: Squeeze your quad to lock the moving knee completely straight!"
                is_bad_rep = True
            elif not self.support_knee_straight:
                self.current_feedback = "Bad rep: Keep your standing leg firm (only a slight soft bend)!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower your leg slowly over a 2-second count! Do not drop it."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected posture or balance breakdown."
                is_bad_rep = True
            elif self.was_half_rep or not self.target_height_reached:
                if self.min_hip_angle <= HALF_RAISE_MAX_HIP_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Lift your straight leg all the way to hip level."
                else:
                    self.current_feedback = "Bad rep: Leg raise was too shallow! Lift higher."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[STANDING LEG RAISE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_hip={self.min_hip_angle:.1f}°, knee={active_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Straight Leg Raise! Great quad lockout and controlled lowering."
                print(
                    f"[STANDING LEG RAISE REP #{rep_num} -> GOOD REP] "
                    f"min_hip={self.min_hip_angle:.1f}°, knee={active_knee_angle:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(active_hip_angle - 90) * 0.4))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 35
        if self.leg_swung_backward: form_score -= 30
        if not self.torso_upright: form_score -= 30
        if not self.lifted_knee_straight: form_score -= 30
        if not self.support_knee_straight: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if (self.was_half_rep or not self.target_height_reached) and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Leg Raise", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "hip_angle": round(active_hip_angle, 1),
            "knee_angle": round(active_knee_angle, 1),
            "depth_reached": self.target_height_reached
        }