import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIDE / SAGITTAL VIEW - SOFT KNEE HINGE)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Stand Tall, Unlocked Soft Knees)
SETUP_HIP_MIN_ANGLE = 160.0          # Hip angle required to confirm standing tall
SETUP_KNEE_MIN_ANGLE = 148.0         # Lower bound for starting soft knee bend
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.30 # Max shoulder width / torso length for valid side view
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames standing upright to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames standing tall to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum landmark visibility for profile detection

# 2. Knee-to-Waist (Thigh) Soft Bend & Freeze Thresholds
MAX_STRAIGHT_KNEE_ANGLE = 173.0      # Above 173° means knees are locked straight (no soft bend!)
MIN_KNEE_ANGLE = 135.0               # Bending below 135° turns the hip hinge into a squat
MAX_KNEE_FREEZE_DELTA = 16.0         # Max degrees knee angle can deviate from initial soft bend
MIN_THIGH_SLANT_DEG = 6.0            # Minimum knee-to-waist (femur) angle from vertical during hinge

# 3. Kinematic & Form Thresholds (Flat Back, Vertical Shins, Controlled Tempo)
MIN_SPINE_NECK_ANGLE = 125.0         # Thoracic/cervical spine alignment threshold
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Maximum drop speed (degrees/frame) before flagging free-fall
MAX_SHIN_DISPLACEMENT_RATIO = 0.25   # Max horizontal shin forward drift / shin length

# 4. Hinge Depth & Range of Motion (Torso Near Parallel -> Drive Hips Forward)
FULL_HINGE_MAX_HIP_ANGLE = 108.0     # Peak hip hinge angle required for full rep credit
MIN_PARALLEL_TORSO_TILT_DEG = 58.0   # Torso forward tilt from vertical (~60°–90° = near parallel)
HALF_HINGE_THRESHOLD_ANGLE = 135.0   # Hinging down past 135° but short of parallel flags a half-rep (bad rep)
HINGE_START_HIP_ANGLE = 152.0        # Transition threshold from STANDING/EXTENDING to HINGING
EXTENDING_START_HIP_ANGLE = 115.0    # Moving above this angle transitions from PEAK_HINGE to EXTENDING
EARLY_REVERSAL_DELTA = 8.0           # Upward bounce in degrees before reaching full depth
REP_COMPLETE_HIP_ANGLE = 160.0       # Standing lockout angle to evaluate rep completion

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4                 # Frames before flagging lost side profile
DEBOUNCE_SHIN = 4                    # Frames before flagging forward shin/knee drift
DEBOUNCE_SQUAT = 4                   # Frames before flagging squatting cheat / unfrozen knee bend
DEBOUNCE_LOCKED_KNEE = 5             # Frames before flagging stiff/locked vertical knees
DEBOUNCE_SPINE = 4                   # Frames before flagging spinal rounding
DEBOUNCE_DROP = 2                    # Frames before flagging fast/uncontrolled hinge
DEBOUNCE_HALF_REP = 3                # Frames before flagging premature upward reversal

# 6. ST-GCN Machine Learning Model Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.90 # Required probability (>= 75%) before considering ML breakdown
DEBOUNCE_ML_FAULT = 5                # Consecutive frames required above threshold to trip ML fault

# 7. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40        # Feedback display duration post-rep

# =====================================================================


class GoodMorningTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.target_hinge_reached = False
        self.half_hinge_reached = False
        self.min_hip_angle = 180.0
        self.max_torso_tilt = 0.0
        self.baseline_knee_angle = None

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.vertical_shins = True         # Shins stay vertical (knees do not shoot forward)
        self.soft_knee_maintained = True   # Knees stay unlocked with thigh slanted back (not vertical/stiff)
        self.not_squatting = True          # Knee angle stays frozen (does not keep bending into a squat)
        self.neutral_spine = True          # Back stays flat and neutral
        self.controlled_hinge = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.shin_frames = 0
        self.squat_frames = 0
        self.locked_knee_frames = 0
        self.spine_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_hip_angle = None
        self.current_feedback = "Stand sideways and unlock your knees (soft 10-15° bend)"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_good_morning.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[GOOD MORNING ERROR] __init__ model_path={model_path} | error={exc}",
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
        self.target_hinge_reached = False
        self.half_hinge_reached = False
        self.min_hip_angle = 180.0
        self.max_torso_tilt = 0.0
        self.prev_hip_angle = None

        self.is_side_view = True
        self.vertical_shins = True
        self.soft_knee_maintained = True
        self.not_squatting = True
        self.neutral_spine = True
        self.controlled_hinge = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.shin_frames = 0
        self.squat_frames = 0
        self.locked_knee_frames = 0
        self.spine_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
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
                "exercise": "Good Morning",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Show full side profile",
                "hip_angle": 0.0,
                "knee_angle": 0.0,
                "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal joint angles
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        spine_neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Torso forward inclination angle from vertical (0° = upright, 90° = parallel to floor)
        torso_dx = abs(active_shoulder.x - active_hip.x)
        torso_dy = max(abs(active_hip.y - active_shoulder.y), 0.05)
        torso_tilt_deg = np.degrees(np.arctan2(torso_dx, torso_dy))

        # Knee-to-Waist (Thigh / Femur) angle from vertical (0° = dead vertical, >6° = angled back with soft bend)
        thigh_dx = abs(active_hip.x - active_knee.x)
        thigh_dy = max(abs(active_knee.y - active_hip.y), 0.05)
        thigh_tilt_deg = np.degrees(np.arctan2(thigh_dx, thigh_dy))

        # Profile check
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Vertical shin check (prevents squatting knees forward over toes)
        shin_len = np.hypot(active_knee.x - active_ankle.x, active_knee.y - active_ankle.y) or 0.35
        horizontal_shin_drift = abs(active_knee.x - active_ankle.x) / shin_len
        shins_vertical = horizontal_shin_drift <= MAX_SHIN_DISPLACEMENT_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Stand Tall, Soft Knee Bend)
        # ---------------------------------------------------------
        is_standing = (
            (hip_angle > SETUP_HIP_MIN_ANGLE)
            and (knee_angle > SETUP_KNEE_MIN_ANGLE)
            and is_side
            and shins_vertical
        )

        if is_standing:
            self.standing_frames += 1
            # Record the starting soft-knee angle so we can verify it stays frozen during the hinge
            if self.baseline_knee_angle is None:
                self.baseline_knee_angle = min(knee_angle, 168.0)
            else:
                self.baseline_knee_angle = 0.90 * self.baseline_knee_angle + 0.10 * min(knee_angle, 168.0)

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif not shins_vertical:
                feedback_msg = "Keep shins vertical with a soft bend in your knees"
            else:
                feedback_msg = "Stand tall sideways and unlock your knees to begin"

            return {
                "exercise": "Good Morning",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1),
                "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
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
                                f"[GOOD MORNING FAULT] AI ST-GCN flagged bad posture: prob={bad_posture_prob:.2f} "
                                f"(threshold={ML_FAULT_CONFIDENCE_THRESHOLD})",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[GOOD MORNING ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["HINGING", "PEAK_HINGE", "EXTENDING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[GOOD MORNING FAULT] Lost side view: shoulder_span_ratio={shoulder_span_ratio:.2f} "
                            f"(max={SETUP_MAX_SHOULDER_SPAN_RATIO})",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # A. Soft Knee Bend & Thigh Slant Check (Catches locked/vertical knees when hinging)
            if torso_tilt_deg > 30.0:
                is_knee_locked_vertical = (knee_angle > MAX_STRAIGHT_KNEE_ANGLE) or (thigh_tilt_deg < MIN_THIGH_SLANT_DEG)
                if is_knee_locked_vertical:
                    self.locked_knee_frames += 1
                    if self.locked_knee_frames >= DEBOUNCE_LOCKED_KNEE:
                        if self.soft_knee_maintained:
                            print(
                                f"[GOOD MORNING FAULT] Knees locked/vertical instead of soft bend: "
                                f"knee_angle={knee_angle:.1f}° (max={MAX_STRAIGHT_KNEE_ANGLE}°), "
                                f"thigh_tilt={thigh_tilt_deg:.1f}° (min={MIN_THIGH_SLANT_DEG}°)",
                                flush=True,
                            )
                        self.soft_knee_maintained = False
                else:
                    self.locked_knee_frames = max(0, self.locked_knee_frames - 1)

            # B. Hold the Angle / Squatting Cheat (Catches continuing to bend knees during the hinge)
            knee_bent_too_much = (knee_angle < MIN_KNEE_ANGLE) or (
                self.baseline_knee_angle is not None and (self.baseline_knee_angle - knee_angle) > MAX_KNEE_FREEZE_DELTA
            )
            if knee_bent_too_much:
                self.squat_frames += 1
                if self.squat_frames >= DEBOUNCE_SQUAT:
                    if self.not_squatting:
                        print(
                            f"[GOOD MORNING FAULT] Squatting / failed to freeze knee angle: "
                            f"knee_angle={knee_angle:.1f}° (min={MIN_KNEE_ANGLE}°, baseline={self.baseline_knee_angle:.1f}°)",
                            flush=True,
                        )
                    self.not_squatting = False
            else:
                self.squat_frames = max(0, self.squat_frames - 1)

            # C. Foot-to-Knee verticality check (catches forward knee drift during hinge)
            if not shins_vertical:
                self.shin_frames += 1
                if self.shin_frames >= DEBOUNCE_SHIN:
                    if self.vertical_shins:
                        print(
                            f"[GOOD MORNING FAULT] Shins drifted forward: drift={horizontal_shin_drift:.2f} "
                            f"(max={MAX_SHIN_DISPLACEMENT_RATIO})",
                            flush=True,
                        )
                    self.vertical_shins = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # D. Rounded spine/neck check
            if spine_neck_angle < MIN_SPINE_NECK_ANGLE:
                self.spine_frames += 1
                if self.spine_frames >= DEBOUNCE_SPINE:
                    if self.neutral_spine:
                        print(
                            f"[GOOD MORNING FAULT] Spine/neck rounded: spine_neck_angle={spine_neck_angle:.1f}° "
                            f"(min={MIN_SPINE_NECK_ANGLE}°)",
                            flush=True,
                        )
                    self.neutral_spine = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

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
                    self.current_feedback = "Reset complete! Unlock knees (10-15° bend) and push hips back."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Good Morning",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "hip_angle": round(hip_angle, 1),
                "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Unlock knees (soft bend), freeze knee angle, and push hips back!"

        # Begin hinging forward at the hips
        elif self.state in ["STANDING", "EXTENDING"] and hip_angle < HINGE_START_HIP_ANGLE:
            self.state = "HINGING"
            self.rep_started = True
            self.min_hip_angle = hip_angle
            self.max_torso_tilt = torso_tilt_deg
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Push hips back, hold soft knee bend frozen..."

        # Full ROM: Hip angle reaches <= 108° OR torso tilt >= 58° (nearly parallel to floor)
        elif self.state == "HINGING" and (hip_angle <= FULL_HINGE_MAX_HIP_ANGLE or torso_tilt_deg >= MIN_PARALLEL_TORSO_TILT_DEG):
            self.target_hinge_reached = True
            self.state = "PEAK_HINGE"
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)
            self.max_torso_tilt = max(self.max_torso_tilt, torso_tilt_deg)
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Parallel stretch reached! Drive heels and squeeze glutes to stand."

        # Premature reversal: User rises before reaching parallel torso (Half rep branch)
        elif self.state == "HINGING" and not self.target_hinge_reached and (hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                if self.min_hip_angle <= HALF_HINGE_THRESHOLD_ANGLE or self.max_torso_tilt >= 40.0:
                    self.half_hinge_reached = True
                self.state = "EXTENDING"
                self.current_feedback = "Incomplete hinge! Rising early..."
                print(
                    f"[GOOD MORNING FAULT] Early reversal before parallel: min_hip_angle={self.min_hip_angle:.1f}°, "
                    f"max_torso_tilt={self.max_torso_tilt:.1f}°",
                    flush=True,
                )

        # Drive hips forward to return to start
        elif self.state == "PEAK_HINGE" and hip_angle > EXTENDING_START_HIP_ANGLE:
            self.state = "EXTENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through heels and squeeze glutes to stand tall"

        # Decay reversal frames if not reversing
        if self.state == "HINGING" and not ((hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA):
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Track minimum hip angle and max torso tilt while in hinging phases
        if self.state in ["HINGING", "PEAK_HINGE"]:
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)
            self.max_torso_tilt = max(self.max_torso_tilt, torso_tilt_deg)

        # Eccentric speed monitor with decaying debounce
        if self.rep_started and self.state in ["EXTENDING", "HINGING"]:
            if self.prev_hip_angle is not None:
                drop_speed = self.prev_hip_angle - hip_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_hinge:
                            print(
                                f"[GOOD MORNING FAULT] Uncontrolled hinge drop speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_hinge = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_hip_angle = hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.state in ["EXTENDING", "PEAK_HINGE", "HINGING"] and self.rep_started and hip_angle >= REP_COMPLETE_HIP_ANGLE:
            is_bad_rep = False
            is_half_rep = False

            # Form Cheats Evaluated First
            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Stand tall to reset."
                is_bad_rep = True
            elif not self.soft_knee_maintained:
                self.current_feedback = "Bad rep: Unlock your knees! Keep a soft 10-15° bend and push hips back."
                is_bad_rep = True
            elif not self.not_squatting:
                self.current_feedback = "Bad rep: Freeze your knee angle! Do not bend knees further into a squat."
                is_bad_rep = True
            elif not self.vertical_shins:
                self.current_feedback = "Bad rep: Keep shins vertical! Push hips back instead of knees forward."
                is_bad_rep = True
            elif not self.neutral_spine:
                self.current_feedback = "Bad rep: Keep your back flat and neutral! Do not round spine."
                is_bad_rep = True
            elif not self.controlled_hinge:
                self.current_feedback = "Bad rep: Control your torso descent! Stand tall to reset."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged spinal rounding or hinge breakdown."
                is_bad_rep = True
            elif not self.target_hinge_reached:
                if self.min_hip_angle <= HALF_HINGE_THRESHOLD_ANGLE or self.half_hinge_reached:
                    self.current_feedback = "Bad rep: Half rep! Hinge torso down until nearly parallel to floor. Stand tall to reset."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Hinge was too shallow! Lower torso toward parallel to reset."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[GOOD MORNING REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_hip={self.min_hip_angle:.1f}°, max_tilt={self.max_torso_tilt:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.bad_reps += 1  # Flag half rep as bad rep and lock state
                print(
                    f"[GOOD MORNING REP #{rep_num} -> BAD REP (HALF REP)] "
                    f"Reason: {self.current_feedback} | min_hip={self.min_hip_angle:.1f}°, max_tilt={self.max_torso_tilt:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Good Morning! Great frozen soft-knee hinge."
                print(
                    f"[GOOD MORNING REP #{rep_num} -> GOOD REP] "
                    f"min_hip={self.min_hip_angle:.1f}°, max_tilt={self.max_torso_tilt:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((165 - hip_angle) / 60 * 100))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 35
        if not self.soft_knee_maintained: form_score -= 30
        if not self.not_squatting: form_score -= 35
        if not self.vertical_shins: form_score -= 30
        if not self.neutral_spine: form_score -= 35
        if not self.controlled_hinge: form_score -= 20
        if not self.target_hinge_reached and self.rep_started: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Good Morning",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "depth_reached": self.target_hinge_reached
        }