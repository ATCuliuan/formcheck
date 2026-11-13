import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (CONCENTRATION CURL - SEATED FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.50 # Must face camera (front/oblique view >= 0.50; side view < 0.50 flags bad rep)
MAX_SEATED_THIGH_DROP_RATIO = 0.70   # Seated check: vertical hip-to-knee distance / torso height (standing is > 0.85)
MAX_SEATED_KNEE_ANGLE = 135.0        # Seated check: knee flexion angle must be bent (<= 135°) when ankle is visible
SETUP_ELBOW_MIN_ANGLE = 130.0        # Arm must hang extended down inside thigh (> 130°)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames hanging arm while seated to become READY
UNLOCK_RESET_FRAMES = 15             # Consecutive frames with arm hanging to clear LOCKED_FAULT (~0.5s)
BASELINE_EXPONENTIAL_ALPHA = 0.85    # Weight for baseline exponential moving average
BASELINE_NEW_SAMPLE_WEIGHT = 0.15    # Weight for new incoming baseline measurement
MIN_VISIBILITY_THRESHOLD = 1.8       # Cumulative visibility score across key joints

# 2. Kinematic & Form Thresholds (From Dumbbell Biceps Curls + Inside Path + Loosened Traps)
MAX_UPPER_ARM_ANGLE = 15.0           # Angle between torso and upper arm before flagging elbow flare/swing
MAX_UPPER_ARM_SWING_DEG = 15.0       # Triceps must remain pinned against inner thigh
MIN_VERTICAL_DROP_RATIO = 0.30       # Vertical drop of elbow below shoulder / upper arm length
MAX_OUTWARD_WRIST_RATIO = - 0.10       # Max wrist drift outward/away from body midline relative to elbow
MAX_BACKWARD_Z_DRIFT = 0.05          # Max wrist Z-depth behind the elbow (prevents backward curling)
SHRUG_EAR_DROP_RATIO = 0.72          # Loosened from 0.85 -> 0.72: allows natural seated head/neck lean without false trap flags
MAX_TORSO_SWAY = 0.75                # Horizontal shoulder-hip displacement / upper arm length
ECCENTRIC_DROP_SPEED_MAX = 30.0      # Angular descent speed (deg/frame) for fast negative warning

# 3. Curl Depth & Range of Motion (From Dumbbell Biceps Curls)
FULL_CURL_PEAK_ANGLE = 75.0          # Maximum elbow angle required to credit a full contraction
HALF_CURL_THRESHOLD_ANGLE = 100.0    # If arm curls to <= 100° but misses 75°, it is flagged as half-rep failure
CURL_START_ELBOW_ANGLE = 125.0       # Moving below this angle transitions to CURLING
LOWERING_START_ELBOW_ANGLE = 80.0    # Moving above this angle transitions from TOP to LOWERING
EARLY_REVERSAL_DELTA = 5.0           # Upward bounce in degrees while curling to catch premature reversal
REP_COMPLETE_ELBOW_ANGLE = 130.0     # Arm extension angle required to evaluate rep completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 2                 # Fast trigger when rotating into side view (must stay front view)
DEBOUNCE_SEATED = 4                  # Frames before flagging standing up / not seated
DEBOUNCE_INSIDE_PATH = 3             # Frames before flagging outward/backward curl direction
DEBOUNCE_ELBOW_ELEVATION = 2         # Frames before flagging elbow unpinning / forward lift
DEBOUNCE_SHRUG = 6                   # Loosened from 3 -> 6 frames before flagging shoulder shrugging
DEBOUNCE_TORSO = 5                   # Frames before flagging torso lean / heave
DEBOUNCE_DROP = 2                    # Frames before flagging fast/uncontrolled drop
DEBOUNCE_ML_FAULT = 5                # Frames before flagging AI posture fault

# 5. Model & Buffer Settings
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class ConcentrationCurlTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.peak_contraction_reached = False
        self.min_elbow_angle = 180.0

        # Strict Tripwires (FRONT VIEW + SEATED + INSIDE CURL + BICEP CURL MECHANICS)
        self.is_front_view = True
        self.stayed_seated = True
        self.curl_inside_path = True
        self.elbow_stayed_down = True
        self.shoulder_not_shrugged = True
        self.torso_stable = True
        self.ml_posture_maintained = True

        # Advisory / Warning Flags
        self.fast_descent_warning = False

        self.baseline_ear_shoulder_dist = None
        self.baseline_shoulder_hip_dist = None
        self.baseline_upper_arm_tilt = None

        # Debounce counters
        self.profile_frames = 0
        self.seated_frames = 0
        self.inside_path_frames = 0
        self.elevation_frames = 0
        self.shrug_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.prev_elbow_angle = None
        self.extended_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Sit down facing camera, brace elbow against inner thigh, extend arm"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_concentration_curl.tflite")
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
        self.peak_contraction_reached = False
        self.min_elbow_angle = 180.0
        self.prev_elbow_angle = None

        self.is_front_view = True
        self.stayed_seated = True
        self.curl_inside_path = True
        self.elbow_stayed_down = True
        self.shoulder_not_shrugged = True
        self.torso_stable = True
        self.ml_posture_maintained = True
        self.fast_descent_warning = False

        self.profile_frames = 0
        self.seated_frames = 0
        self.inside_path_frames = 0
        self.elevation_frames = 0
        self.shrug_frames = 0
        self.torso_frames = 0
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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Concentration Curl",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Sit facing camera and show curling arm and hips/knees clearly",
                "elbow_angle": 0.0
            }

        use_right = r_vis > l_vis
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        opposite_shoulder = l_shoulder if use_right else r_shoulder
        active_elbow = r_elbow if use_right else l_elbow
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee
        active_ankle = r_ankle if use_right else l_ankle

        # Joint angle calculations (Same hand/arm curl logic as Dumbbell Biceps Curls)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)

        upper_arm_len = np.sqrt((active_shoulder.x - active_elbow.x)**2 + (active_shoulder.y - active_elbow.y)**2)
        if upper_arm_len == 0:
            upper_arm_len = 0.25

        dx_upper = active_shoulder.x - active_elbow.x
        dy_upper = max(abs(active_shoulder.y - active_elbow.y), 0.05)
        upper_arm_tilt = np.degrees(np.arctan2(dx_upper, dy_upper))

        ear_shoulder_dist = abs(active_shoulder.y - active_ear.y)
        shoulder_hip_dist = abs(active_hip.y - active_shoulder.y)

        # Strict Front View Verification (Flags side view as invalid)
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        avg_hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_height = max(abs(avg_hip_y - avg_shoulder_y), 0.20)
        is_front = (shoulder_span_x / torso_height) >= SETUP_MIN_SHOULDER_SPAN_RATIO

        # Seated Posture Verification (Checks that user is sitting, not standing)
        # 1. In front-view seated position, thighs project toward camera so vertical hip-to-knee drop is small relative to torso
        # 2. If knees and ankles are visible, knee flexion angle must be bent (<= 135°)
        knee_visible = (active_knee.visibility > 0.35) or (l_knee.visibility > 0.35) or (r_knee.visibility > 0.35)
        ankle_visible = (active_ankle.visibility > 0.35)
        if knee_visible:
            ref_knee_y = active_knee.y if active_knee.visibility > 0.35 else min(l_knee.y, r_knee.y)
            thigh_vertical_drop_ratio = max(0.0, ref_knee_y - avg_hip_y) / torso_height
            thigh_is_seated = thigh_vertical_drop_ratio <= MAX_SEATED_THIGH_DROP_RATIO
            if ankle_visible and active_knee.visibility > 0.35:
                knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
                is_seated = thigh_is_seated or (knee_angle <= MAX_SEATED_KNEE_ANGLE)
            else:
                is_seated = thigh_is_seated
        else:
            # If knees are out of frame while standing close to camera, require knees to be visible in seated position
            is_seated = False

        # Inward vs Outward/Backward Curl Direction Calculation
        inward_dir_x = 1.0 if (opposite_shoulder.x - active_shoulder.x) >= 0 else -1.0
        inward_wrist_ratio = ((active_wrist.x - active_elbow.x) * inward_dir_x) / upper_arm_len
        is_curling_outward_or_backward = (
            (inward_wrist_ratio < -MAX_OUTWARD_WRIST_RATIO)
            or ((active_wrist.z - active_elbow.z) > MAX_BACKWARD_Z_DRIFT)
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        arm_extended = (elbow_angle > SETUP_ELBOW_MIN_ANGLE) and is_front and is_seated

        if arm_extended:
            self.extended_frames += 1
            if self.extended_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
                if self.baseline_ear_shoulder_dist is None:
                    self.baseline_ear_shoulder_dist = ear_shoulder_dist
                if self.baseline_shoulder_hip_dist is None:
                    self.baseline_shoulder_hip_dist = shoulder_hip_dist
                if self.baseline_upper_arm_tilt is None:
                    self.baseline_upper_arm_tilt = upper_arm_tilt
                else:
                    self.baseline_upper_arm_tilt = (
                        BASELINE_EXPONENTIAL_ALPHA * self.baseline_upper_arm_tilt
                        + BASELINE_NEW_SAMPLE_WEIGHT * upper_arm_tilt
                    )
        else:
            self.extended_frames = max(0, self.extended_frames - 1)

        if not self.is_ready and self.state not in ["EXTENDED", "CURLING", "TOP", "LOWERING", "LOCKED_FAULT"]:
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_front:
                feedback_msg = "Face the camera directly (front view)"
            elif not is_seated:
                feedback_msg = "Sit down on a bench or chair (keep knees in view) to start"
            else:
                feedback_msg = "Brace elbow on inner thigh and let arm hang down to start"
            return {
                "exercise": "Concentration Curl",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["EXTENDED", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (FRONT VIEW + SEATED + INSIDE CURL + BICEP CURL LOGIC)
        # ---------------------------------------------------------
        if self.state in ["CURLING", "TOP", "LOWERING"]:
            # 0A. Strict Front View Check (Flags turning sideways as bad rep)
            if not is_front:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_front_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # 0B. Seated Posture Check (Flags standing up as bad rep)
            if not is_seated:
                self.seated_frames += 1
                if self.seated_frames >= DEBOUNCE_SEATED:
                    self.stayed_seated = False
            else:
                self.seated_frames = max(0, self.seated_frames - 1)

            # A. Inside Curl Path Check (Prevents curling outward or backward)
            if is_curling_outward_or_backward:
                self.inside_path_frames += 1
                if self.inside_path_frames >= DEBOUNCE_INSIDE_PATH:
                    self.curl_inside_path = False
            else:
                self.inside_path_frames = max(0, self.inside_path_frames - 1)

            # B. Shoulder-to-Elbow Elevation & Upper Arm Unpinning from Thigh
            vertical_drop_ratio = (active_elbow.y - active_shoulder.y) / upper_arm_len
            swing_delta = (
                abs(upper_arm_tilt - self.baseline_upper_arm_tilt)
                if self.baseline_upper_arm_tilt is not None
                else 0.0
            )
            is_elevating = (
                (upper_arm_angle > MAX_UPPER_ARM_ANGLE)
                or (vertical_drop_ratio < MIN_VERTICAL_DROP_RATIO)
                or (swing_delta > MAX_UPPER_ARM_SWING_DEG)
            )

            if is_elevating:
                self.elevation_frames += 1
                if self.elevation_frames >= DEBOUNCE_ELBOW_ELEVATION:
                    self.elbow_stayed_down = False
            else:
                self.elevation_frames = max(0, self.elevation_frames - 1)

            # C. Shoulder Shrug Check (Loosened threshold: 0.72 ratio + 6-frame debounce)
            if self.baseline_ear_shoulder_dist is not None:
                ear_compressed = ear_shoulder_dist < (self.baseline_ear_shoulder_dist * SHRUG_EAR_DROP_RATIO)
                if ear_compressed:
                    self.shrug_frames += 1
                    if self.shrug_frames >= DEBOUNCE_SHRUG:
                        self.shoulder_not_shrugged = False
                else:
                    self.shrug_frames = max(0, self.shrug_frames - 1)

            # D. Torso Sway / Heave Check
            torso_sway = abs(active_shoulder.x - active_hip.x) / upper_arm_len
            if torso_sway > MAX_TORSO_SWAY:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    self.torso_stable = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (BICEP CURL HAND LOGIC + LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if arm_extended:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "EXTENDED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Curl the weight inward toward your chest."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Concentration Curl",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1)
            }

        # Setup to Initial Extended Transition
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            if self.post_rep_timer == 0:
                self.current_feedback = "Curl the weight inward toward your chest!"

        # Curl initiation
        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < CURL_START_ELBOW_ANGLE:
            self.state = "CURLING"
            self.rep_started = True
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            if self.post_rep_timer == 0:
                self.current_feedback = "Curl inward toward chest, keep elbow pinned..."

        # Full peak contraction reached (<= 75°)
        elif self.state == "CURLING" and elbow_angle <= FULL_CURL_PEAK_ANGLE:
            self.state = "TOP"
            self.peak_contraction_reached = True
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            if self.post_rep_timer == 0:
                self.current_feedback = "Good squeeze! Lower the weight."

        # Premature reversal detection while curling: arm stopped before 75° and reversed downward
        elif self.state == "CURLING" and not self.peak_contraction_reached and (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA:
            self.state = "LOWERING"
            self.current_feedback = "Incomplete curl! Lowering arm early..."

        # Normal descent transition from TOP
        elif self.state == "TOP" and elbow_angle > LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower arm down..."

        # Track minimum angle while in curling phase
        if self.state in ["CURLING", "TOP"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # Eccentric Free-Fall Monitor (Advisory)
        if self.state in ["LOWERING", "CURLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.fast_descent_warning = True
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "CURLING"] and self.rep_started and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False

            # Strict Form Tripwires (Front view, Seated posture, Inside curl path & Bicep curl faults first)
            if not self.is_front_view:
                self.current_feedback = "Bad rep: Stay in front view! Do not turn sideways. Sit and extend arm to reset."
                is_bad_rep = True
            elif not self.stayed_seated:
                self.current_feedback = "Bad rep: Stay seated! Do not stand up during the curl. Sit and extend arm to reset."
                is_bad_rep = True
            elif not self.curl_inside_path:
                self.current_feedback = "Bad rep: Curl inward toward your chest, not outward or backward! Extend arm to reset."
                is_bad_rep = True
            elif not self.elbow_stayed_down:
                self.current_feedback = "Bad rep: Keep elbow pinned against inner thigh! Return arm down to reset."
                is_bad_rep = True
            elif not self.shoulder_not_shrugged:
                self.current_feedback = "Bad rep: Relax your traps! Return arm down to reset."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "Bad rep: Do not heave or sway torso! Return arm down to reset."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected arm/torso mechanics breakdown."
                is_bad_rep = True

            # Incomplete Rep & Half Rep Check (Flagged as Bad Rep)
            elif not self.peak_contraction_reached:
                if self.min_elbow_angle <= HALF_CURL_THRESHOLD_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Curled to ~100°, squeeze to under 75°. Return arm down to reset."
                else:
                    self.current_feedback = "Bad rep: Movement was too shallow! Return arm down to reset."
                is_bad_rep = True

            # Rep Evaluation
            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.fast_descent_warning:
                    self.current_feedback = "Good rep! Reminder: Lower slower for muscle control."
                else:
                    self.current_feedback = "Clean concentration curl! Strict bicep isolation."
                self.state = "EXTENDED"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

            # Baseline updates on rep completion
            if self.baseline_ear_shoulder_dist is not None:
                self.baseline_ear_shoulder_dist = (
                    BASELINE_EXPONENTIAL_ALPHA * self.baseline_ear_shoulder_dist
                    + BASELINE_NEW_SAMPLE_WEIGHT * ear_shoulder_dist
                )
            if self.baseline_shoulder_hip_dist is not None:
                self.baseline_shoulder_hip_dist = (
                    BASELINE_EXPONENTIAL_ALPHA * self.baseline_shoulder_hip_dist
                    + BASELINE_NEW_SAMPLE_WEIGHT * shoulder_hip_dist
                )
            if self.baseline_upper_arm_tilt is not None:
                self.baseline_upper_arm_tilt = (
                    BASELINE_EXPONENTIAL_ALPHA * self.baseline_upper_arm_tilt
                    + BASELINE_NEW_SAMPLE_WEIGHT * upper_arm_tilt
                )

        # Dynamic Scoring
        form_score = max(0, min(100, int(((150 - elbow_angle) / 80) * 100)))
        if not self.is_front_view: form_score -= 35
        if not self.stayed_seated: form_score -= 35
        if not self.curl_inside_path: form_score -= 35
        if not self.elbow_stayed_down: form_score -= 30
        if not self.shoulder_not_shrugged: form_score -= 20
        if not self.torso_stable: form_score -= 20
        if self.fast_descent_warning: form_score -= 8
        if not self.ml_posture_maintained: form_score -= 30
        if not self.peak_contraction_reached and self.rep_started: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Concentration Curl",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1)
        }