import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (REVERSE CURL - DUMBBELL CURL LOGIC)
# =====================================================================

# 1. Setup & Starting Guard (Exact Constants from DumbbellBicepsCurlsTracker)
SETUP_ELBOW_MIN_ANGLE = 150.0        # Arms extended hanging down at start
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.35 # Side profile shoulder span / torso height ratio
SETUP_MIN_READY_FRAMES = 10          # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames held extended to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility sum

# 2. Strict Elbow Elevation, Trap Shrug & Torso Sway Thresholds
MAX_UPPER_ARM_ANGLE = 15.0           # Max hip-shoulder-elbow angle (catches forward/backward elbow elevation)
MIN_VERTICAL_DROP_RATIO = 0.85       # Min (elbow.y - shoulder.y) / upper_arm_len (catches upward elbow lift)
SHRUG_EAR_DROP_RATIO = 0.70          # Ear-to-shoulder distance shrink ratio catching trap shrugs
MAX_TORSO_SWAY = 0.20                # Max |shoulder.x - hip.x| / shoulder_hip_dist catching body heave
ECCENTRIC_DROP_SPEED_MAX = 20.0      # Max deg/frame downward drop speed

# 3. Curl ROM & State Machine Thresholds
CURL_START_ELBOW_ANGLE = 140.0       # Flexing below 140° initiates CURLING
MIN_VALID_REP_FLEXION_ANGLE = 120.0  # Ignore minor resting twitches (> 120°) so shallow sway never flags!
FULL_CURL_PEAK_ANGLE = 75.0          # Top pronated squeeze (<= 75°)
HALF_CURL_THRESHOLD_ANGLE = 115.0    # Stopping <= 115° (but > 75°) flags Half Rep (counted as Bad Rep)
LOWERING_START_ELBOW_ANGLE = 85.0    # Lowering from TOP (> 85°)
EARLY_REVERSAL_DELTA = 12.0          # Reversing downward early before reaching 75° peak
REP_COMPLETE_ELBOW_ANGLE = 148.0     # Full arm extension completes rep

# 4. Exponential Moving Average Weights for Baselines
BASELINE_EXPONENTIAL_ALPHA = 0.90
BASELINE_NEW_SAMPLE_WEIGHT = 0.10

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 5                 # Frames before flagging lost side view
DEBOUNCE_ELBOW_ELEVATION = 5         # Frames before flagging elbow elevation / drift
DEBOUNCE_SHRUG = 5                   # Frames before flagging trap shrug
DEBOUNCE_TORSO = 5                   # Frames before flagging torso momentum swing
DEBOUNCE_DROP = 3                    # Frames before flagging fast descent warning
DEBOUNCE_ML_FAULT = 5

ML_FAULT_CONFIDENCE_THRESHOLD = 0.75
SEQUENCE_LENGTH = 30


class ReverseCurlTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.peak_contraction_reached = False
        self.min_elbow_angle = 180.0

        # Strict Posture Tripwires (Upper Body Only)
        self.is_side_view = True
        self.elbow_stayed_down = True
        self.shoulder_not_shrugged = True
        self.torso_stable = True
        self.ml_posture_maintained = True

        # Advisory Warning
        self.fast_descent_warning = False

        # Dynamic Anatomical Baselines
        self.baseline_ear_shoulder_dist = None
        self.baseline_shoulder_hip_dist = None

        # Debounce Counters
        self.profile_frames = 0
        self.elevation_frames = 0
        self.shrug_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.prev_elbow_angle = None
        self.extended_frames = 0
        self.unlock_frames = 0
        self.is_ready = False

        self.current_feedback = "Show body sideways, let arm hang (palms facing down)"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_reverse_curl.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[REVERSE CURL ERROR] __init__ model_path={model_path} | error={exc}",
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
        self.peak_contraction_reached = False
        self.min_elbow_angle = 180.0
        self.prev_elbow_angle = None

        self.is_side_view = True
        self.elbow_stayed_down = True
        self.shoulder_not_shrugged = True
        self.torso_stable = True
        self.ml_posture_maintained = True
        self.fast_descent_warning = False

        self.profile_frames = 0
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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Reverse Curl",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Show your arm and torso (side view)",
                "elbow_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        # Core Upper-Body Angles & Segment Lengths
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)

        upper_arm_len = np.sqrt(
            (active_shoulder.x - active_elbow.x) ** 2 + (active_shoulder.y - active_elbow.y) ** 2
        )
        if upper_arm_len == 0:
            upper_arm_len = 0.25

        ear_shoulder_dist = abs(active_ear.y - active_shoulder.y)
        shoulder_hip_dist = abs(active_shoulder.y - active_hip.y)
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        torso_height = shoulder_hip_dist if shoulder_hip_dist > 0 else 0.40

        is_side = (shoulder_span_x / torso_height) < SETUP_MAX_SHOULDER_SPAN_RATIO
        arm_extended = (elbow_angle > SETUP_ELBOW_MIN_ANGLE) and is_side

        # ---------------------------------------------------------
        # 1. SETUP GUARD & DYNAMIC BASELINE CALIBRATION
        # ---------------------------------------------------------
        if arm_extended:
            self.extended_frames += 1
            if self.extended_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
                if self.baseline_ear_shoulder_dist is None:
                    self.baseline_ear_shoulder_dist = ear_shoulder_dist
                if self.baseline_shoulder_hip_dist is None:
                    self.baseline_shoulder_hip_dist = shoulder_hip_dist
        else:
            self.extended_frames = max(0, self.extended_frames - 1)

        if not self.is_ready and self.state not in ("LOCKED_FAULT",):
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera (side view)" if not is_side else "Let arm hang down to start"
            return {
                "exercise": "Reverse Curl",
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
        # 2. ST-GCN TEMPORAL BUFFER & INFERENCE
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["EXTENDED", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[REVERSE CURL FAULT] AI ST-GCN flagged mechanics fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[REVERSE CURL ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. STRICT KINEMATIC TRIPWIRES (UPPER BODY ONLY)
        # ---------------------------------------------------------
        if self.state in ("CURLING", "TOP", "LOWERING"):
            # A. Strict Side Profile Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[REVERSE CURL FAULT] Lost side view: ratio={shoulder_span_x / torso_height:.2f}",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Strict Dual-Condition Elbow Elevation Check (Only active once curl is truly underway < 125°)
            if elbow_angle < 125.0:
                vertical_drop_ratio = (active_elbow.y - active_shoulder.y) / upper_arm_len
                is_elevating = (upper_arm_angle > MAX_UPPER_ARM_ANGLE) or (vertical_drop_ratio < MIN_VERTICAL_DROP_RATIO)
                if is_elevating:
                    self.elevation_frames += 1
                    if self.elevation_frames >= DEBOUNCE_ELBOW_ELEVATION:
                        if self.elbow_stayed_down:
                            print(
                                f"[REVERSE CURL FAULT] Elbow elevated/drifted! "
                                f"upper_arm_angle={upper_arm_angle:.1f}° (max={MAX_UPPER_ARM_ANGLE}°), "
                                f"vert_drop_ratio={vertical_drop_ratio:.2f} (min={MIN_VERTICAL_DROP_RATIO})",
                                flush=True,
                            )
                        self.elbow_stayed_down = False
                else:
                    self.elevation_frames = max(0, self.elevation_frames - 1)

            # C. Trap Shrug Check
            if self.baseline_ear_shoulder_dist:
                ear_compressed = ear_shoulder_dist < (self.baseline_ear_shoulder_dist * SHRUG_EAR_DROP_RATIO)
                if ear_compressed:
                    self.shrug_frames += 1
                    if self.shrug_frames >= DEBOUNCE_SHRUG:
                        if self.shoulder_not_shrugged:
                            print("[REVERSE CURL FAULT] Traps shrugged upward!", flush=True)
                        self.shoulder_not_shrugged = False
                else:
                    self.shrug_frames = max(0, self.shrug_frames - 1)

            # D. Torso Sway / Heave Check
            torso_sway = abs(active_shoulder.x - active_hip.x) / torso_height
            if torso_sway > MAX_TORSO_SWAY:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_stable:
                        print(f"[REVERSE CURL FAULT] Torso swayed/leaned: sway={torso_sway:.2f}", flush=True)
                    self.torso_stable = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (LOCKED_FAULT & CURLING -> TOP -> LOWERING)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if arm_extended:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "EXTENDED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Curl the weight up with palms down."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Reverse Curl",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Curl the weight up (knuckles facing ceiling)!"

        # Initiate CURLING only from EXTENDED (prevents ping-pong during LOWERING)
        elif self.state == "EXTENDED" and elbow_angle < CURL_START_ELBOW_ANGLE:
            self.state = "CURLING"
            self.rep_started = True
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.current_feedback = "Curl up to your chest, keep elbows pinned..."

        elif self.state == "CURLING" and elbow_angle <= FULL_CURL_PEAK_ANGLE:
            self.state = "TOP"
            self.peak_contraction_reached = True
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.current_feedback = "Good squeeze! Lower the weight."

        elif self.state == "CURLING" and not self.peak_contraction_reached and (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA:
            # Only transition to a half-rep lowering if the user actually curled past MIN_VALID_REP_FLEXION_ANGLE (120°)
            if self.min_elbow_angle <= MIN_VALID_REP_FLEXION_ANGLE:
                self.state = "LOWERING"
                self.current_feedback = "Incomplete curl! Lowering arm early..."
                print(
                    f"[REVERSE CURL FAULT] Early reversal before peak flexion: min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )

        elif self.state == "TOP" and elbow_angle > LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            self.current_feedback = "Lower arm down..."

        if self.state in ("CURLING", "TOP"):
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # Eccentric Drop Speed Check
        if self.state in ("LOWERING", "CURLING"):
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
        # 5. REP COMPLETION & EVALUATION (SHALLOW TWITCHES IGNORED!)
        # ---------------------------------------------------------
        if self.state in ("LOWERING", "TOP", "CURLING") and self.rep_started and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            # Silently ignore shallow movements / resting arm twitches (> 120°) without flagging a bad rep!
            if self.min_elbow_angle > MIN_VALID_REP_FLEXION_ANGLE:
                self._reset_rep_tracking()
                self.state = "EXTENDED"
                self.current_feedback = "Curl the weight up (knuckles facing ceiling)!"
                return {
                    "exercise": "Reverse Curl",
                    "state": self.state,
                    "good_reps": self.good_reps,
                    "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps,
                    "total_reps": total_reps,
                    "form_score": 100,
                    "feedback": self.current_feedback,
                    "elbow_angle": round(elbow_angle, 1)
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stay in side view! Stand sideways and let arm hang to reset."
                is_bad_rep = True
            elif not self.elbow_stayed_down:
                self.current_feedback = "Bad rep: Keep elbows pinned down! Return arm down to reset."
                is_bad_rep = True
            elif not self.shoulder_not_shrugged:
                self.current_feedback = "Bad rep: Relax your traps! Return arm down to reset."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "Bad rep: Do not heave or lean back! Return arm down to reset."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected arm mechanics breakdown."
                is_bad_rep = True
            elif not self.peak_contraction_reached:
                self.current_feedback = "Bad rep: Half rep! Squeeze all the way up under 75°. Return arm down to reset."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[REVERSE CURL REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.fast_descent_warning:
                    self.current_feedback = "Good rep! Reminder: Lower slower for muscle control."
                else:
                    self.current_feedback = "Great reverse curl! Clean rep."
                print(
                    f"[REVERSE CURL REP #{rep_num} -> GOOD REP] "
                    f"min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "EXTENDED"
                self._reset_rep_tracking()

                # Update anatomical baselines smoothly after a clean rep
                if self.baseline_ear_shoulder_dist:
                    self.baseline_ear_shoulder_dist = (
                        BASELINE_EXPONENTIAL_ALPHA * self.baseline_ear_shoulder_dist
                        + BASELINE_NEW_SAMPLE_WEIGHT * ear_shoulder_dist
                    )
                if self.baseline_shoulder_hip_dist:
                    self.baseline_shoulder_hip_dist = (
                        BASELINE_EXPONENTIAL_ALPHA * self.baseline_shoulder_hip_dist
                        + BASELINE_NEW_SAMPLE_WEIGHT * shoulder_hip_dist
                    )

        # Dynamic Form Score
        form_score = max(0, min(100, int(((150 - elbow_angle) / 80) * 100)))
        if not self.is_side_view: form_score -= 35
        if not self.elbow_stayed_down: form_score -= 30
        if not self.shoulder_not_shrugged: form_score -= 20
        if not self.torso_stable: form_score -= 20
        if self.fast_descent_warning: form_score -= 10
        if not self.peak_contraction_reached and self.rep_started: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Reverse Curl",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1)
        }