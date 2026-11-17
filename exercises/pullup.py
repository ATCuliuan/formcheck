import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (PULL-UP - CORONAL / FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Wider-Than-Shoulder Grip, Dead Hang, Front View)
SETUP_HANG_ELBOW_MIN_ANGLE = 152.0   # Dead hang full arm extension
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.25 # Front view profile validation (16:9 aspect calibrated)
SETUP_MAX_SHOULDER_Y_DELTA = 0.09    # Level shoulders in frontal hang
SETUP_MIN_GRIP_WIDTH_RATIO = 0.85    # Hands at least shoulder-width / slightly wider on bar
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames holding dead hang
UNLOCK_RESET_FRAMES = 12             # Frames held in dead hang to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.40      # Minimum upper-body keypoint visibility

# 2. Kinematic & Form Thresholds (Step 2, 3, 4: No Swinging/Kipping, Even Pull, Controlled Descent)
MAX_HIP_SWING_RATIO = 0.22           # Max horizontal hip swing / torso_len (Step 2: keep body from swinging)
MIN_HIP_KIP_ANGLE = 135.0            # Piking/kipping hips forward toward bar cheat
MAX_ARM_ASYMMETRY_DELTA = 24.0       # Asymmetric arm pull difference (deg)
ECCENTRIC_DROP_SPEED_MAX = 25.0      # Free-fall descent cutoff (deg/frame)
EARLY_REVERSAL_DELTA = 10.0          # Early downward drop before clearing chin over bar

# 3. Pull Depth & Range of Motion (Step 3 & 4: Chin Clears Bar -> Full Dead Hang)
FULL_PULL_MAX_ELBOW_ANGLE = 95.0     # Elbows driven down and back at top of pull-up
CHIN_CLEAR_BAR_MARGIN = 0.04         # Mouth/chin must reach or clear wrist/bar Y level
HALF_PULL_MAX_ELBOW_ANGLE = 125.0    # Pulling past 125° but failing to clear chin flags Half Rep (Bad Rep)
MIN_VALID_REP_FLEXION_ANGLE = 138.0  # Ignore minor shoulder shrugs in dead hang (> 138°)
PULLING_START_ELBOW_ANGLE = 142.0    # Initiating upward pull from dead hang
LOWERING_START_ELBOW_ANGLE = 105.0   # Lowering down from top of bar
REP_COMPLETE_ELBOW_ANGLE = 150.0     # Full extension dead hang evaluating completion

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_KIP = 3
DEBOUNCE_ASYM = 4
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class PullupTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_depth_reached = False
        self.min_elbow_angle = 180.0
        self.baseline_hip_x = None

        # Posture Tripwires (FRONT / CORONAL VIEW)
        self.is_front_view = True
        self.proper_grip_width = True
        self.no_kipping = True             # Step 2: Engage core/glutes to prevent swinging/kipping
        self.is_symmetrical = True         # Step 3: Even elbow drive
        self.controlled_descent = True     # Step 4: Controlled slow negative
        self.ml_posture_maintained = True

        # Debounce counters
        self.front_frames = 0
        self.kip_frames = 0
        self.asym_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.hang_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Face camera and hang from bar with arms fully extended"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_pullup.tflite")
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
        self.half_depth_reached = False
        self.min_elbow_angle = 180.0
        self.reversal_frames = 0
        self.prev_elbow_angle = None

        self.is_front_view = True
        self.proper_grip_width = True
        self.no_kipping = True
        self.is_symmetrical = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.kip_frames = 0
        self.asym_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        mouth_l = landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT.value]
        mouth_r = landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value]
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

        upper_joints = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(upper_joints) < MIN_VISIBILITY_THRESHOLD:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Pull-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Ensure arms, shoulders, and bar are visible",
                "elbow_angle": 0.0, "depth_reached": False
            }

        # Bilateral Elbow Angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        # Body & Front View Geometry
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        torso_len = np.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y) or 0.40
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_width / torso_len

        is_facing_front = (
            (shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO)
            and (abs(l_shoulder.y - r_shoulder.y) < SETUP_MAX_SHOULDER_Y_DELTA)
        )

        # Step 1: Grip Width Check (Hands slightly wider than shoulder-width apart)
        wrist_width = abs(l_wrist.x - r_wrist.x)
        has_proper_grip = wrist_width >= (shoulder_width * SETUP_MIN_GRIP_WIDTH_RATIO)

        # Step 3: FIXED Chin-Over-Bar Check (In MediaPipe, smaller Y = higher on screen!)
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2.0
        chin_proxy_y = (mouth_l.y + mouth_r.y) / 2.0 if (mouth_l.visibility > 0.4 and mouth_r.visibility > 0.4) else nose.y
        is_chin_over_bar = (
            (chin_proxy_y <= (avg_wrist_y + CHIN_CLEAR_BAR_MARGIN))
            and (avg_elbow_angle <= FULL_PULL_MAX_ELBOW_ANGLE)
        )

        # Step 2: Swinging / Kipping Check (Only checks hips/knees if visible in frame)
        hips_visible = (l_hip.visibility > 0.40 and r_hip.visibility > 0.40)
        knees_visible = (l_knee.visibility > 0.45 and r_knee.visibility > 0.45)

        is_swinging_or_kipping = False
        if hips_visible and self.baseline_hip_x is not None:
            hip_swing_ratio = abs(mid_hip_x - self.baseline_hip_x) / torso_len
            if hip_swing_ratio > MAX_HIP_SWING_RATIO:
                is_swinging_or_kipping = True

        if hips_visible and knees_visible:
            l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            if min(l_hip_angle, r_hip_angle) < MIN_HIP_KIP_ANGLE:
                is_swinging_or_kipping = True

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1 & 2: Dead Hang, Front View, Proper Grip)
        # ---------------------------------------------------------
        is_overhead_hang = (l_wrist.y < l_shoulder.y) and (r_wrist.y < r_shoulder.y)
        is_dead_hang = (
            is_overhead_hang
            and (avg_elbow_angle >= SETUP_HANG_ELBOW_MIN_ANGLE)
            and is_facing_front
            and has_proper_grip
        )

        if is_dead_hang:
            self.hang_frames += 1
            if hips_visible:
                if self.baseline_hip_x is None:
                    self.baseline_hip_x = mid_hip_x
                else:
                    self.baseline_hip_x = 0.90 * self.baseline_hip_x + 0.10 * mid_hip_x

            if self.hang_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.hang_frames = max(0, self.hang_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face camera directly (front view required)"
            elif not is_overhead_hang or avg_elbow_angle < SETUP_HANG_ELBOW_MIN_ANGLE:
                feedback_msg = "Hang from bar with arms fully extended (dead hang)"
            elif not has_proper_grip:
                feedback_msg = "Widen grip: hands slightly wider than shoulder-width"
            else:
                feedback_msg = "Hold steady dead hang to begin"

            return {
                "exercise": "Pull-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(avg_elbow_angle, 1), "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["HANGING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[PULLUP FAULT] AI ST-GCN flagged swing/posture fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["PULLING", "TOP", "LOWERING"]:
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[PULLUP FAULT] Lost front view: span_ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            if is_swinging_or_kipping:
                self.kip_frames += 1
                if self.kip_frames >= DEBOUNCE_KIP:
                    if self.no_kipping:
                        print("[PULLUP FAULT] Body swinging or kipping knees/hips detected!", flush=True)
                    self.no_kipping = False
            else:
                self.kip_frames = max(0, self.kip_frames - 1)

            if abs(l_elbow_angle - r_elbow_angle) > MAX_ARM_ASYMMETRY_DELTA:
                self.asym_frames += 1
                if self.asym_frames >= DEBOUNCE_ASYM:
                    if self.is_symmetrical:
                        print(
                            f"[PULLUP FAULT] Uneven arm pull: L={l_elbow_angle:.1f}°, R={r_elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.is_symmetrical = False
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, avg_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_dead_hang:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "HANGING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Pull chest to bar until chin clears it."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Pull-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(avg_elbow_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "HANGING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Engage core and pull chin over the bar!"

        # Step 3: Initiate Pull from Dead Hang
        elif self.state in ["HANGING", "LOWERING"] and avg_elbow_angle < PULLING_START_ELBOW_ANGLE:
            self.state = "PULLING"
            self.rep_started = True
            self.min_elbow_angle = avg_elbow_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive elbows down and back, pull chest to bar..."

        # Step 3 Complete: Chin clears bar AND elbows flexed <= 95°
        elif self.state == "PULLING" and (is_chin_over_bar or avg_elbow_angle <= 78.0):
            self.depth_reached = True
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Chin cleared bar! Lower down slowly under control."

        # Early downward reversal (Half rep branch)
        elif self.state == "PULLING" and not self.depth_reached and (avg_elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_elbow_angle <= HALF_PULL_MAX_ELBOW_ANGLE:
                    self.half_depth_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete pull! Lowering before chin cleared bar..."
                print(
                    f"[PULLUP FAULT] Early reversal before chin cleared bar: min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )

        # Step 4: Controlled Descent from Top
        elif self.state == "TOP" and avg_elbow_angle > LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower slowly all the way back to a full dead hang..."

        if self.state == "PULLING" and not ((avg_elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 4: Eccentric drop speed detector
        if self.rep_started and self.state in ["LOWERING", "PULLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = avg_elbow_angle - self.prev_elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[PULLUP FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = avg_elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "TOP", "PULLING"] and min_elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            # Ignore minor shoulder shrugs in dead hang (> 138°)
            if self.min_elbow_angle > MIN_VALID_REP_FLEXION_ANGLE and self.no_kipping and self.is_front_view:
                self._reset_rep_tracking()
                self.state = "HANGING"
                return {
                    "exercise": "Pull-Up", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(avg_elbow_angle, 1), "depth_reached": False
                }

            is_bad_rep = False
            is_half_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face camera directly! Hang straight to reset."
                is_bad_rep = True
            elif not self.no_kipping:
                self.current_feedback = "Bad rep: Engage core and glutes! Do not swing or kip."
                is_bad_rep = True
            elif not self.is_symmetrical:
                self.current_feedback = "Bad rep: Drive both elbows down evenly!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower yourself slowly with control! Do not drop."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged excessive body swing/breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                if self.min_elbow_angle <= HALF_PULL_MAX_ELBOW_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Bad rep: Half rep! Pull higher until your chin clears the bar."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Pull was too shallow! Clear your chin over the bar."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep or is_half_rep:
                self.bad_reps += 1
                print(
                    f"[PULLUP REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Pull-Up! Chin cleared bar with full dead-hang control."
                print(
                    f"[PULLUP REP #{rep_num} -> GOOD REP] "
                    f"min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "HANGING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((180 - avg_elbow_angle) / 95 * 100))) if self.state != "HANGING" else 100
        if not self.is_front_view: form_score -= 35
        if not self.no_kipping: form_score -= 35
        if not self.is_symmetrical: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "LOWERING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Pull-Up",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(avg_elbow_angle, 1),
            "depth_reached": self.depth_reached
        }