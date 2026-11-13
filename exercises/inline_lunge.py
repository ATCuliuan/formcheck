import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (INLINE LUNGE - SIDE / SAGITTAL VIEW ONLY)
# =====================================================================

# 1. Setup & Starting Guard
#    Tall stance, side profile, torso vertical, feet one shin-length apart,
#    back heel lifted (ball of foot), hands on hips or behind head.
SETUP_KNEE_MIN_ANGLE = 160.0         # Knees extended in tall split stance
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.45 # Shoulder x-span / torso length. Above this = not a side view
SETUP_MAX_TRUNK_LEAN_DEG = 10.0       # Torso must be (almost) vertical at setup. 0 = dead vertical
MIN_STANCE_SHIN_RATIO = 0.60         # Front/back ankle gap must be roughly one shin length...
MAX_STANCE_SHIN_RATIO = 1.60         # ...allowed range, as a multiple of shin length
SETUP_BACK_HEEL_MIN_LIFT_RATIO = 0.15  # Back heel must sit this far above its toe (fraction of shin length)
HANDS_ON_HIPS_MAX_RATIO = 0.45       # Wrist-to-hip distance / torso length counted as "hands on hips"
HANDS_AT_HEAD_Y_MARGIN = 0.05        # Wrist at/above shoulder line (+margin) counted as "hands behind head"
MIN_WRIST_VISIBILITY = 0.30          # Below this the wrist is hidden by the body and hand position is not judged
SETUP_MIN_READY_FRAMES = 12          # Consecutive frames holding the setup
UNLOCK_RESET_FRAMES = 15             # Frames held in setup to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 3.0       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds
MAX_TRUNK_LEAN_DEG = 15.0            # Max torso lean from vertical during the lunge (keep a few degrees above setup value)
MAX_HIP_DRIFT_RATIO = 0.30           # Max horizontal hip drift from standing position / torso length (drop STRAIGHT down)
EARLY_REVERSAL_DELTA = 10.0          # Premature upward push before depth

# 3. Lunge Depth & Range of Motion
FULL_LUNGE_MAX_KNEE_ANGLE = 105.0    # Back and front knee reaching parallel/floor (~90 deg)
HALF_LUNGE_MAX_KNEE_ANGLE = 130.0    # Dropping past 130° but short of 105° awards half rep
DESCENDING_START_KNEE_ANGLE = 140.0  # Flexing below this initiates DESCENDING
ASCENDING_START_KNEE_ANGLE = 105.0   # Rising above bottom transitions to ASCENDING
REP_COMPLETE_KNEE_ANGLE = 155.0      # Lockout angle evaluating rep completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4                 # Frames before flagging a lost side view
DEBOUNCE_TORSO = 4                   # Frames before flagging torso lean
DEBOUNCE_HANDS = 6                   # Frames before flagging hands dropped from hips/head
DEBOUNCE_WOBBLE = 4                  # Frames before flagging hip drift
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class InlineLungeTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_depth_reached = False
        self.min_back_knee_angle = 180.0

        # Posture Tripwires
        self.is_side_view = True
        self.torso_collapsed = False
        self.hands_dropped = False
        self.wobbled = False                 # Hip drifted forward/back instead of dropping straight down
        self.ml_posture_maintained = True

        # Standing baseline (recorded while holding the setup)
        self.baseline_hip_x = None

        # Peak values seen during the current rep (printed in the rep summary so you know what to tune)
        self.max_trunk_lean = 0.0
        self.max_hip_drift = 0.0
        self.last_setup_msg = None

        # Debounce counters
        self.profile_frames = 0
        self.torso_frames = 0
        self.hands_frames = 0
        self.wobble_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Side view: stand tall, feet on one line, hands on hips"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_inline_lunge.tflite")
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
        self.min_back_knee_angle = 180.0
        self.reversal_frames = 0
        self.max_trunk_lean = 0.0
        self.max_hip_drift = 0.0

        self.is_side_view = True
        self.torso_collapsed = False
        self.hands_dropped = False
        self.wobbled = False
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.torso_frames = 0
        self.hands_frames = 0
        self.wobble_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        pose = self.mp_pose.PoseLandmark
        l_shoulder = landmarks[pose.LEFT_SHOULDER.value]
        r_shoulder = landmarks[pose.RIGHT_SHOULDER.value]
        l_wrist = landmarks[pose.LEFT_WRIST.value]
        r_wrist = landmarks[pose.RIGHT_WRIST.value]
        l_hip = landmarks[pose.LEFT_HIP.value]
        r_hip = landmarks[pose.RIGHT_HIP.value]
        l_knee = landmarks[pose.LEFT_KNEE.value]
        r_knee = landmarks[pose.RIGHT_KNEE.value]
        l_ankle = landmarks[pose.LEFT_ANKLE.value]
        r_ankle = landmarks[pose.RIGHT_ANKLE.value]
        l_heel = landmarks[pose.LEFT_HEEL.value]
        r_heel = landmarks[pose.RIGHT_HEEL.value]
        l_toe = landmarks[pose.LEFT_FOOT_INDEX.value]
        r_toe = landmarks[pose.RIGHT_FOOT_INDEX.value]

        total_vis = sum([l_hip.visibility, r_hip.visibility, l_knee.visibility, r_knee.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if total_vis < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Inline Lunge", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Show full body side profile",
                "knee_angle": 0.0, "depth_reached": False
            }

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        if l_knee.y > r_knee.y:
            back_knee_angle = l_knee_angle
            front_knee_angle = r_knee_angle
        else:
            back_knee_angle = r_knee_angle
            front_knee_angle = l_knee_angle

        # Camera-facing side is used for torso / hand references
        l_side_vis = l_shoulder.visibility + l_hip.visibility + l_wrist.visibility
        r_side_vis = r_shoulder.visibility + r_hip.visibility + r_wrist.visibility
        if r_side_vis >= l_side_vis:
            active_shoulder, active_hip, active_wrist = r_shoulder, r_hip, r_wrist
        else:
            active_shoulder, active_hip, active_wrist = l_shoulder, l_hip, l_wrist

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        l_shin_len = np.hypot(l_knee.x - l_ankle.x, l_knee.y - l_ankle.y)
        r_shin_len = np.hypot(r_knee.x - r_ankle.x, r_knee.y - r_ankle.y)
        shin_len = max(l_shin_len, r_shin_len, 0.10)

        # ---------------------------------------------------------
        # SIDE-VIEW MEASUREMENTS
        # ---------------------------------------------------------
        # 1. Side profile: shoulders overlap in x when viewed from the side
        shoulder_span_ratio = abs(l_shoulder.x - r_shoulder.x) / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # 2. Torso vertical (0 deg = dead vertical)
        dx_trunk = abs(active_shoulder.x - active_hip.x)
        dy_trunk = max(abs(active_shoulder.y - active_hip.y), 0.10)
        trunk_lean = np.degrees(np.arctan2(dx_trunk, dy_trunk))

        # 3. Hands on hips OR locked behind the head (arms stay out of the movement)
        if active_wrist.visibility < MIN_WRIST_VISIBILITY:
            hands_ok = True  # Hidden by the body, cannot judge
        else:
            wrist_hip_ratio = np.hypot(active_wrist.x - active_hip.x, active_wrist.y - active_hip.y) / torso_len
            hands_on_hips = wrist_hip_ratio <= HANDS_ON_HIPS_MAX_RATIO
            hands_at_head = active_wrist.y <= (active_shoulder.y + HANDS_AT_HEAD_Y_MARGIN)
            hands_ok = hands_on_hips or hands_at_head

        # 4. Stance length ~ one shin length, measured front ankle to back ankle
        stance_ratio = abs(l_ankle.x - r_ankle.x) / shin_len
        stance_ok = MIN_STANCE_SHIN_RATIO <= stance_ratio <= MAX_STANCE_SHIN_RATIO

        # 5. Back heel lifted (resting on the ball of the foot). The raised heel sits above its toe,
        #    the flat front foot does not, so the larger of the two lifts is the back foot's.
        l_heel_lift = (l_toe.y - l_heel.y) / shin_len
        r_heel_lift = (r_toe.y - r_heel.y) / shin_len
        back_heel_lifted = max(l_heel_lift, r_heel_lift) >= SETUP_BACK_HEEL_MIN_LIFT_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        knees_straight = front_knee_angle > SETUP_KNEE_MIN_ANGLE and back_knee_angle > SETUP_KNEE_MIN_ANGLE
        torso_vertical_setup = trunk_lean <= SETUP_MAX_TRUNK_LEAN_DEG
        is_standing_setup = (
            is_side
            and knees_straight
            and torso_vertical_setup
            and stance_ok
            and back_heel_lifted
            and hands_ok
        )

        if is_standing_setup:
            self.standing_frames += 1
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            if self.baseline_hip_x is None:
                self.baseline_hip_x = mid_hip_x
            else:
                self.baseline_hip_x = 0.90 * self.baseline_hip_x + 0.10 * mid_hip_x
            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                if not self.is_ready:
                    print(
                        f"[INLINE LUNGE] READY: setup held | trunk_lean={trunk_lean:.1f}° "
                        f"stance={stance_ratio:.2f} shin heel_lift={max(l_heel_lift, r_heel_lift):.2f}",
                        flush=True,
                    )
                self.is_ready = True
                self.last_setup_msg = None
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (show full side profile)"
            elif not knees_straight:
                feedback_msg = "Stand completely tall with both knees straight"
            elif not torso_vertical_setup:
                feedback_msg = "Stand upright: keep your torso vertical"
            elif not stance_ok:
                feedback_msg = "Feet on one line, about one shin-length apart"
            elif not back_heel_lifted:
                feedback_msg = "Lift your back heel: rest on the ball of your back foot"
            elif not hands_ok:
                feedback_msg = "Put your hands on your hips or behind your head"
            else:
                feedback_msg = "Hold the position to begin..."

            # Print only when the blocking reason changes (no per-frame spam). Shows every setup
            # measurement next to its limit so you can see which one to tune.
            if feedback_msg != self.last_setup_msg:
                self.last_setup_msg = feedback_msg
                print(
                    f"[INLINE LUNGE SETUP] {feedback_msg} | "
                    f"side_ratio={shoulder_span_ratio:.2f} (max {SETUP_MAX_SHOULDER_SPAN_RATIO}) | "
                    f"knees front/back={front_knee_angle:.0f}°/{back_knee_angle:.0f}° (min {SETUP_KNEE_MIN_ANGLE}°) | "
                    f"trunk_lean={trunk_lean:.1f}° (max {SETUP_MAX_TRUNK_LEAN_DEG}°) | "
                    f"stance={stance_ratio:.2f} shin (range {MIN_STANCE_SHIN_RATIO}-{MAX_STANCE_SHIN_RATIO}) | "
                    f"back_heel_lift={max(l_heel_lift, r_heel_lift):.2f} (min {SETUP_BACK_HEEL_MIN_LIFT_RATIO}) | "
                    f"hands_ok={hands_ok}",
                    flush=True,
                )
            return {
                "exercise": "Inline Lunge", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "knee_angle": round(front_knee_angle, 1), "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[INLINE LUNGE FAULT] AI ST-GCN flagged bad posture: prob={bad_posture_prob:.2f} "
                                f"(threshold={ML_FAULT_CONFIDENCE_THRESHOLD}), state={self.state}",
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
        if self.rep_started and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.max_trunk_lean = max(self.max_trunk_lean, trunk_lean)

            # A. Side view must be held for the whole rep
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[INLINE LUNGE FAULT] Lost side view: shoulder_span_ratio={shoulder_span_ratio:.2f} "
                            f"(max={SETUP_MAX_SHOULDER_SPAN_RATIO})",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Torso stays vertical
            if trunk_lean > MAX_TRUNK_LEAN_DEG:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if not self.torso_collapsed:
                        print(
                            f"[INLINE LUNGE FAULT] Torso leaned: trunk_lean={trunk_lean:.1f}° "
                            f"(max={MAX_TRUNK_LEAN_DEG}°)",
                            flush=True,
                        )
                    self.torso_collapsed = True
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # C. Hands stay on hips / behind head
            if not hands_ok:
                self.hands_frames += 1
                if self.hands_frames >= DEBOUNCE_HANDS:
                    if not self.hands_dropped:
                        print(
                            f"[INLINE LUNGE FAULT] Hands left hips/head: wrist_hip_ratio={wrist_hip_ratio:.2f} "
                            f"(max={HANDS_ON_HIPS_MAX_RATIO}), wrist_y={active_wrist.y:.2f} vs "
                            f"shoulder_y+margin={active_shoulder.y + HANDS_AT_HEAD_Y_MARGIN:.2f}",
                            flush=True,
                        )
                    self.hands_dropped = True
            else:
                self.hands_frames = max(0, self.hands_frames - 1)

            # D. Drop straight down: hips must not drift forward/back from the standing position
            if self.baseline_hip_x is not None:
                hip_drift = abs(((l_hip.x + r_hip.x) / 2.0) - self.baseline_hip_x) / torso_len
                self.max_hip_drift = max(self.max_hip_drift, hip_drift)
                if hip_drift > MAX_HIP_DRIFT_RATIO:
                    self.wobble_frames += 1
                    if self.wobble_frames >= DEBOUNCE_WOBBLE:
                        if not self.wobbled:
                            print(
                                f"[INLINE LUNGE FAULT] Hips drifted (not straight down): hip_drift={hip_drift:.2f} "
                                f"(max={MAX_HIP_DRIFT_RATIO})",
                                flush=True,
                            )
                        self.wobbled = True
                else:
                    self.wobble_frames = max(0, self.wobble_frames - 1)

            self.min_back_knee_angle = min(self.min_back_knee_angle, back_knee_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Drop straight down into lunge."
                    print("[INLINE LUNGE] Reset complete: ready for next rep", flush=True)
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Inline Lunge", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "knee_angle": round(front_knee_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drop straight down..."

        elif front_knee_angle <= DESCENDING_START_KNEE_ANGLE and self.state in ["STANDING", "ASCENDING"]:
            self.state = "DESCENDING"
            self.rep_started = True
            self.min_back_knee_angle = back_knee_angle
            self.reversal_frames = 0
            print(f"[INLINE LUNGE] Descent started: front_knee={front_knee_angle:.1f}° (start <= {DESCENDING_START_KNEE_ANGLE}°)", flush=True)
            if self.post_rep_timer == 0:
                self.current_feedback = "Drop straight down..."

        # Full ROM: Both knees near 90-105°
        elif self.state == "DESCENDING" and back_knee_angle <= FULL_LUNGE_MAX_KNEE_ANGLE and front_knee_angle <= FULL_LUNGE_MAX_KNEE_ANGLE:
            self.depth_reached = True
            self.state = "BOTTOM"
            print(
                f"[INLINE LUNGE] Depth reached: front_knee={front_knee_angle:.1f}° back_knee={back_knee_angle:.1f}° "
                f"(both need <= {FULL_LUNGE_MAX_KNEE_ANGLE}°)",
                flush=True,
            )
            if self.post_rep_timer == 0:
                self.current_feedback = "Depth reached: push back up!"

        # Early upward reversal (Half rep branch)
        elif self.state == "DESCENDING" and not self.depth_reached and (back_knee_angle - self.min_back_knee_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_back_knee_angle <= HALF_LUNGE_MAX_KNEE_ANGLE:
                    self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Incomplete depth! Rising early..."
                print(
                    f"[INLINE LUNGE FAULT] Early reversal: min_back_knee={self.min_back_knee_angle:.1f}° "
                    f"(full depth needs <= {FULL_LUNGE_MAX_KNEE_ANGLE}°, half-rep credit <= {HALF_LUNGE_MAX_KNEE_ANGLE}°)",
                    flush=True,
                )

        elif front_knee_angle > ASCENDING_START_KNEE_ANGLE and self.state in ["BOTTOM", "DESCENDING"] and self.rep_started:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through the front heel and the ball of the back foot!"

        if self.state == "DESCENDING" and not ((back_knee_angle - self.min_back_knee_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and front_knee_angle >= REP_COMPLETE_KNEE_ANGLE and back_knee_angle >= REP_COMPLETE_KNEE_ANGLE:
            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to the camera!"
                is_bad_rep = True
            elif self.torso_collapsed:
                self.current_feedback = "Bad rep: Keep your torso vertical! Do not lean."
                is_bad_rep = True
            elif self.hands_dropped:
                self.current_feedback = "Bad rep: Keep hands on your hips or behind your head!"
                is_bad_rep = True
            elif self.wobbled:
                self.current_feedback = "Bad rep: Drop straight down! Do not drift forward or back."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected form breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                if self.min_back_knee_angle <= HALF_LUNGE_MAX_KNEE_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Half rep: Drop back knee lower (to 90 degrees)."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Lunge was too shallow! Drop down deeper."
                    is_bad_rep = True

            # Rep summary with every peak value next to its limit
            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            rep_stats = (
                f"min_back_knee={self.min_back_knee_angle:.1f}° (full <= {FULL_LUNGE_MAX_KNEE_ANGLE}°), "
                f"max_trunk_lean={self.max_trunk_lean:.1f}° (max {MAX_TRUNK_LEAN_DEG}°), "
                f"max_hip_drift={self.max_hip_drift:.2f} (max {MAX_HIP_DRIFT_RATIO}), "
                f"side_view={self.is_side_view}, hands_ok={not self.hands_dropped}, depth={self.depth_reached}"
            )

            if is_bad_rep:
                self.bad_reps += 1
                print(f"[INLINE LUNGE REP #{rep_num} -> BAD REP] Reason: {self.current_feedback} | {rep_stats}", flush=True)
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.half_reps += 1
                print(f"[INLINE LUNGE REP #{rep_num} -> HALF REP] Reason: {self.current_feedback} | {rep_stats}", flush=True)
                self.state = "STANDING"
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect lunge! Clean rep."
                print(f"[INLINE LUNGE REP #{rep_num} -> GOOD REP] {rep_stats}", flush=True)
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Form Score
        form_score = max(0, min(100, int(100 - abs(front_knee_angle - 90) * 0.3))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 35
        if self.torso_collapsed: form_score -= 40
        if self.hands_dropped: form_score -= 15
        if self.wobbled: form_score -= 30
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Inline Lunge", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": round(front_knee_angle, 1), "depth_reached": self.depth_reached
        }