import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIDE LATERAL RAISE - BALANCED FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Front View, Arms Hanging at Sides)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.23 # Enforces front-facing view (slightly looser)
SETUP_MAX_ABDUCTION_ANGLE = 34.0     # Arms hanging down by thighs at start
SETUP_MIN_ELBOW_ANGLE = 110.0        # Arms extended with comfortable soft bend at start
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held at sides to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35      # Minimum upper-body landmark visibility

# 2. Kinematic & Form Thresholds (Loosened for Natural Movement + Strict Bilateral Raise)
MIN_ELBOW_BEND_ANGLE = 105.0         # Comfortable bend allowed (~105°+), catches deep L-bend curling
MAX_TORSO_SWING_OFFSET = 0.15        # Max horizontal shoulder-to-hip lean
SHRUG_NECK_RATIO_FACTOR = 0.60       # Ear-to-shoulder distance shrink ratio catching heavy trap shrugs
WRIST_HIKE_MARGIN = 0.11             # Wrists hiking significantly higher than elbows
MAX_ARM_ASYMMETRY_DEG = 28.0         # Max left vs right abduction difference (catches single-arm raise!)
SINGLE_ARM_LIFT_TRIGGER_DEG = 42.0   # If one arm passes 42° while the other stays < 30°, flags single-arm fault
MAX_OVER_RAISE_ANGLE = 110.0         # Raising weights way above shoulder level (>110°)
OVER_RAISE_WRIST_MARGIN = 0.12       # Wrists rising way above shoulder landmarks
ECCENTRIC_DROP_SPEED_MAX = 32.0      # Free-fall descent speed cutoff (deg/frame)

# 3. Range of Motion & Half-Rep Thresholds
ASCENDING_START_ANGLE = 32.0         # Lifting either arm past 32° initiates ASCENDING
MIN_VALID_REP_PEAK_ANGLE = 40.0      # Ignore minor resting twitches (< 40°)
FULL_RAISE_MIN_ANGLE = 68.0          # BOTH arms reaching shoulder height (~68°–105°) confirms full ROM
DESCENDING_START_ANGLE = 60.0        # Lowering arms from TOP
EARLY_REVERSAL_DELTA = 10.0          # Dropping > 10° before reaching shoulder height flags Half Rep
REP_COMPLETE_ANGLE = 35.0            # Returning both dumbbells to sides completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_ELBOW = 6
DEBOUNCE_TORSO = 6
DEBOUNCE_SHRUG = 6
DEBOUNCE_WRIST = 5
DEBOUNCE_BOTH_ARMS = 4               # Frames before flagging single-arm / uneven arm raise
DEBOUNCE_OVER_RAISE = 4
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6

ML_FAULT_CONFIDENCE_THRESHOLD = 0.82
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class LateralRaiseTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.target_height_reached = False
        self.half_raise_reached = False
        self.max_abduction = 0.0           # Tracks highest bilateral (min_abduction) peak
        self.max_single_arm_abduction = 0.0# Tracks highest single-arm (max_abduction) peak
        self.reversal_frames = 0

        # Instructor Tripwires (FRONT VIEW)
        self.is_front_view = True
        self.both_arms_raised = True       # Enforces BOTH arms raising together!
        self.arms_straight_enough = True
        self.torso_stable = True
        self.shoulders_down = True
        self.elbows_lead_wrists = True
        self.controlled_descent = True
        self.not_over_raised = True
        self.ml_posture_maintained = True

        # Advisory / Warning Flags
        self.knee_bounce_warning = False
        self.baseline_knee_angle = None
        self.baseline_hip_y = None
        self.baseline_neck_ratio = None

        # Debounce counters
        self.front_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.knee_frames = 0
        self.over_raise_frames = 0
        self.straight_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.wrist_frames = 0
        self.balance_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0
        self.post_rep_timer = 0

        self.is_ready = False
        self.prev_abduction = None

        self.current_feedback = "Face camera directly and let both arms hang down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_side_lateral_raise.tflite")
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
        self.target_height_reached = False
        self.half_raise_reached = False
        self.max_abduction = 0.0
        self.max_single_arm_abduction = 0.0
        self.reversal_frames = 0
        self.prev_abduction = None

        self.is_front_view = True
        self.both_arms_raised = True
        self.arms_straight_enough = True
        self.torso_stable = True
        self.shoulders_down = True
        self.elbows_lead_wrists = True
        self.controlled_descent = True
        self.not_over_raised = True
        self.knee_bounce_warning = False
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.knee_frames = 0
        self.over_raise_frames = 0
        self.straight_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.wrist_frames = 0
        self.balance_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0

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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        vis_all = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(vis_all) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Lateral Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show full torso and both arms",
                "elbow_bend": 0.0, "abduction_angle": 0.0
            }

        # Bilateral Kinematics
        l_abduction = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction = calculate_angle(r_hip, r_shoulder, r_elbow)
        min_abduction = min(l_abduction, r_abduction)
        max_abduction = max(l_abduction, r_abduction)
        arm_diff = abs(l_abduction - r_abduction)

        l_elbow_bend = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_bend = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_bend = min(l_elbow_bend, r_elbow_bend)

        # Front View Verification
        torso_height = max(abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0)), 0.30)
        raw_shoulder_span = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_span / torso_height
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO

        shoulder_width = max(raw_shoulder_span, 0.30)
        current_hip_y = (l_hip.y + r_hip.y) / 2.0

        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height

        has_legs = (l_knee.visibility > 0.35 and l_ankle.visibility > 0.35) or (r_knee.visibility > 0.35 and r_ankle.visibility > 0.35)
        current_knee_angle = None
        if has_legs:
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            current_knee_angle = min(l_knee_angle, r_knee_angle)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Front View + Both Arms Resting at Sides)
        # ---------------------------------------------------------
        is_resting_setup = (
            is_facing_front
            and (max_abduction < SETUP_MAX_ABDUCTION_ANGLE)
            and (min_elbow_bend >= SETUP_MIN_ELBOW_ANGLE)
        )

        if is_resting_setup:
            self.standing_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            else:
                self.baseline_neck_ratio = 0.90 * self.baseline_neck_ratio + 0.10 * neck_ratio

            if self.baseline_hip_y is None:
                self.baseline_hip_y = current_hip_y
            else:
                self.baseline_hip_y = 0.90 * self.baseline_hip_y + 0.10 * current_hip_y

            if current_knee_angle is not None:
                if self.baseline_knee_angle is None:
                    self.baseline_knee_angle = current_knee_angle
                else:
                    self.baseline_knee_angle = 0.90 * self.baseline_knee_angle + 0.10 * current_knee_angle

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            else:
                feedback_msg = "Let both arms hang down by your sides to begin"
            return {
                "exercise": "Lateral Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_bend": round(min_elbow_bend, 1), "abduction_angle": round(min_abduction, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER & INFERENCE
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_started:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_vals = np.exp(output_data - np.max(output_data))
                probs = exp_vals / np.sum(exp_vals)

                if probs[0] >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(f"[LATERAL RAISE FAULT] AI ST-GCN flagged posture fault: prob={probs[0]:.2f}", flush=True)
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (BOTH ARMS MUST RAISE + LOOSENED FORM)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            self.max_abduction = max(self.max_abduction, min_abduction)
            self.max_single_arm_abduction = max(self.max_single_arm_abduction, max_abduction)

            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[LATERAL RAISE FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # A. Both Arms Must Raise Together (Catches single-arm raises or large left/right asymmetry!)
            is_single_arm_or_uneven = (
                (max_abduction >= SINGLE_ARM_LIFT_TRIGGER_DEG and min_abduction < ASCENDING_START_ANGLE)
                or (arm_diff > MAX_ARM_ASYMMETRY_DEG)
            )
            if is_single_arm_or_uneven:
                self.balance_frames += 1
                if self.balance_frames >= DEBOUNCE_BOTH_ARMS:
                    if self.both_arms_raised:
                        print(
                            f"[LATERAL RAISE FAULT] Single-arm or uneven raise! L={l_abduction:.1f}°, R={r_abduction:.1f}°",
                            flush=True,
                        )
                    self.both_arms_raised = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Raise BOTH arms together at the same time!"
            else:
                self.balance_frames = max(0, self.balance_frames - 1)

            # B. Elbow Bend (Loosened to 105° so a comfortable soft bend is always allowed)
            if min_elbow_bend < MIN_ELBOW_BEND_ANGLE:
                self.straight_frames += 1
                if self.straight_frames >= DEBOUNCE_ELBOW:
                    if self.arms_straight_enough:
                        print(f"[LATERAL RAISE FAULT] Excessive elbow bend: angle={min_elbow_bend:.1f}°", flush=True)
                    self.arms_straight_enough = False
            else:
                self.straight_frames = max(0, self.straight_frames - 1)

            # C. Momentum Swing
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            if abs(mid_shoulder_x - mid_hip_x) > MAX_TORSO_SWING_OFFSET:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_stable:
                        print("[LATERAL RAISE FAULT] Torso swung/leaned!", flush=True)
                    self.torso_stable = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # D. Shoulder Shrug
            if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR):
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    if self.shoulders_down:
                        print("[LATERAL RAISE FAULT] Shoulders shrugged upward!", flush=True)
                    self.shoulders_down = False
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # E. Wrist Hiking
            if (l_wrist.y < (l_elbow.y - WRIST_HIKE_MARGIN)) or (r_wrist.y < (r_elbow.y - WRIST_HIKE_MARGIN)):
                self.wrist_frames += 1
                if self.wrist_frames >= DEBOUNCE_WRIST:
                    if self.elbows_lead_wrists:
                        print("[LATERAL RAISE FAULT] Wrists hiked above elbows!", flush=True)
                    self.elbows_lead_wrists = False
            else:
                self.wrist_frames = max(0, self.wrist_frames - 1)

            # F. Over-Raise (> 110° abduction or hands way above shoulders)
            wrist_over_shoulder = (
                (l_wrist.y < l_shoulder.y - OVER_RAISE_WRIST_MARGIN)
                or (r_wrist.y < r_shoulder.y - OVER_RAISE_WRIST_MARGIN)
            )
            if max_abduction > MAX_OVER_RAISE_ANGLE or wrist_over_shoulder:
                self.over_raise_frames += 1
                if self.over_raise_frames >= DEBOUNCE_OVER_RAISE:
                    if self.not_over_raised:
                        print(f"[LATERAL RAISE FAULT] Over-raised above shoulder height: max_abd={max_abduction:.1f}°", flush=True)
                    self.not_over_raised = False
            else:
                self.over_raise_frames = max(0, self.over_raise_frames - 1)

            # G. Knee Bounce Warning
            hip_drop_normalized = (current_hip_y - self.baseline_hip_y) / shoulder_width if self.baseline_hip_y else 0.0
            knee_deflection = (
                abs(current_knee_angle - self.baseline_knee_angle)
                if (self.baseline_knee_angle is not None and current_knee_angle is not None) else 0.0
            )
            if (hip_drop_normalized > 0.07 or knee_deflection > 8.0) and self.state == "ASCENDING":
                self.knee_frames += 1
                if self.knee_frames >= 3:
                    self.knee_bounce_warning = True
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (TRIGGERS ON EITHER ARM SO 1-ARM LIFT IS CAUGHT!)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Raise BOTH arms out to shoulder height."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Lateral Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_bend": round(min_elbow_bend, 1), "abduction_angle": round(min_abduction, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise BOTH arms out to shoulder height!"

        # Initiates ASCENDING from RESTING when max_abduction > 32° (catches single-arm raises too!)
        elif self.state == "RESTING" and max_abduction > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            self.rep_started = True
            self.max_abduction = min_abduction
            self.max_single_arm_abduction = max_abduction
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise both dumbbells evenly to shoulder height..."

        # Full ROM requires BOTH arms (min_abduction) to reach shoulder height (>= 68°)
        elif self.state == "ASCENDING" and min_abduction >= FULL_RAISE_MIN_ANGLE and self.both_arms_raised:
            self.target_height_reached = True
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Shoulder height reached! Lower slowly under control."

        # Early downward reversal (works for both 2-arm half-reps and 1-arm raises returning down)
        elif self.state == "ASCENDING" and not self.target_height_reached and (
            (self.max_single_arm_abduction - max_abduction) > EARLY_REVERSAL_DELTA
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.half_raise_reached = True
                self.state = "DESCENDING"
                if not self.both_arms_raised:
                    self.current_feedback = "One arm stayed down! Raise both arms together."
                else:
                    self.current_feedback = "Half rep: Lowering arms early..."
                print(
                    f"[LATERAL RAISE FAULT] Early reversal: min_peak={self.max_abduction:.1f}°, "
                    f"max_peak={self.max_single_arm_abduction:.1f}°",
                    flush=True,
                )

        elif self.state == "TOP" and max_abduction < DESCENDING_START_ANGLE:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the weight down..."

        if self.state == "ASCENDING" and not ((self.max_single_arm_abduction - max_abduction) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Debounced Eccentric Free-Fall Monitor
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_abduction is not None:
                drop_speed = self.prev_abduction - max_abduction
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(f"[LATERAL RAISE FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame", flush=True)
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_abduction = max_abduction
        else:
            self.prev_abduction = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (BOTH ARMS CHECK PRIORITIZED!)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "TOP", "ASCENDING"] and max_abduction <= REP_COMPLETE_ANGLE:
            # Ignore minor resting twitches at the bottom (< 40° on both arms)
            if self.max_single_arm_abduction < MIN_VALID_REP_PEAK_ANGLE and self.is_front_view:
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Lateral Raise", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_bend": round(min_elbow_bend, 1), "abduction_angle": round(min_abduction, 1)
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Return arms to sides to reset."
                is_bad_rep = True
            elif not self.both_arms_raised:
                self.current_feedback = "Bad rep: Raise BOTH arms together! Do not lift only one arm."
                is_bad_rep = True
            elif not self.not_over_raised:
                self.current_feedback = "Bad rep: Raised too high! Stop at shoulder height (90°)."
                is_bad_rep = True
            elif not self.arms_straight_enough:
                self.current_feedback = "Bad rep: Arms bent too much! Keep a soft bend, do not curl."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "Bad rep: Do not swing your torso! Keep body steady."
                is_bad_rep = True
            elif not self.shoulders_down:
                self.current_feedback = "Bad rep: Keep traps relaxed! Do not shrug your shoulders up."
                is_bad_rep = True
            elif not self.elbows_lead_wrists:
                self.current_feedback = "Bad rep: Lead with your elbows! Do not hike wrists above elbows."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Lower weights slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
            elif not self.target_height_reached:
                self.current_feedback = "Bad rep: Half rep! Lift both arms all the way to shoulder height."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[LATERAL RAISE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_peak={self.max_abduction:.1f}°, "
                    f"max_peak={self.max_single_arm_abduction:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Strict raise only, avoid leg bounce."
                else:
                    self.current_feedback = "Clean Lateral Raise! Great bilateral shoulder isolation."
                print(f"[LATERAL RAISE REP #{rep_num} -> GOOD REP] max_abd={self.max_abduction:.1f}°", flush=True)
                self.state = "RESTING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = 100 if self.state == "RESTING" else max(0, min(100, int((min_abduction / 70) * 100)))
        if not self.is_front_view: form_score -= 35
        if not self.both_arms_raised: form_score -= 40
        if not self.arms_straight_enough: form_score -= 20
        if not self.torso_stable: form_score -= 20
        if not self.shoulders_down: form_score -= 20
        if not self.elbows_lead_wrists: form_score -= 15
        if not self.controlled_descent: form_score -= 15
        if not self.not_over_raised: form_score -= 20
        if self.knee_bounce_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 25
        if not self.target_height_reached and self.state == "DESCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Lateral Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_bend": round(min_elbow_bend, 1),
            "abduction_angle": round(min_abduction, 1)
        }