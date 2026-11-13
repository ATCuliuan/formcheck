import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (HURDLE STEP - SIDE / SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Stand Tall, Feet Together, Side View)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.45 # Enforces side profile view
SETUP_MIN_SUPPORT_KNEE_ANGLE = 158.0 # Standing leg must be straight and tall
SETUP_MAX_TRUNK_LEAN_DEG = 16.0      # Torso upright at setup
SETUP_MAX_ANKLE_SPREAD_RATIO = 0.28  # Feet together with toes aligned at base line
SETUP_WRIST_SHOULDER_MARGIN = 0.15   # Hands holding dowel at upper back/shoulders (or behind head)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 12             # Frames standing tall with feet together to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.2       # Minimum side-profile landmark visibility sum

# 2. Hurdle Clearance & Foot Slide Thresholds
# Hurdle is at the tibial tuberosity (just below the standing kneecap)
MIN_KNEE_DRIVE_RATIO = 0.35          # Stepping knee must rise at least 35% of thigh length toward hip
MIN_FOOT_CLEARANCE_RATIO = 0.22      # Stepping ankle must lift at least 22% of shin length off floor to clear hurdle
SLIDE_STRIDE_RATIO = 0.36            # Horizontal foot movement ratio where foot MUST already be elevated (kept above SETUP_MAX_ANKLE_SPREAD_RATIO so a normal stance never counts as a stride)
SLIDE_MIN_ANKLE_LIFT_RATIO = 0.15    # If ankle moves forward > SLIDE_STRIDE_RATIO with < 15% lift -> Foot Slide Fault!

# 3. Step Progression (Lift -> Heel Tap on Other Side -> Return to Start)
STEP_START_LIFT_RATIO = 0.30         # Foot lift (vs its own standing baseline) needed to start a step. Must sit well above resting jitter
STEP_START_KNEE_LIFT_RATIO = 0.15    # Knee lift (vs its own standing baseline) needed to start a step
MIN_ATTEMPT_FOOT_LIFT_RATIO = 0.30   # A movement that never reached this foot lift AND the knee lift below is a false start (ignored, not a bad rep)
MIN_ATTEMPT_KNEE_LIFT_RATIO = 0.15
MIN_TAP_FORWARD_STRIDE_RATIO = 0.32  # Minimum horizontal distance stepping foot must reach past standing foot for heel tap
TAP_FLOOR_TOLERANCE_RATIO = 0.15     # How close the heel (or ankle) must be to its standing height to count as touching the floor
RETURN_FEET_TOGETHER_RATIO = 0.25    # Horizontal ankle distance ratio to confirm foot returned to start

# 4. Kinematic Form Thresholds (Standing Leg Alignment & Upright Torso)
MIN_SUPPORT_KNEE_ANGLE = 142.0       # Standing leg must stay straight/aligned (no knee caving/buckling)
MAX_TRUNK_LEAN_DEG = 9.0            # Max forward/backward torso lean during step
MAX_HIP_WOBBLE_RATIO = 0.22          # Max horizontal hip shift of standing leg (prevents twisting/wobbling)

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 5
DEBOUNCE_SUPPORT_KNEE = 5
DEBOUNCE_TORSO = 5
DEBOUNCE_HANDS = 6
DEBOUNCE_SLIDE = 2                   # Fast trigger to catch sliding foot along floor
DEBOUNCE_WOBBLE = 5
DEBOUNCE_STEP_START = 4              # Frames the lift must be sustained before a step starts
DEBOUNCE_FALSE_START = 4             # Frames feet must be back together before a tiny movement is cancelled
DEBOUNCE_ML_FAULT = 6

# 6. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class HurdleStepTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        # Hurdle Step Cycle Stages:
        # "STANDING"      -> Feet together at base line
        # "LIFTING_OVER"  -> Elevating knee & foot up over tibial-tuberosity hurdle
        # "HEEL_TAPPED"   -> Foot tapped floor on the far side of the hurdle
        # "RETURNING"     -> Lifting foot back over hurdle to starting position
        # "LOCKED_FAULT"  -> Frozen after bad rep until user resets in standing position
        self.rep_started = False
        self.elevated_first = False          # Confirms foot lifted UP before moving forward
        self.hurdle_cleared_forward = False  # Confirms foot/knee cleared hurdle height going forward
        self.heel_tapped_front = False       # Confirms heel tapped floor on the other side
        self.max_knee_lift_ratio = 0.0
        self.max_foot_lift_ratio = 0.0
        self.max_forward_stride_ratio = 0.0

        # Sagittal Posture Tripwires
        self.is_side_view = True
        self.support_leg_aligned = True      # Standing ankle-knee-hip stays straight
        self.torso_upright = True            # Torso stays upright
        self.hands_positioned = True         # Hands stay at shoulders/dowel
        self.foot_slid = False               # Catches sliding foot forward without elevating first
        self.no_hip_wobble = True            # Catches excessive pelvis drift/twisting
        self.ml_posture_maintained = True

        # Baselines recorded at standing setup
        # Per-leg baselines: in a side view one foot/knee always appears lower than the other,
        # so lift must be measured against each leg's own resting position.
        self.baseline_l_ankle_y = None
        self.baseline_r_ankle_y = None
        self.baseline_l_knee_y = None
        self.baseline_r_knee_y = None
        self.baseline_l_heel_y = None
        self.baseline_r_heel_y = None
        self.baseline_hip_x = None

        # Debounce counters
        self.profile_frames = 0
        self.support_bend_frames = 0
        self.torso_lean_frames = 0
        self.hands_drop_frames = 0
        self.slide_frames = 0
        self.wobble_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.tap_frames = 0
        self.return_frames = 0
        self.start_frames = 0
        self.false_start_frames = 0
        self.min_far_floor_contact = 9.99    # Diagnostic: closest the foot got to the floor while past the tap distance
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Side view: Stand tall with feet together, hands at shoulders"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_hurdle_step.tflite")
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
        self.elevated_first = False
        self.hurdle_cleared_forward = False
        self.heel_tapped_front = False
        self.max_knee_lift_ratio = 0.0
        self.max_foot_lift_ratio = 0.0
        self.max_forward_stride_ratio = 0.0

        self.is_side_view = True
        self.support_leg_aligned = True
        self.torso_upright = True
        self.hands_positioned = True
        self.foot_slid = False
        self.no_hip_wobble = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.support_bend_frames = 0
        self.torso_lean_frames = 0
        self.hands_drop_frames = 0
        self.slide_frames = 0
        self.wobble_frames = 0
        self.ml_fault_frames = 0
        self.tap_frames = 0
        self.return_frames = 0
        self.start_frames = 0
        self.false_start_frames = 0
        self.min_far_floor_contact = 9.99

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value]
        r_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Hurdle Step",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Show full body side profile",
                "knee_angle": 0.0,
                "hip_angle": 0.0,
                "depth_reached": False
            }

        # Primary visible side for torso/arm reference
        if r_vis >= l_vis:
            active_shoulder, active_hip, active_wrist = r_shoulder, r_hip, r_wrist
        else:
            active_shoulder, active_hip, active_wrist = l_shoulder, l_hip, l_wrist

        # Body scale references
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        l_shin_len = np.hypot(l_knee.x - l_ankle.x, l_knee.y - l_ankle.y)
        r_shin_len = np.hypot(r_knee.x - r_ankle.x, r_knee.y - r_ankle.y)
        shin_len = max(l_shin_len, r_shin_len, 0.30)

        # 1. Side Profile Check
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # 2. Sagittal Trunk Inclination (0° = dead vertical upright)
        dx_trunk = abs(active_shoulder.x - active_hip.x)
        dy_trunk = max(abs(active_shoulder.y - active_hip.y), 0.10)
        trunk_lean_from_vertical = np.degrees(np.arctan2(dx_trunk, dy_trunk))

        # 3. Dowel / Hands at Upper Back or Shoulders Check
        wrist_at_shoulders = active_wrist.y <= (active_shoulder.y + SETUP_WRIST_SHOULDER_MARGIN)

        # 4. Bilateral Knee Angles & Dynamic Standing vs Stepping Leg Identification
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # The stepping leg is whichever knee/ankle is lifted higher (smaller Y) or extended further forward
        if (l_knee.y + l_ankle.y) < (r_knee.y + r_ankle.y):
            stepping_knee, stepping_ankle, stepping_hip = l_knee, l_ankle, l_hip
            support_knee, support_ankle, support_hip = r_knee, r_ankle, r_hip
            support_knee_angle = r_knee_angle
        else:
            stepping_knee, stepping_ankle, stepping_hip = r_knee, r_ankle, r_hip
            support_knee, support_ankle, support_hip = l_knee, l_ankle, l_hip
            support_knee_angle = l_knee_angle

        # Horizontal & Vertical Foot/Knee Metrics
        ankle_stride_ratio = abs(l_ankle.x - r_ankle.x) / torso_len
        feet_together_x = ankle_stride_ratio <= SETUP_MAX_ANKLE_SPREAD_RATIO

        # How high each foot/knee has lifted above ITS OWN standing baseline.
        # (Old version compared against max(ankle y), so the naturally higher-looking leg in a
        # side view read as permanently "lifted" and tiny movements started false reps.)
        if self.baseline_l_ankle_y is not None:
            l_foot_lift = max(0.0, (self.baseline_l_ankle_y - l_ankle.y) / shin_len)
            r_foot_lift = max(0.0, (self.baseline_r_ankle_y - r_ankle.y) / shin_len)
            l_knee_lift = max(0.0, (self.baseline_l_knee_y - l_knee.y) / torso_len)
            r_knee_lift = max(0.0, (self.baseline_r_knee_y - r_knee.y) / torso_len)
            foot_lift_ratio = max(l_foot_lift, r_foot_lift)
            knee_lift_ratio = max(l_knee_lift, r_knee_lift)

            # Floor contact: a heel tap lands with the toes up, so the ANKLE stays well above its
            # standing height even though the heel is on the floor. Count a foot as touching down
            # if EITHER its heel or its ankle is back near standing height.
            l_heel_lift = max(0.0, (self.baseline_l_heel_y - l_heel.y) / shin_len)
            r_heel_lift = max(0.0, (self.baseline_r_heel_y - r_heel.y) / shin_len)
            floor_contact_ratio = max(min(l_foot_lift, l_heel_lift), min(r_foot_lift, r_heel_lift))
        else:
            # No standing baseline captured yet
            foot_lift_ratio = 0.0
            knee_lift_ratio = 0.0
            floor_contact_ratio = 0.0

        both_feet_on_floor = floor_contact_ratio <= TAP_FLOOR_TOLERANCE_RATIO
        is_standing_setup = (
            is_side
            and (l_knee_angle > SETUP_MIN_SUPPORT_KNEE_ANGLE)
            and (r_knee_angle > SETUP_MIN_SUPPORT_KNEE_ANGLE)
            and (trunk_lean_from_vertical <= SETUP_MAX_TRUNK_LEAN_DEG)
            and feet_together_x
            and both_feet_on_floor
            and wrist_at_shoulders
        )

        # ---------------------------------------------------------
        # 1. SAGITTAL SETUP GUARD
        # ---------------------------------------------------------
        if is_standing_setup:
            self.standing_frames += 1
            if self.baseline_l_ankle_y is None:
                self.baseline_l_ankle_y = l_ankle.y
                self.baseline_r_ankle_y = r_ankle.y
                self.baseline_l_knee_y = l_knee.y
                self.baseline_r_knee_y = r_knee.y
                self.baseline_l_heel_y = l_heel.y
                self.baseline_r_heel_y = r_heel.y
                self.baseline_hip_x = active_hip.x
            else:
                self.baseline_l_ankle_y = 0.90 * self.baseline_l_ankle_y + 0.10 * l_ankle.y
                self.baseline_r_ankle_y = 0.90 * self.baseline_r_ankle_y + 0.10 * r_ankle.y
                self.baseline_l_knee_y = 0.90 * self.baseline_l_knee_y + 0.10 * l_knee.y
                self.baseline_r_knee_y = 0.90 * self.baseline_r_knee_y + 0.10 * r_knee.y
                self.baseline_l_heel_y = 0.90 * self.baseline_l_heel_y + 0.10 * l_heel.y
                self.baseline_r_heel_y = 0.90 * self.baseline_r_heel_y + 0.10 * r_heel.y
                self.baseline_hip_x = 0.90 * self.baseline_hip_x + 0.10 * active_hip.x

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (show full side profile)"
            elif not wrist_at_shoulders:
                feedback_msg = "Rest dowel/hands across upper back/shoulders"
            elif not feet_together_x:
                feedback_msg = "Stand tall with feet together touching the base line"
            else:
                feedback_msg = "Stand tall with knees straight to begin"

            return {
                "exercise": "Hurdle Step",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "knee_angle": round(support_knee_angle, 1),
                "hip_angle": round(trunk_lean_from_vertical, 1),
                "depth_reached": False
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
                out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[HURDLE STEP FAULT] AI ST-GCN flagged poor stability/posture: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. SAGITTAL POSTURE TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LIFTING_OVER", "HEEL_TAPPED", "RETURNING"]:
            self.max_knee_lift_ratio = max(self.max_knee_lift_ratio, knee_lift_ratio)
            self.max_foot_lift_ratio = max(self.max_foot_lift_ratio, foot_lift_ratio)
            self.max_forward_stride_ratio = max(self.max_forward_stride_ratio, ankle_stride_ratio)

            # Check if user elevated foot vertically BEFORE sliding forward
            if foot_lift_ratio >= MIN_FOOT_CLEARANCE_RATIO and knee_lift_ratio >= (MIN_KNEE_DRIVE_RATIO * 0.6):
                self.elevated_first = True

            # Check full hurdle clearance (tibial tuberosity height)
            if self.elevated_first and foot_lift_ratio >= MIN_FOOT_CLEARANCE_RATIO and knee_lift_ratio >= MIN_KNEE_DRIVE_RATIO:
                self.hurdle_cleared_forward = True

            # A. Foot Slide Cheat: Foot moves horizontally out without elevating off the floor first!
            if not self.elevated_first and ankle_stride_ratio > SLIDE_STRIDE_RATIO and foot_lift_ratio < SLIDE_MIN_ANKLE_LIFT_RATIO:
                self.slide_frames += 1
                if self.slide_frames >= DEBOUNCE_SLIDE:
                    if not self.foot_slid:
                        print(
                            f"[HURDLE STEP FAULT] Foot slid forward without elevating first! "
                            f"stride_ratio={ankle_stride_ratio:.2f}, foot_lift_ratio={foot_lift_ratio:.2f}",
                            flush=True,
                        )
                    self.foot_slid = True
            else:
                self.slide_frames = max(0, self.slide_frames - 1)

            # B. Lost Side Profile
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[HURDLE STEP FAULT] Lost side view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # C. Standing (Supporting) Leg Buckle / Alignment Loss
            if support_knee_angle < MIN_SUPPORT_KNEE_ANGLE:
                self.support_bend_frames += 1
                if self.support_bend_frames >= DEBOUNCE_SUPPORT_KNEE:
                    if self.support_leg_aligned:
                        print(
                            f"[HURDLE STEP FAULT] Standing leg buckled/bent: support_knee={support_knee_angle:.1f}° "
                            f"(min={MIN_SUPPORT_KNEE_ANGLE}°)",
                            flush=True,
                        )
                    self.support_leg_aligned = False
            else:
                self.support_bend_frames = max(0, self.support_bend_frames - 1)

            # D. Upright Torso Check (No bowing forward or leaning back)
            if trunk_lean_from_vertical > MAX_TRUNK_LEAN_DEG:
                self.torso_lean_frames += 1
                if self.torso_lean_frames >= DEBOUNCE_TORSO:
                    if self.torso_upright:
                        print(
                            f"[HURDLE STEP FAULT] Torso leaned: lean={trunk_lean_from_vertical:.1f}° "
                            f"(max={MAX_TRUNK_LEAN_DEG}°)",
                            flush=True,
                        )
                    self.torso_upright = False
            else:
                self.torso_lean_frames = max(0, self.torso_lean_frames - 1)

            # E. Standing Leg Weight Centered / No Hip Wobble
            if self.baseline_hip_x is not None:
                hip_wobble = abs(support_hip.x - self.baseline_hip_x) / torso_len
                if hip_wobble > MAX_HIP_WOBBLE_RATIO:
                    self.wobble_frames += 1
                    if self.wobble_frames >= DEBOUNCE_WOBBLE:
                        if self.no_hip_wobble:
                            print(f"[HURDLE STEP FAULT] Excessive hip wobble/shift: {hip_wobble:.2f}", flush=True)
                        self.no_hip_wobble = False
                else:
                    self.wobble_frames = max(0, self.wobble_frames - 1)

            # F. Hands Dropped From Shoulders/Dowel
            if not wrist_at_shoulders:
                self.hands_drop_frames += 1
                if self.hands_drop_frames >= DEBOUNCE_HANDS:
                    if self.hands_positioned:
                        print("[HURDLE STEP FAULT] Hands dropped from dowel/shoulders", flush=True)
                    self.hands_positioned = False
            else:
                self.hands_drop_frames = max(0, self.hands_drop_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (LIFT -> HEEL TAP -> RETURN TO START = 1 REP)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Lift leg up over hurdle, tap heel, and return."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Hurdle Step",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "knee_angle": round(support_knee_angle, 1),
                "hip_angle": round(trunk_lean_from_vertical, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lift one leg up over hurdle, tap heel in front, and return!"

        # Phase 1: Start Step (Either foot lifts up OR foot slides out)
        elif self.state == "STANDING":
            is_starting = (
                foot_lift_ratio > STEP_START_LIFT_RATIO
                or knee_lift_ratio > STEP_START_KNEE_LIFT_RATIO
                or ankle_stride_ratio > SLIDE_STRIDE_RATIO
            )
            if is_starting:
                self.start_frames += 1
            else:
                self.start_frames = max(0, self.start_frames - 1)

            if self.start_frames >= DEBOUNCE_STEP_START:
                self.start_frames = 0
                self.state = "LIFTING_OVER"
                self.rep_started = True
                if foot_lift_ratio >= SLIDE_MIN_ANKLE_LIFT_RATIO or knee_lift_ratio >= STEP_START_KNEE_LIFT_RATIO:
                    self.elevated_first = True
                if self.post_rep_timer == 0:
                    self.current_feedback = "Lift knee high over hurdle, then tap heel on the other side..."

        # Phase 2: Heel Tap on the Far Side (Foot touches down horizontally forward of standing foot)
        elif self.state == "LIFTING_OVER":
            if ankle_stride_ratio >= MIN_TAP_FORWARD_STRIDE_RATIO:
                self.min_far_floor_contact = min(self.min_far_floor_contact, floor_contact_ratio)

            is_tapping_far_side = (
                ankle_stride_ratio >= MIN_TAP_FORWARD_STRIDE_RATIO
                and both_feet_on_floor
            )
            if is_tapping_far_side:
                self.tap_frames += 1
                if self.tap_frames >= 2:
                    self.heel_tapped_front = True
                    self.state = "HEEL_TAPPED"
                    self.tap_frames = 0
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Heel tapped! Lift leg back over hurdle to starting position."
            else:
                # Decay instead of hard reset so one noisy frame does not wipe out a real tap
                self.tap_frames = max(0, self.tap_frames - 1)

            # Premature abort: user puts foot right back down at start without ever reaching far side
            if (
                not self.heel_tapped_front
                and (self.max_foot_lift_ratio > 0.15 or self.max_forward_stride_ratio > 0.20)
                and feet_together_x
                and both_feet_on_floor
            ):
                self.return_frames += 1
                if self.return_frames >= 4:
                    self.state = "RETURNING"
            else:
                self.return_frames = 0

        # Phase 3: Returning Leg Back Over Hurdle to Starting Point
        elif self.state == "HEEL_TAPPED":
            if foot_lift_ratio > STEP_START_LIFT_RATIO or ankle_stride_ratio < (MIN_TAP_FORWARD_STRIDE_RATIO - 0.05):
                self.state = "RETURNING"
                if self.post_rep_timer == 0:
                    self.current_feedback = "Return foot with control back to starting line..."

        # False start guard: weight shifts, landmark jitter or a tiny foot lift that never became
        # a real step attempt. Cancel silently: no rep counted and no LOCKED_FAULT.
        # (A foot slide is NOT a false start, it must still be flagged as a bad rep.)
        if (
            self.rep_started
            and self.state in ["LIFTING_OVER", "RETURNING"]
            and not self.heel_tapped_front
            and not self.foot_slid
            and self.max_foot_lift_ratio < MIN_ATTEMPT_FOOT_LIFT_RATIO
            and self.max_knee_lift_ratio < MIN_ATTEMPT_KNEE_LIFT_RATIO
            and self.max_forward_stride_ratio < SLIDE_STRIDE_RATIO
            and feet_together_x
            and both_feet_on_floor
        ):
            self.false_start_frames += 1
            if self.false_start_frames >= DEBOUNCE_FALSE_START:
                self.state = "STANDING"
                self._reset_rep_tracking()
        else:
            self.false_start_frames = max(0, self.false_start_frames - 1)

        # ---------------------------------------------------------
        # 5. REP COMPLETION (WHEN FOOT RETURNS TO STARTING POINT)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["RETURNING", "HEEL_TAPPED"] and feet_together_x and both_feet_on_floor:
            self.return_frames += 1
            if self.return_frames >= 3:
                is_bad_rep = False
                is_half_rep = False

                if self.foot_slid or not self.elevated_first:
                    self.current_feedback = "Bad rep: Do not slide foot! Elevate knee and foot over hurdle first."
                    is_bad_rep = True
                elif not self.is_side_view:
                    self.current_feedback = "Bad rep: Stay sideways to the camera! Stand tall to reset."
                    is_bad_rep = True
                elif not self.hurdle_cleared_forward:
                    self.current_feedback = "Bad rep: Lift knee and foot higher to clear tibial hurdle height!"
                    is_bad_rep = True
                elif not self.heel_tapped_front:
                    self.current_feedback = "Bad rep: Half rep! Step over hurdle and tap heel in front before returning."
                    is_half_rep = True
                elif not self.support_leg_aligned:
                    self.current_feedback = "Bad rep: Keep standing leg straight and aligned! Do not bend support knee."
                    is_bad_rep = True
                elif not self.torso_upright:
                    self.current_feedback = "Bad rep: Keep torso upright! Do not bow forward or lean back."
                    is_bad_rep = True
                elif not self.no_hip_wobble:
                    self.current_feedback = "Bad rep: Keep weight centered on standing leg! Avoid wobbling."
                    is_bad_rep = True
                elif not self.hands_positioned:
                    self.current_feedback = "Bad rep: Keep hands/dowel across your shoulders!"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "Bad rep: AI flagged balance or posture breakdown."
                    is_bad_rep = True

                rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
                if is_bad_rep or is_half_rep:
                    self.bad_reps += 1
                    print(
                        f"[HURDLE STEP REP #{rep_num} -> BAD REP] "
                        f"Reason: {self.current_feedback} | max_knee_lift={self.max_knee_lift_ratio:.2f}, "
                        f"max_foot_lift={self.max_foot_lift_ratio:.2f}, tapped={self.heel_tapped_front}, "
                        f"max_stride={self.max_forward_stride_ratio:.2f} (need >={MIN_TAP_FORWARD_STRIDE_RATIO}), "
                        f"closest_floor_contact_past_tap_distance={self.min_far_floor_contact:.2f} (need <={TAP_FLOOR_TOLERANCE_RATIO})",
                        flush=True,
                    )
                    self.state = "LOCKED_FAULT"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                else:
                    self.good_reps += 1
                    self.current_feedback = "Clean Hurdle Step! Great elevation, heel tap, and return."
                    print(
                        f"[HURDLE STEP REP #{rep_num} -> GOOD REP] "
                        f"max_knee_lift={self.max_knee_lift_ratio:.2f}, max_foot_lift={self.max_foot_lift_ratio:.2f}",
                        flush=True,
                    )
                    self.state = "STANDING"
                    self._reset_rep_tracking()

                self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = 100
        if self.foot_slid: form_score -= 40
        if not self.support_leg_aligned: form_score -= 30
        if not self.torso_upright: form_score -= 25
        if not self.no_hip_wobble: form_score -= 20
        if not self.hands_positioned: form_score -= 15
        if not self.is_side_view: form_score -= 35
        form_score = max(0, form_score)

        return {
            "exercise": "Hurdle Step",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(support_knee_angle, 1),
            "hip_angle": round(trunk_lean_from_vertical, 1),
            "depth_reached": self.hurdle_cleared_forward and self.heel_tapped_front
        }