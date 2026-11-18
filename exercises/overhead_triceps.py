import math
import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (OVERHEAD TRICEPS - BEGINNER SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Side View, Standing Tall, Arms Overhead & Tucked)
SETUP_MAX_SIDE_SPAN_RATIO = 0.48      # Enforces side profile (allows slight 3/4 home angle)
SETUP_MIN_ELBOW_ANGLE = 142.0         # Relaxed top lockout (~142 deg+) safe for beginner elbows
SETUP_MIN_SHOULDER_FLEXION = 138.0    # Upper arms raised overhead near ears
SETUP_MAX_BACK_ARCH_RATIO = 0.35      # Neutral vertical torso at setup
SETUP_MIN_UPPER_ARM_RATIO = 0.42      # Rejects starting with wide flared elbows (2D foreshortening)
SETUP_MIN_ELBOW_ABOVE_EAR = 0.04      # Elbow must sit above ear/head level at setup
SETUP_MAX_ELBOW_HEAD_X_RATIO = 0.32   # Elbow must stay horizontally close to head at setup
SETUP_MIN_READY_FRAMES = 6            # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10              # Frames held at overhead top to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35       # Minimum active-side upper-body landmark visibility
OUT_OF_VIEW_RESET_FRAMES = 15         # Frames out of view before full state reset
REST_EXIT_SHOULDER_FLEXION = 105.0    # Dropping arms below 105 deg while resting exits to WAITING
REST_EXIT_FRAMES = 12                 # Frames with arms lowered to exit cleanly between sets

# 2. Kinematic & Form Thresholds (Beginner-Friendly Overhead Isolation)
MIN_SHOULDER_FLEXION = 135.0          # Upper arm dropping below 135 deg = elbows drifting forward
MIN_UPPER_ARM_RATIO = 0.40            # Hard floor for 2D upper-arm length / torso height (wide flare)
FLARE_UPPER_ARM_SHRINK_FACTOR = 0.80  # >20% drop from setup upper-arm length flags lateral elbow flare
MIN_ELBOW_ABOVE_EAR_RATIO = 0.02      # Elbow dropping to/below ear height flags wide/collapsed elbows
MAX_ELBOW_HEAD_X_RATIO = 0.35         # Elbow drifting horizontally far from head/ear
ELBOW_SPREAD_CHECK_MIN_SPAN = 0.20    # Min shoulder span ratio (3/4 view) to enable bilateral elbow spread
MAX_ELBOW_SPREAD_TO_SHOULDER = 1.48   # Bilateral elbow width > 1.48x shoulder span flags wide elbows
MAX_WRIST_FORWARD_RATIO = 0.14        # Wrist folding forward in front of elbow/face instead of behind head
MIN_FACING_NOSE_OFFSET = 0.05         # Minimum nose-to-shoulder X offset to trust facing direction
ADVISORY_ELBOW_HEAD_X_RATIO = 0.26    # Mild elbow separation reminder without failing rep
MAX_BACK_ARCH_RATIO = 0.38            # Horizontal shoulder-to-hip lean/arch ratio
ADVISORY_BACK_ARCH_RATIO = 0.28       # Mild arch reminder without failing rep
ADVISORY_HEAD_FORWARD_RATIO = 0.28    # Forward head poke / neck jutting reminder
ECCENTRIC_DROP_SPEED_MAX = 26.0       # Dropping weight behind head too fast (deg/frame)

# 3. Range of Motion & Half-Rep Thresholds
DESCENDING_START_ANGLE = 132.0        # Bending elbow below 132 deg initiates DESCENDING
MIN_START_SHOULDER_FLEXION = 110.0    # Arms must still be overhead to start a rep from RESTING
MIN_VALID_REP_ELBOW_ANGLE = 120.0     # Ignore minor resting twitches overhead (> 120 deg)
BOTTOM_TARGET_ANGLE = 100.0           # Forgiving beginner bend (<= 100 deg behind head)
ASCENDING_START_ANGLE = 110.0         # Extending from BOTTOM initiates ASCENDING
EARLY_REVERSAL_DELTA = 12.0           # Pressing back up > 12 deg before depth flags Half Rep
REP_COMPLETE_ANGLE = 142.0            # Extending arms overhead completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_SIDE = 4
DEBOUNCE_ELBOW_DRIFT = 5
DEBOUNCE_ELBOW_FLARE = 5
DEBOUNCE_FORWARD_FOLD = 5
DEBOUNCE_BACK_ARCH = 5
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6

ML_FAULT_CONFIDENCE_THRESHOLD = 0.82
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class OverheadTricepsTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_ext_reached = False
        self.min_elbow_angle = 180.0
        self.min_upper_arm_ratio = 1.0
        self.max_elbow_head_x = 0.0
        self.reversal_frames = 0
        self.active_side = None

        # Instructor Tripwires (SIDE VIEW)
        self.is_side_view = True
        self.elbows_tucked = True         # Elbows stay close to head (no wide lateral flare)
        self.behind_head_path = True      # Forearm lowers behind head, not forward in front of face
        self.upper_arm_stable = True      # Upper arms stay pointing up (no forward drift)
        self.spine_neutral = True         # No excessive lower-back arch
        self.controlled_descent = True    # No uncontrolled drop behind head
        self.ml_posture_maintained = True

        # Advisory / Warning Flags & Baselines
        self.mild_arch_warning = False
        self.mild_flare_warning = False
        self.head_forward_warning = False
        self.knee_bounce_warning = False
        self.baseline_upper_arm_ratio = None
        self.baseline_knee_angle = None

        # Debounce & Lifecycle counters
        self.side_frames = 0
        self.setup_frames = 0
        self.unlock_frames = 0
        self.out_of_view_frames = 0
        self.rest_exit_frames = 0
        self.drift_frames = 0
        self.flare_frames = 0
        self.mild_flare_frames = 0
        self.forward_fold_frames = 0
        self.arch_frames = 0
        self.mild_arch_frames = 0
        self.head_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0
        self.post_rep_timer = 0

        self.is_ready = False
        self.prev_elbow_angle = None

        self.current_feedback = "Turn sideways, stand tall, and tuck arms overhead close to head"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_overhead_triceps.tflite")
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
        self.half_ext_reached = False
        self.min_elbow_angle = 180.0
        self.min_upper_arm_ratio = 1.0
        self.max_elbow_head_x = 0.0
        self.reversal_frames = 0
        self.prev_elbow_angle = None

        self.is_side_view = True
        self.elbows_tucked = True
        self.behind_head_path = True
        self.upper_arm_stable = True
        self.spine_neutral = True
        self.controlled_descent = True
        self.mild_arch_warning = False
        self.mild_flare_warning = False
        self.head_forward_warning = False
        self.knee_bounce_warning = False
        self.ml_posture_maintained = True

        self.side_frames = 0
        self.drift_frames = 0
        self.flare_frames = 0
        self.mild_flare_frames = 0
        self.forward_fold_frames = 0
        self.arch_frames = 0
        self.mild_arch_frames = 0
        self.head_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
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

        # Ear excluded from hard visibility guard because tucked overhead arms can occlude the ear
        l_core = (l_shoulder, l_elbow, l_wrist, l_hip)
        r_core = (r_shoulder, r_elbow, r_wrist, r_hip)
        l_vis_sum = sum(p.visibility for p in l_core)
        r_vis_sum = sum(p.visibility for p in r_core)
        l_vis_min = min(p.visibility for p in l_core)
        r_vis_min = min(p.visibility for p in r_core)

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis_min, r_vis_min) < MIN_VISIBILITY_THRESHOLD:
            self.out_of_view_frames += 1
            self.prev_elbow_angle = None
            self.frame_buffer.clear()
            if self.out_of_view_frames >= OUT_OF_VIEW_RESET_FRAMES:
                self.state = "WAITING"
                self.is_ready = False
                self.setup_frames = 0
                self.unlock_frames = 0
                self.active_side = None
                self._reset_rep_tracking()
            return {
                "exercise": "Overhead Triceps", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Full side profile and overhead arms must be visible",
                "elbow_angle": 0.0, "shoulder_flexion": 0.0, "upper_arm_ratio": 0.0
            }

        self.out_of_view_frames = 0

        # Dynamic Side Selection with Active-Rep Lock (prevents mid-rep left/right flicker noise)
        if not self.rep_started or self.active_side is None:
            self.active_side = "RIGHT" if r_vis_sum > l_vis_sum else "LEFT"
        elif self.active_side == "RIGHT" and r_vis_min < MIN_VISIBILITY_THRESHOLD and l_vis_min >= MIN_VISIBILITY_THRESHOLD:
            self.active_side = "LEFT"
            self.prev_elbow_angle = None
        elif self.active_side == "LEFT" and l_vis_min < MIN_VISIBILITY_THRESHOLD and r_vis_min >= MIN_VISIBILITY_THRESHOLD:
            self.active_side = "RIGHT"
            self.prev_elbow_angle = None

        if self.active_side == "RIGHT":
            active_ear = r_ear
            active_shoulder, active_elbow, active_wrist, active_hip = r_core
            active_knee, active_ankle = r_knee, r_ankle
        else:
            active_ear = l_ear
            active_shoulder, active_elbow, active_wrist, active_hip = l_core
            active_knee, active_ankle = l_knee, l_ankle

        # Core Side-View Kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        shoulder_flexion = calculate_angle(active_hip, active_shoulder, active_elbow)
        torso_height = max(abs(active_shoulder.y - active_hip.y), 0.20)
        raw_shoulder_span = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_span / torso_height
        is_facing_side = shoulder_span_ratio <= SETUP_MAX_SIDE_SPAN_RATIO

        back_arch_ratio = abs(active_hip.x - active_shoulder.x) / torso_height

        # Head reference with nose/shoulder fallback when tucked bicep occludes ear
        ear_visible = active_ear.visibility >= MIN_VISIBILITY_THRESHOLD
        head_ref_y = active_ear.y if ear_visible else nose.y
        head_ref_x = active_ear.x if ear_visible else active_shoulder.x

        # Facing Direction (to verify forearm lowers BEHIND head, not forward in front of face)
        nose_offset_ratio = (nose.x - active_shoulder.x) / torso_height
        facing_known = nose.visibility >= MIN_VISIBILITY_THRESHOLD and abs(nose_offset_ratio) >= MIN_FACING_NOSE_OFFSET
        facing_dir = 1.0 if nose_offset_ratio >= 0 else -1.0
        wrist_forward_ratio = ((active_wrist.x - active_elbow.x) * facing_dir) / torso_height if facing_known else 0.0

        # Wide / Flared Elbow Kinematics (Side View + 3/4 View)
        # 1. 2D upper-arm length shrinks when elbow flares laterally out of the sagittal plane
        upper_arm_len_2d = math.hypot(active_shoulder.x - active_elbow.x, active_shoulder.y - active_elbow.y)
        upper_arm_ratio = upper_arm_len_2d / torso_height
        # 2. Elbow drops toward/below ear level when flared wide out to the sides
        elbow_above_ear_ratio = (head_ref_y - active_elbow.y) / torso_height
        # 3. Horizontal distance between elbow and head/ear
        elbow_head_x_ratio = abs(active_elbow.x - head_ref_x) / torso_height
        # 4. Bilateral elbow spread when angled slightly (3/4 view) and both elbows are visible
        both_elbows_visible = min(l_elbow.visibility, r_elbow.visibility) >= 0.45
        elbow_spread_ratio = 0.0
        if both_elbows_visible and shoulder_span_ratio >= ELBOW_SPREAD_CHECK_MIN_SPAN:
            elbow_spread_ratio = abs(l_elbow.x - r_elbow.x) / max(raw_shoulder_span, 0.10)

        # Optional Forward Head & Knee Bounce Checks
        head_forward_ratio = abs(active_ear.x - active_shoulder.x) / torso_height if ear_visible else 0.0
        has_legs = active_knee.visibility >= MIN_VISIBILITY_THRESHOLD and active_ankle.visibility >= MIN_VISIBILITY_THRESHOLD
        current_knee_angle = calculate_angle(active_hip, active_knee, active_ankle) if has_legs else None

        # Combined Wide-Elbow Check (Setup & Active)
        is_wide_at_setup = (
            (upper_arm_ratio < SETUP_MIN_UPPER_ARM_RATIO)
            or (elbow_above_ear_ratio < SETUP_MIN_ELBOW_ABOVE_EAR)
            or (elbow_head_x_ratio > SETUP_MAX_ELBOW_HEAD_X_RATIO)
            or (elbow_spread_ratio > MAX_ELBOW_SPREAD_TO_SHOULDER)
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Side View + Standing Tall + Arms Overhead & Tucked)
        # ---------------------------------------------------------
        is_resting_setup = (
            is_facing_side
            and (elbow_angle >= SETUP_MIN_ELBOW_ANGLE)
            and (shoulder_flexion >= SETUP_MIN_SHOULDER_FLEXION)
            and (back_arch_ratio <= SETUP_MAX_BACK_ARCH_RATIO)
            and not is_wide_at_setup
        )

        if is_resting_setup:
            self.setup_frames += 1
            if self.baseline_upper_arm_ratio is None:
                self.baseline_upper_arm_ratio = upper_arm_ratio
            elif not self.rep_started:
                self.baseline_upper_arm_ratio = 0.90 * self.baseline_upper_arm_ratio + 0.10 * upper_arm_ratio

            if current_knee_angle is not None:
                if self.baseline_knee_angle is None:
                    self.baseline_knee_angle = current_knee_angle
                elif not self.rep_started:
                    self.baseline_knee_angle = 0.90 * self.baseline_knee_angle + 0.10 * current_knee_angle

            if self.setup_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.setup_frames = max(0, self.setup_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_side:
                feedback_msg = "Turn sideways to the camera (side view required)"
            elif back_arch_ratio > SETUP_MAX_BACK_ARCH_RATIO:
                feedback_msg = "Brace your core and keep your torso upright"
            elif is_wide_at_setup and elbow_angle >= SETUP_MIN_ELBOW_ANGLE:
                feedback_msg = "Tuck your elbows close to your head to begin"
            else:
                feedback_msg = "Raise arms straight overhead close to your ears to begin"
            return {
                "exercise": "Overhead Triceps", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "shoulder_flexion": round(shoulder_flexion, 1),
                "upper_arm_ratio": round(upper_arm_ratio, 2)
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

                if probs[0] >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(f"[OVERHEAD TRICEPS FAULT] AI ST-GCN flagged posture fault: prob={probs[0]:.2f}", flush=True)
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (SIDE VIEW)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.min_upper_arm_ratio = min(self.min_upper_arm_ratio, upper_arm_ratio)
            self.max_elbow_head_x = max(self.max_elbow_head_x, elbow_head_x_ratio)

            if not is_facing_side:
                self.side_frames += 1
                if self.side_frames >= DEBOUNCE_SIDE:
                    if self.is_side_view:
                        print(f"[OVERHEAD TRICEPS FAULT] Lost side view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.side_frames = max(0, self.side_frames - 1)

            # A. Wide / Flared Elbows (Elbows moving far away from the head)
            shrunk_from_baseline = (
                self.baseline_upper_arm_ratio is not None
                and upper_arm_ratio < (self.baseline_upper_arm_ratio * FLARE_UPPER_ARM_SHRINK_FACTOR)
            )
            is_elbow_flaring_wide = (
                (upper_arm_ratio < MIN_UPPER_ARM_RATIO)
                or shrunk_from_baseline
                or (elbow_above_ear_ratio < MIN_ELBOW_ABOVE_EAR_RATIO)
                or (elbow_head_x_ratio > MAX_ELBOW_HEAD_X_RATIO)
                or (elbow_spread_ratio > MAX_ELBOW_SPREAD_TO_SHOULDER)
            )
            if is_elbow_flaring_wide:
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_ELBOW_FLARE:
                    if self.elbows_tucked:
                        print(
                            f"[OVERHEAD TRICEPS FAULT] Wide/flared elbows away from head! "
                            f"arm_ratio={upper_arm_ratio:.2f} (base={self.baseline_upper_arm_ratio or 0:.2f}), "
                            f"above_ear={elbow_above_ear_ratio:.2f}, head_x={elbow_head_x_ratio:.2f}, "
                            f"spread={elbow_spread_ratio:.2f}",
                            flush=True,
                        )
                    self.elbows_tucked = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Tuck your elbows closer to your head!"
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            if not is_elbow_flaring_wide and elbow_head_x_ratio > ADVISORY_ELBOW_HEAD_X_RATIO:
                self.mild_flare_frames += 1
                if self.mild_flare_frames >= 5:
                    self.mild_flare_warning = True
            else:
                self.mild_flare_frames = max(0, self.mild_flare_frames - 1)

            # B. Forward-Fold Cheat (Lowering forearm in front of face instead of behind head)
            if facing_known and elbow_angle < 130.0 and wrist_forward_ratio > MAX_WRIST_FORWARD_RATIO:
                self.forward_fold_frames += 1
                if self.forward_fold_frames >= DEBOUNCE_FORWARD_FOLD:
                    if self.behind_head_path:
                        print(
                            f"[OVERHEAD TRICEPS FAULT] Forearm lowered in front of face! "
                            f"wrist_fwd={wrist_forward_ratio:.2f}, elbow={elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.behind_head_path = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Lower behind your head, not in front of your face!"
            else:
                self.forward_fold_frames = max(0, self.forward_fold_frames - 1)

            # C. Upper Arm / Elbow Drift Forward (Swinging elbows down in front of face)
            if shoulder_flexion < MIN_SHOULDER_FLEXION:
                self.drift_frames += 1
                if self.drift_frames >= DEBOUNCE_ELBOW_DRIFT:
                    if self.upper_arm_stable:
                        print(f"[OVERHEAD TRICEPS FAULT] Elbows drifted forward: flexion={shoulder_flexion:.1f}°", flush=True)
                    self.upper_arm_stable = False
            else:
                self.drift_frames = max(0, self.drift_frames - 1)

            # D. Lower-Back Arch / Forward Lean
            if back_arch_ratio > MAX_BACK_ARCH_RATIO:
                self.arch_frames += 1
                if self.arch_frames >= DEBOUNCE_BACK_ARCH:
                    if self.spine_neutral:
                        print(f"[OVERHEAD TRICEPS FAULT] Excessive back arch/lean: ratio={back_arch_ratio:.2f}", flush=True)
                    self.spine_neutral = False
            else:
                self.arch_frames = max(0, self.arch_frames - 1)

            if ADVISORY_BACK_ARCH_RATIO < back_arch_ratio <= MAX_BACK_ARCH_RATIO:
                self.mild_arch_frames += 1
                if self.mild_arch_frames >= 4:
                    self.mild_arch_warning = True
            else:
                self.mild_arch_frames = max(0, self.mild_arch_frames - 1)

            # E. Forward Head Poke Advisory (Dodging weight with neck)
            if head_forward_ratio > ADVISORY_HEAD_FORWARD_RATIO:
                self.head_frames += 1
                if self.head_frames >= 5:
                    self.head_forward_warning = True
            else:
                self.head_frames = max(0, self.head_frames - 1)

            # F. Knee Bounce Advisory (Using legs to pop weight overhead)
            if self.baseline_knee_angle is not None and current_knee_angle is not None and self.state == "ASCENDING":
                if abs(current_knee_angle - self.baseline_knee_angle) > 10.0:
                    self.knee_frames += 1
                    if self.knee_frames >= 3:
                        self.knee_bounce_warning = True
                else:
                    self.knee_frames = max(0, self.knee_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep elbows tucked close to your head."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Overhead Triceps", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "shoulder_flexion": round(shoulder_flexion, 1),
                "upper_arm_ratio": round(upper_arm_ratio, 2)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.rest_exit_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower slowly behind your head, keeping elbows tucked!"

        elif self.state == "RESTING":
            # Graceful set-break exit: lowering arms down to chest/sides returns to WAITING without a false rep
            if shoulder_flexion < REST_EXIT_SHOULDER_FLEXION:
                self.rest_exit_frames += 1
                if self.rest_exit_frames >= REST_EXIT_FRAMES:
                    self.state = "WAITING"
                    self.is_ready = False
                    self.setup_frames = 0
                    self.rest_exit_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Arms lowered. Raise arms overhead close to ears to resume."
            else:
                self.rest_exit_frames = 0
                if elbow_angle < DESCENDING_START_ANGLE and shoulder_flexion >= MIN_START_SHOULDER_FLEXION:
                    self.state = "DESCENDING"
                    self.rep_started = True
                    self.min_elbow_angle = elbow_angle
                    self.min_upper_arm_ratio = upper_arm_ratio
                    self.max_elbow_head_x = elbow_head_x_ratio
                    self.reversal_frames = 0
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Lower behind your head with elbows close to ears..."

        elif self.state == "DESCENDING" and elbow_angle <= BOTTOM_TARGET_ANGLE:
            self.depth_reached = True
            self.state = "BOTTOM"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Good stretch! Press straight back up."

        elif self.state == "DESCENDING" and not self.depth_reached and (
            (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.half_ext_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Pressing up too early..."
                print(f"[OVERHEAD TRICEPS FAULT] Early reversal: min_elbow={self.min_elbow_angle:.1f}°", flush=True)

        elif self.state == "BOTTOM" and elbow_angle > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Extend up cleanly, keep elbows hugged to head..."

        # Allow recovery from brief fatigue tremor on descent if user still completes full depth
        elif self.state == "ASCENDING" and not self.depth_reached and elbow_angle <= BOTTOM_TARGET_ANGLE:
            self.depth_reached = True
            self.half_ext_reached = False
            self.state = "BOTTOM"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Good stretch! Press straight back up."

        if self.state == "DESCENDING" and not ((elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Debounced Eccentric Free-Fall Monitor
        if self.rep_started and self.state == "DESCENDING":
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(f"[OVERHEAD TRICEPS FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame", flush=True)
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and elbow_angle >= REP_COMPLETE_ANGLE:
            # Ignore minor resting twitches overhead
            if self.min_elbow_angle > MIN_VALID_REP_ELBOW_ANGLE and not self.depth_reached and self.is_side_view:
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Overhead Triceps", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(elbow_angle, 1),
                    "shoulder_flexion": round(shoulder_flexion, 1),
                    "upper_arm_ratio": round(upper_arm_ratio, 2)
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Keep a side profile to the camera! Hold overhead to reset."
                is_bad_rep = True
            elif not self.behind_head_path:
                self.current_feedback = "Bad rep: Lower behind your head, not in front of your face!"
                is_bad_rep = True
            elif not self.elbows_tucked:
                self.current_feedback = "Bad rep: Elbows flared too wide! Keep elbows tucked close to your head."
                is_bad_rep = True
            elif not self.upper_arm_stable:
                self.current_feedback = "Bad rep: Elbows drifted forward! Keep upper arms pointed straight up."
                is_bad_rep = True
            elif not self.spine_neutral:
                self.current_feedback = "Bad rep: Do not arch your lower back! Brace your core."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Lower slowly behind your head."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor overhead triceps mechanics."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "Bad rep: Half rep! Lower deeper behind your head."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[OVERHEAD TRICEPS REP #{rep_num} -> BAD REP] Reason: {self.current_feedback} | "
                    f"min_elbow={self.min_elbow_angle:.1f}°, min_arm_ratio={self.min_upper_arm_ratio:.2f}, "
                    f"max_head_x={self.max_elbow_head_x:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.mild_flare_warning:
                    self.current_feedback = "Good rep! Reminder: Hug your elbows even closer to your ears."
                elif self.mild_arch_warning:
                    self.current_feedback = "Good rep! Reminder: Keep ribs tucked to avoid arching."
                elif self.head_forward_warning:
                    self.current_feedback = "Good rep! Reminder: Keep your neck neutral, avoid poking head forward."
                elif self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Avoid bouncing your knees for momentum."
                else:
                    self.current_feedback = "Great Overhead Extension! Clean triceps isolation."
                print(
                    f"[OVERHEAD TRICEPS REP #{rep_num} -> GOOD REP] min_elbow={self.min_elbow_angle:.1f}°, "
                    f"min_arm_ratio={self.min_upper_arm_ratio:.2f}",
                    flush=True,
                )
                self.state = "RESTING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        total_reps = self.good_reps + self.bad_reps + self.half_reps
        if self.state == "RESTING":
            form_score = 100
        else:
            progress = max(0.0, min(1.0, (REP_COMPLETE_ANGLE - elbow_angle) / 45.0))
            form_score = int(60 + progress * 40)
        if not self.is_side_view: form_score -= 35
        if not self.behind_head_path: form_score -= 35
        if not self.elbows_tucked: form_score -= 30
        if not self.upper_arm_stable: form_score -= 25
        if not self.spine_neutral: form_score -= 25
        if not self.controlled_descent: form_score -= 15
        if self.mild_flare_warning: form_score -= 10
        if self.mild_arch_warning: form_score -= 10
        if self.head_forward_warning: form_score -= 5
        if self.knee_bounce_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 25
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 35
        form_score = max(0, form_score)

        return {
            "exercise": "Overhead Triceps",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "shoulder_flexion": round(shoulder_flexion, 1),
            "upper_arm_ratio": round(upper_arm_ratio, 2)
        }
