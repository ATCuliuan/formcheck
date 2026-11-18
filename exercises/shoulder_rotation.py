import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SHOULDER ROTATION - FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Front View, One Elbow Pinned to Ribs at ~90 deg)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.23  # Enforces front-facing view
SETUP_MAX_WRIST_Y_DRIFT = 0.18        # Forearm roughly level with elbow (Z-foreshortening safe!)
SETUP_MIN_UPPER_ARM_DROP = 0.25       # Upper arm hanging down (elbow clearly below shoulder)
SETUP_MAX_SHOULDER_ABDUCTION = 26.0   # Elbow pinned to ribs (hip-shoulder-elbow angle)
SETUP_MAX_OUTWARD_FLARE = 0.26        # Signed outward elbow drift / shoulder width at setup
SETUP_NEUTRAL_MAX_RATIO = 0.16        # Forearm pointing roughly forward at setup (~13 deg)
SETUP_MIN_READY_FRAMES = 6            # Consecutive frames in neutral setup to confirm READY
UNLOCK_RESET_FRAMES = 10              # Frames held in neutral setup to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35       # Minimum upper-body + hip landmark visibility
OUT_OF_VIEW_RESET_FRAMES = 15         # Frames out of view before full state reset
REST_EXIT_WRIST_DROP = 0.24           # Lowering active arm to side while resting exits to WAITING
REST_EXIT_FRAMES = 12                 # Frames with arm lowered to cleanly exit / switch arms

# 2. Kinematic & Form Thresholds (Beginner-Friendly Swinging Door Mechanics)
MAX_SHOULDER_ABDUCTION_DEG = 30.0     # Elbow lifting/flaring off ribs (> 30 deg hip-shoulder-elbow)
ADVISORY_SHOULDER_ABDUCTION = 23.0    # Mild elbow separation reminder without failing rep
MAX_OUTWARD_FLARE_RATIO = 0.30        # Signed outward elbow X drift (> 30% shoulder width)
ADVISORY_OUTWARD_FLARE = 0.22         # Mild outward elbow drift reminder
MIN_ACTIVE_UPPER_ARM_DROP = 0.24      # Hard floor catching elbow lifting forward in front of chest
UPPER_ARM_DROP_SHRINK_FACTOR = 0.72   # >28% drop from setup upper-arm drop flags forward elbow lift
LATERAL_ELBOW_CHECK_MIN_ROT = 0.20    # Only check 2D elbow angle when forearm is angled sideways
MIN_ACTIVE_ELBOW_ANGLE = 55.0         # Excessive curling of forearm (when laterally visible)
MAX_ACTIVE_ELBOW_ANGLE = 132.0        # Straightening arm out sideways (when laterally visible)
MAX_WRIST_Y_DRIFT_RATIO = 0.22        # Forearm drooping below or hiking above elbow height
ADVISORY_WRIST_Y_DRIFT = 0.16         # Mild forearm droop/hike reminder
TORSO_TWIST_RATIO_FACTOR = 0.80       # Camera-distance-invariant span/torso ratio shrink < 80% flags twist
MAX_TORSO_SIDE_LEAN_RATIO = 0.16      # Lateral leaning of shoulders away from hips
MAX_SHOULDER_TILT_RATIO = 0.11        # Hiking one shoulder noticeably higher than the other
SHRUG_NECK_RATIO_FACTOR = 0.65        # Ear-to-shoulder vertical shrink catching trap shrug
MIN_ABSOLUTE_NECK_RATIO = 0.13        # Absolute floor catching users who start already shrugged
MAX_ROTATION_SPEED = 0.24             # Jerking/whipping forearm too fast (ratio/frame)

# 3. Range of Motion & Half-Rep Thresholds (Normalized Wrist-to-Elbow X / Shoulder Width)
# Biomechanics: forearm ~0.68x shoulder width -> ratio ~ 0.68 * sin(angle)
ROTATION_START_RATIO = 0.16           # Moving forearm > 16% shoulder width starts rep
SECOND_SIDE_COMMIT_RATIO = 0.20       # Clear excursion required to commit visit to second side
MIN_VALID_REP_PEAK_RATIO = 0.19       # Ignore tiny resting hand twitches (< 19%)
EXTERNAL_TARGET_RATIO = 0.34          # Swinging forearm outward (~30-33 deg external rotation)
INTERNAL_TARGET_RATIO = -0.28         # Swinging forearm inward across stomach (~25-28 deg)
NEUTRAL_RETURN_RATIO = 0.11           # Clean hysteresis gap from ROTATION_START_RATIO (0.16)
SHALLOW_RETURN_CONFIRM_FRAMES = 4     # Debounce incomplete both-sides return against 1-frame jitter
NEUTRAL_ABANDON_FRAMES = 35           # Stopping at center after only one side (~1.2s) flags Half Rep
SAME_SIDE_RETURN_FRAMES = 5           # Neutral frames before re-entering same side
SAME_SIDE_REENTRY_RATIO = 0.24        # Must clearly re-enter same side (filters jitter)

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_FLARE = 5
DEBOUNCE_ELBOW = 5
DEBOUNCE_WRIST = 5
DEBOUNCE_TWIST = 5
DEBOUNCE_LEAN = 5
DEBOUNCE_SHRUG = 5
DEBOUNCE_SPEED = 3
DEBOUNCE_ML_FAULT = 6

ML_FAULT_CONFIDENCE_THRESHOLD = 0.82
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class ShoulderRotationTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.hit_external = False
        self.hit_internal = False
        self.visited_external = False
        self.visited_internal = False
        self.max_external_peak = 0.0
        self.max_internal_peak = 0.0
        self.max_abduction_deg = 0.0
        self.active_side = None
        self.neutral_hold_frames = 0

        # Instructor Tripwires (FRONT VIEW)
        self.is_front_view = True
        self.elbow_pinned = True          # Elbow stays glued to ribs (no lateral flare or forward lift)
        self.elbow_bent_90 = True         # Steady ~90 deg bend (checked when laterally visible)
        self.forearm_level = True         # Forearm stays parallel to floor (no droop or hike)
        self.torso_stable = True          # No spinal twisting or side-bending lean
        self.shoulders_relaxed = True     # No shrugging/hiking working shoulder
        self.controlled_tempo = True      # No jerking/whipping momentum
        self.ml_posture_maintained = True

        # Advisory / Camera-Distance-Invariant Baselines
        self.mild_flare_warning = False
        self.mild_wrist_warning = False
        self.baseline_span_ratio = None
        self.baseline_neck_ratio = None
        self.baseline_upper_arm_drop = None

        # Debounce & Lifecycle counters
        self.front_frames = 0
        self.setup_frames = 0
        self.unlock_frames = 0
        self.out_of_view_frames = 0
        self.rest_exit_frames = 0
        self.flare_frames = 0
        self.mild_flare_frames = 0
        self.elbow_frames = 0
        self.wrist_frames = 0
        self.mild_wrist_frames = 0
        self.twist_frames = 0
        self.lean_frames = 0
        self.shrug_frames = 0
        self.speed_frames = 0
        self.ml_bad_frames = 0
        self.post_rep_timer = 0

        self.is_ready = False
        self.prev_rotation_val = None

        self.current_feedback = "Face camera, pin one elbow to your ribs at 90 degrees"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_shoulder_rotation.tflite")
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
        self.hit_external = False
        self.hit_internal = False
        self.visited_external = False
        self.visited_internal = False
        self.max_external_peak = 0.0
        self.max_internal_peak = 0.0
        self.max_abduction_deg = 0.0
        self.neutral_hold_frames = 0
        self.prev_rotation_val = None

        self.is_front_view = True
        self.elbow_pinned = True
        self.elbow_bent_90 = True
        self.forearm_level = True
        self.torso_stable = True
        self.shoulders_relaxed = True
        self.controlled_tempo = True
        self.mild_flare_warning = False
        self.mild_wrist_warning = False
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.flare_frames = 0
        self.mild_flare_frames = 0
        self.elbow_frames = 0
        self.wrist_frames = 0
        self.mild_wrist_frames = 0
        self.twist_frames = 0
        self.lean_frames = 0
        self.shrug_frames = 0
        self.speed_frames = 0
        self.ml_bad_frames = 0

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
            self.out_of_view_frames += 1
            self.prev_rotation_val = None
            self.frame_buffer.clear()
            if self.out_of_view_frames >= OUT_OF_VIEW_RESET_FRAMES:
                self.state = "WAITING"
                self.is_ready = False
                self.setup_frames = 0
                self.unlock_frames = 0
                self.active_side = None
                self.baseline_span_ratio = None
                self.baseline_neck_ratio = None
                self.baseline_upper_arm_drop = None
                self._reset_rep_tracking()
            return {
                "exercise": "Shoulder Rotation", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Shoulders, arms, and torso must be visible",
                "rotation_ratio": 0.0, "flare_ratio": 0.0, "abduction_angle": 0.0
            }

        self.out_of_view_frames = 0

        # Front View & Normalization
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_height = max(abs(mid_shoulder_y - mid_hip_y), 0.20)
        raw_shoulder_span = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_span / torso_height
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO
        shoulder_width = max(raw_shoulder_span, 0.16)

        # Z-Foreshortening-Safe Active Arm Detection (with stickiness bias to avoid resting flicker)
        l_outward_sign = 1.0 if l_shoulder.x > r_shoulder.x else -1.0
        r_outward_sign = 1.0 if r_shoulder.x > l_shoulder.x else -1.0

        l_wrist_y_drift = abs(l_wrist.y - l_elbow.y) / torso_height
        r_wrist_y_drift = abs(r_wrist.y - r_elbow.y) / torso_height
        l_outward_flare = max(0.0, ((l_elbow.x - l_shoulder.x) * l_outward_sign) / shoulder_width)
        r_outward_flare = max(0.0, ((r_elbow.x - r_shoulder.x) * r_outward_sign) / shoulder_width)
        l_abduction = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction = calculate_angle(r_hip, r_shoulder, r_elbow)

        if not self.rep_started or self.active_side is None:
            l_score = (l_wrist_y_drift * 2.5) + l_outward_flare + (l_abduction / 180.0)
            r_score = (r_wrist_y_drift * 2.5) + r_outward_flare + (r_abduction / 180.0)
            if self.active_side == "LEFT":
                l_score -= 0.08
            elif self.active_side == "RIGHT":
                r_score -= 0.08
            self.active_side = "LEFT" if l_score < r_score else "RIGHT"

        if self.active_side == "LEFT":
            active_ear = l_ear
            active_shoulder, active_elbow, active_wrist, active_hip = l_shoulder, l_elbow, l_wrist, l_hip
            outward_sign = l_outward_sign
            shoulder_abduction = l_abduction
            wrist_y_drift = l_wrist_y_drift
            outward_flare = l_outward_flare
        else:
            active_ear = r_ear
            active_shoulder, active_elbow, active_wrist, active_hip = r_shoulder, r_elbow, r_wrist, r_hip
            outward_sign = r_outward_sign
            shoulder_abduction = r_abduction
            wrist_y_drift = r_wrist_y_drift
            outward_flare = r_outward_flare

        # Camera-distance-invariant AND twist-proof reference width for rotation ratio
        if self.baseline_span_ratio is not None:
            ref_width = max(self.baseline_span_ratio * torso_height, 0.16)
        else:
            ref_width = shoulder_width

        # Positive = External Rotation (swinging outward), Negative = Internal Rotation (across belly)
        rotation_val = ((active_wrist.x - active_elbow.x) * outward_sign) / ref_width
        elbow_angle_2d = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_drop = (active_elbow.y - active_shoulder.y) / torso_height
        wrist_below_elbow = (active_wrist.y - active_elbow.y) / torso_height

        # Torso & Shoulder Stability Kinematics
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        torso_side_lean = abs(mid_shoulder_x - mid_hip_x) / torso_height
        shoulder_tilt_ratio = abs(l_shoulder.y - r_shoulder.y) / torso_height

        ear_visible = active_ear.visibility >= MIN_VISIBILITY_THRESHOLD
        neck_ratio = ((active_shoulder.y - active_ear.y) / torso_height) if ear_visible else None

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Front View + Elbow Pinned at ~90 deg, Neutral Forearm)
        # ---------------------------------------------------------
        is_resting_setup = (
            is_facing_front
            and (upper_arm_drop >= SETUP_MIN_UPPER_ARM_DROP)
            and (wrist_y_drift <= SETUP_MAX_WRIST_Y_DRIFT)
            and (shoulder_abduction <= SETUP_MAX_SHOULDER_ABDUCTION)
            and (outward_flare <= SETUP_MAX_OUTWARD_FLARE)
            and (abs(rotation_val) <= SETUP_NEUTRAL_MAX_RATIO)
        )

        if is_resting_setup:
            self.setup_frames += 1
            if not self.rep_started:
                if self.baseline_span_ratio is None:
                    self.baseline_span_ratio = shoulder_span_ratio
                else:
                    self.baseline_span_ratio = 0.90 * self.baseline_span_ratio + 0.10 * shoulder_span_ratio

                if self.baseline_upper_arm_drop is None:
                    self.baseline_upper_arm_drop = upper_arm_drop
                else:
                    self.baseline_upper_arm_drop = 0.90 * self.baseline_upper_arm_drop + 0.10 * upper_arm_drop

                if neck_ratio is not None and neck_ratio > 0.05:
                    if self.baseline_neck_ratio is None:
                        self.baseline_neck_ratio = neck_ratio
                    else:
                        self.baseline_neck_ratio = 0.90 * self.baseline_neck_ratio + 0.10 * neck_ratio

            if self.setup_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.setup_frames = max(0, self.setup_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            elif shoulder_abduction > SETUP_MAX_SHOULDER_ABDUCTION or outward_flare > SETUP_MAX_OUTWARD_FLARE or upper_arm_drop < SETUP_MIN_UPPER_ARM_DROP:
                feedback_msg = "Pin your elbow tightly against your ribs"
            elif wrist_y_drift > SETUP_MAX_WRIST_Y_DRIFT:
                feedback_msg = "Bend one elbow to 90 degrees with forearm level to floor"
            else:
                feedback_msg = "Point forearm straight forward to begin"
            return {
                "exercise": "Shoulder Rotation", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "rotation_ratio": round(rotation_val, 2),
                "flare_ratio": round(outward_flare, 2),
                "abduction_angle": round(shoulder_abduction, 1)
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

                if probs[0] >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state in ["EXTERNAL", "INTERNAL", "CROSSING"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(f"[SHOULDER ROTATION FAULT] AI ST-GCN flagged posture fault: prob={probs[0]:.2f}", flush=True)
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (FRONT VIEW)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["EXTERNAL", "INTERNAL", "CROSSING"]:
            self.max_external_peak = max(self.max_external_peak, max(0.0, rotation_val))
            self.max_internal_peak = max(self.max_internal_peak, max(0.0, -rotation_val))
            self.max_abduction_deg = max(self.max_abduction_deg, shoulder_abduction)

            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[SHOULDER ROTATION FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # A. Elbow Unpinned (Lateral flare OR lifting elbow forward in front of chest)
            elbow_lifted_forward = (
                upper_arm_drop < MIN_ACTIVE_UPPER_ARM_DROP
                or (
                    self.baseline_upper_arm_drop is not None
                    and upper_arm_drop < (self.baseline_upper_arm_drop * UPPER_ARM_DROP_SHRINK_FACTOR)
                )
            )
            is_flaring = (
                (shoulder_abduction > MAX_SHOULDER_ABDUCTION_DEG)
                or (outward_flare > MAX_OUTWARD_FLARE_RATIO)
                or elbow_lifted_forward
            )
            if is_flaring:
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_FLARE:
                    if self.elbow_pinned:
                        print(
                            f"[SHOULDER ROTATION FAULT] Elbow unpinned from ribs: "
                            f"abd={shoulder_abduction:.1f}°, flare={outward_flare:.2f}, "
                            f"arm_drop={upper_arm_drop:.2f} (base={self.baseline_upper_arm_drop or 0:.2f})",
                            flush=True,
                        )
                    self.elbow_pinned = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Keep your elbow glued to your ribs!"
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            is_mild_flare = (
                not is_flaring
                and (shoulder_abduction > ADVISORY_SHOULDER_ABDUCTION or outward_flare > ADVISORY_OUTWARD_FLARE)
            )
            if is_mild_flare:
                self.mild_flare_frames += 1
                if self.mild_flare_frames >= 5:
                    self.mild_flare_warning = True
            else:
                self.mild_flare_frames = max(0, self.mild_flare_frames - 1)

            # B. Forearm Drooping / Hiking (Works at ALL rotation angles including straight-at-camera!)
            if wrist_y_drift > MAX_WRIST_Y_DRIFT_RATIO:
                self.wrist_frames += 1
                if self.wrist_frames >= DEBOUNCE_WRIST:
                    if self.forearm_level:
                        print(f"[SHOULDER ROTATION FAULT] Forearm dropped/hiked: drift={wrist_y_drift:.2f}", flush=True)
                    self.forearm_level = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Keep your forearm level with your elbow!"
            else:
                self.wrist_frames = max(0, self.wrist_frames - 1)

            if ADVISORY_WRIST_Y_DRIFT < wrist_y_drift <= MAX_WRIST_Y_DRIFT_RATIO:
                self.mild_wrist_frames += 1
                if self.mild_wrist_frames >= 5:
                    self.mild_wrist_warning = True
            else:
                self.mild_wrist_frames = max(0, self.mild_wrist_frames - 1)

            # C. 2D Elbow Bend Check (ONLY evaluated when forearm is angled laterally, avoiding Z-axis bug!)
            if abs(rotation_val) >= LATERAL_ELBOW_CHECK_MIN_ROT:
                if elbow_angle_2d < MIN_ACTIVE_ELBOW_ANGLE or elbow_angle_2d > MAX_ACTIVE_ELBOW_ANGLE:
                    self.elbow_frames += 1
                    if self.elbow_frames >= DEBOUNCE_ELBOW:
                        if self.elbow_bent_90:
                            print(
                                f"[SHOULDER ROTATION FAULT] Lost 90 deg bend laterally: "
                                f"angle={elbow_angle_2d:.1f}°, rot={rotation_val:.2f}",
                                flush=True,
                            )
                        self.elbow_bent_90 = False
                else:
                    self.elbow_frames = max(0, self.elbow_frames - 1)
            else:
                self.elbow_frames = max(0, self.elbow_frames - 1)

            # D. Torso Twist (Camera-distance-invariant span/torso ratio shrink!)
            if self.baseline_span_ratio and shoulder_span_ratio < (self.baseline_span_ratio * TORSO_TWIST_RATIO_FACTOR):
                self.twist_frames += 1
                if self.twist_frames >= DEBOUNCE_TWIST:
                    if self.torso_stable:
                        print(
                            f"[SHOULDER ROTATION FAULT] Torso twisted: span_ratio={shoulder_span_ratio:.2f} "
                            f"vs base={self.baseline_span_ratio:.2f}",
                            flush=True,
                        )
                    self.torso_stable = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Do not twist your torso! Rotate only the shoulder."
            else:
                self.twist_frames = max(0, self.twist_frames - 1)

            # E. Lateral Torso Side-Lean (Leaning body sideways to fake extra range)
            if torso_side_lean > MAX_TORSO_SIDE_LEAN_RATIO:
                self.lean_frames += 1
                if self.lean_frames >= DEBOUNCE_LEAN:
                    if self.torso_stable:
                        print(f"[SHOULDER ROTATION FAULT] Torso leaned sideways: lean={torso_side_lean:.2f}", flush=True)
                    self.torso_stable = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Keep your torso upright! Do not lean sideways."
            else:
                self.lean_frames = max(0, self.lean_frames - 1)

            # F. Shoulder Shrug / Hiking (Relative shrink + absolute floor + bilateral tilt)
            shrugged_neck = (
                neck_ratio is not None
                and (
                    (self.baseline_neck_ratio is not None and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR))
                    or (neck_ratio < MIN_ABSOLUTE_NECK_RATIO)
                )
            )
            hiked_shoulder = shoulder_tilt_ratio > MAX_SHOULDER_TILT_RATIO
            if shrugged_neck or hiked_shoulder:
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    if self.shoulders_relaxed:
                        print(
                            f"[SHOULDER ROTATION FAULT] Shoulder shrugged/hiked: "
                            f"tilt={shoulder_tilt_ratio:.2f}, neck={neck_ratio or 0:.2f}",
                            flush=True,
                        )
                    self.shoulders_relaxed = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Relax your shoulder down away from your ear!"
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # G. Controlled Tempo Monitor (Catches jerking/whipping momentum)
            if self.prev_rotation_val is not None:
                rot_speed = abs(rotation_val - self.prev_rotation_val)
                if rot_speed > MAX_ROTATION_SPEED:
                    self.speed_frames += 1
                    if self.speed_frames >= DEBOUNCE_SPEED:
                        if self.controlled_tempo:
                            print(f"[SHOULDER ROTATION FAULT] Jerking/too fast: speed={rot_speed:.2f}/frame", flush=True)
                        self.controlled_tempo = False
                else:
                    self.speed_frames = max(0, self.speed_frames - 1)
            self.prev_rotation_val = rotation_val
        else:
            self.prev_rotation_val = None

        # ---------------------------------------------------------
        # 4. STATE MACHINE (SUPPORTS OUT-THEN-IN OR IN-THEN-OUT)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Swing forearm smoothly outward and inward."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Shoulder Rotation", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "rotation_ratio": round(rotation_val, 2),
                "flare_ratio": round(outward_flare, 2),
                "abduction_angle": round(shoulder_abduction, 1)
            }

        same_side_repeat = False

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.rest_exit_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Rotate forearm outward and inward like a swinging door!"

        elif self.state == "RESTING":
            # Graceful set-break / arm-switch exit: dropping arm down to side resets cleanly to WAITING
            if wrist_below_elbow > REST_EXIT_WRIST_DROP:
                self.rest_exit_frames += 1
                if self.rest_exit_frames >= REST_EXIT_FRAMES:
                    self.state = "WAITING"
                    self.is_ready = False
                    self.setup_frames = 0
                    self.rest_exit_frames = 0
                    self.active_side = None
                    self._reset_rep_tracking()
                    self.current_feedback = "Arm lowered. Pin an elbow at 90 degrees to resume or switch arms."
            else:
                self.rest_exit_frames = 0
                if abs(rotation_val) > ROTATION_START_RATIO and wrist_y_drift <= MAX_WRIST_Y_DRIFT_RATIO:
                    self.rep_started = True
                    self.neutral_hold_frames = 0
                    self.prev_rotation_val = rotation_val
                    if rotation_val > 0:
                        self.state = "EXTERNAL"
                        self.visited_external = True
                        if self.post_rep_timer == 0:
                            self.current_feedback = "Swinging outward... keep elbow glued to ribs"
                    else:
                        self.state = "INTERNAL"
                        self.visited_internal = True
                        if self.post_rep_timer == 0:
                            self.current_feedback = "Swinging inward across stomach..."

        elif self.rep_started:
            if rotation_val >= EXTERNAL_TARGET_RATIO:
                if self.visited_external and not self.visited_internal and self.neutral_hold_frames >= SAME_SIDE_RETURN_FRAMES:
                    same_side_repeat = True
                self.state = "EXTERNAL"
                self.visited_external = True
                self.hit_external = True
                self.neutral_hold_frames = 0
                if self.post_rep_timer == 0:
                    self.current_feedback = (
                        "Return to center to finish!" if self.hit_internal
                        else "External reached! Now rotate inward across stomach."
                    )
            elif rotation_val <= INTERNAL_TARGET_RATIO:
                if self.visited_internal and not self.visited_external and self.neutral_hold_frames >= SAME_SIDE_RETURN_FRAMES:
                    same_side_repeat = True
                self.state = "INTERNAL"
                self.visited_internal = True
                self.hit_internal = True
                self.neutral_hold_frames = 0
                if self.post_rep_timer == 0:
                    self.current_feedback = (
                        "Return to center to finish!" if self.hit_external
                        else "Internal reached! Now rotate outward."
                    )
            elif rotation_val > SECOND_SIDE_COMMIT_RATIO:
                if (
                    self.visited_external
                    and not self.visited_internal
                    and self.neutral_hold_frames >= SAME_SIDE_RETURN_FRAMES
                    and rotation_val >= SAME_SIDE_REENTRY_RATIO
                ):
                    same_side_repeat = True
                self.state = "EXTERNAL"
                self.visited_external = True
                self.neutral_hold_frames = 0
            elif rotation_val < -SECOND_SIDE_COMMIT_RATIO:
                if (
                    self.visited_internal
                    and not self.visited_external
                    and self.neutral_hold_frames >= SAME_SIDE_RETURN_FRAMES
                    and rotation_val <= -SAME_SIDE_REENTRY_RATIO
                ):
                    same_side_repeat = True
                self.state = "INTERNAL"
                self.visited_internal = True
                self.neutral_hold_frames = 0
            elif abs(rotation_val) <= NEUTRAL_RETURN_RATIO:
                self.state = "CROSSING"
                self.neutral_hold_frames += 1
            else:
                # Hysteresis transition band (0.11 < |rot| <= 0.20): hold steady without jitter false-triggers
                if abs(rotation_val) > ROTATION_START_RATIO:
                    self.neutral_hold_frames = 0

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        full_rep_return = (
            self.rep_started
            and self.hit_external
            and self.hit_internal
            and abs(rotation_val) <= NEUTRAL_RETURN_RATIO
        )
        # Debounced incomplete both-sides return so 1-frame crossing jitter never aborts a clean rep
        shallow_both_sides_return = (
            self.rep_started
            and self.visited_external
            and self.visited_internal
            and not (self.hit_external and self.hit_internal)
            and self.neutral_hold_frames >= SHALLOW_RETURN_CONFIRM_FRAMES
        )
        abandoned_half_rep = (
            self.rep_started
            and (not (self.hit_external and self.hit_internal))
            and self.neutral_hold_frames >= NEUTRAL_ABANDON_FRAMES
        )

        if full_rep_return or shallow_both_sides_return or abandoned_half_rep or same_side_repeat:
            peak_total = max(self.max_external_peak, self.max_internal_peak)
            # Ignore minor resting twitches
            if peak_total < MIN_VALID_REP_PEAK_RATIO and self.is_front_view and self.elbow_pinned and self.forearm_level:
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Shoulder Rotation", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "rotation_ratio": round(rotation_val, 2),
                    "flare_ratio": round(outward_flare, 2),
                    "abduction_angle": round(shoulder_abduction, 1)
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Reset at center."
                is_bad_rep = True
            elif not self.elbow_pinned:
                self.current_feedback = "Bad rep: Keep your elbow glued to your ribs!"
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "Bad rep: Keep torso steady! Do not twist or lean sideways."
                is_bad_rep = True
            elif not self.shoulders_relaxed:
                self.current_feedback = "Bad rep: Do not shrug or hike your shoulder up!"
                is_bad_rep = True
            elif not self.forearm_level:
                self.current_feedback = "Bad rep: Keep your forearm level with your elbow at 90 degrees."
                is_bad_rep = True
            elif not self.elbow_bent_90:
                self.current_feedback = "Bad rep: Maintain a steady 90-degree bend in your elbow."
                is_bad_rep = True
            elif not self.controlled_tempo:
                self.current_feedback = "Bad rep: Slow down! Rotate smoothly without jerking."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor shoulder rotation mechanics."
                is_bad_rep = True
            elif not (self.hit_external and self.hit_internal):
                self.current_feedback = "Bad rep: Half rep! Rotate fully both outward and inward."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SHOULDER ROTATION REP #{rep_num} -> BAD REP] Reason: {self.current_feedback} | "
                    f"ext={self.max_external_peak:.2f}, int={self.max_internal_peak:.2f}, "
                    f"max_abd={self.max_abduction_deg:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.mild_flare_warning:
                    self.current_feedback = "Good rep! Reminder: Keep elbow tucked tightly to ribs."
                elif self.mild_wrist_warning:
                    self.current_feedback = "Good rep! Reminder: Keep wrist level with your elbow."
                else:
                    self.current_feedback = "Great Rotation! Clean shoulder isolation."
                print(
                    f"[SHOULDER ROTATION REP #{rep_num} -> GOOD REP] "
                    f"ext={self.max_external_peak:.2f}, int={self.max_internal_peak:.2f}, "
                    f"max_abd={self.max_abduction_deg:.1f}°",
                    flush=True,
                )
                self.state = "RESTING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        total_reps = self.good_reps + self.bad_reps + self.half_reps
        form_score = 100
        if not self.is_front_view: form_score -= 35
        if not self.elbow_pinned: form_score -= 30
        if not self.torso_stable: form_score -= 30
        if not self.shoulders_relaxed: form_score -= 20
        if not self.forearm_level: form_score -= 20
        if not self.elbow_bent_90: form_score -= 20
        if not self.controlled_tempo: form_score -= 15
        if self.mild_flare_warning: form_score -= 10
        if self.mild_wrist_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Rotation",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "rotation_ratio": round(rotation_val, 2),
            "flare_ratio": round(outward_flare, 2),
            "abduction_angle": round(shoulder_abduction, 1)
        }
