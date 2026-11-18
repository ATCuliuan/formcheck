import math
import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (TRICEPS DIPS - BEGINNER SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Side View, Hands Planted on Chair/Bench, Legs Optional)
SETUP_MAX_SIDE_SPAN_RATIO = 0.48      # Enforces side profile (loose enough for slight 3/4 angle)
SETUP_MIN_ELBOW_ANGLE = 148.0         # Arms mostly straight at top (no hyperextension needed)
SETUP_MAX_HIP_DRIFT_RATIO = 0.55      # Hips stay close to bench/hands at setup
SETUP_MAX_TORSO_LEAN_DEG = 26.0       # Torso upright at setup (prevents hunched/reclined start)
SETUP_MIN_WRIST_BELOW_SHOULDER = 0.45 # Hands planted below shoulders near hip/chair height
SETUP_MIN_READY_FRAMES = 6            # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10              # Frames held at top to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35       # Minimum active-side core landmark visibility
OUT_OF_VIEW_RESET_FRAMES = 15         # Frames out of view before full state reset
REST_EXIT_WRIST_SHIFT_RATIO = 0.25    # Standing up OR sitting down onto new height exits to WAITING
REST_EXIT_FRAMES = 12                 # Frames with shifted hands/height to exit cleanly between sets

# 2. Kinematic & Form Thresholds (Beginner-Friendly Chair/Bench Dip)
MAX_HIP_DRIFT_RATIO = 0.58            # Horizontal hip-to-wrist gap / torso height (strains shoulders)
ADVISORY_HIP_DRIFT_RATIO = 0.45       # Mild drift reminder without failing rep
SHRUG_NECK_RATIO_FACTOR = 0.62        # Signed ear-to-shoulder shrink ratio catching shoulder sinking
MIN_ABSOLUTE_NECK_RATIO = 0.12        # Absolute floor catching users who start already slumped/shrugged
MIN_SAFE_ELBOW_ANGLE = 52.0           # Extreme acute elbow collapse (< 52 deg) risks shoulder strain
MAX_SHOULDER_BELOW_ELBOW_RATIO = 0.08 # Shoulder dropping clearly below elbow flags excessive depth
MIN_UPPER_ARM_RATIO = 0.34            # Hard floor for 2D upper-arm length / torso height (elbow flare)
FLARE_UPPER_ARM_SHRINK_FACTOR = 0.74  # >26% drop from setup upper-arm length flags chicken-wing flare
ELBOW_SPREAD_CHECK_MIN_SPAN = 0.20    # Min shoulder span ratio (3/4 view) for bilateral elbow spread
MAX_ELBOW_SPREAD_TO_SHOULDER = 1.48   # Bilateral elbow width > 1.48x shoulder span flags wide elbows
MAX_ELBOW_FORWARD_RATIO = 0.12        # Elbow bending clearly in front of torso (unsigned-angle cheat)
MIN_FACING_NOSE_OFFSET = 0.05         # Minimum nose-to-shoulder X offset to trust facing direction
MAX_TORSO_LEAN_DEG = 32.0             # Excessive forward hunch or backward slump (> 32 deg from vertical)
ADVISORY_TORSO_LEAN_DEG = 24.0        # Mild torso lean reminder without failing rep
ADVISORY_HEAD_FORWARD_RATIO = 0.32    # Chin-to-chest / forward head poke reminder
ADVISORY_KNEE_EXTEND_DELTA = 18.0     # Knee extending > 18 deg past setup baseline flags leg drive
ADVISORY_KNEE_KICK_SPEED = 6.0        # Fast knee snap (> 6 deg/frame) on ascent flags leg drive
MAX_WRIST_LIFT_DURING_REP = 0.22      # Hands must stay planted on chair (rejects curls/hand lift)
MIN_SHOULDER_DROP_FOR_DEPTH = 0.08    # Torso must actually descend toward hands for valid depth
ECCENTRIC_DROP_SPEED_MAX = 25.0       # Free-fall drop into bottom (deg/frame)

# 3. Range of Motion & Half-Rep Thresholds
DESCENDING_START_ANGLE = 138.0        # Bending elbow below 138 deg initiates DESCENDING
MIN_VALID_REP_ELBOW_ANGLE = 124.0     # Ignore minor resting twitches at top (> 124 deg)
BOTTOM_TARGET_ANGLE = 102.0           # Forgiving beginner depth (<= 102 deg instead of harsh 90 deg)
ASCENDING_START_ANGLE = 112.0         # Extending elbow from BOTTOM initiates ASCENDING
EARLY_REVERSAL_DELTA = 12.0           # Pressing up > 12 deg before reaching depth flags Half Rep
REP_COMPLETE_ANGLE = 148.0            # Extending arms back to top completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_SIDE = 4
DEBOUNCE_HIP_DRIFT = 5
DEBOUNCE_SHRUG = 5
DEBOUNCE_OVER_DEPTH = 4
DEBOUNCE_ELBOW_FLARE = 5
DEBOUNCE_ELBOW_FORWARD = 5
DEBOUNCE_TORSO_LEAN = 5
DEBOUNCE_HAND_LIFT = 4
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6

ML_FAULT_CONFIDENCE_THRESHOLD = 0.82
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class TricepsDipsTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_dip_reached = False
        self.min_elbow_angle = 180.0
        self.max_hip_drift = 0.0
        self.max_torso_lean = 0.0
        self.min_upper_arm_ratio = 1.0
        self.reversal_frames = 0
        self.active_side = None

        # Instructor Tripwires (SIDE VIEW)
        self.is_side_view = True
        self.hands_planted = True         # Wrists stay anchored on chair (no lifting/curling)
        self.elbows_behind_torso = True   # Elbows bend backward behind torso, not forward
        self.hips_close = True            # Hips stay close to chair edge
        self.shoulders_depressed = True   # No sinking/shrugging ears into shoulders
        self.safe_depth = True            # No dangerous bottoming out below parallel
        self.elbows_tucked = True         # Elbows point straight back (no chicken-wing flare)
        self.torso_upright = True         # No excessive forward hunch or backward slump
        self.controlled_descent = True    # No uncontrolled drop into bottom
        self.ml_posture_maintained = True

        # Advisory / Warning Flags & Baselines
        self.mild_drift_warning = False
        self.mild_lean_warning = False
        self.head_forward_warning = False
        self.leg_drive_warning = False
        self.baseline_neck_ratio = None
        self.baseline_upper_arm_ratio = None
        self.baseline_wrist_y = None
        self.baseline_shoulder_y = None
        self.baseline_knee_angle = None

        # Debounce & Lifecycle counters
        self.side_frames = 0
        self.setup_frames = 0
        self.unlock_frames = 0
        self.out_of_view_frames = 0
        self.rest_exit_frames = 0
        self.drift_frames = 0
        self.mild_drift_frames = 0
        self.shrug_frames = 0
        self.over_depth_frames = 0
        self.flare_frames = 0
        self.elbow_fwd_frames = 0
        self.lean_frames = 0
        self.mild_lean_frames = 0
        self.head_frames = 0
        self.leg_frames = 0
        self.hand_lift_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0
        self.post_rep_timer = 0

        self.is_ready = False
        self.prev_elbow_angle = None
        self.prev_knee_angle = None

        self.current_feedback = "Turn sideways, place hands on chair, and press arms straight"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_triceps_dips.tflite")
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
        self.half_dip_reached = False
        self.min_elbow_angle = 180.0
        self.max_hip_drift = 0.0
        self.max_torso_lean = 0.0
        self.min_upper_arm_ratio = 1.0
        self.reversal_frames = 0
        self.prev_elbow_angle = None
        self.prev_knee_angle = None

        self.is_side_view = True
        self.hands_planted = True
        self.elbows_behind_torso = True
        self.hips_close = True
        self.shoulders_depressed = True
        self.safe_depth = True
        self.elbows_tucked = True
        self.torso_upright = True
        self.controlled_descent = True
        self.mild_drift_warning = False
        self.mild_lean_warning = False
        self.head_forward_warning = False
        self.leg_drive_warning = False
        self.ml_posture_maintained = True

        self.side_frames = 0
        self.drift_frames = 0
        self.mild_drift_frames = 0
        self.shrug_frames = 0
        self.over_depth_frames = 0
        self.flare_frames = 0
        self.elbow_fwd_frames = 0
        self.lean_frames = 0
        self.mild_lean_frames = 0
        self.head_frames = 0
        self.leg_frames = 0
        self.hand_lift_frames = 0
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

        # Ear excluded from hard visibility guard so hair/head tilt does not drop tracking
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
            self.prev_knee_angle = None
            self.frame_buffer.clear()
            if self.out_of_view_frames >= OUT_OF_VIEW_RESET_FRAMES:
                self.state = "WAITING"
                self.is_ready = False
                self.setup_frames = 0
                self.unlock_frames = 0
                self.active_side = None
                self.baseline_wrist_y = None
                self.baseline_shoulder_y = None
                self._reset_rep_tracking()
            return {
                "exercise": "Triceps Dips", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Full side profile must be visible",
                "elbow_angle": 0.0, "hip_drift": 0.0, "torso_lean": 0.0
            }

        self.out_of_view_frames = 0

        # Dynamic Side Selection with Active-Rep Lock (prevents mid-rep left/right flicker)
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
        torso_height = max(abs(active_shoulder.y - active_hip.y), 0.20)
        raw_shoulder_span = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_span / torso_height
        is_facing_side = shoulder_span_ratio <= SETUP_MAX_SIDE_SPAN_RATIO

        # Facing Direction (verifies elbows bend BEHIND torso, not forward in front)
        nose_offset_ratio = (nose.x - active_shoulder.x) / torso_height
        facing_known = nose.visibility >= MIN_VISIBILITY_THRESHOLD and abs(nose_offset_ratio) >= MIN_FACING_NOSE_OFFSET
        facing_dir = 1.0 if nose_offset_ratio >= 0 else -1.0
        elbow_forward_ratio = ((active_elbow.x - active_shoulder.x) * facing_dir) / torso_height if facing_known else 0.0

        # Horizontal hip distance from planted hands
        hip_drift_ratio = abs(active_hip.x - active_wrist.x) / torso_height

        # Torso lean angle from vertical (catches hunched chest / excessive lean)
        torso_dy = max(abs(active_hip.y - active_shoulder.y), 1e-6)
        torso_dx = abs(active_shoulder.x - active_hip.x)
        torso_lean_deg = math.degrees(math.atan2(torso_dx, torso_dy))

        # Signed vertical neck distance (shoulder.y - ear.y > 0 normally; <= 0 when deeply shrugged)
        ear_visible = active_ear.visibility >= MIN_VISIBILITY_THRESHOLD
        neck_ratio = ((active_shoulder.y - active_ear.y) / torso_height) if ear_visible else None
        head_forward_ratio = (abs(active_ear.x - active_shoulder.x) / torso_height) if ear_visible else 0.0

        # Wide / Chicken-Wing Elbow Flare Kinematics
        # At top (vertical) and proper bottom (horizontal back), upper arm stays in side plane.
        # Flaring elbows laterally toward/away from camera foreshortens 2D upper-arm length.
        upper_arm_len_2d = math.hypot(active_shoulder.x - active_elbow.x, active_shoulder.y - active_elbow.y)
        upper_arm_ratio = upper_arm_len_2d / torso_height
        both_elbows_visible = min(l_elbow.visibility, r_elbow.visibility) >= 0.45
        elbow_spread_ratio = 0.0
        if both_elbows_visible and shoulder_span_ratio >= ELBOW_SPREAD_CHECK_MIN_SPAN:
            elbow_spread_ratio = abs(l_elbow.x - r_elbow.x) / max(raw_shoulder_span, 0.10)

        # Excessive Depth Check (shoulder dropping below elbow strains anterior shoulder capsule)
        shoulder_below_elbow_ratio = (active_shoulder.y - active_elbow.y) / torso_height

        # Wrist anchor / body descent checks
        hands_below_shoulders = (active_wrist.y - active_shoulder.y) / torso_height >= SETUP_MIN_WRIST_BELOW_SHOULDER
        wrist_lift_ratio = (
            (self.baseline_wrist_y - active_wrist.y) / torso_height
            if self.baseline_wrist_y is not None else 0.0
        )
        wrist_shift_abs_ratio = abs(wrist_lift_ratio)
        shoulder_drop_ratio = (
            (active_shoulder.y - self.baseline_shoulder_y) / torso_height
            if self.baseline_shoulder_y is not None else 0.0
        )

        # Optional leg-drive advisory when lower body is in frame
        has_legs = active_knee.visibility >= MIN_VISIBILITY_THRESHOLD and active_ankle.visibility >= MIN_VISIBILITY_THRESHOLD
        current_knee_angle = calculate_angle(active_hip, active_knee, active_ankle) if has_legs else None

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Side View + Straight Arms Planted on Chair)
        # ---------------------------------------------------------
        is_resting_setup = (
            is_facing_side
            and hands_below_shoulders
            and (elbow_angle >= SETUP_MIN_ELBOW_ANGLE)
            and (hip_drift_ratio <= SETUP_MAX_HIP_DRIFT_RATIO)
            and (torso_lean_deg <= SETUP_MAX_TORSO_LEAN_DEG)
        )

        if is_resting_setup:
            self.setup_frames += 1
            if not self.rep_started:
                if neck_ratio is not None and neck_ratio > 0.05:
                    if self.baseline_neck_ratio is None:
                        self.baseline_neck_ratio = neck_ratio
                    else:
                        self.baseline_neck_ratio = 0.90 * self.baseline_neck_ratio + 0.10 * neck_ratio

                if self.baseline_upper_arm_ratio is None:
                    self.baseline_upper_arm_ratio = upper_arm_ratio
                else:
                    self.baseline_upper_arm_ratio = 0.90 * self.baseline_upper_arm_ratio + 0.10 * upper_arm_ratio

                if self.baseline_wrist_y is None or self.state in ("WAITING", "LOCKED_FAULT"):
                    self.baseline_wrist_y = (
                        active_wrist.y if self.baseline_wrist_y is None
                        else 0.90 * self.baseline_wrist_y + 0.10 * active_wrist.y
                    )
                    self.baseline_shoulder_y = (
                        active_shoulder.y if self.baseline_shoulder_y is None
                        else 0.90 * self.baseline_shoulder_y + 0.10 * active_shoulder.y
                    )
                elif wrist_shift_abs_ratio < 0.12:
                    self.baseline_wrist_y = 0.95 * self.baseline_wrist_y + 0.05 * active_wrist.y
                    self.baseline_shoulder_y = 0.95 * self.baseline_shoulder_y + 0.05 * active_shoulder.y

                if current_knee_angle is not None:
                    if self.baseline_knee_angle is None:
                        self.baseline_knee_angle = current_knee_angle
                    else:
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
            elif not hands_below_shoulders:
                feedback_msg = "Place your hands firmly on the chair edge behind you"
            elif hip_drift_ratio > SETUP_MAX_HIP_DRIFT_RATIO:
                feedback_msg = "Scoot your hips closer to the chair/bench"
            elif torso_lean_deg > SETUP_MAX_TORSO_LEAN_DEG:
                feedback_msg = "Sit tall with your torso upright to begin"
            else:
                feedback_msg = "Press your arms straight at the top to begin"
            return {
                "exercise": "Triceps Dips", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "hip_drift": round(hip_drift_ratio, 2),
                "torso_lean": round(torso_lean_deg, 1)
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
                            print(f"[TRICEPS DIPS FAULT] AI ST-GCN flagged posture fault: prob={probs[0]:.2f}", flush=True)
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
            self.max_hip_drift = max(self.max_hip_drift, hip_drift_ratio)
            self.max_torso_lean = max(self.max_torso_lean, torso_lean_deg)
            self.min_upper_arm_ratio = min(self.min_upper_arm_ratio, upper_arm_ratio)

            if not is_facing_side:
                self.side_frames += 1
                if self.side_frames >= DEBOUNCE_SIDE:
                    if self.is_side_view:
                        print(f"[TRICEPS DIPS FAULT] Lost side view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.side_frames = max(0, self.side_frames - 1)

            # A. Hands Planted Check (Prevents standing curls / lifting hands off chair)
            if wrist_lift_ratio > MAX_WRIST_LIFT_DURING_REP:
                self.hand_lift_frames += 1
                if self.hand_lift_frames >= DEBOUNCE_HAND_LIFT:
                    if self.hands_planted:
                        print(f"[TRICEPS DIPS FAULT] Wrists lifted off chair: lift={wrist_lift_ratio:.2f}", flush=True)
                    self.hands_planted = False
            else:
                self.hand_lift_frames = max(0, self.hand_lift_frames - 1)

            # B. Backward-Elbow Path Check (Catches unsigned-angle forward-arm crouch/curl)
            if facing_known and elbow_angle < 125.0 and elbow_forward_ratio > MAX_ELBOW_FORWARD_RATIO:
                self.elbow_fwd_frames += 1
                if self.elbow_fwd_frames >= DEBOUNCE_ELBOW_FORWARD:
                    if self.elbows_behind_torso:
                        print(
                            f"[TRICEPS DIPS FAULT] Elbows bent forward in front of torso! "
                            f"elbow_fwd={elbow_forward_ratio:.2f}, elbow={elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.elbows_behind_torso = False
            else:
                self.elbow_fwd_frames = max(0, self.elbow_fwd_frames - 1)

            # C. Hip Drift (Hips pushing too far forward from chair)
            if hip_drift_ratio > MAX_HIP_DRIFT_RATIO:
                self.drift_frames += 1
                if self.drift_frames >= DEBOUNCE_HIP_DRIFT:
                    if self.hips_close:
                        print(f"[TRICEPS DIPS FAULT] Hips drifted too far forward: ratio={hip_drift_ratio:.2f}", flush=True)
                    self.hips_close = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Keep your hips close to the chair!"
            else:
                self.drift_frames = max(0, self.drift_frames - 1)

            if ADVISORY_HIP_DRIFT_RATIO < hip_drift_ratio <= MAX_HIP_DRIFT_RATIO:
                self.mild_drift_frames += 1
                if self.mild_drift_frames >= 4:
                    self.mild_drift_warning = True
            else:
                self.mild_drift_frames = max(0, self.mild_drift_frames - 1)

            # D. Shoulder Sinking / Shrugging (Relative drop + absolute floor if user started slumped)
            if neck_ratio is not None:
                below_relative = (
                    self.baseline_neck_ratio is not None
                    and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR)
                )
                below_absolute = neck_ratio < MIN_ABSOLUTE_NECK_RATIO
                if below_relative or below_absolute:
                    self.shrug_frames += 1
                    if self.shrug_frames >= DEBOUNCE_SHRUG:
                        if self.shoulders_depressed:
                            print(
                                f"[TRICEPS DIPS FAULT] Shoulders sank/shrugged: "
                                f"neck={neck_ratio:.2f} vs base={self.baseline_neck_ratio or 0:.2f}",
                                flush=True,
                            )
                        self.shoulders_depressed = False
                        if self.post_rep_timer == 0:
                            self.current_feedback = "Push shoulders down away from your ears!"
                else:
                    self.shrug_frames = max(0, self.shrug_frames - 1)
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # E. Excessive Depth / Bottoming Out (Anterior Shoulder Strain Risk)
            is_too_deep = (
                (elbow_angle < MIN_SAFE_ELBOW_ANGLE)
                or (shoulder_below_elbow_ratio > MAX_SHOULDER_BELOW_ELBOW_RATIO)
            )
            if is_too_deep:
                self.over_depth_frames += 1
                if self.over_depth_frames >= DEBOUNCE_OVER_DEPTH:
                    if self.safe_depth:
                        print(
                            f"[TRICEPS DIPS FAULT] Dipped too deep! "
                            f"elbow={elbow_angle:.1f}°, sh_below_el={shoulder_below_elbow_ratio:.2f}",
                            flush=True,
                        )
                    self.safe_depth = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Too deep! Stop around 90 degrees to protect shoulders."
            else:
                self.over_depth_frames = max(0, self.over_depth_frames - 1)

            # F. Wide / Chicken-Wing Elbow Flare (Elbows flaring out instead of pointing back)
            shrunk_from_baseline = (
                self.baseline_upper_arm_ratio is not None
                and upper_arm_ratio < (self.baseline_upper_arm_ratio * FLARE_UPPER_ARM_SHRINK_FACTOR)
            )
            is_elbow_flaring = (
                (upper_arm_ratio < MIN_UPPER_ARM_RATIO)
                or shrunk_from_baseline
                or (elbow_spread_ratio > MAX_ELBOW_SPREAD_TO_SHOULDER)
            )
            if is_elbow_flaring and elbow_angle < 130.0:
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_ELBOW_FLARE:
                    if self.elbows_tucked:
                        print(
                            f"[TRICEPS DIPS FAULT] Elbows flared out wide! "
                            f"arm_ratio={upper_arm_ratio:.2f} (base={self.baseline_upper_arm_ratio or 0:.2f}), "
                            f"spread={elbow_spread_ratio:.2f}",
                            flush=True,
                        )
                    self.elbows_tucked = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Point elbows straight back, do not flare out!"
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # G. Excessive Torso Lean / Forward Hunch
            if torso_lean_deg > MAX_TORSO_LEAN_DEG:
                self.lean_frames += 1
                if self.lean_frames >= DEBOUNCE_TORSO_LEAN:
                    if self.torso_upright:
                        print(f"[TRICEPS DIPS FAULT] Excessive torso lean/hunch: angle={torso_lean_deg:.1f}°", flush=True)
                    self.torso_upright = False
            else:
                self.lean_frames = max(0, self.lean_frames - 1)

            if ADVISORY_TORSO_LEAN_DEG < torso_lean_deg <= MAX_TORSO_LEAN_DEG:
                self.mild_lean_frames += 1
                if self.mild_lean_frames >= 5:
                    self.mild_lean_warning = True
            else:
                self.mild_lean_frames = max(0, self.mild_lean_frames - 1)

            # H. Forward Head / Chin-to-Chest Advisory
            if head_forward_ratio > ADVISORY_HEAD_FORWARD_RATIO:
                self.head_frames += 1
                if self.head_frames >= 5:
                    self.head_forward_warning = True
            else:
                self.head_frames = max(0, self.head_frames - 1)

            # I. Leg Drive / Knee Kick Advisory (Extending knees past baseline or fast snap on ascent)
            if current_knee_angle is not None and self.state == "ASCENDING":
                pushed_past_base = (
                    self.baseline_knee_angle is not None
                    and (current_knee_angle - self.baseline_knee_angle) > ADVISORY_KNEE_EXTEND_DELTA
                )
                fast_knee_kick = (
                    self.prev_knee_angle is not None
                    and (current_knee_angle - self.prev_knee_angle) > ADVISORY_KNEE_KICK_SPEED
                )
                if pushed_past_base or fast_knee_kick:
                    self.leg_frames += 1
                    if self.leg_frames >= 3:
                        self.leg_drive_warning = True
                else:
                    self.leg_frames = max(0, self.leg_frames - 1)
                self.prev_knee_angle = current_knee_angle
            else:
                self.prev_knee_angle = current_knee_angle

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
                    self.current_feedback = "Reset complete! Lower your hips straight down."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Triceps Dips", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "hip_drift": round(hip_drift_ratio, 2),
                "torso_lean": round(torso_lean_deg, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.rest_exit_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower slowly straight down along the chair!"

        elif self.state == "RESTING":
            # Graceful set-break / height-change exit: standing up OR sitting down onto a new height resets cleanly
            if wrist_shift_abs_ratio > REST_EXIT_WRIST_SHIFT_RATIO or not hands_below_shoulders:
                self.rest_exit_frames += 1
                if self.rest_exit_frames >= REST_EXIT_FRAMES:
                    self.state = "WAITING"
                    self.is_ready = False
                    self.setup_frames = 0
                    self.rest_exit_frames = 0
                    self.baseline_wrist_y = None
                    self.baseline_shoulder_y = None
                    self._reset_rep_tracking()
                    self.current_feedback = "Position changed. Place hands on chair and straighten arms to resume."
            else:
                self.rest_exit_frames = 0
                if elbow_angle < DESCENDING_START_ANGLE and wrist_lift_ratio <= MAX_WRIST_LIFT_DURING_REP:
                    self.state = "DESCENDING"
                    self.rep_started = True
                    self.min_elbow_angle = elbow_angle
                    self.max_hip_drift = hip_drift_ratio
                    self.max_torso_lean = torso_lean_deg
                    self.min_upper_arm_ratio = upper_arm_ratio
                    self.reversal_frames = 0
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Lower under control, elbows straight back..."

        elif self.state == "DESCENDING" and elbow_angle <= BOTTOM_TARGET_ANGLE and shoulder_drop_ratio >= MIN_SHOULDER_DROP_FOR_DEPTH:
            self.depth_reached = True
            self.state = "BOTTOM"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Good depth! Press back up smoothly."

        elif self.state == "DESCENDING" and not self.depth_reached and (
            (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.half_dip_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Pressing up too early..."
                print(f"[TRICEPS DIPS FAULT] Early reversal: min_elbow={self.min_elbow_angle:.1f}°", flush=True)

        elif self.state == "BOTTOM" and elbow_angle > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive up through your palms..."

        # Allow recovery from brief fatigue tremor on descent if user still completes full depth
        elif (
            self.state == "ASCENDING"
            and not self.depth_reached
            and elbow_angle <= BOTTOM_TARGET_ANGLE
            and shoulder_drop_ratio >= MIN_SHOULDER_DROP_FOR_DEPTH
        ):
            self.depth_reached = True
            self.half_dip_reached = False
            self.state = "BOTTOM"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Good depth! Press back up smoothly."

        if self.state == "DESCENDING" and not ((elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Debounced Eccentric Free-Fall Monitor (Dropping too fast into bottom)
        if self.rep_started and self.state == "DESCENDING":
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(f"[TRICEPS DIPS FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame", flush=True)
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
            # Ignore minor resting twitches at the top
            if (
                self.min_elbow_angle > MIN_VALID_REP_ELBOW_ANGLE
                and not self.depth_reached
                and self.is_side_view
                and self.hands_planted
                and self.elbows_behind_torso
            ):
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Triceps Dips", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(elbow_angle, 1),
                    "hip_drift": round(hip_drift_ratio, 2),
                    "torso_lean": round(torso_lean_deg, 1)
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Keep a side profile to the camera! Hold top to reset."
                is_bad_rep = True
            elif not self.hands_planted:
                self.current_feedback = "Bad rep: Keep hands planted on the chair and lower your body!"
                is_bad_rep = True
            elif not self.elbows_behind_torso:
                self.current_feedback = "Bad rep: Bend elbows backward behind your body, not forward!"
                is_bad_rep = True
            elif not self.safe_depth:
                self.current_feedback = "Bad rep: Dipped too deep! Stop near 90 degrees to protect shoulders."
                is_bad_rep = True
            elif not self.elbows_tucked:
                self.current_feedback = "Bad rep: Elbows flared out wide! Point elbows straight back."
                is_bad_rep = True
            elif not self.hips_close:
                self.current_feedback = "Bad rep: Hips too far forward! Keep your back close to the chair."
                is_bad_rep = True
            elif not self.shoulders_depressed:
                self.current_feedback = "Bad rep: Do not sink into your shoulders! Push shoulders down."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Keep your torso upright! Do not hunch forward."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower too fast! Control the descent to protect shoulders."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor dip posture/mechanics."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "Bad rep: Half rep! Lower deeper until elbows approach 90 degrees."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[TRICEPS DIPS REP #{rep_num} -> BAD REP] Reason: {self.current_feedback} | "
                    f"min_elbow={self.min_elbow_angle:.1f}°, max_drift={self.max_hip_drift:.2f}, "
                    f"max_lean={self.max_torso_lean:.1f}°, min_arm={self.min_upper_arm_ratio:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.mild_drift_warning:
                    self.current_feedback = "Good rep! Reminder: Keep hips grazing close to the chair."
                elif self.mild_lean_warning:
                    self.current_feedback = "Good rep! Reminder: Keep chest proud and torso upright."
                elif self.leg_drive_warning:
                    self.current_feedback = "Good rep! Reminder: Press with your triceps, avoid pushing with legs."
                elif self.head_forward_warning:
                    self.current_feedback = "Good rep! Reminder: Look forward, keep neck neutral."
                else:
                    self.current_feedback = "Great Triceps Dip! Clean depth and shoulder control."
                print(
                    f"[TRICEPS DIPS REP #{rep_num} -> GOOD REP] min_elbow={self.min_elbow_angle:.1f}°, "
                    f"max_drift={self.max_hip_drift:.2f}, max_lean={self.max_torso_lean:.1f}°",
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
            progress = max(0.0, min(1.0, (REP_COMPLETE_ANGLE - elbow_angle) / 50.0))
            form_score = int(60 + progress * 40)
        if not self.is_side_view: form_score -= 35
        if not self.hands_planted: form_score -= 35
        if not self.elbows_behind_torso: form_score -= 35
        if not self.safe_depth: form_score -= 30
        if not self.elbows_tucked: form_score -= 25
        if not self.hips_close: form_score -= 25
        if not self.shoulders_depressed: form_score -= 25
        if not self.torso_upright: form_score -= 20
        if not self.controlled_descent: form_score -= 15
        if self.mild_drift_warning: form_score -= 10
        if self.mild_lean_warning: form_score -= 10
        if self.leg_drive_warning: form_score -= 10
        if self.head_forward_warning: form_score -= 5
        if not self.ml_posture_maintained: form_score -= 25
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 35
        form_score = max(0, form_score)

        return {
            "exercise": "Triceps Dips",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "hip_drift": round(hip_drift_ratio, 2),
            "torso_lean": round(torso_lean_deg, 1)
        }
