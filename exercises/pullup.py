import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (PULLUP - BEGINNER-FRIENDLY FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Front View, Overhead Bar Hang, Bent Knees Allowed)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.23     # Enforces front-facing view
SETUP_MIN_ELBOW_ANGLE = 148.0            # Relaxed dead hang (~148 deg+) keeps scapular tension
SETUP_WRIST_OVER_SHOULDER_MARGIN = 0.05  # Wrists must be clearly above shoulders (holding bar)
SETUP_MIN_READY_FRAMES = 6               # Consecutive frames in hang to confirm READY
UNLOCK_RESET_FRAMES = 10                 # Frames held in dead hang to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35          # Minimum upper-body + hip landmark visibility
OUT_OF_VIEW_RESET_FRAMES = 15            # Frames out of view before full state reset
REST_EXIT_WRIST_SHIFT_RATIO = 0.26       # Letting go or switching bar height while resting exits cleanly
REST_EXIT_FRAMES = 12                    # Frames with hands off bar / shifted to exit between sets

# 2. Kinematic & Form Thresholds (Beginner-Friendly Home/Doorframe Bar)
MAX_TORSO_SWAY_RATIO = 0.38              # Horizontal shoulder-to-hip sway ratio (lateral kipping)
ADVISORY_SWAY_RATIO = 0.25               # Mild sway reminder without failing rep
SAGITTAL_KIP_TORSO_FACTOR = 0.72         # Torso/shoulder ratio shrinking < 72% flags forward/back swing
KNEE_DRIVE_VISIBILITY = 0.50             # Only check knee drive if knees are clearly visible
KNEE_DRIVE_HIP_MARGIN = 0.12             # Knees rising within 12% torso height of hips flags knee kip
MAX_SHOULDER_TILT_RATIO = 0.14           # Vertical L/R shoulder height gap / torso height
MAX_ELBOW_ASYMMETRY_DEG = 34.0           # L vs R elbow angle gap (catches one-arm chicken-wing pull)
TORSO_TWIST_RATIO_FACTOR = 0.75          # 2D shoulder span ratio shrinking < 75% flags torso twisting
MAX_WRIST_DROP_DURING_REP = 0.25         # Wrists dropping > 25% torso height rejects air-pulldown cheats
MIN_SHOULDER_RISE_FOR_TOP = 0.16         # Shoulders must actually rise upward toward the bar
ECCENTRIC_DROP_SPEED_MAX = 28.0          # Uncontrolled elbow-opening drop speed (deg/frame)
ECCENTRIC_SHOULDER_DROP_MIN = 0.05       # Corroborates real physical free-fall (prevents Z-flip false alarms)
ADVISORY_MIN_GRIP_RATIO = 0.60           # Very narrow grip reminder (hands nearly touching)
ADVISORY_MAX_GRIP_RATIO = 2.25           # Overly wide grip reminder
ADVISORY_TOP_SHRUG_NECK_RATIO = 0.08     # Shoulders bunched into ears at top reminder

# 3. Range of Motion & Half-Rep Thresholds
PULLING_START_ELBOW_ANGLE = 138.0        # Bending elbows below 138 deg initiates PULLING
PULLING_START_SHOULDER_RISE = 0.16       # Forward-elbow/chin-up fallback start trigger
MIN_VALID_REP_ELBOW_ANGLE = 122.0        # Ignore minor dead-hang twitches (> 122 deg min angle)
MIN_VALID_REP_SHOULDER_RISE = 0.18       # Ignore minor scapular shrug twitches (< 18% torso height)
NOSE_BAR_CLEARANCE_MARGIN = 0.04         # Nose within 4% torso height of wrist line confirms clearance
MAX_TOP_ELBOW_ANGLE_WITH_NOSE = 108.0    # Prevents chin-craning at near-straight arms from counting
TOP_ELBOW_FALLBACK_ANGLE = 78.0          # Fallback top depth if bar occludes nose
LOWERING_START_OFFSET = 0.06             # Nose dropping below bar line initiates LOWERING
LOWERING_START_ELBOW_ANGLE = 98.0        # Elbow opening past 98 deg also transitions TOP -> LOWERING
EARLY_REVERSAL_DELTA = 14.0              # Re-extending > 14 deg before clearing bar flags Half Rep
REP_COMPLETE_ELBOW_ANGLE = 148.0         # Returning to relaxed dead hang completes rep
REP_COMPLETE_MIN_NOSE_GAP = 0.12         # Head must be back below bar to prevent mid-air Z-flip completion
REP_COMPLETE_MAX_SHOULDER_RISE = 0.22    # Shoulders must lower back near hang (allows active-shoulder hang)

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_SWAY = 5
DEBOUNCE_KIP = 5
DEBOUNCE_TWIST = 5
DEBOUNCE_ASYMMETRY = 5
DEBOUNCE_HAND_ANCHOR = 4
DEBOUNCE_GRIP = 5
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6

ML_FAULT_CONFIDENCE_THRESHOLD = 0.82
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class PullupTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_pull_reached = False
        self.min_elbow_angle = 180.0
        self.min_nose_bar_gap = 1.0
        self.max_sway_ratio = 0.0
        self.max_elbow_diff = 0.0
        self.max_shoulder_rise = 0.0
        self.reversal_frames = 0

        # Instructor Tripwires (FRONT VIEW - Upper Body + Hips Required, Knees Optional)
        self.is_front_view = True
        self.hands_anchored = True        # Hands stay fixed on bar until hang completion
        self.torso_stable = True          # No lateral sway, sagittal swing, or torso twisting
        self.no_knee_kipping = True       # No driving knees up toward chest for momentum
        self.balanced_pull = True         # Both arms & shoulders pull evenly
        self.controlled_descent = True    # No uncontrolled drop back to dead hang
        self.ml_posture_maintained = True

        # Advisory / Warning Flags & Baselines
        self.mild_sway_warning = False
        self.top_shrug_warning = False
        self.grip_width_warning = None    # 'narrow' or 'wide'
        self.baseline_bar_y = None
        self.baseline_shoulder_y = None
        self.baseline_span_ratio = None
        self.baseline_torso_to_shoulder = None

        # Debounce & Lifecycle counters
        self.front_frames = 0
        self.setup_frames = 0
        self.unlock_frames = 0
        self.out_of_view_frames = 0
        self.rest_exit_frames = 0
        self.sway_frames = 0
        self.mild_sway_frames = 0
        self.kip_frames = 0
        self.knee_kip_frames = 0
        self.twist_frames = 0
        self.asymmetry_frames = 0
        self.hand_anchor_frames = 0
        self.off_bar_frames = 0
        self.narrow_grip_frames = 0
        self.wide_grip_frames = 0
        self.shrug_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0
        self.post_rep_timer = 0

        self.is_ready = False
        self.prev_elbow_angle = None
        self.prev_shoulder_y = None

        self.current_feedback = "Face camera, grab the bar, and hang with arms straight"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

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
        self.half_pull_reached = False
        self.min_elbow_angle = 180.0
        self.min_nose_bar_gap = 1.0
        self.max_sway_ratio = 0.0
        self.max_elbow_diff = 0.0
        self.max_shoulder_rise = 0.0
        self.reversal_frames = 0
        self.prev_elbow_angle = None
        self.prev_shoulder_y = None

        self.is_front_view = True
        self.hands_anchored = True
        self.torso_stable = True
        self.no_knee_kipping = True
        self.balanced_pull = True
        self.controlled_descent = True
        self.mild_sway_warning = False
        self.top_shrug_warning = False
        self.grip_width_warning = None
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.sway_frames = 0
        self.mild_sway_frames = 0
        self.kip_frames = 0
        self.knee_kip_frames = 0
        self.twist_frames = 0
        self.asymmetry_frames = 0
        self.hand_anchor_frames = 0
        self.off_bar_frames = 0
        self.narrow_grip_frames = 0
        self.wide_grip_frames = 0
        self.shrug_frames = 0
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

        # Nose excluded from hard visibility guard because the pull-up bar can briefly block the face at TOP!
        # Knees & ankles excluded so bent-knee home doorframe setups work reliably.
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
            self.prev_elbow_angle = None
            self.prev_shoulder_y = None
            self.frame_buffer.clear()
            if self.out_of_view_frames >= OUT_OF_VIEW_RESET_FRAMES:
                self.state = "WAITING"
                self.is_ready = False
                self.setup_frames = 0
                self.unlock_frames = 0
                self.baseline_bar_y = None
                self.baseline_shoulder_y = None
                self.baseline_span_ratio = None
                self.baseline_torso_to_shoulder = None
                self._reset_rep_tracking()
            return {
                "exercise": "Pullup", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Arms, shoulders, and torso must be visible",
                "elbow_angle": 0.0, "torso_sway": 0.0, "nose_bar_gap": 0.0
            }

        self.out_of_view_frames = 0

        # Bilateral Upper-Body Kinematics
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        elbow_diff_deg = abs(l_elbow_angle - r_elbow_angle)

        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0
        raw_torso_height = abs(mid_shoulder_y - mid_hip_y)
        torso_height = max(raw_torso_height, 0.20)

        raw_shoulder_span = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_span / torso_height
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO
        shoulder_width = max(raw_shoulder_span, 0.16)
        torso_to_shoulder_ratio = raw_torso_height / shoulder_width

        # Head / Bar Proxy Tracking (Ear fallback if bar briefly occludes nose at TOP)
        bar_height_y = (l_wrist.y + r_wrist.y) / 2.0
        hands_overhead = bar_height_y < (mid_shoulder_y - SETUP_WRIST_OVER_SHOULDER_MARGIN)
        ears_visible = min(l_ear.visibility, r_ear.visibility) >= MIN_VISIBILITY_THRESHOLD
        mid_ear_y = (l_ear.y + r_ear.y) / 2.0

        if nose.visibility >= MIN_VISIBILITY_THRESHOLD:
            head_y = nose.y
            head_visible = True
        elif ears_visible:
            head_y = mid_ear_y
            head_visible = True
        else:
            head_y = mid_shoulder_y
            head_visible = False

        nose_bar_gap = (head_y - bar_height_y) / torso_height  # <= 0 means nose/head at or above wrists

        # Wrist Anchor & Body Rise Checks (Rejects fake standing air-pulldowns & early bar release)
        wrist_drop_ratio = (
            (bar_height_y - self.baseline_bar_y) / torso_height
            if self.baseline_bar_y is not None else 0.0
        )
        wrist_shift_abs_ratio = abs(wrist_drop_ratio)
        shoulder_rise_ratio = (
            (self.baseline_shoulder_y - mid_shoulder_y) / torso_height
            if self.baseline_shoulder_y is not None else 0.0
        )

        # Torso Sway, Symmetry, Grip Width & Optional Knee-Drive Kipping
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        torso_sway_ratio = abs(mid_shoulder_x - mid_hip_x) / shoulder_width
        shoulder_tilt_ratio = abs(l_shoulder.y - r_shoulder.y) / torso_height
        grip_width_ratio = abs(l_wrist.x - r_wrist.x) / shoulder_width
        neck_ratio = ((mid_shoulder_y - mid_ear_y) / torso_height) if ears_visible else None

        knees_visible = min(l_knee.visibility, r_knee.visibility) >= KNEE_DRIVE_VISIBILITY
        knee_hip_gap_ratio = (
            (min(l_knee.y, r_knee.y) - mid_hip_y) / torso_height
            if knees_visible else 1.0
        )  # Normal hang (even bent-knee behind you) stays > 0.35; knee-drive kip drops <= 0.12

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Front View + Overhead Relaxed Dead Hang)
        # ---------------------------------------------------------
        is_resting_setup = (
            is_facing_front
            and hands_overhead
            and (avg_elbow_angle >= SETUP_MIN_ELBOW_ANGLE)
        )

        if is_resting_setup:
            self.setup_frames += 1
            if not self.rep_started:
                if self.baseline_bar_y is None or self.state in ("WAITING", "LOCKED_FAULT"):
                    self.baseline_bar_y = (
                        bar_height_y if self.baseline_bar_y is None
                        else 0.90 * self.baseline_bar_y + 0.10 * bar_height_y
                    )
                    self.baseline_shoulder_y = (
                        mid_shoulder_y if self.baseline_shoulder_y is None
                        else 0.90 * self.baseline_shoulder_y + 0.10 * mid_shoulder_y
                    )
                elif wrist_shift_abs_ratio < 0.12:
                    self.baseline_bar_y = 0.95 * self.baseline_bar_y + 0.05 * bar_height_y
                    self.baseline_shoulder_y = 0.95 * self.baseline_shoulder_y + 0.05 * mid_shoulder_y

                if self.baseline_span_ratio is None:
                    self.baseline_span_ratio = shoulder_span_ratio
                else:
                    self.baseline_span_ratio = 0.90 * self.baseline_span_ratio + 0.10 * shoulder_span_ratio

                if self.baseline_torso_to_shoulder is None:
                    self.baseline_torso_to_shoulder = torso_to_shoulder_ratio
                else:
                    self.baseline_torso_to_shoulder = (
                        0.90 * self.baseline_torso_to_shoulder + 0.10 * torso_to_shoulder_ratio
                    )

            if self.setup_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.setup_frames = max(0, self.setup_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            elif not hands_overhead:
                feedback_msg = "Grab the bar overhead to begin"
            else:
                feedback_msg = "Lower into a controlled hang with arms straight to start"
            return {
                "exercise": "Pullup", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(avg_elbow_angle, 1),
                "torso_sway": round(torso_sway_ratio, 2),
                "nose_bar_gap": round(nose_bar_gap, 2)
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

                if probs[0] >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state in ["PULLING", "TOP", "LOWERING"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(f"[PULLUP FAULT] AI ST-GCN flagged posture fault: prob={probs[0]:.2f}", flush=True)
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (FRONT VIEW - HOME BAR COMPATIBLE)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["PULLING", "TOP", "LOWERING"]:
            self.min_elbow_angle = min(self.min_elbow_angle, avg_elbow_angle)
            self.min_nose_bar_gap = min(self.min_nose_bar_gap, nose_bar_gap)
            self.max_sway_ratio = max(self.max_sway_ratio, torso_sway_ratio)
            self.max_elbow_diff = max(self.max_elbow_diff, elbow_diff_deg)
            self.max_shoulder_rise = max(self.max_shoulder_rise, shoulder_rise_ratio)

            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[PULLUP FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # A. Hands Anchored on Bar (Rejects air-pulldown cheats & letting go early mid-rep)
            if wrist_drop_ratio > MAX_WRIST_DROP_DURING_REP:
                self.hand_anchor_frames += 1
                if self.hand_anchor_frames >= DEBOUNCE_HAND_ANCHOR:
                    if self.hands_anchored:
                        print(f"[PULLUP FAULT] Wrists dropped off bar line: drop={wrist_drop_ratio:.2f}", flush=True)
                    self.hands_anchored = False
            else:
                self.hand_anchor_frames = max(0, self.hand_anchor_frames - 1)

            if not hands_overhead:
                self.off_bar_frames += 1
                if self.off_bar_frames >= DEBOUNCE_HAND_ANCHOR:
                    if self.hands_anchored:
                        print("[PULLUP FAULT] Hands left the bar before completing the hang!", flush=True)
                    self.hands_anchored = False
            else:
                self.off_bar_frames = max(0, self.off_bar_frames - 1)

            # B. Lateral Torso Sway / Side Kipping
            if torso_sway_ratio > MAX_TORSO_SWAY_RATIO:
                self.sway_frames += 1
                if self.sway_frames >= DEBOUNCE_SWAY:
                    if self.torso_stable:
                        print(f"[PULLUP FAULT] Excessive lateral torso sway: ratio={torso_sway_ratio:.2f}", flush=True)
                    self.torso_stable = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Tighten your core and stop swinging!"
            else:
                self.sway_frames = max(0, self.sway_frames - 1)

            if ADVISORY_SWAY_RATIO < torso_sway_ratio <= MAX_TORSO_SWAY_RATIO:
                self.mild_sway_frames += 1
                if self.mild_sway_frames >= 4:
                    self.mild_sway_warning = True
            else:
                self.mild_sway_frames = max(0, self.mild_sway_frames - 1)

            # C. Forward/Backward Sagittal Kipping (Torso foreshortening out of vertical plane)
            if (
                self.baseline_torso_to_shoulder is not None
                and torso_to_shoulder_ratio < (self.baseline_torso_to_shoulder * SAGITTAL_KIP_TORSO_FACTOR)
                and self.state in ["PULLING", "TOP"]
            ):
                self.kip_frames += 1
                if self.kip_frames >= DEBOUNCE_KIP:
                    if self.torso_stable:
                        print(
                            f"[PULLUP FAULT] Forward/backward kipping swing! "
                            f"torso_ratio={torso_to_shoulder_ratio:.2f} vs base={self.baseline_torso_to_shoulder:.2f}",
                            flush=True,
                        )
                    self.torso_stable = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Avoid swinging your hips forward! Strict pull."
            else:
                self.kip_frames = max(0, self.kip_frames - 1)

            # D. Optional Knee-Drive Kipping (Only if knees are visible AND kick up near hip height)
            if knees_visible and knee_hip_gap_ratio <= KNEE_DRIVE_HIP_MARGIN and self.state in ["PULLING", "TOP"]:
                self.knee_kip_frames += 1
                if self.knee_kip_frames >= DEBOUNCE_KIP:
                    if self.no_knee_kipping:
                        print(f"[PULLUP FAULT] Knee-drive kipping detected: knee_hip_gap={knee_hip_gap_ratio:.2f}", flush=True)
                    self.no_knee_kipping = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Do not kick your knees up for momentum!"
            else:
                self.knee_kip_frames = max(0, self.knee_kip_frames - 1)

            # E. Torso Twisting Mid-Pull (Camera-distance-invariant span ratio shrink)
            if self.baseline_span_ratio and shoulder_span_ratio < (self.baseline_span_ratio * TORSO_TWIST_RATIO_FACTOR):
                self.twist_frames += 1
                if self.twist_frames >= DEBOUNCE_TWIST:
                    if self.torso_stable:
                        print(
                            f"[PULLUP FAULT] Torso twisted mid-pull: span_ratio={shoulder_span_ratio:.2f} "
                            f"vs base={self.baseline_span_ratio:.2f}",
                            flush=True,
                        )
                    self.torso_stable = False
            else:
                self.twist_frames = max(0, self.twist_frames - 1)

            # F. Asymmetrical Pulling (Uneven shoulder height OR one-arm chicken-wing bend)
            is_asymmetrical = (shoulder_tilt_ratio > MAX_SHOULDER_TILT_RATIO) or (elbow_diff_deg > MAX_ELBOW_ASYMMETRY_DEG)
            if is_asymmetrical and self.state in ["PULLING", "TOP"]:
                self.asymmetry_frames += 1
                if self.asymmetry_frames >= DEBOUNCE_ASYMMETRY:
                    if self.balanced_pull:
                        print(
                            f"[PULLUP FAULT] Uneven pull: tilt={shoulder_tilt_ratio:.2f}, "
                            f"elbow_diff={elbow_diff_deg:.1f}°",
                            flush=True,
                        )
                    self.balanced_pull = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Pull evenly with both arms!"
            else:
                self.asymmetry_frames = max(0, self.asymmetry_frames - 1)

            # G. Top Shoulder Shrug Advisory (Shoulders still bunched near ears at TOP)
            if self.state == "TOP" and neck_ratio is not None and neck_ratio < ADVISORY_TOP_SHRUG_NECK_RATIO:
                self.shrug_frames += 1
                if self.shrug_frames >= 3:
                    self.top_shrug_warning = True
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # H. Debounced Grip Width Advisory (Prevents 1-frame wrist jitter false warnings)
            if grip_width_ratio < ADVISORY_MIN_GRIP_RATIO:
                self.narrow_grip_frames += 1
                if self.narrow_grip_frames >= DEBOUNCE_GRIP:
                    self.grip_width_warning = "narrow"
            else:
                self.narrow_grip_frames = max(0, self.narrow_grip_frames - 1)

            if grip_width_ratio > ADVISORY_MAX_GRIP_RATIO:
                self.wide_grip_frames += 1
                if self.wide_grip_frames >= DEBOUNCE_GRIP:
                    self.grip_width_warning = "wide"
            else:
                self.wide_grip_frames = max(0, self.wide_grip_frames - 1)

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
                    self.current_feedback = "Reset complete! Pull straight up with control."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Pullup", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(avg_elbow_angle, 1),
                "torso_sway": round(torso_sway_ratio, 2),
                "nose_bar_gap": round(nose_bar_gap, 2)
            }

        top_clearance_reached = (
            shoulder_rise_ratio >= MIN_SHOULDER_RISE_FOR_TOP
            and (
                (head_visible and nose_bar_gap <= NOSE_BAR_CLEARANCE_MARGIN and avg_elbow_angle <= MAX_TOP_ELBOW_ANGLE_WITH_NOSE)
                or (avg_elbow_angle <= TOP_ELBOW_FALLBACK_ANGLE)
            )
        )

        started_this_frame = False

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.rest_exit_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull up smoothly until your chin clears the bar!"

        elif self.state == "RESTING":
            # Graceful set-break / bar-height change exit: releasing bar or moving bar line resets cleanly
            if not hands_overhead or wrist_shift_abs_ratio > REST_EXIT_WRIST_SHIFT_RATIO:
                self.rest_exit_frames += 1
                if self.rest_exit_frames >= REST_EXIT_FRAMES:
                    self.state = "WAITING"
                    self.is_ready = False
                    self.setup_frames = 0
                    self.rest_exit_frames = 0
                    self.baseline_bar_y = None
                    self.baseline_shoulder_y = None
                    self._reset_rep_tracking()
                    self.current_feedback = "Stepped off bar. Grab the bar overhead and hang to resume."
            else:
                self.rest_exit_frames = 0
                started_pull = (
                    avg_elbow_angle < PULLING_START_ELBOW_ANGLE
                    or shoulder_rise_ratio >= PULLING_START_SHOULDER_RISE
                )
                if started_pull and wrist_drop_ratio <= MAX_WRIST_DROP_DURING_REP:
                    self.state = "PULLING"
                    self.rep_started = True
                    started_this_frame = True
                    self.min_elbow_angle = avg_elbow_angle
                    self.min_nose_bar_gap = nose_bar_gap
                    self.max_sway_ratio = torso_sway_ratio
                    self.max_elbow_diff = elbow_diff_deg
                    self.max_shoulder_rise = shoulder_rise_ratio
                    self.reversal_frames = 0
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Pull with your back, keep your core steady..."

        elif self.state == "PULLING" and top_clearance_reached:
            self.depth_reached = True
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Chin cleared! Lower yourself slowly."

        elif self.state == "PULLING" and not self.depth_reached and (
            (avg_elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.half_pull_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Half rep: Lowering before clearing the bar..."
                print(
                    f"[PULLUP FAULT] Early reversal: min_elbow={self.min_elbow_angle:.1f}°, "
                    f"min_nose_gap={self.min_nose_bar_gap:.2f}",
                    flush=True,
                )

        elif self.state == "TOP" and (
            (head_visible and nose_bar_gap > LOWERING_START_OFFSET)
            or avg_elbow_angle > LOWERING_START_ELBOW_ANGLE
        ):
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent back to a hang..."

        # Allow recovery from brief fatigue sticking-point shake if user still clears the bar!
        elif self.state == "LOWERING" and not self.depth_reached and top_clearance_reached:
            self.depth_reached = True
            self.half_pull_reached = False
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Chin cleared! Lower yourself slowly."

        if self.state == "PULLING" and not ((avg_elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Debounced Eccentric Free-Fall Monitor (requires both elbow snap AND physical torso drop)
        if self.rep_started and self.state in ["LOWERING", "PULLING"]:
            if self.prev_elbow_angle is not None and self.prev_shoulder_y is not None:
                drop_speed = avg_elbow_angle - self.prev_elbow_angle
                shoulder_drop_speed = (mid_shoulder_y - self.prev_shoulder_y) / torso_height
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX and shoulder_drop_speed > ECCENTRIC_SHOULDER_DROP_MIN:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[PULLUP FAULT] Uncontrolled drop speed: "
                                f"elbow={drop_speed:.1f} deg/frame, sh_drop={shoulder_drop_speed:.2f}",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = avg_elbow_angle
            self.prev_shoulder_y = mid_shoulder_y
        else:
            self.prev_elbow_angle = None
            self.prev_shoulder_y = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        head_back_below_bar = (not head_visible) or (nose_bar_gap >= REP_COMPLETE_MIN_NOSE_GAP)
        shoulders_back_to_hang = shoulder_rise_ratio <= REP_COMPLETE_MAX_SHOULDER_RISE
        returned_to_hang = (
            hands_overhead
            and avg_elbow_angle >= REP_COMPLETE_ELBOW_ANGLE
            and head_back_below_bar
            and shoulders_back_to_hang
        )
        dropped_off_bar_mid_rep = self.off_bar_frames >= DEBOUNCE_HAND_ANCHOR

        if (
            self.rep_started
            and not started_this_frame
            and self.state in ["LOWERING", "TOP", "PULLING"]
            and (returned_to_hang or dropped_off_bar_mid_rep)
        ):
            # Ignore minor resting twitches at the bottom
            if (
                self.min_elbow_angle > MIN_VALID_REP_ELBOW_ANGLE
                and self.max_shoulder_rise < MIN_VALID_REP_SHOULDER_RISE
                and not self.depth_reached
                and self.is_front_view
                and self.hands_anchored
            ):
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Pullup", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(avg_elbow_angle, 1),
                    "torso_sway": round(torso_sway_ratio, 2),
                    "nose_bar_gap": round(nose_bar_gap, 2)
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Hang to reset."
                is_bad_rep = True
            elif not self.hands_anchored:
                self.current_feedback = "Bad rep: Keep hands on the bar until you finish lowering to a hang!"
                is_bad_rep = True
            elif not self.no_knee_kipping:
                self.current_feedback = "Bad rep: Do not kick or drive your knees up! Strict pull."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "Bad rep: Excessive swinging or twisting! Tighten your core."
                is_bad_rep = True
            elif not self.balanced_pull:
                self.current_feedback = "Bad rep: Uneven pull! Pull smoothly with both arms together."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Lower yourself slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged unstable pull-up mechanics."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "Bad rep: Half rep! Pull higher so your chin clears the bar."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[PULLUP REP #{rep_num} -> BAD REP] Reason: {self.current_feedback} | "
                    f"min_elbow={self.min_elbow_angle:.1f}°, min_nose_gap={self.min_nose_bar_gap:.2f}, "
                    f"max_sway={self.max_sway_ratio:.2f}, max_el_diff={self.max_elbow_diff:.1f}°, "
                    f"rise={self.max_shoulder_rise:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.mild_sway_warning:
                    self.current_feedback = "Good rep! Reminder: Keep your core tight to minimize sway."
                elif self.top_shrug_warning:
                    self.current_feedback = "Good rep! Reminder: Pull shoulders down away from your ears at top."
                elif self.grip_width_warning == "narrow":
                    self.current_feedback = "Good rep! Reminder: Try a slightly wider shoulder-width grip."
                elif self.grip_width_warning == "wide":
                    self.current_feedback = "Good rep! Reminder: Bring grip slightly closer to shoulder width."
                else:
                    self.current_feedback = "Great Pull-up! Strong height and controlled hang."
                print(
                    f"[PULLUP REP #{rep_num} -> GOOD REP] min_elbow={self.min_elbow_angle:.1f}°, "
                    f"min_nose_gap={self.min_nose_bar_gap:.2f}, rise={self.max_shoulder_rise:.2f}",
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
            progress = max(0.0, min(1.0, (REP_COMPLETE_ELBOW_ANGLE - avg_elbow_angle) / 70.0))
            form_score = int(60 + progress * 40)
        if not self.is_front_view: form_score -= 35
        if not self.hands_anchored: form_score -= 35
        if not self.no_knee_kipping: form_score -= 30
        if not self.torso_stable: form_score -= 30
        if not self.balanced_pull: form_score -= 20
        if not self.controlled_descent: form_score -= 15
        if self.mild_sway_warning: form_score -= 10
        if self.top_shrug_warning: form_score -= 5
        if self.grip_width_warning is not None: form_score -= 5
        if not self.ml_posture_maintained: form_score -= 25
        if not self.depth_reached and self.state == "LOWERING": form_score -= 35
        form_score = max(0, form_score)

        return {
            "exercise": "Pullup",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(avg_elbow_angle, 1),
            "torso_sway": round(torso_sway_ratio, 2),
            "nose_bar_gap": round(nose_bar_gap, 2)
        }
