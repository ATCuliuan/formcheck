import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SHOULDER INT/EXT ROTATION - FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Front View, Elbow Pinned to Ribs at 90° Level)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.24 # Enforces strict front-facing view
SETUP_MAX_ELBOW_RIB_DRIFT = 0.35     # Step 2: Elbow tucked against ribs/towel (loosened from 0.28)
SETUP_MAX_WRIST_ELBOW_Y_DIFF = 0.34  # Step 2: Forearm horizontal at 90° elbow height (normalized by torso)
SETUP_MIN_UPPER_ARM_DROP = 0.42      # Upper arm hanging vertically down from shoulder to ribs
SETUP_MIN_READY_FRAMES = 6           # Frames holding 90° tucked position to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held in 90° tucked setup to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 1.8       # Minimum arm + torso keypoint visibility

# 2. Kinematic & Form Thresholds (Step 2 & 3: Towel Pin, Fixed 90° Bend, Upright Torso, No Twist)
MAX_ALLOWED_ELBOW_DRIFT = 0.40       # Step 2 & 3: Keep elbow pressed against towel/ribs (no chicken-wing flare)
MAX_WRIST_ELBOW_Y_DRIFT = 0.40       # Step 2: Maintain 90° elbow bend (wrist stays near elbow height)
MAX_TORSO_SWAY_RATIO = 0.15          # Keep spine tall and steady (no lateral lean)
MAX_SHOULDER_TWIST_SHRINK = 0.22     # Catches twisting the torso/shoulders to pull the band!
SHRUG_NECK_RATIO_FACTOR = 0.65       # Catches hiking/shrugging the working shoulder toward the ear
ECCENTRIC_SNAP_SPEED_MAX = 22.0      # Catches letting the band snap the forearm back without control (deg/frame)

# 3. Transverse-Plane Rotation Thresholds (Supports Both Part A External & Part B Internal)
# Rotation is converted to degrees (-65° Internal across belly <-> 0° Neutral <-> +65° External outward)
ROTATION_START_DELTA_DEG = 15.0      # Rotating >= 15° from frozen starting anchor initiates active rep
MIN_VALID_REP_SWEEP_DEG = 18.0       # Ignore tiny hand tremors (< 18° total sweep)
HALF_ROTATION_MIN_SWEEP_DEG = 22.0   # Sweeping 22°–34° without hitting full ROM flags Half Rep (Bad Rep)
FULL_ROTATION_MIN_SWEEP_DEG = 35.0   # Full external door-open OR full internal belly-sweep (>= 35° arc)
RETURN_REVERSAL_DELTA_DEG = 9.0      # Reversing >= 9° from peak transitions to ROTATING_IN
REP_COMPLETE_RETURN_DEG = 14.0       # Returning within 14° (or past start anchor) completes the rep

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 3
DEBOUNCE_ELBOW_DRIFT = 4
DEBOUNCE_ARM_STRAIGHTEN = 4
DEBOUNCE_TORSO = 4
DEBOUNCE_SHRUG = 4
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingShoulderInternalExternalRotationTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.external_reached = False      # True when full External (Part A) or Internal (Part B) sweep is hit
        self.was_half_rep = False
        self.max_rotation = 0.0            # Tracks peak directional angular sweep (deg) achieved during the rep
        self.start_rotation_deg = 0.0      # Frozen anchor where the forearm started in NEUTRAL
        self.peak_rotation_deg = 0.0
        self.prev_sweep_deg = None
        self.rotation_mode = "EXTERNAL"    # Automatically detects 'EXTERNAL' (Part A) or 'INTERNAL' (Part B)
        self.locked_working_arm = None     # Locks 'LEFT' or 'RIGHT' working arm during the rep

        # Form tripwires (STRICT FRONT VIEW)
        self.is_front_view = True
        self.elbow_pinned = True           # Step 2 & 3: Elbow pressed against towel/ribs
        self.elbow_angle_maintained = True # Step 2: Strict 90° elbow bend (wrist stays at elbow height)
        self.torso_steady = True           # No torso twisting or lateral swaying
        self.shoulder_depressed = True     # No shrugging the working shoulder
        self.controlled_return = True      # Do not let band snap forearm back
        self.ml_posture_maintained = True

        self.baseline_shoulder_width = None
        self.baseline_neck_ratio = None

        # Debounce counters
        self.front_frames = 0
        self.elbow_drift_frames = 0
        self.arm_angle_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Face camera: bend elbow 90° at waist level and pin elbow to your ribs"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_internal_external_rotation.tflite")
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
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.external_reached = False
        self.was_half_rep = False
        self.max_rotation = 0.0
        self.peak_rotation_deg = self.start_rotation_deg
        self.prev_sweep_deg = None
        self.locked_working_arm = None

        self.is_front_view = True
        self.elbow_pinned = True
        self.elbow_angle_maintained = True
        self.torso_steady = True
        self.shoulder_depressed = True
        self.controlled_return = True
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.elbow_drift_frames = 0
        self.arm_angle_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
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
                "exercise": "Shoulder Int/Ext Rotation", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Face camera with torso and rotating arm in frame",
                "elbow_angle": 0.0, "rotation_angle": 0.0
            }

        # 1. Front View & Body Scale Geometry
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        torso_height = max(abs(mid_shoulder_y - mid_hip_y), 0.30)
        raw_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_width / torso_height
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO
        shoulder_width = max(raw_shoulder_width, 0.22)

        # 2. Smart Working-Arm Selection (Finds whichever arm is bent at 90° elbow height!)
        l_wrist_elbow_y_diff = abs(l_wrist.y - l_elbow.y) / torso_height
        r_wrist_elbow_y_diff = abs(r_wrist.y - r_elbow.y) / torso_height

        if self.rep_started and self.locked_working_arm is not None:
            use_left_arm = (self.locked_working_arm == "LEFT")
        else:
            if l_vis > 1.8 and r_vis > 1.8:
                use_left_arm = (l_wrist_elbow_y_diff < r_wrist_elbow_y_diff)
            else:
                use_left_arm = (l_vis > r_vis)

        if use_left_arm:
            active_ear, active_shoulder, active_elbow, active_wrist, active_hip = l_ear, l_shoulder, l_elbow, l_wrist, l_hip
            arm_side_sign = 1.0 if (l_shoulder.x >= r_shoulder.x) else -1.0
        else:
            active_ear, active_shoulder, active_elbow, active_wrist, active_hip = r_ear, r_shoulder, r_elbow, r_wrist, r_hip
            arm_side_sign = 1.0 if (r_shoulder.x >= l_shoulder.x) else -1.0

        ear_shoulder_ratio = abs(active_ear.y - active_shoulder.y) / torso_height

        # 3. True Transverse-Plane 90° Elbow & Rotation Kinematics
        # A. Rib-Pin Check: Horizontal distance of elbow from the shoulder/rib vertical line
        rib_anchor_x = 0.72 * active_shoulder.x + 0.28 * active_hip.x
        elbow_drift_ratio = abs(active_elbow.x - rib_anchor_x) / shoulder_width
        upper_arm_vertical_drop = (active_elbow.y - active_shoulder.y) / torso_height
        is_elbow_tucked = (elbow_drift_ratio <= SETUP_MAX_ELBOW_RIB_DRIFT) and (upper_arm_vertical_drop >= SETUP_MIN_UPPER_ARM_DROP)

        # B. 90° Elbow Flexion Check in Front View
        wrist_elbow_y_ratio = abs(active_wrist.y - active_elbow.y) / torso_height
        effective_elbow_angle = 90.0 + np.degrees(np.arctan2(active_wrist.y - active_elbow.y, max(shoulder_width * 0.75, 0.15)))

        # C. Signed Transverse Rotation Angle (-65° Internal across belly <-> 0° Neutral <-> +65° External outward)
        signed_dx_outward = (active_wrist.x - active_elbow.x) * arm_side_sign
        max_forearm_reach = max(shoulder_width * 0.85, 0.18)
        clamped_ratio = np.clip(signed_dx_outward / max_forearm_reach, -1.2, 1.2)
        signed_rotation_deg = float(np.degrees(np.arcsin(np.clip(clamped_ratio / 1.2, -1.0, 1.0))) * 1.15)

        # Directional sweep relative to the locked rotation mode (prevents opposite-direction swing from counting!)
        if self.rep_started:
            if self.rotation_mode == "EXTERNAL":
                current_sweep_deg = max(0.0, signed_rotation_deg - self.start_rotation_deg)
            else:
                current_sweep_deg = max(0.0, self.start_rotation_deg - signed_rotation_deg)
        else:
            current_sweep_deg = abs(signed_rotation_deg - self.start_rotation_deg)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1 & 2: Front View, Elbow Pinned at 90°)
        # ---------------------------------------------------------
        is_90_bend_valid = wrist_elbow_y_ratio <= SETUP_MAX_WRIST_ELBOW_Y_DIFF
        is_setup_ready = is_facing_front and is_elbow_tucked and is_90_bend_valid

        if is_setup_ready:
            self.ready_frames += 1
            # ONLY update baseline anchor during WAITING or when holding completely still (< 5° delta) in NEUTRAL
            # so slow controlled rotations never drag start_rotation_deg along with the moving hand!
            if not self.is_ready:
                self.start_rotation_deg = signed_rotation_deg
                self.baseline_shoulder_width = raw_shoulder_width
                self.baseline_neck_ratio = ear_shoulder_ratio
            elif self.state == "NEUTRAL" and not self.rep_started and abs(signed_rotation_deg - self.start_rotation_deg) < 5.0:
                self.start_rotation_deg = 0.92 * self.start_rotation_deg + 0.08 * signed_rotation_deg
                self.baseline_shoulder_width = 0.92 * self.baseline_shoulder_width + 0.08 * raw_shoulder_width
                self.baseline_neck_ratio = 0.92 * self.baseline_neck_ratio + 0.08 * ear_shoulder_ratio

            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            elif not is_elbow_tucked:
                feedback_msg = "Pin your elbow tightly against your ribs (tuck towel)"
            else:
                feedback_msg = "Bend your elbow to a 90° angle (forearm level with elbow)"
            return {
                "exercise": "Shoulder Int/Ext Rotation", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(effective_elbow_angle, 1),
                "rotation_angle": round(abs(signed_rotation_deg), 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL EVALUATION
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["NEUTRAL", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[SHOULDER ROTATION FAULT] AI ST-GCN flagged posture/momentum fault: prob={bad_posture_prob:.2f}",
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
        if self.rep_started and self.state in ["ROTATING_OUT", "EXTERNAL_PEAK", "ROTATING_IN"]:
            if current_sweep_deg > self.max_rotation:
                self.max_rotation = current_sweep_deg
                self.peak_rotation_deg = signed_rotation_deg

            # A. Strict Front View Check
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[SHOULDER ROTATION FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # B. Step 2 & 3: Elbow Pinned Against Towel/Ribs (Catches chicken-wing abduction)
            if elbow_drift_ratio > MAX_ALLOWED_ELBOW_DRIFT or upper_arm_vertical_drop < (SETUP_MIN_UPPER_ARM_DROP - 0.10):
                self.elbow_drift_frames += 1
                if self.elbow_drift_frames >= DEBOUNCE_ELBOW_DRIFT:
                    if self.elbow_pinned:
                        print(
                            f"[SHOULDER ROTATION FAULT] Elbow flared away from ribs: drift_ratio={elbow_drift_ratio:.2f} "
                            f"(max={MAX_ALLOWED_ELBOW_DRIFT})",
                            flush=True,
                        )
                    self.elbow_pinned = False
            else:
                self.elbow_drift_frames = max(0, self.elbow_drift_frames - 1)

            # C. Step 2: Strict 90° Elbow Bend Maintained (Catches dropping hand down or curling up)
            if wrist_elbow_y_ratio > MAX_WRIST_ELBOW_Y_DRIFT:
                self.arm_angle_frames += 1
                if self.arm_angle_frames >= DEBOUNCE_ARM_STRAIGHTEN:
                    if self.elbow_angle_maintained:
                        print(
                            f"[SHOULDER ROTATION FAULT] Lost 90° elbow angle: wrist_y_drift={wrist_elbow_y_ratio:.2f} "
                            f"(effective_angle={effective_elbow_angle:.1f}°)",
                            flush=True,
                        )
                    self.elbow_angle_maintained = False
            else:
                self.arm_angle_frames = max(0, self.arm_angle_frames - 1)

            # D. Torso Upright & No Axial Spine Twist (Catches leaning OR rotating shoulders/spine!)
            torso_sway = abs(mid_shoulder_x - mid_hip_x) / torso_height
            shoulder_twist_shrink = (
                (self.baseline_shoulder_width - raw_shoulder_width) / self.baseline_shoulder_width
                if self.baseline_shoulder_width else 0.0
            )
            if torso_sway > MAX_TORSO_SWAY_RATIO or shoulder_twist_shrink > MAX_SHOULDER_TWIST_SHRINK:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_steady:
                        print(
                            f"[SHOULDER ROTATION FAULT] Torso leaned/twisted: sway={torso_sway:.2f}, "
                            f"twist_shrink={shoulder_twist_shrink:.2f}",
                            flush=True,
                        )
                    self.torso_steady = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # E. Shoulder Shrug Check
            if self.baseline_neck_ratio and ear_shoulder_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR):
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    if self.shoulder_depressed:
                        print("[SHOULDER ROTATION FAULT] Working shoulder shrugged upward toward ear!", flush=True)
                    self.shoulder_depressed = False
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # F. Controlled Eccentric Return Speed (Catches letting the band snap the forearm back)
            if self.prev_sweep_deg is not None and self.state in ["ROTATING_IN", "ROTATING_OUT"]:
                snap_speed = self.prev_sweep_deg - current_sweep_deg
                if snap_speed > ECCENTRIC_SNAP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_return:
                            print(
                                f"[SHOULDER ROTATION FAULT] Forearm snapped back too fast: speed={snap_speed:.1f} deg/frame",
                                flush=True,
                            )
                        self.controlled_return = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_sweep_deg = current_sweep_deg
        else:
            self.prev_sweep_deg = None

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FIXED NON-OVERLAPPING FLOW FOR BOTH EXTERNAL & INTERNAL)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_setup_ready:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "NEUTRAL"
                    self.unlock_frames = 0
                    self.start_rotation_deg = signed_rotation_deg
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep elbow pinned at 90° and rotate forearm smoothly."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Shoulder Int/Ext Rotation", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(effective_elbow_angle, 1),
                "rotation_angle": round(current_sweep_deg, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "NEUTRAL"
            self.start_rotation_deg = signed_rotation_deg
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep elbow pinned to ribs: rotate forearm outward (or inward across stomach)!"

        # Initiate Rotation (ONLY from NEUTRAL so ROTATING_IN never ping-pongs!)
        elif self.state == "NEUTRAL" and abs(signed_rotation_deg - self.start_rotation_deg) >= ROTATION_START_DELTA_DEG:
            self.state = "ROTATING_OUT"
            self.rep_started = True
            self.locked_working_arm = "LEFT" if use_left_arm else "RIGHT"
            self.rotation_mode = "EXTERNAL" if (signed_rotation_deg > self.start_rotation_deg) else "INTERNAL"
            self.max_rotation = abs(signed_rotation_deg - self.start_rotation_deg)
            self.peak_rotation_deg = signed_rotation_deg
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                if self.rotation_mode == "EXTERNAL":
                    self.current_feedback = "External Rotation: Sweep forearm outward like an opening door..."
                else:
                    self.current_feedback = "Internal Rotation: Sweep forearm inward across your stomach..."

        # Full Peak Contraction Reached (Sweep >= 35° in the locked direction)
        elif self.state == "ROTATING_OUT" and current_sweep_deg >= FULL_ROTATION_MIN_SWEEP_DEG:
            self.external_reached = True
            self.state = "EXTERNAL_PEAK"
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Peak contraction! Pause 1s, then slowly return to starting position."

        # Premature Reversal (Half Rep Branch: Returning before full rotational sweep)
        elif self.state == "ROTATING_OUT" and not self.external_reached and (self.max_rotation - current_sweep_deg) >= RETURN_REVERSAL_DELTA_DEG:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                if self.max_rotation >= HALF_ROTATION_MIN_SWEEP_DEG:
                    self.was_half_rep = True
                self.state = "ROTATING_IN"
                self.current_feedback = "Incomplete rotation range! Returning early..."
                print(
                    f"[SHOULDER ROTATION FAULT] Early reversal before peak sweep: "
                    f"max_sweep={self.max_rotation:.1f}° (required>={FULL_ROTATION_MIN_SWEEP_DEG}°)",
                    flush=True,
                )

        # Controlled Return from EXTERNAL_PEAK ONLY
        elif self.state == "EXTERNAL_PEAK" and (self.max_rotation - current_sweep_deg) >= RETURN_REVERSAL_DELTA_DEG:
            self.state = "ROTATING_IN"
            if self.post_rep_timer == 0:
                self.current_feedback = "Slowly return forearm back to starting position..."

        if self.state == "ROTATING_OUT" and not ((self.max_rotation - current_sweep_deg) >= RETURN_REVERSAL_DELTA_DEG):
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (INCLUDES ROTATING_OUT TO PREVENT STUCK REPS!)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ROTATING_IN", "EXTERNAL_PEAK", "ROTATING_OUT"] and current_sweep_deg <= REP_COMPLETE_RETURN_DEG:
            # Ignore tiny setup tremors (< 18° sweep)
            if self.max_rotation < MIN_VALID_REP_SWEEP_DEG and self.is_front_view and self.elbow_pinned:
                self.start_rotation_deg = signed_rotation_deg
                self._reset_rep_tracking()
                self.state = "NEUTRAL"
                return {
                    "exercise": "Shoulder Int/Ext Rotation", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(effective_elbow_angle, 1),
                    "rotation_angle": round(current_sweep_deg, 1)
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Pin elbow to reset."
                is_bad_rep = True
            elif not self.elbow_pinned:
                self.current_feedback = "Bad rep: Elbow flared away from ribs! Keep elbow pressed against the towel."
                is_bad_rep = True
            elif not self.elbow_angle_maintained:
                self.current_feedback = "Bad rep: Maintain a strict 90° elbow bend (keep forearm at elbow height)!"
                is_bad_rep = True
            elif not self.torso_steady:
                self.current_feedback = "Bad rep: Keep your torso still! Rotate from the shoulder, not your spine."
                is_bad_rep = True
            elif not self.shoulder_depressed:
                self.current_feedback = "Bad rep: Keep your shoulder blade down! Do not shrug toward your ear."
                is_bad_rep = True
            elif not self.controlled_return:
                self.current_feedback = "Bad rep: Control the return! Do not let the band snap your arm back."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected torso twisting or momentum cheat."
                is_bad_rep = True
            elif not self.external_reached:
                if self.max_rotation >= HALF_ROTATION_MIN_SWEEP_DEG or self.was_half_rep:
                    self.current_feedback = "Bad rep: Half rep! Rotate your forearm through a full range of motion."
                else:
                    self.current_feedback = "Bad rep: Rotation was too shallow! Sweep forearm fully."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SHOULDER ROTATION REP #{rep_num} ({self.rotation_mode}) -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_sweep={self.max_rotation:.1f}°, "
                    f"elbow_angle={effective_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = f"Clean {self.rotation_mode.title()} Shoulder Rotation! Great rotator cuff isolation."
                print(
                    f"[SHOULDER ROTATION REP #{rep_num} ({self.rotation_mode}) -> GOOD REP] "
                    f"max_sweep={self.max_rotation:.1f}°, elbow_angle={effective_elbow_angle:.1f}°",
                    flush=True,
                )
                self.start_rotation_deg = signed_rotation_deg
                self.state = "NEUTRAL"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = 100 if self.state == "NEUTRAL" else max(0, min(100, int((current_sweep_deg / FULL_ROTATION_MIN_SWEEP_DEG) * 100)))
        if not self.is_front_view: form_score -= 35
        if not self.elbow_pinned: form_score -= 35
        if not self.elbow_angle_maintained: form_score -= 25
        if not self.torso_steady: form_score -= 25
        if not self.shoulder_depressed: form_score -= 20
        if not self.controlled_return: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.external_reached and self.state == "ROTATING_IN": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Int/Ext Rotation",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(effective_elbow_angle, 1),
            "rotation_angle": round(current_sweep_deg, 1)
        }