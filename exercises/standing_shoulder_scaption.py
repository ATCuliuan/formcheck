import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SHOULDER SCAPTION - STRICT FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1-3: Stand Tall Facing Camera, Arms at Sides, Soft Elbows)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.24 # Enforces front-facing view
SETUP_MAX_ELEVATION_ANGLE = 35.0     # Step 1: Dumbbells resting at outer thighs/sides
SETUP_MIN_ELBOW_ANGLE = 120.0        # Step 3: Soft bend in elbows (calibrated for 2D scaption plane)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held at sides to clear LOCKED_FAULT
MIN_JOINT_VISIBILITY = 0.35          # Every core upper-body joint must be visible (prevents 1-arm off-screen bug)

# 2. Scaption Plane & Kinematic Thresholds (Step 2, 3, 4, 5)
MIN_ELBOW_STRAIGHT_ANGLE = 52.0      # Step 3: Loosened from 112.0° so 2D forward foreshortening (55°-83°) passes cleanly
MIN_SCAPTION_WRIST_SPAN_RATIO = 0.90 # Step 3: Loosened from 1.10 so span_ratio 0.97-1.07 passes while rejecting narrow cross-body raises
MAX_SCAPTION_WRIST_SPAN_RATIO = 2.18 # Step 3: 30-45° scaption Y-plane (rejects wide 90° T-pose lateral raise)
MAX_WRIST_HORIZONTAL_DRIFT_RATIO = 0.30 # NEW: Enforces vertical wrist path (flags horizontal sweeping / jumping jacks)
THUMB_UP_TOLERANCE_MARGIN = 0.06     # Step 2: Thumbs pointing up (thumb.y <= pinky.y + margin when clearly visible)
SHRUG_NECK_RATIO_FACTOR = 0.62       # Step 4: Keep shoulders packed down and relaxed (no shrugging)
MAX_TORSO_SWAY_RATIO = 0.14          # Step 1: Core engaged, no leaning/swaying
MAX_ASYMMETRY_DELTA = 35.0           # Step 4: Loosened from 24.0 so natural L/R difference doesn't false-flag
SINGLE_ARM_TRIGGER_DEG = 52.0        # Loosened from 42.0: still catches lifting only 1 arm while other stays < 33°
MAX_OVER_RAISE_ANGLE = 148.0         # Step 4: Loosened from 112.0° so 118°-144° peak elevation passes without false over-raise flags
OVER_RAISE_WRIST_MARGIN = 0.24       # Loosened from 0.13 so hands at high Y-plane shoulder height don't false-flag
ECCENTRIC_DROP_SPEED_MAX = 25.0      # Step 5: Lower slowly over a 2-3 second count (deg/frame)
EARLY_REVERSAL_DELTA = 9.0           # Early downward drop before reaching 90° shoulder height

# 3. Scaption Target Height & Range of Motion (Step 4 & 5: 90° Shoulder Height in Y-Plane)
LIFTING_START_ANGLE = 33.0           # Raising either arm past 33° initiates LIFTING
MIN_VALID_REP_PEAK_ANGLE = 40.0      # Ignore minor resting arm twitches (< 40°)
HALF_SCAPTION_MIN_ANGLE = 50.0       # Raising past 50° without reaching shoulder height flags Half Rep (Bad Rep)
FULL_SCAPTION_MIN_ANGLE = 66.0       # Step 4: Loosened slightly from 70.0° so 67°+ hits PEAK smoothly
LOWERING_START_ANGLE = 62.0          # Step 5: Lowering dumbbells from PEAK
REP_COMPLETE_ANGLE = 35.0            # Step 5: Both dumbbells returned to outer thighs completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_THUMB = 5
DEBOUNCE_PLANE = 7                   # Increased from 5 to 7 frames
DEBOUNCE_ELBOW = 8                   # Increased from 5 to 8 frames to ignore brief 2D foreshortening dips
DEBOUNCE_SHRUG = 5
DEBOUNCE_TORSO = 5
DEBOUNCE_ASYM = 8                    # Loosened from 4 to 8 frames to ignore brief left/right lag
DEBOUNCE_OVER_RAISE = 6              # Increased from 4 to 6 frames
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6
DEBOUNCE_WRIST_PATH = 3              # NEW: Fast response if wrists deviate from vertical path

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.78

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingShoulderScaptionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.rep_started = False
        self.target_height_reached = False # Step 4: 90° shoulder height reached
        self.was_half_rep = False
        self.max_elevation = 0.0           # Tracks highest bilateral (min_elevation) peak
        self.max_single_arm_elevation = 0.0# Tracks highest single-arm (max_elevation) peak

        # Core Exercise Constraints (STRICT FRONT VIEW)
        self.is_front_view = True
        self.wrists_vertical = True        # Step 2: Thumbs pointing up (semi-pronated / neutral)
        self.arms_y_shape = True           # Step 3: 30-45° wide "Y" scapular plane
        self.hands_too_close = False       # Step 3: Rejects narrow front raise
        self.shoulders_packed = True       # Step 4: Keep shoulders packed down and relaxed
        self.not_over_raised = True        # Step 4: Stop at shoulder height (90°)
        self.controlled_descent = True     # Step 5: Lower slowly over 2-3 seconds
        self.wrists_moving_vertically = True # NEW: Enforces strict vertical wrist movement path

        # Posture Tripwires
        self.arms_bent = False             # Step 3: Soft bend allowed, catches bicep curling
        self.torso_leaned = False          # Step 1: Core engaged, tall spine
        self.is_asymmetrical = False       # Catches single-arm or uneven bilateral lift
        self.ml_posture_maintained = True  # ST-GCN AI flag

        self.baseline_neck_ratio = None
        self.start_l_wrist_x = None        # NEW
        self.start_r_wrist_x = None        # NEW

        # Debounce counters
        self.front_frames = 0
        self.wrist_frames = 0
        self.y_shape_frames = 0
        self.inner_body_frames = 0
        self.shrug_frames = 0
        self.over_raise_frames = 0
        self.bent_frames = 0
        self.lean_frames = 0
        self.asym_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.wrist_path_frames = 0         # NEW
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elevation = None
        self.current_feedback = "Face camera: stand tall, both arms at sides with thumbs pointing up"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_scaption.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[SCAPTION ERROR] __init__ model_path={model_path} | error={exc}",
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
        self.target_height_reached = False
        self.was_half_rep = False
        self.max_elevation = 0.0
        self.max_single_arm_elevation = 0.0
        self.reversal_frames = 0
        self.prev_elevation = None
        self.start_l_wrist_x = None        # NEW
        self.start_r_wrist_x = None        # NEW

        self.is_front_view = True
        self.wrists_vertical = True
        self.arms_y_shape = True
        self.hands_too_close = False
        self.shoulders_packed = True
        self.not_over_raised = True
        self.controlled_descent = True
        self.wrists_moving_vertically = True # NEW
        self.arms_bent = False
        self.torso_leaned = False
        self.is_asymmetrical = False
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.wrist_frames = 0
        self.y_shape_frames = 0
        self.inner_body_frames = 0
        self.shrug_frames = 0
        self.over_raise_frames = 0
        self.bent_frames = 0
        self.lean_frames = 0
        self.asym_frames = 0
        self.drop_frames = 0
        self.wrist_path_frames = 0         # NEW
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
        l_thumb = landmarks[self.mp_pose.PoseLandmark.LEFT_THUMB.value]
        r_thumb = landmarks[self.mp_pose.PoseLandmark.RIGHT_THUMB.value]
        l_pinky = landmarks[self.mp_pose.PoseLandmark.LEFT_PINKY.value]
        r_pinky = landmarks[self.mp_pose.PoseLandmark.RIGHT_PINKY.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        vis_joints = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(vis_joints) < MIN_JOINT_VISIBILITY:
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Scaption",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and both arms must be fully visible",
                "elevation_angle": 0.0, "depth_reached": False
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

        # 2. Elevation & Elbow Angles
        l_elevation_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_elevation_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_elevation = (l_elevation_angle + r_elevation_angle) / 2.0
        min_elevation = min(l_elevation_angle, r_elevation_angle)
        max_elevation = max(l_elevation_angle, r_elevation_angle)
        arm_diff = abs(l_elevation_angle - r_elevation_angle)

        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        # 3. Step 2: Occlusion-Safe Thumbs-Up Check (Only evaluated when hand landmarks have high confidence)
        thumbs_vis = (
            l_thumb.visibility > 0.65 and r_thumb.visibility > 0.65
            and l_pinky.visibility > 0.65 and r_pinky.visibility > 0.65
        )
        if thumbs_vis:
            l_thumb_down = l_thumb.y > (l_pinky.y + THUMB_UP_TOLERANCE_MARGIN)
            r_thumb_down = r_thumb.y > (r_pinky.y + THUMB_UP_TOLERANCE_MARGIN)
            wrist_fails_vertical = l_thumb_down or r_thumb_down
        else:
            wrist_fails_vertical = False

        # 4. Step 3: Mirror-Safe Scaption Plane ("Wide Y" at 30-45° forward from side plane)
        wrist_span_x = abs(l_wrist.x - r_wrist.x)
        wrist_span_ratio = wrist_span_x / shoulder_width
        wrist_crosses_inner_body = wrist_span_ratio < MIN_SCAPTION_WRIST_SPAN_RATIO
        arm_fails_y_shape = wrist_span_ratio > MAX_SCAPTION_WRIST_SPAN_RATIO

        # 5. Step 4: Scapular Depression / Anti-Shrug Ratio
        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Stand Tall Facing Camera, Both Arms at Sides)
        # ---------------------------------------------------------
        torso_sway_ratio = abs(mid_shoulder_x - mid_hip_x) / torso_height
        is_resting_setup = (
            is_facing_front
            and (max_elevation < SETUP_MAX_ELEVATION_ANGLE)
            and (min_elbow_angle >= SETUP_MIN_ELBOW_ANGLE)
            and (torso_sway_ratio <= MAX_TORSO_SWAY_RATIO)
        )

        if is_resting_setup:
            self.standing_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            elif not self.rep_started and max_elevation < 24.0:
                self.baseline_neck_ratio = 0.92 * self.baseline_neck_ratio + 0.08 * neck_ratio

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            elif min_elbow_angle < SETUP_MIN_ELBOW_ANGLE:
                feedback_msg = "Extend both arms down at your sides with a soft bend"
            else:
                feedback_msg = "Stand tall with both dumbbells at your sides and thumbs pointing up"
            return {
                "exercise": "Shoulder Scaption",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "elevation_angle": round(min_elevation, 1), "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER (SOFTMAX + DEBOUNCE)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[SCAPTION FAULT] AI ST-GCN flagged posture/shrug fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[SCAPTION ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES (ACTIVE ACROSS LIFTING, PEAK, AND LOWERING)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LIFTING", "PEAK", "LOWERING"]:
            self.max_elevation = max(self.max_elevation, min_elevation)
            self.max_single_arm_elevation = max(self.max_single_arm_elevation, max_elevation)

            # A. Strict Front View Check
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[SCAPTION FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # B. Bilateral Symmetry & Single-Arm Raise Check (Evaluated early so 1-arm raises are caught!)
            is_single_arm_or_uneven = (
                (max_elevation >= SINGLE_ARM_TRIGGER_DEG and min_elevation < LIFTING_START_ANGLE)
                or (arm_diff > MAX_ASYMMETRY_DELTA)
            )
            if is_single_arm_or_uneven:
                self.asym_frames += 1
                if self.asym_frames >= DEBOUNCE_ASYM:
                    if not self.is_asymmetrical:
                        print(
                            f"[SCAPTION FAULT] Single-arm or uneven raise: L={l_elevation_angle:.1f}°, R={r_elevation_angle:.1f}°",
                            flush=True,
                        )
                    self.is_asymmetrical = True
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Raise BOTH arms evenly together!"
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            # NEW: Strict Vertical Wrist Path Check (Flags horizontal sweeping / jumping jacks)
            if self.start_l_wrist_x is not None and self.start_r_wrist_x is not None:
                l_wrist_horizontal_drift = abs(l_wrist.x - self.start_l_wrist_x) / shoulder_width
                r_wrist_horizontal_drift = abs(r_wrist.x - self.start_r_wrist_x) / shoulder_width
                if l_wrist_horizontal_drift > MAX_WRIST_HORIZONTAL_DRIFT_RATIO or r_wrist_horizontal_drift > MAX_WRIST_HORIZONTAL_DRIFT_RATIO:
                    self.wrist_path_frames += 1
                    if self.wrist_path_frames >= DEBOUNCE_WRIST_PATH:
                        if self.wrists_moving_vertically:
                            print(f"[SCAPTION FAULT] Wrist path not vertical (horizontal drift): L={l_wrist_horizontal_drift:.2f}, R={r_wrist_horizontal_drift:.2f}", flush=True)
                        self.wrists_moving_vertically = False
                        if self.post_rep_timer == 0:
                            self.current_feedback = "Keep wrist movement strictly vertical!"
                else:
                    self.wrist_path_frames = max(0, self.wrist_path_frames - 1)

            # C. Step 2: Thumbs-Up Hand Angle Check (Active between 50° and 130°)
            if 50.0 <= min_elevation <= 130.0:
                if wrist_fails_vertical:
                    self.wrist_frames += 1
                    if self.wrist_frames >= DEBOUNCE_THUMB:
                        if self.wrists_vertical:
                            print("[SCAPTION FAULT] Thumbs rotated down! Keep thumbs pointing up.", flush=True)
                        self.wrists_vertical = False
                else:
                    self.wrist_frames = max(0, self.wrist_frames - 1)

            # D. Step 3: Wide "Y" Scaption Path Check (Active near top when both arms are raised)
            if 55.0 <= min_elevation <= 135.0 and not self.is_asymmetrical:
                if arm_fails_y_shape:
                    self.y_shape_frames += 1
                    if self.y_shape_frames >= DEBOUNCE_PLANE:
                        if self.arms_y_shape:
                            print(f"[SCAPTION FAULT] Arms flared too wide laterally: span_ratio={wrist_span_ratio:.2f}", flush=True)
                        self.arms_y_shape = False
                else:
                    self.y_shape_frames = max(0, self.y_shape_frames - 1)

                if wrist_crosses_inner_body:
                    self.inner_body_frames += 1
                    if self.inner_body_frames >= DEBOUNCE_PLANE:
                        if not self.hands_too_close:
                            print(f"[SCAPTION FAULT] Hands too narrow in front of chest: span_ratio={wrist_span_ratio:.2f}", flush=True)
                        self.hands_too_close = True
                else:
                    self.inner_body_frames = max(0, self.inner_body_frames - 1)

            # E. Step 3: Soft Locked Elbows Check (Calibrated for 2D diagonal foreshortening)
            if min_elbow_angle < MIN_ELBOW_STRAIGHT_ANGLE:
                self.bent_frames += 1
                if self.bent_frames >= DEBOUNCE_ELBOW:
                    if not self.arms_bent:
                        print(f"[SCAPTION FAULT] Elbows bent too much: min_elbow={min_elbow_angle:.1f}°", flush=True)
                    self.arms_bent = True
            else:
                self.bent_frames = max(0, self.bent_frames - 1)

            # F. Step 4: Keep Shoulders Packed Down & Relaxed (Anti-Shrug)
            if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR):
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    if self.shoulders_packed:
                        print("[SCAPTION FAULT] Shoulders shrugged toward ears!", flush=True)
                    self.shoulders_packed = False
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # G. Step 4: Stop at Shoulder Height (90° Parallel — Do Not Over-Raise Overhead)
            wrist_way_above_shoulder = (
                (l_wrist.y < l_shoulder.y - OVER_RAISE_WRIST_MARGIN)
                or (r_wrist.y < r_shoulder.y - OVER_RAISE_WRIST_MARGIN)
            )
            if max_elevation > MAX_OVER_RAISE_ANGLE or wrist_way_above_shoulder:
                self.over_raise_frames += 1
                if self.over_raise_frames >= DEBOUNCE_OVER_RAISE:
                    if self.not_over_raised:
                        print(f"[SCAPTION FAULT] Over-raised past 90° shoulder height: max_elev={max_elevation:.1f}°", flush=True)
                    self.not_over_raised = False
            else:
                self.over_raise_frames = max(0, self.over_raise_frames - 1)

            # H. Step 1: Torso Upright & Steady Check
            if torso_sway_ratio > MAX_TORSO_SWAY_RATIO:
                self.lean_frames += 1
                if self.lean_frames >= DEBOUNCE_TORSO:
                    if not self.torso_leaned:
                        print(f"[SCAPTION FAULT] Torso swayed/leaned: sway_ratio={torso_sway_ratio:.2f}", flush=True)
                    self.torso_leaned = True
            else:
                self.lean_frames = max(0, self.lean_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (TRIGGERS ON EITHER ARM SO 1-ARM LIFT IS CAUGHT!)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Point thumbs up and raise BOTH arms into a 90° wide Y."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Shoulder Scaption",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elevation_angle": round(min_elevation, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Thumbs up: raise both arms forward & outward (30-45° Y) to shoulder height!"

        # Step 4: Initiate Scaption Raise (Triggers on max_elevation >= 33° so single-arm raises are caught!)
        elif self.state == "STANDING" and max_elevation >= LIFTING_START_ANGLE:
            self.state = "LIFTING"
            self.rep_started = True
            self.max_elevation = min_elevation
            self.max_single_arm_elevation = max_elevation
            self.start_l_wrist_x = l_wrist.x  # NEW: Lock starting wrist X positions
            self.start_r_wrist_x = r_wrist.x  # NEW: Lock starting wrist X positions
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise both arms along scapular Y-plane to shoulder height (90°)..."

        # Step 4 Complete: BOTH Arms Reach Shoulder Height (min_elevation >= 66° in 2D Y-plane) AND wrists moved vertically
        elif self.state == "LIFTING" and min_elevation >= FULL_SCAPTION_MIN_ANGLE and not self.is_asymmetrical and self.wrists_moving_vertically:
            self.target_height_reached = True
            self.state = "PEAK"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "90° shoulder height reached! Lower slowly over 2-3 seconds."

        # Premature Reversal (Half Rep OR Single-Arm Raise returning down early)
        elif self.state == "LIFTING" and not self.target_height_reached and (
            (self.max_single_arm_elevation - max_elevation) > EARLY_REVERSAL_DELTA
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_elevation >= HALF_SCAPTION_MIN_ANGLE:
                    self.was_half_rep = True
                self.state = "LOWERING"
                if self.is_asymmetrical:
                    self.current_feedback = "Uneven lift! Raise both arms together."
                elif not self.wrists_moving_vertically:
                    self.current_feedback = "Wrist movement must be strictly vertical!"
                else:
                    self.current_feedback = "Incomplete scaption height! Lowering early..."
                print(
                    f"[SCAPTION FAULT] Early reversal: min_peak={self.max_elevation:.1f}°, "
                    f"max_peak={self.max_single_arm_elevation:.1f}°",
                    flush=True,
                )

        # Step 5: Lower Under Control from PEAK ONLY
        elif self.state == "PEAK" and max_elevation < LOWERING_START_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower slowly over a 2-3 second count to your outer thighs..."

        if self.state == "LIFTING" and not ((self.max_single_arm_elevation - max_elevation) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 5: Eccentric drop speed detector
        if self.rep_started and self.state in ["LOWERING", "LIFTING"]:
            if self.prev_elevation is not None:
                drop_speed = self.prev_elevation - max_elevation
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[SCAPTION FAULT] Dropped weights too fast: speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elevation = max_elevation
        else:
            self.prev_elevation = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS & 1-ARM RAISES FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "PEAK", "LIFTING"] and max_elevation <= REP_COMPLETE_ANGLE:
            # Ignore minor resting arm twitches (< 40° on both arms)
            if self.max_single_arm_elevation < MIN_VALID_REP_PEAK_ANGLE and self.is_front_view and not self.arms_bent:
                self._reset_rep_tracking()
                self.state = "STANDING"
                return {
                    "exercise": "Shoulder Scaption",
                    "state": self.state,
                    "good_reps": self.good_reps,
                    "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps,
                    "total_reps": total_reps,
                    "form_score": 100,
                    "feedback": self.current_feedback,
                    "elevation_angle": round(min_elevation, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Rest arms at sides to reset."
                is_bad_rep = True
            elif self.is_asymmetrical:
                self.current_feedback = "Bad rep: Raise and lower BOTH arms evenly together!"
                is_bad_rep = True
            elif not self.wrists_moving_vertically:
                self.current_feedback = "Bad rep: Wrist movement must be strictly vertical! Avoid horizontal swinging."
                is_bad_rep = True
            elif not self.not_over_raised:
                self.current_feedback = "Bad rep: Stop at shoulder height (90° parallel to floor)! Do not raise overhead."
                is_bad_rep = True
            elif self.hands_too_close:
                self.current_feedback = "Bad rep: Arms too narrow! Lift outward at a 30-45° wide Y-angle."
                is_bad_rep = True
            elif not self.arms_y_shape:
                self.current_feedback = "Bad rep: Keep arms in the 30-45° scapular Y-plane (not straight out to sides)!"
                is_bad_rep = True
            elif not self.wrists_vertical:
                self.current_feedback = "Bad rep: Keep thumbs pointing up toward the ceiling!"
                is_bad_rep = True
            elif not self.shoulders_packed:
                self.current_feedback = "Bad rep: Keep your shoulders packed down and relaxed! Do not shrug."
                is_bad_rep = True
            elif self.arms_bent:
                self.current_feedback = "Bad rep: Keep arms straight with only a soft bend in the elbows!"
                is_bad_rep = True
            elif self.torso_leaned:
                self.current_feedback = "Bad rep: Engage your core and stand tall! Do not lean or sway."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower slowly over a 2-3 second count! Keep constant tension."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected shoulder shrugging or posture breakdown."
                is_bad_rep = True
            elif not self.target_height_reached:
                if self.max_elevation >= HALF_SCAPTION_MIN_ANGLE or self.was_half_rep:
                    self.current_feedback = "Bad rep: Half rep! Raise both arms all the way to shoulder height (90°)."
                else:
                    self.current_feedback = "Bad rep: Scaption raise was too shallow! Lift to shoulder height."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SCAPTION REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_peak={self.max_elevation:.1f}°, "
                    f"max_peak={self.max_single_arm_elevation:.1f}°, min_elbow={min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect 90° Shoulder Scaption! Great thumbs-up Y-plane control."
                print(
                    f"[SCAPTION REP #{rep_num} -> GOOD REP] "
                    f"max_elev={self.max_elevation:.1f}°, min_elbow={min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Form Score
        form_score = max(0, min(100, int((min_elevation / 75.0) * 100))) if self.state != "STANDING" else 100
        if not self.is_front_view: form_score -= 35
        if self.is_asymmetrical: form_score -= 40
        if not self.wrists_moving_vertically: form_score -= 45
        if not self.not_over_raised: form_score -= 25
        if self.hands_too_close: form_score -= 30
        if not self.wrists_vertical: form_score -= 25
        if not self.arms_y_shape: form_score -= 30
        if not self.shoulders_packed: form_score -= 25
        if self.torso_leaned: form_score -= 30
        if self.arms_bent: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.target_height_reached and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Scaption",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elevation_angle": round(min_elevation, 1),
            "depth_reached": self.target_height_reached
        }