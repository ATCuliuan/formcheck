import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (RENEGADE ROW - STRICT PRONE SIDE PLANK)
# =====================================================================

# 1. Setup & Starting Guard (Step 1-3: Prone High Plank on Dumbbells, Hands Under Shoulders)
SETUP_PLANK_ELBOW_MIN_ANGLE = 148.0  # Arms straight in high plank on dumbbells[cite: 20]
SETUP_PLANK_BODY_MIN_ANGLE = 150.0   # Rigid body line from head/shoulders to heels (Step 3)[cite: 20]
SETUP_KNEE_EXT_MIN_ANGLE = 148.0     # Legs locked out on toes (knees off floor)[cite: 20]
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.24 # Strict flat-shoulders side profile at setup
SETUP_MAX_HIP_SPAN_RATIO = 0.20      # Hips squared flat to floor at setup
SETUP_MAX_ARM_VERTICAL_TILT_DEG = 28.0 # Step 1: Hands stacked directly under shoulders[cite: 20]
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames holding high plank to confirm READY[cite: 20]
UNLOCK_RESET_FRAMES = 12             # Frames held in high plank to clear LOCKED_FAULT[cite: 20]
MIN_VISIBILITY_THRESHOLD = 2.4       # Minimum side-profile landmark visibility sum[cite: 20]

# 2. Kinematic & Form Thresholds (Prone Position, Anti-Twist, No Sag/Pike, Support Arm Locked)
MIN_KNEE_OFF_FLOOR_ANGLE = 142.0     # Knees bending past this indicates kneeling[cite: 20]
MIN_SUPPORT_ARM_ANGLE = 138.0        # Planted support arm must stay straight (rejects push-up/collapse)[cite: 20]
MAX_HIP_TWIST_DELTA = 0.10           # Step 4: Max pelvis vertical height difference (anti-rotation)
MAX_SHOULDER_TWIST_DELTA = 0.15      # Step 4: Max shoulder vertical height difference (anti-torso twist)
MAX_BODY_TWIST_SPAN_RATIO = 0.30     # Step 4: Max horizontal shoulder/hip opening during row before flagging body twist
MAX_CAMERA_POV_LOSS_RATIO = 0.45     # Only flags camera POV error if user completely rotates to front view
MIN_CORE_SAG_BODY_ANGLE = 140.0      # Hips sagging below the shoulder-to-ankle line[cite: 20]
MIN_CORE_PIKE_BODY_ANGLE = 135.0     # Butt piking upward above the shoulder-to-ankle line[cite: 20]
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Step 6: Lower with control (deg/frame cutoff)[cite: 20]
EARLY_REVERSAL_DELTA = 10.0          # Early downward drop before reaching rib cage[cite: 20]

# 3. Pull Depth & Rib-Cage Range of Motion (Step 4 & 6)
FULL_ROW_MAX_ELBOW_ANGLE = 95.0      # Rowing elbow flexed <= 95° at top of pull[cite: 20]
MIN_WRIST_LIFT_RATIO = 0.22          # Dumbbell wrist must lift >= 22% of arm length off floor toward ribs[cite: 20]
HALF_ROW_MAX_ELBOW_ANGLE = 122.0     # Pulling past 122° without reaching ribs flags Half Rep (Bad Rep)[cite: 20]
MIN_VALID_REP_FLEXION_ANGLE = 132.0  # Ignore minor plank weight shifts (> 132°)[cite: 20]
PULLING_START_ELBOW_ANGLE = 138.0    # Initiating pull from floor[cite: 20]
LOWERING_START_ELBOW_ANGLE = 105.0   # Lowering dumbbell from rib cage[cite: 20]
REP_COMPLETE_ELBOW_ANGLE = 148.0     # Dumbbell returned to floor in high plank lockout[cite: 20]

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_KNEE_FLOOR = 3              # Frames before flagging knees on floor[cite: 20]
DEBOUNCE_ELBOW_PLANK = 3             # Frames before flagging dropping to elbows[cite: 20]
DEBOUNCE_TWIST = 4                   # Fast 2-frame trigger for twisting torso or hips!
DEBOUNCE_SAG = 4                     # Frames before flagging hip sag[cite: 20]
DEBOUNCE_PIKE = 3                    # Frames before flagging hip pike[cite: 20]
DEBOUNCE_SUPPORT = 3                 # Frames before flagging supporting arm collapse[cite: 20]
DEBOUNCE_DROP = 2                    # Frames before flagging slamming weight down[cite: 20]
DEBOUNCE_HALF_REP = 3                # Frames before confirming premature lowering[cite: 20]
DEBOUNCE_ML_FAULT = 5                #[cite: 20]

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 #[cite: 20]

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 #[cite: 20]
POST_REP_FEEDBACK_FRAMES = 40        #[cite: 20]

# =====================================================================


class RenegadeRowTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility (half reps count as bad reps)[cite: 20]

        self.rep_started = False
        self.depth_reached = False
        self.was_half_rep = False
        self.min_row_elbow = 180.0
        self.max_wrist_lift_ratio = 0.0
        self.locked_rowing_side = None     # Locks 'LEFT' or 'RIGHT' arm during active row[cite: 20]
        self.baseline_floor_wrist_y = None

        # Posture Tripwires (STRICT PRONE SIDE PLANK VIEW)
        self.is_side_view = True
        self.is_prone_plank = True         # Enforces face-down prone high plank position
        self.hands_under_shoulders = True  # Step 1: Hands stacked directly under shoulders[cite: 20]
        self.knees_off_ground = True       # Step 1 & 3: High plank on toes, not knees[cite: 20]
        self.no_elbow_plank = True         # Step 1: High plank on hands/dumbbells, not forearms[cite: 20]
        self.no_hip_rotation = True        # Step 4: Do not twist body (catches BOTH hip and torso/shoulder twist!)
        self.core_maintained = True        # Step 3: Straight line from head to heels (no sag)[cite: 20]
        self.no_pike = True                # Step 3: No piking hips up[cite: 20]
        self.support_arm_locked = True     # Step 4: Planted arm stays straight while opposite arm rows[cite: 20]
        self.controlled_descent = True     # Step 6: Lower dumbbell with control[cite: 20]
        self.ml_posture_maintained = True  #[cite: 20]

        # Debounce counters
        self.profile_frames = 0
        self.hand_stack_frames = 0
        self.knee_floor_frames = 0
        self.elbow_plank_frames = 0
        self.twist_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.support_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.plank_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_row_elbow = None
        self.current_feedback = "Get in a prone (face-down) high plank sideways to camera"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model[cite: 20]
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_renegade_row.tflite")
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
        self.was_half_rep = False
        self.min_row_elbow = 180.0
        self.max_wrist_lift_ratio = 0.0
        self.locked_rowing_side = None
        self.reversal_frames = 0
        self.prev_row_elbow = None

        self.is_side_view = True
        self.is_prone_plank = True
        self.hands_under_shoulders = True
        self.knees_off_ground = True
        self.no_elbow_plank = True
        self.no_hip_rotation = True
        self.core_maintained = True
        self.no_pike = True
        self.support_arm_locked = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.hand_stack_frames = 0
        self.knee_floor_frames = 0
        self.elbow_plank_frames = 0
        self.twist_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.support_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Renegade Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile in prone high plank",
                "row_elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        # Primary camera-facing side for plank body line[cite: 20]
        use_right_profile = (r_vis >= l_vis)
        active_ear = r_ear if use_right_profile else l_ear
        active_shoulder = r_shoulder if use_right_profile else l_shoulder
        active_elbow = r_elbow if use_right_profile else l_elbow
        active_wrist = r_wrist if use_right_profile else l_wrist
        active_hip = r_hip if use_right_profile else l_hip
        active_knee = r_knee if use_right_profile else l_knee
        active_ankle = r_ankle if use_right_profile else l_ankle

        # Occlusion-safe bilateral arm angles[cite: 20]
        l_arm_valid = (l_elbow.visibility > 0.45 and l_wrist.visibility > 0.45)
        r_arm_valid = (r_elbow.visibility > 0.45 and r_wrist.visibility > 0.45)

        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist) if l_arm_valid else 165.0
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist) if r_arm_valid else 165.0

        # Lock onto whichever arm initiates the row so mid-rep occlusion never swaps arms[cite: 20]
        if self.rep_started and self.locked_rowing_side is not None:
            row_with_left = (self.locked_rowing_side == "LEFT")
        else:
            if l_arm_valid and r_arm_valid:
                row_with_left = (l_elbow_angle < r_elbow_angle)
            else:
                row_with_left = not use_right_profile

        if row_with_left:
            rowing_elbow_angle = l_elbow_angle
            rowing_wrist = l_wrist
            rowing_shoulder = l_shoulder
            support_elbow_angle = r_elbow_angle if r_arm_valid else 165.0
            support_wrist = r_wrist if r_arm_valid else active_wrist
            support_shoulder = r_shoulder if r_arm_valid else active_shoulder
            both_arms_visible = (l_arm_valid and r_arm_valid)
        else:
            rowing_elbow_angle = r_elbow_angle
            rowing_wrist = r_wrist
            rowing_shoulder = r_shoulder
            support_elbow_angle = l_elbow_angle if l_arm_valid else 165.0
            support_wrist = l_wrist if l_arm_valid else active_wrist
            support_shoulder = l_shoulder if l_arm_valid else active_shoulder
            both_arms_visible = (l_arm_valid and r_arm_valid)

        # Body & Knee Angles[cite: 20]
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        knee_extension_angle = calculate_angle(active_hip, active_knee, active_ankle)

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        hip_span_ratio = abs(l_hip.x - r_hip.x) / torso_len

        # Flat neutral side plank at setup (both shoulders and hips squared to the floor)
        is_flat_side_setup = (
            (shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO)
            and (hip_span_ratio < SETUP_MAX_HIP_SPAN_RATIO)
        )
        # Only flags camera POV loss if the user completely faces the camera (not just twisting during a row)
        is_side_camera_pov = shoulder_span_ratio < MAX_CAMERA_POV_LOSS_RATIO

        # =========================================================
        # STRICT PRONE (FACE-DOWN) PLANK VERIFICATION
        # =========================================================
        dx_body = abs(active_shoulder.x - active_ankle.x)
        dy_body = abs(active_shoulder.y - active_ankle.y)
        is_horizontal = dx_body > (dy_body * 1.20)

        # In a prone plank on hands/toes:
        # 1. Support wrist is on the floor well below the shoulder (larger Y = lower on screen)
        # 2. Ankles/toes are on the floor below the shoulder line
        # 3. Head/nose is facing downward toward the floor (not looking up at ceiling in a supine plank)
        hands_below_shoulders = support_wrist.y > (support_shoulder.y + 0.08)
        feet_on_floor_level = active_ankle.y > (active_shoulder.y + 0.04)
        face_pointed_down = nose.y >= (active_ear.y - 0.03)
        is_prone = is_horizontal and hands_below_shoulders and feet_on_floor_level and face_pointed_down

        # Step 1: Hands stacked directly under shoulders check[cite: 20]
        dx_support_arm = abs(support_shoulder.x - support_wrist.x)
        dy_support_arm = max(abs(support_wrist.y - support_shoulder.y), 0.05)
        support_arm_tilt_deg = np.degrees(np.arctan2(dx_support_arm, dy_support_arm))
        is_hand_under_shoulder = support_arm_tilt_deg <= SETUP_MAX_ARM_VERTICAL_TILT_DEG

        # Step 3: True Shoulder-to-Ankle Line Deviation for Sag vs Pike[cite: 20]
        expected_hip_y = (active_shoulder.y + active_ankle.y) / 2.0
        hip_vertical_deviation = active_hip.y - expected_hip_y  # Positive = sagging down, Negative = piking up[cite: 20]

        # Step 4: Dumbbell Wrist Lift Toward Rib Cage[cite: 20]
        floor_wrist_y = self.baseline_floor_wrist_y if self.baseline_floor_wrist_y is not None else max(l_wrist.y, r_wrist.y)
        arm_len = max(abs(floor_wrist_y - rowing_shoulder.y), 0.25)
        wrist_lift_ratio = max(0.0, (floor_wrist_y - rowing_wrist.y) / arm_len)

        # Ground contact guards[cite: 20]
        is_kneeling = (knee_extension_angle < MIN_KNEE_OFF_FLOOR_ANGLE) or (active_knee.y > max(active_hip.y, active_ankle.y) + 0.02)
        is_on_elbows = (active_elbow.y >= active_wrist.y - 0.02) and (rowing_elbow_angle < 125.0) and (wrist_lift_ratio < 0.10)

        # =========================================================
        # BODY TWIST DETECTION (TORSO + SHOULDER + PELVIS ROTATION)
        # =========================================================
        hip_level_offset = abs(l_hip.y - r_hip.y)
        shoulder_level_offset = abs(l_shoulder.y - r_shoulder.y)
        is_body_twisting = (
            (hip_level_offset > MAX_HIP_TWIST_DELTA)
            or (shoulder_level_offset > MAX_SHOULDER_TWIST_DELTA)
            or (shoulder_span_ratio > MAX_BODY_TWIST_SPAN_RATIO)
            or (hip_span_ratio > MAX_BODY_TWIST_SPAN_RATIO)
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Prone High Plank on Dumbbells, Side View)
        # ---------------------------------------------------------
        is_plank = (
            is_flat_side_setup
            and is_prone
            and is_hand_under_shoulder
            and (rowing_elbow_angle >= SETUP_PLANK_ELBOW_MIN_ANGLE)
            and (support_elbow_angle >= SETUP_PLANK_ELBOW_MIN_ANGLE)
            and (body_angle >= SETUP_PLANK_BODY_MIN_ANGLE)
            and (knee_extension_angle >= SETUP_KNEE_EXT_MIN_ANGLE)
            and (not is_kneeling)
            and (not is_on_elbows)
            and (not is_body_twisting)
        )

        if is_plank:
            self.plank_frames += 1
            if self.baseline_floor_wrist_y is None:
                self.baseline_floor_wrist_y = active_wrist.y
            else:
                self.baseline_floor_wrist_y = 0.90 * self.baseline_floor_wrist_y + 0.10 * active_wrist.y

            if self.plank_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.plank_frames = max(0, self.plank_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            if not is_side_camera_pov:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            elif not is_prone:
                feedback_msg = "Get down into a prone (face-down) high plank on your dumbbells"
            elif is_kneeling:
                feedback_msg = "Lift knees off the floor into a full toe plank"
            elif is_on_elbows:
                feedback_msg = "Hold a high plank on your hands/dumbbells (not on elbows)"
            elif not is_hand_under_shoulder:
                feedback_msg = "Stack your hands directly under your shoulders"
            elif is_body_twisting or not is_flat_side_setup:
                feedback_msg = "Square your shoulders and hips flat to the floor (do not twist body)"
            elif body_angle < SETUP_PLANK_BODY_MIN_ANGLE:
                feedback_msg = "Engage core and glutes: form a straight line from head to heels"
            else:
                feedback_msg = "Hold steady prone high plank with arms straight to begin"

            return {
                "exercise": "Renegade Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "row_elbow_angle": round(rowing_elbow_angle, 1),
                "body_angle": round(body_angle, 1),
                "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["PLANKING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[RENEGADE ROW FAULT] AI ST-GCN flagged plank/rotation breakdown: prob={bad_posture_prob:.2f}",
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
            self.max_wrist_lift_ratio = max(self.max_wrist_lift_ratio, wrist_lift_ratio)

            # A. Step 4: Body & Hip Twist Check (Checked FIRST so twisting in plank flags "Do not twist the body!")
            if is_body_twisting:
                self.twist_frames += 1
                if self.twist_frames >= DEBOUNCE_TWIST:
                    if self.no_hip_rotation:
                        print(
                            f"[RENEGADE ROW FAULT] Body/hips twisted during row! "
                            f"hip_y_diff={hip_level_offset:.2f}, shldr_y_diff={shoulder_level_offset:.2f}, "
                            f"shldr_span={shoulder_span_ratio:.2f}",
                            flush=True,
                        )
                    self.no_hip_rotation = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Do not twist your body! Keep shoulders and hips flat to the floor."
            else:
                self.twist_frames = max(0, self.twist_frames - 1)

            # B. Prone Plank & Camera POV Check
            if not is_side_camera_pov or not is_prone:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if not is_prone:
                        self.is_prone_plank = False
                        print("[RENEGADE ROW FAULT] Left prone high plank position!", flush=True)
                    else:
                        self.is_side_view = False
                        print(f"[RENEGADE ROW FAULT] Lost side camera view: ratio={shoulder_span_ratio:.2f}", flush=True)
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # C. Knees Touching Floor[cite: 20]
            if is_kneeling:
                self.knee_floor_frames += 1
                if self.knee_floor_frames >= DEBOUNCE_KNEE_FLOOR:
                    if self.knees_off_ground:
                        print(f"[RENEGADE ROW FAULT] Knees dropped to floor: knee_angle={knee_extension_angle:.1f}°", flush=True)
                    self.knees_off_ground = False
            else:
                self.knee_floor_frames = max(0, self.knee_floor_frames - 1)

            # D. Dropping to Forearm Plank[cite: 20]
            if is_on_elbows:
                self.elbow_plank_frames += 1
                if self.elbow_plank_frames >= DEBOUNCE_ELBOW_PLANK:
                    if self.no_elbow_plank:
                        print("[RENEGADE ROW FAULT] Dropped onto forearms/elbows!", flush=True)
                    self.no_elbow_plank = False
            else:
                self.elbow_plank_frames = max(0, self.elbow_plank_frames - 1)

            # E. Step 3: Core Sag Check (relative to shoulder-to-ankle line)[cite: 20]
            if (hip_vertical_deviation > 0.03) and (body_angle < MIN_CORE_SAG_BODY_ANGLE):
                self.sag_frames += 1
                if self.sag_frames >= DEBOUNCE_SAG:
                    if self.core_maintained:
                        print(f"[RENEGADE ROW FAULT] Core/hips sagged: body_angle={body_angle:.1f}°", flush=True)
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # F. Step 3: Hip Pike Check (relative to shoulder-to-ankle line)[cite: 20]
            if (hip_vertical_deviation < -0.04) and (body_angle < MIN_CORE_PIKE_BODY_ANGLE):
                self.pike_frames += 1
                if self.pike_frames >= DEBOUNCE_PIKE:
                    if self.no_pike:
                        print(f"[RENEGADE ROW FAULT] Hips piked upward: body_angle={body_angle:.1f}°", flush=True)
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # G. Supporting Arm Locked[cite: 20]
            is_doing_pushup_cheat = (wrist_lift_ratio < 0.10) and (rowing_elbow_angle < 115.0)
            if (both_arms_visible and support_elbow_angle < MIN_SUPPORT_ARM_ANGLE) or is_doing_pushup_cheat:
                self.support_frames += 1
                if self.support_frames >= DEBOUNCE_SUPPORT:
                    if self.support_arm_locked:
                        print(
                            f"[RENEGADE ROW FAULT] Supporting arm bent or push-up performed instead of row! "
                            f"support_elbow={support_elbow_angle:.1f}°, wrist_lift={wrist_lift_ratio:.2f}",
                            flush=True,
                        )
                    self.support_arm_locked = False
            else:
                self.support_frames = max(0, self.support_frames - 1)

            self.min_row_elbow = min(self.min_row_elbow, rowing_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_plank:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "PLANKING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Row dumbbell to rib cage without twisting your body."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Renegade Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "row_elbow_angle": round(rowing_elbow_angle, 1),
                "body_angle": round(body_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "PLANKING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Brace core and pull dumbbell up toward your rib cage (do not twist body)!"

        # Step 4: Initiate Row Pull (ONLY from PLANKING so LOWERING never ping-pongs)
        elif self.state == "PLANKING" and rowing_elbow_angle < PULLING_START_ELBOW_ANGLE:
            self.state = "PULLING"
            self.rep_started = True
            self.locked_rowing_side = "LEFT" if row_with_left else "RIGHT"
            self.min_row_elbow = rowing_elbow_angle
            self.max_wrist_lift_ratio = wrist_lift_ratio
            self.reversal_frames = 0
            if self.post_rep_timer == 0 and self.no_hip_rotation:
                self.current_feedback = "Pull dumbbell to rib cage, keep body flat (do not twist)..."

        # Step 4 Complete: Elbow <= 95° AND dumbbell lifted off floor toward rib cage[cite: 20]
        elif self.state == "PULLING" and rowing_elbow_angle <= FULL_ROW_MAX_ELBOW_ANGLE and self.max_wrist_lift_ratio >= MIN_WRIST_LIFT_RATIO:
            self.depth_reached = True
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0 and self.no_hip_rotation:
                self.current_feedback = "Rib-cage squeeze! Slowly return dumbbell to the floor."

        # Early downward reversal (Half Rep Branch -> Bad Rep)[cite: 20]
        elif self.state == "PULLING" and not self.depth_reached and (rowing_elbow_angle - self.min_row_elbow) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.was_half_rep = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete row! Lowering weight early..."
                print(
                    f"[RENEGADE ROW FAULT] Early reversal before rib cage: min_elbow={self.min_row_elbow:.1f}°, "
                    f"wrist_lift={self.max_wrist_lift_ratio:.2f}",
                    flush=True,
                )

        # Step 6: Lower with Control[cite: 20]
        elif self.state == "TOP" and rowing_elbow_angle > LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0 and self.no_hip_rotation:
                self.current_feedback = "Lower dumbbell slowly back to the floor..."

        if self.state == "PULLING" and not ((rowing_elbow_angle - self.min_row_elbow) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 6: Eccentric drop speed detector[cite: 20]
        if self.rep_started and self.state in ["LOWERING", "PULLING"]:
            if self.prev_row_elbow is not None:
                drop_speed = rowing_elbow_angle - self.prev_row_elbow
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[RENEGADE ROW FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_row_elbow = rowing_elbow_angle
        else:
            self.prev_row_elbow = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (BODY TWIST PRIORITIZED!)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "TOP", "PULLING"] and rowing_elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            # Ignore minor weight shifts in high plank (> 132°) ONLY if no twist or plank fault occurred
            if (
                self.min_row_elbow > MIN_VALID_REP_FLEXION_ANGLE
                and self.no_hip_rotation
                and self.knees_off_ground
                and self.is_prone_plank
                and self.is_side_view
            ):
                self._reset_rep_tracking()
                self.state = "PLANKING"
                return {
                    "exercise": "Renegade Row", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "row_elbow_angle": round(rowing_elbow_angle, 1),
                    "body_angle": round(body_angle, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.no_hip_rotation:
                self.current_feedback = "Bad rep: Do not twist the body! Keep your shoulders and hips flat to the floor."
                is_bad_rep = True
            elif not self.is_prone_plank:
                self.current_feedback = "Bad rep: Stay in a prone (face-down) high plank position!"
                is_bad_rep = True
            elif not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to camera in high plank!"
                is_bad_rep = True
            elif not self.knees_off_ground:
                self.current_feedback = "Bad rep: Knees touched the floor! Stay in a full plank on toes."
                is_bad_rep = True
            elif not self.no_elbow_plank:
                self.current_feedback = "Bad rep: Do not drop to elbows! Stay in a high plank on your hands."
                is_bad_rep = True
            elif not self.support_arm_locked:
                self.current_feedback = "Bad rep: Keep supporting arm straight and row the dumbbell up to your ribs!"
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "Bad rep: Hips sagged! Brace core and squeeze glutes."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "Bad rep: Butt too high! Keep body in a straight line from head to heels."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower the dumbbell with control! Do not slam weight down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: Do not twist the body! AI flagged torso rotation."
                is_bad_rep = True
            elif self.was_half_rep or not self.depth_reached:
                if self.min_row_elbow <= HALF_ROW_MAX_ELBOW_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Pull dumbbell all the way up to your rib cage."
                else:
                    self.current_feedback = "Bad rep: Row was too shallow! Pull dumbbell up toward your rib cage."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[RENEGADE ROW REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow={self.min_row_elbow:.1f}°, "
                    f"wrist_lift={self.max_wrist_lift_ratio:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Renegade Row! Great anti-rotation core stability."
                print(
                    f"[RENEGADE ROW REP #{rep_num} -> GOOD REP] "
                    f"min_elbow={self.min_row_elbow:.1f}°, wrist_lift={self.max_wrist_lift_ratio:.2f}",
                    flush=True,
                )
                self.state = "PLANKING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring[cite: 20]
        form_score = max(0, min(100, int(100 - abs(rowing_elbow_angle - 90) * 0.4))) if self.state != "PLANKING" else 100
        if not self.no_hip_rotation: form_score -= 40
        if not self.is_prone_plank: form_score -= 40
        if not self.is_side_view: form_score -= 35
        if not self.knees_off_ground: form_score -= 40
        if not self.no_elbow_plank: form_score -= 40
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 20
        if not self.support_arm_locked: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if (self.was_half_rep or not self.depth_reached) and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Renegade Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "row_elbow_angle": round(rowing_elbow_angle, 1),
            "body_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }