import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (TRICEP DIPS - STRICT SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1-3: Hands on Bench Next to Hips, Arms Locked, Side View)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.30 # Strict side profile (16:9 webcam calibrated)
SETUP_MAX_HIP_SPAN_RATIO = 0.24      # Hips aligned sideways to camera
SETUP_ELBOW_MIN_ANGLE = 144.0        # Step 1 & 6: Arms locked out supporting body on bench
SETUP_TORSO_VERTICAL_MAX = 28.0      # Step 3: Torso upright close to edge of bench
SETUP_MAX_HIP_WRIST_Y_RATIO = 0.55   # Step 1 & 2: Hands flat on bench edge next to hips (normalized by torso_len)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held at top lockout to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum upper-body keypoint visibility

# 2. Kinematic & Form Thresholds (Step 3, 4, 5: Spine Close to Bench, Shoulders Down, Elbows Pointed Back)
MAX_TORSO_TILT_ACTIVE = 34.0         # Step 3: Keep torso upright (do not fold forward)
MAX_HIP_BENCH_DRIFT_RATIO = 0.62     # Step 3: Keep hips close to bench edge (loosened from 0.45 to prevent false flags)
SHRUG_NECK_RATIO_FACTOR = 0.58       # Step 3: Roll shoulders back and down (catches true shoulder dumping into ears)
MIN_ELBOW_BACKWARD_RATIO = -0.04     # Step 4: Active elbow must point backward behind shoulder/wrist line
MIN_BODY_DIP_DROP_RATIO = 0.07       # Step 4: Both hips & shoulders must lower toward floor
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Step 4: Slowly bend elbows under control (deg/frame)
EARLY_ASCENT_DELTA = 8.0             # Early upward push before reaching ~90° parallel upper arms

# 3. Dip Depth & Range of Motion (Step 4, 5, 6: Upper Arms Parallel ~90° -> Full Lockout)
DESCENDING_START_ELBOW_ANGLE = 136.0 # Bending elbows below 136° initiates DESCENDING
MIN_VALID_REP_FLEXION_ANGLE = 128.0  # Ignore minor seated weight shifts (> 128°)
HALF_DIP_MAX_ELBOW_ANGLE = 120.0     # Stopping between 99° and 120° flags Half Rep (Bad Rep)
FULL_DEPTH_MAX_ELBOW_ANGLE = 98.0    # Step 4: Upper arms parallel to floor (<= 98° in 2D side projection)
ASCENDING_START_ELBOW_ANGLE = 105.0  # Step 5: Pressing up from BOTTOM
REP_COMPLETE_ELBOW_ANGLE = 144.0     # Step 6: Full tricep lockout at top

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_SHOULDER_DUMP = 5
DEBOUNCE_TORSO = 4
DEBOUNCE_HIP_DRIFT = 5               # Frames before flagging hips drifting far away from bench
DEBOUNCE_ELBOW_DIR = 4               # Frames before flagging elbows collapsing forward instead of pointing back
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class TricepDipsTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.was_half_rep = False
        self.min_elbow_angle = 180.0
        self.max_shoulder_drop_ratio = 0.0
        self.max_hip_drop_ratio = 0.0

        # Posture Tripwires (STRICT SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.torso_upright = True          # Step 3: Keep torso upright
        self.spine_close_to_bench = True   # Step 3: Keep spine/hips close to edge of bench
        self.shoulders_packed = True       # Step 3: Roll shoulders back and down (no shrugging)
        self.elbows_pointed_back = True    # Step 4: Elbows hinge backward behind shoulders
        self.controlled_descent = True     # Step 4: Lower slowly under control
        self.ml_posture_maintained = True

        self.baseline_neck_ratio = None
        self.baseline_shoulder_y = None
        self.baseline_hip_y = None
        self.baseline_wrist_x = None

        # Debounce counters
        self.profile_frames = 0
        self.shoulder_dump_frames = 0
        self.torso_frames = 0
        self.hip_drift_frames = 0
        self.flare_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Sit on bench edge sideways to camera: hands next to hips, arms locked"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_tricep_dips.tflite")
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
        self.depth_reached = False
        self.was_half_rep = False
        self.min_elbow_angle = 180.0
        self.max_shoulder_drop_ratio = 0.0
        self.max_hip_drop_ratio = 0.0
        self.prev_elbow_angle = None

        self.is_side_view = True
        self.torso_upright = True
        self.spine_close_to_bench = True
        self.shoulders_packed = True
        self.elbows_pointed_back = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.shoulder_dump_frames = 0
        self.torso_frames = 0
        self.hip_drift_frames = 0
        self.flare_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Tricep Dips", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Show full side profile on bench",
                "elbow_angle": 0.0, "depth_reached": False
            }

        use_right = (r_vis >= l_vis)
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_elbow = r_elbow if use_right else l_elbow
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee

        # 1. Sagittal Elbow & Torso Angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        torso_len = max(np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y), 0.35)
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = max(abs(active_shoulder.y - active_hip.y), 0.10)
        torso_tilt = np.degrees(np.arctan2(dx_torso, dy_torso))

        # 2. Strict Side Profile Verification (16:9 Webcam Calibrated)
        shoulder_aspect = abs(l_shoulder.x - r_shoulder.x) / torso_len
        hip_aspect = abs(l_hip.x - r_hip.x) / torso_len
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        # 3. Step 1 & 2: Normalized Bench-Dip Base Verification (Hands on bench next to hips, legs forward)
        hands_next_to_hips = (abs(active_wrist.y - active_hip.y) / torso_len) <= SETUP_MAX_HIP_WRIST_Y_RATIO
        if active_knee.visibility > 0.40:
            is_bench_hover_posture = (
                (abs(active_knee.x - active_hip.x) > (0.20 * torso_len))
                and (active_shoulder.y < active_hip.y)
            )
        else:
            is_bench_hover_posture = (active_shoulder.y < active_hip.y)

        # 4. Step 3 & 4: Packed Shoulders, Spine-to-Bench Distance & True Sagittal Backward Elbow Point
        ear_shoulder_dist = abs(active_ear.y - active_shoulder.y)
        neck_ratio = ear_shoulder_dist / torso_len

        bench_anchor_x = self.baseline_wrist_x if self.baseline_wrist_x is not None else active_wrist.x
        hip_bench_drift_ratio = abs(active_hip.x - bench_anchor_x) / torso_len

        # True Sagittal Elbow Direction Check:
        # In side view, the elbow must bend BACKWARD (opposite the direction of the nose/knees)
        facing_right = (active_knee.x > active_hip.x) if active_knee.visibility > 0.40 else (nose.x > active_ear.x)
        if facing_right:
            # Facing right (+X): Behind the shoulder is toward -X
            elbow_backward_ratio = (active_shoulder.x - active_elbow.x) / torso_len
        else:
            # Facing left (-X): Behind the shoulder is toward +X
            elbow_backward_ratio = (active_elbow.x - active_shoulder.x) / torso_len

        is_elbow_pointing_back = elbow_backward_ratio >= MIN_ELBOW_BACKWARD_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Side View, Hands on Bench Next to Hips, Arms Locked)
        # ---------------------------------------------------------
        is_setup_ready = (
            is_side
            and is_bench_hover_posture
            and hands_next_to_hips
            and (elbow_angle >= SETUP_ELBOW_MIN_ANGLE)
            and (torso_tilt <= SETUP_TORSO_VERTICAL_MAX)
        )

        if is_setup_ready:
            self.ready_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
                self.baseline_shoulder_y = active_shoulder.y
                self.baseline_hip_y = active_hip.y
                self.baseline_wrist_x = active_wrist.x
            elif not self.rep_started and elbow_angle >= 150.0:
                self.baseline_neck_ratio = 0.92 * self.baseline_neck_ratio + 0.08 * neck_ratio
                self.baseline_shoulder_y = 0.92 * self.baseline_shoulder_y + 0.08 * active_shoulder.y
                self.baseline_hip_y = 0.92 * self.baseline_hip_y + 0.08 * active_hip.y
                self.baseline_wrist_x = 0.92 * self.baseline_wrist_x + 0.08 * active_wrist.x

            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            elif not is_bench_hover_posture:
                feedback_msg = "Sit on edge of bench and extend feet forward"
            elif torso_tilt > SETUP_TORSO_VERTICAL_MAX:
                feedback_msg = "Keep torso upright close to the edge of the bench"
            else:
                feedback_msg = "Place hands on bench next to hips and lock arms out to start"
            return {
                "exercise": "Tricep Dips", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["TOP", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[TRICEP DIPS FAULT] AI ST-GCN flagged shoulder/posture fault: prob={bad_posture_prob:.2f}",
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
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            if self.baseline_shoulder_y is not None and self.baseline_hip_y is not None:
                shoulder_drop_ratio = max(0.0, (active_shoulder.y - self.baseline_shoulder_y) / torso_len)
                hip_drop_ratio = max(0.0, (active_hip.y - self.baseline_hip_y) / torso_len)
                self.max_shoulder_drop_ratio = max(self.max_shoulder_drop_ratio, shoulder_drop_ratio)
                self.max_hip_drop_ratio = max(self.max_hip_drop_ratio, hip_drop_ratio)

            # A. Strict Side View Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[TRICEP DIPS FAULT] Lost side view: ratio={shoulder_aspect:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Step 3: Keep Torso Upright
            if torso_tilt > MAX_TORSO_TILT_ACTIVE:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_upright:
                        print(f"[TRICEP DIPS FAULT] Torso leaned forward: tilt={torso_tilt:.1f}°", flush=True)
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # C. Step 3: Keep Spine Close to the Edge of the Bench
            if hip_bench_drift_ratio > MAX_HIP_BENCH_DRIFT_RATIO:
                self.hip_drift_frames += 1
                if self.hip_drift_frames >= DEBOUNCE_HIP_DRIFT:
                    if self.spine_close_to_bench:
                        print(
                            f"[TRICEP DIPS FAULT] Hips drifted too far forward from bench: ratio={hip_bench_drift_ratio:.2f}",
                            flush=True,
                        )
                    self.spine_close_to_bench = False
            else:
                self.hip_drift_frames = max(0, self.hip_drift_frames - 1)

            # D. Step 3: Roll Shoulders Back and Down (Anti-Shrug / Anti-Dump)
            if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR):
                self.shoulder_dump_frames += 1
                if self.shoulder_dump_frames >= DEBOUNCE_SHOULDER_DUMP:
                    if self.shoulders_packed:
                        print("[TRICEP DIPS FAULT] Shoulders shrugged/dumped into ears!", flush=True)
                    self.shoulders_packed = False
            else:
                self.shoulder_dump_frames = max(0, self.shoulder_dump_frames - 1)

            # E. Step 4: Keep Elbows Pointing Straight Backward (Sagittal Direction Check)
            if elbow_angle < 125.0 and not is_elbow_pointing_back:
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_ELBOW_DIR:
                    if self.elbows_pointed_back:
                        print(
                            f"[TRICEP DIPS FAULT] Elbows did not point backward: back_ratio={elbow_backward_ratio:.2f}",
                            flush=True,
                        )
                    self.elbows_pointed_back = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FIXED TOP -> DESCENDING -> BOTTOM -> ASCENDING FLOW)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_setup_ready:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "TOP"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Roll shoulders down and dip to 90°."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Tricep Dips", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep spine close to bench and bend elbows to 90°!"

        # Step 4: Initiate Lowering (ONLY from TOP so ASCENDING never ping-pongs!)
        elif self.state == "TOP" and elbow_angle <= DESCENDING_START_ELBOW_ANGLE:
            self.state = "DESCENDING"
            self.rep_started = True
            self.min_elbow_angle = elbow_angle
            self.max_shoulder_drop_ratio = 0.0
            self.max_hip_drop_ratio = 0.0
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower hips until upper arms are parallel to the floor (~90°)..."

        # Step 4 Complete: Upper arms parallel to floor (elbow <= 98° AND both hips & shoulders lowered)
        elif (
            self.state == "DESCENDING"
            and elbow_angle <= FULL_DEPTH_MAX_ELBOW_ANGLE
            and self.max_shoulder_drop_ratio >= MIN_BODY_DIP_DROP_RATIO
            and self.max_hip_drop_ratio >= MIN_BODY_DIP_DROP_RATIO
        ):
            self.depth_reached = True
            self.half_rep_frames = 0
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "90° parallel depth reached! Drive firmly through your palms."

        # Step 5: Drive Upward from BOTTOM
        elif self.state == "BOTTOM" and elbow_angle > ASCENDING_START_ELBOW_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Contract triceps and lock out completely at the top!"

        # Premature Reversal (Half Rep Branch: Pushing up before reaching 90° parallel depth)
        elif self.state == "DESCENDING" and not self.depth_reached and (elbow_angle - self.min_elbow_angle) > EARLY_ASCENT_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                if self.min_elbow_angle <= HALF_DIP_MAX_ELBOW_ANGLE:
                    self.was_half_rep = True
                self.state = "ASCENDING"
                self.current_feedback = "Incomplete depth! Pushing up early..."
                print(
                    f"[TRICEP DIPS FAULT] Early reversal before 90° depth: min_elbow={self.min_elbow_angle:.1f}° "
                    f"(required<={FULL_DEPTH_MAX_ELBOW_ANGLE}°)",
                    flush=True,
                )
        else:
            if self.state == "DESCENDING":
                self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Step 4: Eccentric drop speed detector
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[TRICEP DIPS FAULT] Dropped too fast: speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            # Ignore minor seated weight shifts (> 128°)
            if self.min_elbow_angle > MIN_VALID_REP_FLEXION_ANGLE and self.is_side_view and self.shoulders_packed:
                self._reset_rep_tracking()
                self.state = "TOP"
                return {
                    "exercise": "Tricep Dips", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(elbow_angle, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to the camera! Lock out on bench to reset."
                is_bad_rep = True
            elif not self.shoulders_packed:
                self.current_feedback = "Bad rep: Roll your shoulders back and down! Do not shrug into your ears."
                is_bad_rep = True
            elif not self.spine_close_to_bench:
                self.current_feedback = "Bad rep: Keep your spine and hips close to the edge of the bench!"
                is_bad_rep = True
            elif not self.elbows_pointed_back:
                self.current_feedback = "Bad rep: Keep your elbows pointing straight backward! Do not flare out."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Keep your torso upright! Do not lean excessively forward."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower your body slowly under control! Do not drop."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected shoulder instability or posture breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                if self.min_elbow_angle <= FULL_DEPTH_MAX_ELBOW_ANGLE and (
                    self.max_hip_drop_ratio < MIN_BODY_DIP_DROP_RATIO
                    or self.max_shoulder_drop_ratio < MIN_BODY_DIP_DROP_RATIO
                ):
                    self.current_feedback = "Bad rep: Lower your hips and torso straight down below the bench!"
                elif self.min_elbow_angle <= HALF_DIP_MAX_ELBOW_ANGLE or self.was_half_rep:
                    self.current_feedback = "Bad rep: Half rep! Lower until your upper arms are parallel to the floor (90°)."
                else:
                    self.current_feedback = "Bad rep: Dip was too shallow! Bend elbows down to 90 degrees."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[TRICEP DIPS REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow={self.min_elbow_angle:.1f}°, "
                    f"shoulder_drop={self.max_shoulder_drop_ratio:.2f}, hip_drop={self.max_hip_drop_ratio:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Tricep Dip! Great 90° depth, packed shoulders, and full lockout."
                print(
                    f"[TRICEP DIPS REP #{rep_num} -> GOOD REP] "
                    f"min_elbow={self.min_elbow_angle:.1f}°, shoulder_drop={self.max_shoulder_drop_ratio:.2f}, "
                    f"hip_drop={self.max_hip_drop_ratio:.2f}",
                    flush=True,
                )
                self.state = "TOP"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4))) if self.state != "TOP" else 100
        if not self.is_side_view: form_score -= 35
        if not self.shoulders_packed: form_score -= 30
        if not self.spine_close_to_bench: form_score -= 25
        if not self.elbows_pointed_back: form_score -= 25
        if not self.torso_upright: form_score -= 30
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Tricep Dips",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "depth_reached": self.depth_reached
        }