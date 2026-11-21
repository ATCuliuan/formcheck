import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (BODYWEIGHT SQUAT - SIDE VIEW & HANDS ON CHEST)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Stand Tall on Sideways, Hands on Chest, Braced Core)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.25 # Strict side profile (prevents false flags at bottom of squat)
SETUP_MAX_HIP_SPAN_RATIO = 0.22      # Hips aligned sideways to camera
SETUP_KNEE_MIN_ANGLE = 155.0         # Upright standing knee extension
SETUP_HIP_MIN_ANGLE = 150.0          # Upright standing hip extension
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames standing upright
UNLOCK_RESET_FRAMES = 12             # Frames held standing tall to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 1.8       # Minimum keypoint visibility for profile

# 2. Loosened Hands-on-Chest Guard (Catches Dropping/Swinging Arms Without False Bottom Flags!)
MAX_HAND_CHEST_RATIO = 0.65          # Loosened from 0.45 so deep squat torso lean (0.35-0.55) always passes!
MAX_CHEST_ELBOW_ANGLE = 135.0        # Loosened from 115° (catches straight-arm swinging/dropping > 135°)
DEBOUNCE_HANDS_CHEST = 6             # Forgiving 6-frame debounce before flagging hands leaving chest

# 3. Kinematic & Form Thresholds (Step 2, 3, 4, 5: Proud Chest, Heels Flat, Hip Sit-Back)
MAX_TORSO_FORWARD_TILT = 52.0        # Step 2 & 4: Good morning squat pitch cutoff
MIN_SPINE_NEUTRAL_ANGLE = 115.0      # Step 2 & 4: Keep chest proud, look straight ahead (no spine rounding)
MAX_HEEL_LIFT_DELTA = 0.05           # Step 4 & 6: Drive through heels and mid-foot (no heel pop)
MAX_FOOT_ALIGN_OFFSET = 0.16         # Step 1: Allows wider-than-shoulder stance in side view without false faults
MAX_HIP_ANGLE_AT_BOTTOM = 122.0      # Step 3: Ensures hips actually hinge/sit back into the invisible chair
ECCENTRIC_DROP_SPEED_MAX = 22.0      # Dropping into the hole speed cutoff (deg/frame)
EARLY_REVERSAL_DELTA = 9.0           # Early upward push before hitting parallel

# 4. Squat Depth & Range of Motion (Step 4 & 6: Thighs Parallel -> Full Standing Glute Squeeze)
FULL_DEPTH_MAX_KNEE_ANGLE = 104.0    # Parallel or sub-parallel knee flexion
PARALLEL_HIP_KNEE_Y_MARGIN = 0.05    # Hip crease approaching knee Y level (thigh parallel to floor)
HALF_DEPTH_MAX_KNEE_ANGLE = 132.0    # Flexing past 132° but short of parallel flags Half Rep (Bad Rep)
MIN_VALID_REP_FLEXION_ANGLE = 138.0  # Ignore minor standing knee shifts (> 138°)
DESCENDING_START_KNEE_ANGLE = 144.0  # Flexing below this initiates DESCENDING
ASCENDING_START_KNEE_ANGLE = 108.0   # Rising out of BOTTOM transitions to ASCENDING
REP_COMPLETE_KNEE_ANGLE = 155.0      # Step 6: Full standing lockout
REP_COMPLETE_HIP_ANGLE = 150.0       # Step 6: Hips driven straight up with glutes squeezed at top

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4                 # Frames before flagging lost sideways view
DEBOUNCE_TORSO = 4
DEBOUNCE_SPINE = 4                   # Frames before flagging rounded back / dropped head
DEBOUNCE_HEEL = 4
DEBOUNCE_FOOT_ALIGN = 4
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6

# 6. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 7. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40        # Duration feedback message stays visible

# =====================================================================


class SquatTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_depth_reached = False
        self.hip_hinged_back = False
        self.min_knee_angle = 180.0
        self.min_hip_angle = 180.0

        # Instructor Tripwires (SAGITTAL / SIDE VIEW + HANDS ON CHEST)
        self.is_side_view = True
        self.hands_on_chest = True         # Enforces keeping hands on the chest throughout the squat
        self.torso_upright = True          # Step 2: Keep chest proud
        self.spine_neutral = True          # Step 2 & 4: Look straight ahead, do not round spine
        self.heels_flat = True             # Step 4 & 6: Weight across whole foot, heels flat
        self.feet_aligned = True           # Step 1: Balanced stance
        self.controlled_descent = True     # Step 3 & 4: Lower under control
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.hand_chest_frames = 0
        self.torso_frames = 0
        self.spine_frames = 0
        self.heel_frames = 0
        self.foot_align_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_bad_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_knee_angle = None
        self.baseline_ankle_y = None

        self.current_feedback = "Stand on sideways to camera with hands on your chest"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_squat.tflite")
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
        self.half_depth_reached = False
        self.hip_hinged_back = False
        self.min_knee_angle = 180.0
        self.min_hip_angle = 180.0
        self.reversal_frames = 0
        self.prev_knee_angle = None

        self.is_side_view = True
        self.hands_on_chest = True
        self.torso_upright = True
        self.spine_neutral = True
        self.heels_flat = True
        self.feet_aligned = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.hand_chest_frames = 0
        self.torso_frames = 0
        self.spine_frames = 0
        self.heel_frames = 0
        self.foot_align_frames = 0
        self.drop_frames = 0
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Squat",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back and stand on sideways: Show full side profile",
                "knee_angle": 0.0,
                "depth_reached": False
            }

        use_right = (r_vis >= l_vis)
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_elbow = r_elbow if use_right else l_elbow
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee
        active_ankle = r_ankle if use_right else l_ankle

        # 1. Sagittal Joint Angles
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        spine_neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        # 2. Side Profile Verification
        torso_len = max(np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y), 0.35)
        shoulder_aspect = abs(l_shoulder.x - r_shoulder.x) / torso_len
        hip_aspect = abs(l_hip.x - r_hip.x) / torso_len
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        # 3. Balanced Hands-on-Chest Verification (Safe at Deep Squat Bottom!)
        chest_anchor_x = 0.75 * active_shoulder.x + 0.25 * active_hip.x
        chest_anchor_y = 0.75 * active_shoulder.y + 0.25 * active_hip.y
        dist_wrist_to_chest = np.hypot(active_wrist.x - chest_anchor_x, active_wrist.y - chest_anchor_y) / torso_len
        dist_wrist_to_shoulder = np.hypot(active_wrist.x - active_shoulder.x, active_wrist.y - active_shoulder.y) / torso_len
        hand_chest_ratio = min(dist_wrist_to_chest, dist_wrist_to_shoulder)

        if active_wrist.visibility > 0.35:
            is_hands_on_chest = (
                (hand_chest_ratio <= MAX_HAND_CHEST_RATIO)
                and (elbow_angle <= MAX_CHEST_ELBOW_ANGLE)
                and (active_wrist.y <= active_hip.y + 0.05)
            )
        else:
            is_hands_on_chest = True

        # 4. Torso Forward Pitch & Stance Alignment
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = max(abs(active_shoulder.y - active_hip.y), 0.05)
        torso_tilt = np.degrees(np.arctan2(dx_torso, dy_torso))

        both_ankles_vis = (l_ankle.visibility > 0.50 and r_ankle.visibility > 0.50)
        foot_offset = abs(l_ankle.x - r_ankle.x) if both_ankles_vis else 0.0

        # Step 4: Thigh Parallel Check (Knee flexion <= 104° OR hip crease descends to knee level)
        is_thigh_parallel = (
            (knee_angle <= FULL_DEPTH_MAX_KNEE_ANGLE)
            or (active_hip.y >= (active_knee.y - PARALLEL_HIP_KNEE_Y_MARGIN) and knee_angle <= 114.0)
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Stand Tall on Sideways + Hands on Chest)
        # ---------------------------------------------------------
        is_standing_lockout = (knee_angle >= REP_COMPLETE_KNEE_ANGLE) and (hip_angle >= REP_COMPLETE_HIP_ANGLE)
        is_setup_ready = (
            is_side
            and is_hands_on_chest
            and (knee_angle > SETUP_KNEE_MIN_ANGLE)
            and (hip_angle > SETUP_HIP_MIN_ANGLE)
        )

        if is_setup_ready:
            self.standing_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Stand on sideways! Strict side view required"
            elif not is_hands_on_chest:
                feedback_msg = "Place and keep your hands on your chest to begin"
            else:
                feedback_msg = "Stand tall on sideways with hands on chest to begin"
            return {
                "exercise": "Squat",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "knee_angle": round(knee_angle, 1),
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
                probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[SQUAT FAULT] AI ST-GCN flagged spinal/knee mechanics fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

            if hip_angle <= MAX_HIP_ANGLE_AT_BOTTOM:
                self.hip_hinged_back = True

            # A. Side Profile Tripwire ("Stand on sideways")
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[SQUAT FAULT] Not standing sideways! shoulder_aspect={shoulder_aspect:.2f}, hip_aspect={hip_aspect:.2f}",
                            flush=True,
                        )
                    self.is_side_view = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Stand on sideways! Keep strict side profile."
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Hands-on-Chest Tripwire (Loosened so deep squats don't false-flag!)
            if not is_hands_on_chest:
                self.hand_chest_frames += 1
                if self.hand_chest_frames >= DEBOUNCE_HANDS_CHEST:
                    if self.hands_on_chest:
                        print(
                            f"[SQUAT FAULT] Hands left chest! chest_ratio={hand_chest_ratio:.2f}, elbow={elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.hands_on_chest = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Keep your hands on your chest!"
            else:
                self.hand_chest_frames = max(0, self.hand_chest_frames - 1)

            # C. Step 2 & 4: Proud Chest / Forward Torso Pitch Check
            if torso_tilt > MAX_TORSO_FORWARD_TILT:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_upright:
                        print(
                            f"[SQUAT FAULT] Chest collapsed forward: torso_tilt={torso_tilt:.1f}° "
                            f"(max={MAX_TORSO_FORWARD_TILT}°)",
                            flush=True,
                        )
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # D. Step 2 & 4: Neutral Spine / Look Straight Ahead Check
            if spine_neck_angle < MIN_SPINE_NEUTRAL_ANGLE:
                self.spine_frames += 1
                if self.spine_frames >= DEBOUNCE_SPINE:
                    if self.spine_neutral:
                        print(
                            f"[SQUAT FAULT] Spine rounded or head dropped: spine_angle={spine_neck_angle:.1f}°",
                            flush=True,
                        )
                    self.spine_neutral = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

            # E. Step 4 & 6: Heels Flat on Floor Check
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > MAX_HEEL_LIFT_DELTA:
                self.heel_frames += 1
                if self.heel_frames >= DEBOUNCE_HEEL:
                    if self.heels_flat:
                        print("[SQUAT FAULT] Heels lifted off the floor!", flush=True)
                    self.heels_flat = False
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

            # F. Step 1: Stance Alignment (Only when both ankles are clearly visible)
            if both_ankles_vis and foot_offset > MAX_FOOT_ALIGN_OFFSET:
                self.foot_align_frames += 1
                if self.foot_align_frames >= DEBOUNCE_FOOT_ALIGN:
                    if self.feet_aligned:
                        print(f"[SQUAT FAULT] Stance staggered too wide in X: offset={foot_offset:.2f}", flush=True)
                    self.feet_aligned = False
            else:
                self.foot_align_frames = max(0, self.foot_align_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_setup_ready:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep hands on chest and squat to parallel."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Squat",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep hands on chest, hinge hips back, and sink into squat!"

        # Step 3: Initiate Descent ONLY from STANDING
        elif self.state == "STANDING" and knee_angle < DESCENDING_START_KNEE_ANGLE:
            self.state = "DESCENDING"
            self.rep_started = True
            self.min_knee_angle = knee_angle
            self.min_hip_angle = hip_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0 and self.hands_on_chest and self.is_side_view:
                self.current_feedback = "Sit hips back into an invisible chair, keep hands on chest..."

        # Step 4 Complete: Thighs parallel to floor AND hips hinged back
        elif self.state == "DESCENDING" and is_thigh_parallel and self.hip_hinged_back:
            self.depth_reached = True
            self.state = "BOTTOM"
            self.reversal_frames = 0
            if self.post_rep_timer == 0 and self.hands_on_chest and self.is_side_view:
                self.current_feedback = "Parallel depth reached! Drive forcefully through your heels."

        # Early upward reversal (Half rep branch -> Routed to Bad Rep)
        elif self.state == "DESCENDING" and not self.depth_reached and (knee_angle - self.min_knee_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_knee_angle <= HALF_DEPTH_MAX_KNEE_ANGLE:
                    self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Incomplete squat: Rising before parallel..."
                print(
                    f"[SQUAT FAULT] Early reversal before parallel depth: "
                    f"min_knee={self.min_knee_angle:.1f}°, min_hip={self.min_hip_angle:.1f}°",
                    flush=True,
                )

        # Step 6: Drive Back Up from BOTTOM
        elif self.state == "BOTTOM" and knee_angle > ASCENDING_START_KNEE_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0 and self.hands_on_chest and self.is_side_view:
                self.current_feedback = "Drive hips straight up, keep hands on chest!"

        if self.state == "DESCENDING" and not ((knee_angle - self.min_knee_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop speed detector
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_knee_angle is not None:
                drop_speed = self.prev_knee_angle - knee_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[SQUAT FAULT] Dropped too fast into the hole: speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_knee_angle = knee_angle
        else:
            self.prev_knee_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and is_standing_lockout and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"]:
            # Ignore minor standing knee shifts (> 138°) ONLY if side view & hands on chest held
            if self.min_knee_angle > MIN_VALID_REP_FLEXION_ANGLE and self.is_side_view and self.hands_on_chest and self.heels_flat:
                self._reset_rep_tracking()
                self.state = "STANDING"
                return {
                    "exercise": "Squat", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "knee_angle": round(knee_angle, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand on sideways to the camera! Stand tall to reset."
                is_bad_rep = True
            elif not self.hands_on_chest:
                self.current_feedback = "Bad rep: Keep your hands on your chest! Do not drop or swing arms."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Chest fell forward! Keep torso upright and chest proud."
                is_bad_rep = True
            elif not self.spine_neutral:
                self.current_feedback = "Bad rep: Keep spine neutral and look straight ahead! Do not round your back."
                is_bad_rep = True
            elif not self.heels_flat:
                self.current_feedback = "Bad rep: Heels came off floor! Drive through your heels and mid-foot."
                is_bad_rep = True
            elif not self.feet_aligned:
                self.current_feedback = "Bad rep: Stance staggered! Align your feet evenly."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Dropped too fast! Control your descent into the squat."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor spinal mechanics or knee collapse."
                is_bad_rep = True
            elif not self.depth_reached:
                if not self.hip_hinged_back and self.min_knee_angle <= FULL_DEPTH_MAX_KNEE_ANGLE:
                    self.current_feedback = "Bad rep: Hinge your hips backward into an invisible chair!"
                elif self.min_knee_angle <= HALF_DEPTH_MAX_KNEE_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Bad rep: Half rep! Lower until your thighs are parallel to the floor."
                else:
                    self.current_feedback = "Bad rep: Squat was too shallow! Lower until thighs are parallel."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SQUAT REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_knee={self.min_knee_angle:.1f}°, min_hip={self.min_hip_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Squat! Hands stayed on chest with parallel depth and strong lockout."
                print(
                    f"[SQUAT REP #{rep_num} -> GOOD REP] "
                    f"min_knee={self.min_knee_angle:.1f}°, min_hip={self.min_hip_angle:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((160 - knee_angle) / 60 * 100))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 40
        if not self.hands_on_chest: form_score -= 40
        if not self.torso_upright: form_score -= 30
        if not self.spine_neutral: form_score -= 25
        if not self.heels_flat: form_score -= 25
        if not self.feet_aligned: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Squat",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "depth_reached": self.depth_reached
        }