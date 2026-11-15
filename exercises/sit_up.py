import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIT-UP - STRICT SAGITTAL / SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Lie Flat Supine on Mat, ~90° Bent Knees, Hands at Ears/Chest)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.34 # Side profile tolerance (prevents false rotation flags at top)
SETUP_MAX_HIP_SPAN_RATIO = 0.28      # Hips aligned sideways to camera
SETUP_MIN_SUPINE_TILT_DEG = 66.0     # Torso lying flat on mat (90° = horizontal, 0° = vertical)
SETUP_MAX_SHOULDER_HIP_Y_DIFF = 0.14 # Shoulders and hips resting on the same floor plane
SETUP_MAX_ANKLE_HIP_Y_DIFF = 0.16    # Feet planted on the same floor plane as hips
SETUP_HIP_MIN_ANGLE = 100.0          # Lying flat on floor with knees bent
SETUP_KNEE_MIN_ANGLE = 50.0          # Step 1: Knees bent at roughly 90° (50° to 125° window)
SETUP_KNEE_MAX_ANGLE = 125.0         # Rejects straight-leg sit-ups[cite: 14]
SETUP_MIN_READY_FRAMES = 6           # Frames lying down to confirm READY[cite: 14]
UNLOCK_RESET_FRAMES = 10             # Frames lying flat to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.2       # Minimum keypoint visibility for profile

# 2. Arm Placement & Floor Support Guard (Step 2: Fingertips at Ears OR Crossed Over Chest)
MAX_HAND_HEAD_CHEST_RATIO = 0.68     # Wrists must stay near ears or chest (catches throwing arms / grabbing thighs)
FLOOR_PUSH_ELBOW_HIP_MARGIN = 0.04   # Catches propping elbows down into the mat behind the torso to push up[cite: 14]
DEBOUNCE_ARM_SUPPORT = 3             # Debounce filter for hand/elbow cheat detection[cite: 14]

# 3. Kinematic & Form Thresholds (Step 1, 3, 5: Chin Tuck Allowed, Feet & Hips Flat, Knees Bent, Controlled Unroll)
MIN_NECK_NEUTRAL_ANGLE = 90.0        # Step 3: Allows slight protective chin tuck while catching violent neck yanks
MAX_FOOT_LIFT_DELTA = 0.07           # Step 1: Feet must stay planted flat on the floor[cite: 14]
MAX_HIP_LIFT_DELTA = 0.07            # Step 1: Hips/butt must stay planted on the mat (no hip bridging)
ACTIVE_KNEE_MAX_ANGLE = 135.0        # Step 1: Knees must stay bent (catches kicking legs straight out for momentum)
ECCENTRIC_DROP_SPEED_MAX = 26.0      # Step 5: Unroll spine slowly (do not slap back hard against mat)[cite: 14]
EARLY_REVERSAL_DELTA = 10.0          # Early downward reversal delta[cite: 14]

# 4. Curl-Up & Unroll Range of Motion (Step 4 & 5: Vertical Torso -> Full Unroll to Mat)
ASCENDING_START_TILT_DEG = 60.0      # Lifting head/shoulders off mat (< 60° from vertical) initiates ASCENDING
MIN_VALID_REP_TILT_DEG = 54.0        # Ignore minor head/shoulder shifts on the mat (> 54°)
HALF_SITUP_MAX_TILT_DEG = 48.0       # Crunching partway (<= 48° tilt) without sitting vertical flags Half Rep[cite: 14]
FULL_COMPRESSION_MAX_HIP_ANGLE = 82.0# Step 4: Chest brought up close to thighs (<= 82°)
FULL_VERTICAL_MAX_TILT_DEG = 30.0    # Step 4: Torso curled all the way up near vertical (<= 30° from vertical)
DESCENDING_START_TILT_DEG = 38.0     # Step 5: Unrolling spine back down from TOP
REP_COMPLETE_TILT_DEG = 65.0         # Step 5: Shoulders unrolled all the way back to mat completes rep

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_NECK = 4                    #[cite: 14]
DEBOUNCE_FEET = 4                    #[cite: 14]
DEBOUNCE_KNEES = 4                   # Frames before flagging straightening legs mid-rep
DEBOUNCE_DROP = 2                    #[cite: 14]
DEBOUNCE_HALF_REP = 3                #[cite: 14]
DEBOUNCE_ML_FAULT = 6                #[cite: 14]

# 6. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.78 #[cite: 14]

# 7. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 #[cite: 14]
POST_REP_FEEDBACK_FRAMES = 35        #[cite: 14]

# =====================================================================


class SitUpTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility[cite: 14]

        self.rep_started = False
        self.depth_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0
        self.min_torso_tilt_deg = 90.0

        # Posture Tripwires (STRICT SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.arms_overhead = True          # Step 2: Hands stay at ears or chest (no floor push or leg grab)[cite: 14]
        self.neck_neutral = True           # Step 2 & 3: No yanking neck forward[cite: 14]
        self.feet_planted = True           # Step 1: Feet & hips planted flat on floor[cite: 14]
        self.knees_stayed_bent = True     # Step 1: Knees remain bent ~90° throughout rep
        self.controlled_descent = True     # Step 5: Unroll spine segment by segment[cite: 14]
        self.ml_posture_maintained = True  #[cite: 14]

        # Debounce counters
        self.profile_frames = 0
        self.arm_support_frames = 0
        self.neck_frames = 0
        self.feet_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.lying_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_ankle_y = None
        self.baseline_hip_y = None
        self.prev_torso_tilt = None
        self.current_feedback = "Lie flat on back sideways to camera, knees bent ~90°, hands at ears or chest"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model[cite: 14]
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_sit_up.tflite")
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
        self.min_hip_angle = 180.0
        self.min_torso_tilt_deg = 90.0
        self.reversal_frames = 0
        self.prev_torso_tilt = None

        self.is_side_view = True
        self.arms_overhead = True
        self.neck_neutral = True
        self.feet_planted = True
        self.knees_stayed_bent = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.arm_support_frames = 0
        self.neck_frames = 0
        self.feet_frames = 0
        self.knee_frames = 0
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

        l_vis = sum([
            l_ear.visibility, l_shoulder.visibility, l_elbow.visibility,
            l_wrist.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility
        ])
        r_vis = sum([
            r_ear.visibility, r_shoulder.visibility, r_elbow.visibility,
            r_wrist.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility
        ])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Sit-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Full side profile not visible",
                "hip_angle": 0.0, "neck_angle": 0.0, "depth_reached": False
            }

        use_right = (r_vis >= l_vis)
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_elbow = r_elbow if use_right else l_elbow
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee
        active_ankle = r_ankle if use_right else l_ankle

        # 1. Sagittal Angles[cite: 14]
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # 2. Torso Tilt From Vertical (90° = lying flat on mat, 0° = sitting completely vertical)[cite: 14]
        torso_len = max(np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y), 0.32)
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = max(abs(active_hip.y - active_shoulder.y), 0.02)
        torso_tilt_deg = np.degrees(np.arctan2(dx_torso, dy_torso))

        # 3. Strict Side Profile Verification[cite: 14]
        shoulder_aspect = abs(l_shoulder.x - r_shoulder.x) / torso_len
        hip_aspect = abs(l_hip.x - r_hip.x) / torso_len
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        # 4. Supine (Face-Up on Floor) Verification
        # When lying supine on back: shoulders, hips, and ankles share the mat plane, knees point up (< hip.y), nose is above/level with ear
        on_floor_plane = (
            (abs(active_shoulder.y - active_hip.y) <= SETUP_MAX_SHOULDER_HIP_Y_DIFF)
            and (abs(active_ankle.y - active_hip.y) <= SETUP_MAX_ANKLE_HIP_Y_DIFF)
        )
        is_supine_face_up = nose.y <= (active_ear.y + 0.04)

        # 5. Step 2: Occlusion-Safe Hand Position Check (Fingertips behind ears OR arms crossed over chest)
        if active_wrist.visibility > 0.35:
            dist_wrist_to_ear = np.hypot(active_wrist.x - active_ear.x, active_wrist.y - active_ear.y) / torso_len
            dist_wrist_to_shoulder = np.hypot(active_wrist.x - active_shoulder.x, active_wrist.y - active_shoulder.y) / torso_len
            hand_anchor_ratio = min(dist_wrist_to_ear, dist_wrist_to_shoulder)
            # Also check that wrist is not reaching past the hip toward the knees/thighs!
            leg_dir_sign = 1.0 if (active_knee.x >= active_hip.x) else -1.0
            reaching_for_thighs = ((active_wrist.x - active_hip.x) * leg_dir_sign) > (0.15 * torso_len)
            is_hands_in_position = (hand_anchor_ratio <= MAX_HAND_HEAD_CHEST_RATIO) and (not reaching_for_thighs)
        else:
            # If far/near wrist is occluded directly behind head/chest while elbow is bent, do not false-fail
            hand_anchor_ratio = 0.0
            is_hands_in_position = True

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1 & 2: Supine on Mat, ~90° Bent Knees, Feet Flat, Hands at Ears/Chest)
        # ---------------------------------------------------------
        is_knees_bent_90 = (SETUP_KNEE_MIN_ANGLE <= knee_angle <= SETUP_KNEE_MAX_ANGLE) and (active_knee.y < active_hip.y - 0.04)
        is_lying = (
            is_side
            and is_supine_face_up
            and on_floor_plane
            and (torso_tilt_deg >= SETUP_MIN_SUPINE_TILT_DEG)
            and (hip_angle >= SETUP_HIP_MIN_ANGLE)
            and is_knees_bent_90
            and is_hands_in_position
        )

        if is_lying:
            self.lying_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
                self.baseline_hip_y = active_hip.y
            else:
                if not self.rep_started:
                    self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y
                    self.baseline_hip_y = 0.90 * self.baseline_hip_y + 0.10 * active_hip.y
            if self.lying_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.lying_frames = max(0, self.lying_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            elif not is_supine_face_up or torso_tilt_deg < SETUP_MIN_SUPINE_TILT_DEG or not on_floor_plane:
                feedback_msg = "Lie flat on your back (face-up) on the mat to begin"
            elif not is_knees_bent_90:
                feedback_msg = "Bend your knees at ~90° and plant your feet flat on the floor"
            elif not is_hands_in_position:
                feedback_msg = "Place fingertips behind ears or cross arms over your chest"
            else:
                feedback_msg = "Hold steady starting position to begin"
            return {
                "exercise": "Sit-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["BOTTOM", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[SIT-UP FAULT] AI ST-GCN flagged posture/momentum fault: prob={bad_posture_prob:.2f}",
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
        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)
            self.min_torso_tilt_deg = min(self.min_torso_tilt_deg, torso_tilt_deg)

            # A. Strict Side View Check[cite: 14]
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[SIT-UP FAULT] Lost side view: ratio={shoulder_aspect:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Step 2: Arm Cheat Detection (Catches pushing elbows into floor OR throwing hands to grab thighs)[cite: 14]
            elbow_pushing_mat = (torso_tilt_deg > 35.0) and (active_elbow.y > active_hip.y + FLOOR_PUSH_ELBOW_HIP_MARGIN)
            hands_thrown_or_grabbing_legs = not is_hands_in_position

            if elbow_pushing_mat or hands_thrown_or_grabbing_legs:
                self.arm_support_frames += 1
                if self.arm_support_frames >= DEBOUNCE_ARM_SUPPORT:
                    if self.arms_overhead:
                        print(
                            f"[SIT-UP FAULT] Arm cheat detected! hand_ratio={hand_anchor_ratio:.2f}, "
                            f"elbow_push={elbow_pushing_mat}",
                            flush=True,
                        )
                    self.arms_overhead = False
            else:
                self.arm_support_frames = max(0, self.arm_support_frames - 1)

            # C. Step 2 & 3: Neck Yank Detection[cite: 14]
            if neck_angle < MIN_NECK_NEUTRAL_ANGLE and torso_tilt_deg > 30.0:
                self.neck_frames += 1
                if self.neck_frames >= DEBOUNCE_NECK:
                    if self.neck_neutral:
                        print(f"[SIT-UP FAULT] Neck yanked forward: neck_angle={neck_angle:.1f}°", flush=True)
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            # D. Step 1: Feet & Hips Planted Flat on Floor (Catches foot lift OR hip-bridge momentum)
            foot_lifted = self.baseline_ankle_y is not None and (self.baseline_ankle_y - active_ankle.y) > MAX_FOOT_LIFT_DELTA
            hip_bridged = self.baseline_hip_y is not None and (self.baseline_hip_y - active_hip.y) > MAX_HIP_LIFT_DELTA
            if foot_lifted or hip_bridged:
                self.feet_frames += 1
                if self.feet_frames >= DEBOUNCE_FEET:
                    if self.feet_planted:
                        print(f"[SIT-UP FAULT] Feet or hips lifted off floor! foot_lift={foot_lifted}, hip_lift={hip_bridged}", flush=True)
                    self.feet_planted = False
            else:
                self.feet_frames = max(0, self.feet_frames - 1)

            # E. Step 1: Knees Must Stay Bent ~90° (Catches kicking legs out straight to counterweight)
            if knee_angle > ACTIVE_KNEE_MAX_ANGLE or active_knee.y >= active_hip.y:
                self.knee_frames += 1
                if self.knee_frames >= DEBOUNCE_KNEES:
                    if self.knees_stayed_bent:
                        print(f"[SIT-UP FAULT] Legs straightened mid-rep: knee_angle={knee_angle:.1f}°", flush=True)
                    self.knees_stayed_bent = False
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FIXED NON-OVERLAPPING BOTTOM -> ASCENDING -> TOP -> DESCENDING)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_lying:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "BOTTOM"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep hands at ears or chest and curl all the way up."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Sit-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Tuck chin slightly and curl torso all the way up to vertical!"

        # Step 4: Initiate Curl Up ONLY from BOTTOM (Fixes DESCENDING ping-pong bug!)
        elif self.state == "BOTTOM" and torso_tilt_deg < ASCENDING_START_TILT_DEG:
            self.state = "ASCENDING"
            self.rep_started = True
            self.min_hip_angle = hip_angle
            self.min_torso_tilt_deg = torso_tilt_deg
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Contract abs and curl all the way up until torso is vertical..."

        # Step 4 Complete: Torso vertical (<= 30° tilt) AND chest compressed toward thighs (hip_angle <= 82°)
        elif self.state == "ASCENDING" and (
            torso_tilt_deg <= FULL_VERTICAL_MAX_TILT_DEG
            and hip_angle <= FULL_COMPRESSION_MAX_HIP_ANGLE
        ):
            self.depth_reached = True
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Vertical top reached! Unroll spine slowly back down to mat."

        # Early downward reversal (Half Rep / Crunch Only -> Flagged as Bad Rep)[cite: 14]
        elif self.state == "ASCENDING" and not self.depth_reached and (torso_tilt_deg - self.min_torso_tilt_deg) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_torso_tilt_deg <= HALF_SITUP_MAX_TILT_DEG:
                    self.was_half_rep = True
                self.state = "DESCENDING"
                self.current_feedback = "Incomplete sit-up! Lowering back early..."
                print(
                    f"[SIT-UP FAULT] Early reversal before vertical top: "
                    f"min_tilt={self.min_torso_tilt_deg:.1f}°, min_hip={self.min_hip_angle:.1f}°",
                    flush=True,
                )

        # Step 5: Unroll Spine Back Down from TOP ONLY[cite: 14]
        elif self.state == "TOP" and torso_tilt_deg > DESCENDING_START_TILT_DEG:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Unroll spine segment by segment back to the mat..."

        if self.state == "ASCENDING" and not ((torso_tilt_deg - self.min_torso_tilt_deg) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 5: Eccentric drop detector (catches slapping back hard against floor)[cite: 14]
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_torso_tilt is not None:
                drop_speed = torso_tilt_deg - self.prev_torso_tilt
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[SIT-UP FAULT] Slapped back hard onto mat: drop_speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_torso_tilt = torso_tilt_deg
        else:
            self.prev_torso_tilt = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (WHEN SHOULDERS UNROLL BACK TO MAT)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "TOP", "ASCENDING"] and torso_tilt_deg >= REP_COMPLETE_TILT_DEG:
            # Ignore minor head/neck shifts while lying on the mat (> 54° tilt)
            if self.min_torso_tilt_deg > MIN_VALID_REP_TILT_DEG and self.is_side_view and self.arms_overhead:
                self._reset_rep_tracking()
                self.state = "BOTTOM"
                return {
                    "exercise": "Sit-Up", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Lie sideways to camera! Return to start to reset."
                is_bad_rep = True
            elif not self.arms_overhead:
                self.current_feedback = "Bad rep: Keep hands at ears or across chest! Do not push off floor or grab legs."
                is_bad_rep = True
            elif not self.feet_planted:
                self.current_feedback = "Bad rep: Keep your feet and hips planted flat on the floor!"
                is_bad_rep = True
            elif not self.knees_stayed_bent:
                self.current_feedback = "Bad rep: Keep your knees bent at ~90°! Do not kick legs straight."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "Bad rep: Do not yank your head/neck! Lift with your abs."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Unroll your spine slowly! Do not let your back slap the mat."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged body twist or momentum swing."
                is_bad_rep = True
            elif self.was_half_rep or not self.depth_reached:
                if self.min_torso_tilt_deg <= HALF_SITUP_MAX_TILT_DEG:
                    self.current_feedback = "Bad rep: Half rep! Curl all the way up until your torso is vertical."
                else:
                    self.current_feedback = "Bad rep: Sit-up was too shallow! Curl chest all the way to your thighs."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SIT-UP REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_hip={self.min_hip_angle:.1f}°, "
                    f"min_tilt={self.min_torso_tilt_deg:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Great Sit-Up! Clean vertical curl and controlled unroll."
                print(
                    f"[SIT-UP REP #{rep_num} -> GOOD REP] "
                    f"min_hip={self.min_hip_angle:.1f}°, min_tilt={self.min_torso_tilt_deg:.1f}°",
                    flush=True,
                )
                self.state = "BOTTOM"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring[cite: 14]
        form_score = max(0, min(100, int(((90.0 - torso_tilt_deg) / 65.0) * 100))) if self.state != "BOTTOM" else 100
        if not self.is_side_view: form_score -= 35
        if not self.arms_overhead: form_score -= 40
        if not self.feet_planted: form_score -= 30
        if not self.knees_stayed_bent: form_score -= 30
        if not self.neck_neutral: form_score -= 30
        if not self.controlled_descent: form_score -= 20
        if (self.was_half_rep or not self.depth_reached) and self.state == "DESCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Sit-Up", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1), "depth_reached": self.depth_reached
        }