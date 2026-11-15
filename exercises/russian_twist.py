import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (RUSSIAN TWIST - STRICT SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1-3: Side View, ~45° V-Sit Lean, Bent Knees, Hands Clasped)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.35 # Neutral side-view starting profile
SETUP_MIN_TORSO_LEAN_DEG = 22.0      # Leaning back at roughly 45° from vertical (22° to 65° window)
SETUP_MAX_TORSO_LEAN_DEG = 65.0      # Prevents lying completely flat on floor
SETUP_MIN_KNEE_BEND_ANGLE = 55.0     # Knees bent in V-sit
SETUP_MAX_KNEE_BEND_ANGLE = 140.0    # Knees must be bent (not locked straight out)
SETUP_MAX_WRIST_CLASP_DIST = 0.35    # Hands interlocked / holding weight together in front of chest
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held in neutral V-sit to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 1.8       # Minimum side-profile landmark visibility sum

# 2. Kinematic & Form Thresholds (Step 1, 4, 5: Straight Spine, Rib Rotation, Steady Knees/Hips)
MIN_TORSO_LEAN_ACTIVE_DEG = 18.0     # Sitting straight up (< 18°) loses V-sit core tension
MAX_TORSO_LEAN_ACTIVE_DEG = 72.0     # Collapsing backward onto floor (> 72°)
MIN_SPINE_STRAIGHT_ANGLE = 118.0     # Ear-shoulder-hip angle (catches hunched/rounded spine)
MIN_RIB_ROTATION_SPAN_RATIO = 0.16   # Shoulders MUST open up in side view when hands drop to hip (catches arm-only waving)
MAX_KNEE_SWING_DRIFT_RATIO = 0.25    # Step 5: Keep knees and hips steady (no wild leg swinging)
MAX_HIP_SLIDE_DRIFT_RATIO = 0.20     # Sit bones stay planted on mat

# 3. Side-View Twist Detection (Step 4: Touch Floor Next to Hip on Each Side)
# In side view, twisting to Side A vs Side B flips the signed X-offset between left/right shoulders
# AND brings the clasped hands down/back toward the hip!
TWIST_SHOULDER_SIGNED_RATIO = 0.12   # Signed (l_shoulder.x - r_shoulder.x) / torso_len threshold for each side
HAND_NEAR_HIP_MAX_DIST_RATIO = 0.78  # Hands must move toward hip/floor during the twist
CENTER_RETURN_SPAN_RATIO = 0.10      # Returning through center between twists

# 4. Debounce Frame Thresholds
DEBOUNCE_LEAN = 4                    # Frames before flagging lost 45° V-sit lean
DEBOUNCE_SPINE = 4                   # Frames before flagging hunched/rounded back
DEBOUNCE_ARM_ONLY = 3                # Frames before flagging arm waving without rib rotation[cite: 11]
DEBOUNCE_LEG_SWING = 4               # Frames before flagging wild knee/leg swinging
DEBOUNCE_ML_FAULT = 5                # Consecutive frames for ST-GCN fault[cite: 11]

ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 # Softmax confidence threshold[cite: 11]
SEQUENCE_LENGTH = 30                 # ST-GCN temporal window[cite: 11]
POST_REP_FEEDBACK_FRAMES = 40        # Post-rep feedback hold frames[cite: 11]


class RussianTwistTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.hit_side_a = False          # Twisted toward Side A (signed shoulder rotation > +threshold)
        self.hit_side_b = False          # Twisted toward Side B (signed shoulder rotation < -threshold)
        self.partial_twist_detected = False
        self.max_rotation_span = 0.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.v_sit_maintained = True     # Step 1: Maintain ~45° torso lean
        self.spine_straight = True       # Step 1: Spine straight and long (not rounded/hunched)
        self.chest_rotates_with_hands = True  # Step 4: Rotate from ribs, not just waving arms[cite: 11]
        self.hips_knees_steady = True    # Step 5: Keep knees and hips steady
        self.ml_posture_maintained = True

        self.baseline_knee_x = None
        self.baseline_knee_y = None
        self.baseline_hip_x = None

        # Debounce counters
        self.lean_frames = 0
        self.spine_frames = 0
        self.arm_swing_frames = 0
        self.leg_swing_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.center_hold_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Sit sideways to camera: bend knees and lean torso back ~45°"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_russian_twist.tflite")
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
        self.hit_side_a = False
        self.hit_side_b = False
        self.partial_twist_detected = False
        self.max_rotation_span = 0.0
        self.center_hold_frames = 0

        self.v_sit_maintained = True
        self.spine_straight = True
        self.chest_rotates_with_hands = True
        self.hips_knees_steady = True
        self.ml_posture_maintained = True

        self.lean_frames = 0
        self.spine_frames = 0
        self.arm_swing_frames = 0
        self.leg_swing_frames = 0
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
                "exercise": "Russian Twist", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile seated on the floor",
                "torso_angle": 0.0, "knee_angle": 0.0
            }

        # Select primary camera-facing side for sagittal V-sit angles
        use_right = (r_vis >= l_vis)
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_elbow = r_elbow if use_right else l_elbow
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee
        active_ankle = r_ankle if use_right else l_ankle

        # 1. Sagittal V-Sit Geometry (Step 1 & 2: ~45° Torso Lean, Bent Knees, Straight Spine)
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        torso_len = np.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y) or 0.38
        dx_torso = abs(mid_shoulder_x - mid_hip_x)
        dy_torso = max(abs(mid_hip_y - mid_shoulder_y), 0.05)
        torso_lean_deg = np.degrees(np.arctan2(dx_torso, dy_torso))  # 0° = vertical upright, 45° = textbook V-sit

        torso_hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        spine_neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 2. Side Profile & Signed Rib Rotation Metrics (Step 4)
        # In neutral side view, l_shoulder.x and r_shoulder.x overlap (signed_shoulder_ratio ~ 0).
        # Twisting ribs left vs right rotates one shoulder forward and the other back, flipping the sign!
        signed_shoulder_ratio = (l_shoulder.x - r_shoulder.x) / torso_len
        abs_shoulder_span_ratio = abs(signed_shoulder_ratio)
        is_side_neutral = abs_shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Hands clasped / holding single weight check (Step 3)
        both_wrists_vis = (l_wrist.visibility > 0.40 and r_wrist.visibility > 0.40)
        if both_wrists_vis:
            mid_hands_x = (l_wrist.x + r_wrist.x) / 2.0
            mid_hands_y = (l_wrist.y + r_wrist.y) / 2.0
            wrist_clasp_ratio = np.hypot(l_wrist.x - r_wrist.x, l_wrist.y - r_wrist.y) / torso_len
            hands_clasped = wrist_clasp_ratio <= SETUP_MAX_WRIST_CLASP_DIST
        else:
            mid_hands_x, mid_hands_y = active_wrist.x, active_wrist.y
            hands_clasped = True

        # Distance from hands to hip (Step 4: moving hands to touch floor next to hip)
        hand_to_hip_dist_ratio = np.hypot(mid_hands_x - active_hip.x, mid_hands_y - active_hip.y) / torso_len
        hand_dropped_near_hip = (mid_hands_y >= active_hip.y - 0.12) and (hand_to_hip_dist_ratio <= HAND_NEAR_HIP_MAX_DIST_RATIO)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Seated Side View, ~45° Lean, Bent Knees)
        # ---------------------------------------------------------
        is_seated_on_floor = (active_hip.y > active_shoulder.y) and (active_knee.y < active_hip.y + 0.12)
        is_lean_valid = (SETUP_MIN_TORSO_LEAN_DEG <= torso_lean_deg <= SETUP_MAX_TORSO_LEAN_DEG)
        is_knees_bent = (SETUP_MIN_KNEE_BEND_ANGLE <= knee_angle <= SETUP_MAX_KNEE_BEND_ANGLE)

        is_setup_ready = (
            is_side_neutral
            and is_seated_on_floor
            and is_lean_valid
            and is_knees_bent
            and hands_clasped
        )

        if is_setup_ready:
            self.ready_frames += 1
            if self.baseline_knee_x is None:
                self.baseline_knee_x = active_knee.x
                self.baseline_knee_y = active_knee.y
                self.baseline_hip_x = active_hip.x
            else:
                self.baseline_knee_x = 0.90 * self.baseline_knee_x + 0.10 * active_knee.x
                self.baseline_knee_y = 0.90 * self.baseline_knee_y + 0.10 * active_knee.y
                self.baseline_hip_x = 0.90 * self.baseline_hip_x + 0.10 * active_hip.x

            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_seated_on_floor:
                feedback_msg = "Sit on the floor sideways to the camera"
            elif not is_side_neutral:
                feedback_msg = "Turn sideways to camera (show side profile to start)"
            elif torso_lean_deg < SETUP_MIN_TORSO_LEAN_DEG:
                feedback_msg = "Lean your upper body back at roughly a 45° angle"
            elif torso_lean_deg > SETUP_MAX_TORSO_LEAN_DEG:
                feedback_msg = "Sit up slightly: hold a 45° V-sit lean (do not lie flat)"
            elif not is_knees_bent:
                feedback_msg = "Bend your knees in front of you"
            else:
                feedback_msg = "Hold steady 45° V-sit with hands in front of chest"

            return {
                "exercise": "Russian Twist", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "torso_angle": round(torso_lean_deg, 1), "knee_angle": round(knee_angle, 1)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["CENTER", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[RUSSIAN TWIST FAULT] AI ST-GCN flagged posture/balance fault: prob={bad_posture_prob:.2f}",
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
        if self.rep_started and self.state in ["TWIST_SIDE_A", "TWIST_SIDE_B", "CENTER"]:
            self.max_rotation_span = max(self.max_rotation_span, abs_shoulder_span_ratio)

            # A. Step 1: Maintain ~45° V-Sit Torso Lean (Neither sitting straight up nor collapsing flat)
            if not (MIN_TORSO_LEAN_ACTIVE_DEG <= torso_lean_deg <= MAX_TORSO_LEAN_ACTIVE_DEG):
                self.lean_frames += 1
                if self.lean_frames >= DEBOUNCE_LEAN:
                    if self.v_sit_maintained:
                        print(
                            f"[RUSSIAN TWIST FAULT] Lost 45° V-sit torso lean: lean={torso_lean_deg:.1f}°",
                            flush=True,
                        )
                    self.v_sit_maintained = False
            else:
                self.lean_frames = max(0, self.lean_frames - 1)

            # B. Step 1: Keep Spine Straight and Long (No rounding/hunching)
            if spine_neck_angle < MIN_SPINE_STRAIGHT_ANGLE:
                self.spine_frames += 1
                if self.spine_frames >= DEBOUNCE_SPINE:
                    if self.spine_straight:
                        print(
                            f"[RUSSIAN TWIST FAULT] Spine rounded/hunched: spine_angle={spine_neck_angle:.1f}°",
                            flush=True,
                        )
                    self.spine_straight = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

            # C. Step 4: Rotate From the Ribs (Catches dropping hands to hips without rotating shoulders/chest)
            if hand_dropped_near_hip and abs_shoulder_span_ratio < (TWIST_SHOULDER_SIGNED_RATIO * 0.65):
                self.arm_swing_frames += 1
                if self.arm_swing_frames >= DEBOUNCE_ARM_ONLY:
                    if self.chest_rotates_with_hands:
                        print(
                            f"[RUSSIAN TWIST FAULT] Arm waving without rib rotation! "
                            f"shoulder_span={abs_shoulder_span_ratio:.2f}",
                            flush=True,
                        )
                    self.chest_rotates_with_hands = False
            else:
                self.arm_swing_frames = max(0, self.arm_swing_frames - 1)

            # D. Step 5: Control the Hips & Knees (Avoid swinging legs wildly)
            if self.baseline_knee_x is not None and self.baseline_hip_x is not None:
                knee_drift = np.hypot(active_knee.x - self.baseline_knee_x, active_knee.y - self.baseline_knee_y) / torso_len
                hip_drift = abs(active_hip.x - self.baseline_hip_x) / torso_len
                if knee_drift > MAX_KNEE_SWING_DRIFT_RATIO or hip_drift > MAX_HIP_SLIDE_DRIFT_RATIO:
                    self.leg_swing_frames += 1
                    if self.leg_swing_frames >= DEBOUNCE_LEG_SWING:
                        if self.hips_knees_steady:
                            print(
                                f"[RUSSIAN TWIST FAULT] Knees/hips swung wildly: knee_drift={knee_drift:.2f}, hip_drift={hip_drift:.2f}",
                                flush=True,
                            )
                        self.hips_knees_steady = False
                else:
                    self.leg_swing_frames = max(0, self.leg_swing_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (SIDE VIEW RIB ROTATION: SIDE A -> SIDE B -> CENTER)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_setup_ready:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "CENTER"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Lean back 45° and rotate ribs side to side."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Russian Twist", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "torso_angle": round(torso_lean_deg, 1), "knee_angle": round(knee_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "CENTER"
            if self.post_rep_timer == 0:
                self.current_feedback = "Rotate ribs & shoulders to touch next to each hip!"

        # Twist to Side A (Signed shoulder rotation positive OR near-side hip touch)
        if self.state in ["CENTER", "TWIST_SIDE_B"] and (
            signed_shoulder_ratio > TWIST_SHOULDER_SIGNED_RATIO
            or (hand_dropped_near_hip and signed_shoulder_ratio > 0.06)
        ):
            self.state = "TWIST_SIDE_A"
            self.rep_started = True
            self.hit_side_a = True
            self.center_hold_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "First side reached! Now rotate across to the opposite hip..."

        # Twist to Side B (Signed shoulder rotation negative OR far-side hip rotation)
        elif self.state in ["CENTER", "TWIST_SIDE_A"] and (
            signed_shoulder_ratio < -TWIST_SHOULDER_SIGNED_RATIO
            or (hand_dropped_near_hip and signed_shoulder_ratio < -0.06)
        ):
            self.state = "TWIST_SIDE_B"
            self.rep_started = True
            self.hit_side_b = True
            self.center_hold_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Opposite side reached! Return to center or keep twisting..."

        # Detect Half Rep: User twisted to only ONE side, returned to neutral center, and stopped for > 18 frames
        in_neutral_center = abs_shoulder_span_ratio <= CENTER_RETURN_SPAN_RATIO
        if self.rep_started and (self.hit_side_a ^ self.hit_side_b) and in_neutral_center:
            self.center_hold_frames += 1
            if self.center_hold_frames >= 18:
                self.partial_twist_detected = True
        else:
            self.center_hold_frames = 0

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (BOTH SIDES TAPPED OR HALF-REP ABORT)
        # ---------------------------------------------------------
        completed_full_cycle = (self.rep_started and self.hit_side_a and self.hit_side_b and in_neutral_center)
        if completed_full_cycle or self.partial_twist_detected:
            is_bad_rep = False

            if not self.v_sit_maintained:
                self.current_feedback = "Bad rep: Hold your 45° V-sit lean! Do not sit straight up or lie flat."
                is_bad_rep = True
            elif not self.spine_straight:
                self.current_feedback = "Bad rep: Keep your spine straight and long! Do not hunch your back."
                is_bad_rep = True
            elif not self.chest_rotates_with_hands or self.max_rotation_span < MIN_RIB_ROTATION_SPAN_RATIO:
                self.current_feedback = "Bad rep: Rotate from your ribs and shoulders! Do not just wave your arms."
                is_bad_rep = True
            elif not self.hips_knees_steady:
                self.current_feedback = "Bad rep: Keep your knees and hips steady! Avoid swinging your legs."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected posture rounding or loss of V-sit balance."
                is_bad_rep = True
            elif self.partial_twist_detected or not (self.hit_side_a and self.hit_side_b):
                self.current_feedback = "Bad rep: Half rep! Rotate all the way to both sides before stopping."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[RUSSIAN TWIST REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | lean={torso_lean_deg:.1f}°, max_span={self.max_rotation_span:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Russian Twist! Great 45° V-sit and rib rotation."
                print(
                    f"[RUSSIAN TWIST REP #{rep_num} -> GOOD REP] "
                    f"lean={torso_lean_deg:.1f}°, max_span={self.max_rotation_span:.2f}",
                    flush=True,
                )
                self.state = "CENTER"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = 100 if self.state == "CENTER" else 90
        if not self.v_sit_maintained: form_score -= 35
        if not self.spine_straight: form_score -= 30
        if not self.chest_rotates_with_hands: form_score -= 35
        if not self.hips_knees_steady: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Russian Twist",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "torso_angle": round(torso_lean_deg, 1),
            "knee_angle": round(knee_angle, 1)
        }