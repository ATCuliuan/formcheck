import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIDE LUNGE - STRICT FRONTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1: Front View, Feet Hip-Width, Hands at Chest)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.25 # Enforces front-facing view (rejects side view)
SETUP_KNEE_MIN_ANGLE = 160.0         # Both legs fully extended standing tall
SETUP_MAX_HAND_CHEST_DIST = 0.45     # Step 1: Hands clasped in front of chest for balance
SETUP_MIN_READY_FRAMES = 8           # Frames holding standing position to confirm READY
UNLOCK_RESET_FRAMES = 12             # Frames standing tall to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.8       # Minimum keypoint visibility for lower body

# 2. Kinematic & Form Thresholds (Step 2, 3, 4: Wide Step, Trailing Leg Straight, Flat Feet)
MIN_WIDE_STEP_ANKLE_RATIO = 0.48     # Step 2: Large lateral step width (ankle_span / torso_len)
MIN_LATERAL_HIP_SHIFT_RATIO = 0.12   # Step 3: Pelvis shifts laterally toward lunging leg
MIN_TRAILING_LEG_ANGLE = 150.0       # Step 4: Trailing leg must remain straight and locked out
MIN_ACTIVE_HIP_ANGLE = 55.0          # Step 3: Hinge at hips without collapsing chest onto floor
MAX_HEEL_LIFT_DELTA = 0.05           # Step 2 & 4: Both feet stay flat on the floor
EARLY_REVERSAL_DELTA = 8.0           # Early upward bounce before reaching depth

# 3. Lunge Depth & Range of Motion (Step 3 & 5)
FULL_LUNGE_MAX_KNEE_ANGLE = 112.0    # Deep active knee flexion
HALF_LUNGE_MAX_KNEE_ANGLE = 135.0    # Flexing past 135° but short of 112° flags Half Rep (Bad Rep)
MIN_VALID_REP_FLEXION_ANGLE = 142.0  # Ignore minor standing knee shifts (> 142°)
DESCENDING_START_KNEE_ANGLE = 145.0  # Flexing below this initiates DESCENDING
ASCENDING_START_KNEE_ANGLE = 118.0   # Rising out of BOTTOM transitions to ASCENDING
REP_COMPLETE_KNEE_ANGLE = 156.0      # Step 5: Drive back up to standing lockout

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 3
DEBOUNCE_HANDS = 5
DEBOUNCE_TRAILING_LEG = 3
DEBOUNCE_TORSO = 3
DEBOUNCE_HEEL = 3
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40        # Duration feedback message stays visible

# =====================================================================


class SideLungeTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_depth_reached = False
        self.wide_step_taken = False
        self.weight_shifted = False
        self.min_active_knee_angle = 180.0
        self.max_ankle_span_ratio = 0.0

        # Posture Tripwires (STRICT FRONT VIEW)
        self.is_front_view = True
        self.hands_at_chest = True        # Step 1: Hands clasped in front of chest
        self.trailing_leg_bent = False    # Step 4: Keep trailing leg straight
        self.torso_collapsed = False      # Step 3: Chest proud, hinge at hips
        self.heel_lifted = False          # Step 2 & 4: Keep both feet flat on floor
        self.ml_posture_maintained = True

        # Debounce counters
        self.front_frames = 0
        self.hand_frames = 0
        self.trailing_frames = 0
        self.torso_frames = 0
        self.heel_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_ankle_y = None
        self.active_leg = None

        self.current_feedback = "Face camera: stand tall with hands clasped in front of chest"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_side_lunge.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[SIDE LUNGE ERROR] __init__ model_path={model_path} | error={exc}",
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
        self.depth_reached = False
        self.half_depth_reached = False
        self.wide_step_taken = False
        self.weight_shifted = False
        self.min_active_knee_angle = 180.0
        self.max_ankle_span_ratio = 0.0
        self.reversal_frames = 0
        self.active_leg = None

        self.is_front_view = True
        self.hands_at_chest = True
        self.trailing_leg_bent = False
        self.torso_collapsed = False
        self.heel_lifted = False
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.hand_frames = 0
        self.trailing_frames = 0
        self.torso_frames = 0
        self.heel_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        total_vis = sum([l_hip.visibility, r_hip.visibility, l_knee.visibility, r_knee.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if total_vis < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Side Lunge",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full lower body and torso must be visible",
                "knee_angle": 0.0, "depth_reached": False
            }

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

        # Front View & Body Scale Metrics
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        torso_len = np.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y) or 0.40
        shoulder_span_ratio = abs(l_shoulder.x - r_shoulder.x) / torso_len
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO

        # Step 1: Hands Clasped in Front of Chest Check
        wrists_vis = (l_wrist.visibility > 0.35 and r_wrist.visibility > 0.35)
        if wrists_vis:
            hands_above_hips = (l_wrist.y < l_hip.y) and (r_wrist.y < r_hip.y)
            wrist_clasp_dist = np.hypot(l_wrist.x - r_wrist.x, l_wrist.y - r_wrist.y) / torso_len
            are_hands_at_chest = hands_above_hips and (wrist_clasp_dist <= SETUP_MAX_WRIST_CLASP_DIST if hasattr(self, 'SETUP_MAX_WRIST_CLASP_DIST') else wrist_clasp_dist <= SETUP_MAX_HAND_CHEST_DIST)
        else:
            are_hands_at_chest = True

        # Step 2: Lateral Step Width Ratio
        ankle_span_ratio = abs(l_ankle.x - r_ankle.x) / torso_len

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1: Stand Tall Facing Camera, Hands at Chest)
        # ---------------------------------------------------------
        is_standing_tall = (
            is_facing_front
            and (l_knee_angle > SETUP_KNEE_MIN_ANGLE)
            and (r_knee_angle > SETUP_KNEE_MIN_ANGLE)
        )

        if is_standing_tall:
            self.standing_frames += 1
            if not self.rep_started:
                self.active_leg = None
            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
                current_ankle_y = max(l_ankle.y, r_ankle.y)
                if self.baseline_ankle_y is None:
                    self.baseline_ankle_y = current_ankle_y
                else:
                    self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * current_ankle_y
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            else:
                feedback_msg = "Stand tall with legs straight and hands at chest to begin"
            return {
                "exercise": "Side Lunge", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "knee_angle": round(min(l_knee_angle, r_knee_angle), 1), "depth_reached": False
            }

        # Lock onto the active lunging leg during the rep so it doesn't swap mid-lunge
        if self.active_leg is None:
            if l_knee_angle < r_knee_angle - 10.0:
                self.active_leg = "LEFT"
            elif r_knee_angle < l_knee_angle - 10.0:
                self.active_leg = "RIGHT"

        if self.active_leg == "LEFT":
            active_knee_angle = l_knee_angle
            trailing_knee_angle = r_knee_angle
            active_hip_angle = l_hip_angle
            active_ankle = l_ankle
            trailing_ankle = r_ankle
        elif self.active_leg == "RIGHT":
            active_knee_angle = r_knee_angle
            trailing_knee_angle = l_knee_angle
            active_hip_angle = r_hip_angle
            active_ankle = r_ankle
            trailing_ankle = l_ankle
        else:
            active_knee_angle = min(l_knee_angle, r_knee_angle)
            trailing_knee_angle = max(l_knee_angle, r_knee_angle)
            active_hip_angle = min(l_hip_angle, r_hip_angle)
            active_ankle = l_ankle if l_knee_angle < r_knee_angle else r_ankle
            trailing_ankle = r_ankle if l_knee_angle < r_knee_angle else l_ankle

        # Step 3: Lateral Weight Shift Check (Pelvis shifts closer to active ankle than trailing ankle)
        dist_hip_to_active = abs(mid_hip_x - active_ankle.x)
        dist_hip_to_trailing = abs(mid_hip_x - trailing_ankle.x)
        lateral_shift_ratio = (dist_hip_to_trailing - dist_hip_to_active) / torso_len

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
                                f"[SIDE LUNGE FAULT] AI ST-GCN flagged posture breakdown: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[SIDE LUNGE ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.max_ankle_span_ratio = max(self.max_ankle_span_ratio, ankle_span_ratio)
            if ankle_span_ratio >= MIN_WIDE_STEP_ANKLE_RATIO:
                self.wide_step_taken = True
            if lateral_shift_ratio >= MIN_LATERAL_HIP_SHIFT_RATIO:
                self.weight_shifted = True

            # A. Front View Check
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[SIDE LUNGE FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # B. Step 1: Hands Clasped at Chest Check
            if not are_hands_at_chest:
                self.hand_frames += 1
                if self.hand_frames >= DEBOUNCE_HANDS:
                    if self.hands_at_chest:
                        print("[SIDE LUNGE FAULT] Hands dropped from chest!", flush=True)
                    self.hands_at_chest = False
            else:
                self.hand_frames = max(0, self.hand_frames - 1)

            # C. Step 4: Trailing Leg Must Stay Straight (Catches regular two-legged squat!)
            if trailing_knee_angle < MIN_TRAILING_LEG_ANGLE:
                self.trailing_frames += 1
                if self.trailing_frames >= DEBOUNCE_TRAILING_LEG:
                    if not self.trailing_leg_bent:
                        print(
                            f"[SIDE LUNGE FAULT] Trailing leg bent: trailing_knee={trailing_knee_angle:.1f}° "
                            f"(min={MIN_TRAILING_LEG_ANGLE}°)",
                            flush=True,
                        )
                    self.trailing_leg_bent = True
            else:
                self.trailing_frames = max(0, self.trailing_frames - 1)

            # D. Step 3: Torso Collapse Check
            if active_hip_angle < MIN_ACTIVE_HIP_ANGLE:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if not self.torso_collapsed:
                        print(
                            f"[SIDE LUNGE FAULT] Chest collapsed forward: hip_angle={active_hip_angle:.1f}°",
                            flush=True,
                        )
                    self.torso_collapsed = True
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # E. Step 2 & 4: Both Feet Flat on Floor Check
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > MAX_HEEL_LIFT_DELTA:
                self.heel_frames += 1
                if self.heel_frames >= DEBOUNCE_HEEL:
                    if not self.heel_lifted:
                        print("[SIDE LUNGE FAULT] Active heel lifted off the floor!", flush=True)
                    self.heel_lifted = True
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

            self.min_active_knee_angle = min(self.min_active_knee_angle, active_knee_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FIXED DESCENDING -> BOTTOM -> ASCENDING FLOW)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_tall:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Step out wide, sit hips back, and keep trailing leg straight."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Side Lunge", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "knee_angle": round(active_knee_angle, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Step out wide to the side, sit back into hip, keep trailing leg straight!"

        # Step 2 & 3: Initiate Side Lunge Descent
        elif self.state in ["STANDING", "ASCENDING"] and active_knee_angle <= DESCENDING_START_KNEE_ANGLE:
            self.state = "DESCENDING"
            self.rep_started = True
            self.min_active_knee_angle = active_knee_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Sit hips back into the lunging leg, keep trailing leg locked..."

        # Step 3 Complete: Active knee reaches <= 112°
        elif self.state == "DESCENDING" and active_knee_angle <= FULL_LUNGE_MAX_KNEE_ANGLE:
            self.depth_reached = True
            self.state = "BOTTOM"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Depth reached! Drive forcefully through your heel to stand tall."

        # Premature Reversal (Half Rep Branch: Rising from DESCENDING before reaching depth)
        elif self.state == "DESCENDING" and not self.depth_reached and (active_knee_angle - self.min_active_knee_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_active_knee_angle <= HALF_LUNGE_MAX_KNEE_ANGLE:
                    self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Incomplete depth! Pushing back early..."
                print(
                    f"[SIDE LUNGE FAULT] Early reversal before full lunge depth: "
                    f"min_knee={self.min_active_knee_angle:.1f}° (required<={FULL_LUNGE_MAX_KNEE_ANGLE}°)",
                    flush=True,
                )

        # Step 5: Drive Back Up from BOTTOM ONLY (Fixed ping-pong bug!)
        elif self.state == "BOTTOM" and active_knee_angle > ASCENDING_START_KNEE_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through your heel back to starting stance!"

        if self.state == "DESCENDING" and not ((active_knee_angle - self.min_active_knee_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and active_knee_angle >= REP_COMPLETE_KNEE_ANGLE:
            # Ignore minor standing knee shifts (> 142°)
            if self.min_active_knee_angle > MIN_VALID_REP_FLEXION_ANGLE and self.is_front_view and not self.trailing_leg_bent:
                self._reset_rep_tracking()
                self.state = "STANDING"
                return {
                    "exercise": "Side Lunge", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "knee_angle": round(active_knee_angle, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Stand tall to reset."
                is_bad_rep = True
            elif self.trailing_leg_bent:
                self.current_feedback = "Bad rep: Keep your trailing leg straight and locked out! Do not squat with both legs."
                is_bad_rep = True
            elif not self.wide_step_taken:
                self.current_feedback = "Bad rep: Take a wider step out to the side!"
                is_bad_rep = True
            elif not self.weight_shifted:
                self.current_feedback = "Bad rep: Shift your hips and weight laterally over your lunging leg!"
                is_bad_rep = True
            elif self.heel_lifted:
                self.current_feedback = "Bad rep: Keep both feet completely flat on the floor! Do not lift heel."
                is_bad_rep = True
            elif self.torso_collapsed:
                self.current_feedback = "Bad rep: Keep your chest up! Hinge at the hips without collapsing forward."
                is_bad_rep = True
            elif not self.hands_at_chest:
                self.current_feedback = "Bad rep: Keep your hands clasped in front of your chest for balance!"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected broken side-lunge alignment."
                is_bad_rep = True
            elif not self.depth_reached:
                if self.min_active_knee_angle <= HALF_LUNGE_MAX_KNEE_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Bad rep: Half rep! Sit deeper into the lunging hip."
                else:
                    self.current_feedback = "Bad rep: Side lunge was too shallow! Sit deeper into your hip."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SIDE LUNGE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_knee={self.min_active_knee_angle:.1f}°, "
                    f"step_width={self.max_ankle_span_ratio:.2f}",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Side Lunge! Great hip hinge, straight trailing leg, and heel drive."
                print(
                    f"[SIDE LUNGE REP #{rep_num} -> GOOD REP] "
                    f"min_knee={self.min_active_knee_angle:.1f}°, step_width={self.max_ankle_span_ratio:.2f}",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(active_knee_angle - 90) * 0.3))) if self.state != "STANDING" else 100
        if not self.is_front_view: form_score -= 35
        if self.trailing_leg_bent: form_score -= 40
        if not self.wide_step_taken and self.rep_started: form_score -= 25
        if self.torso_collapsed: form_score -= 30
        if self.heel_lifted: form_score -= 25
        if not self.hands_at_chest: form_score -= 15
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Side Lunge", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": round(active_knee_angle, 1), "depth_reached": self.depth_reached
        }