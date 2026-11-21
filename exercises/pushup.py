import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (PUSH-UP - SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_ELBOW_MIN_ANGLE = 150.0        # Straight arms in high plank
SETUP_BODY_MIN_ANGLE = 150.0         # Rigid horizontal plank line
SETUP_KNEE_MIN_ANGLE = 150.0         # Knees locked straight off floor
SETUP_MIN_READY_FRAMES = 8           # Frames holding plank to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames held in high plank to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.5       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds
MIN_KNEE_STRAIGHT_ANGLE = 145.0      # Bending knees / knee push-up cheat
MIN_CORE_SAG_BODY_ANGLE = 140.0      # Hips collapsing toward floor (banana back)
MIN_CORE_PIKE_BODY_ANGLE = 140.0     # Butt peaking upward into an A-frame
MIN_NECK_NEUTRAL_ANGLE = 130.0       # Chin poking forward to fake chest depth
MAX_WRIST_Y_DIFF = 0.08              # Vertical height gap allowed between hands on floor
MAX_WRIST_X_STAGGER = 0.12           # Anterior/posterior horizontal stagger gap allowed between hands
ECCENTRIC_DROP_SPEED_MAX = 22.0      # Free-falling into floor cutoff (deg/frame)
EARLY_ASCENT_DELTA = 8.0             # Early upward push before hitting 90°

# 3. Pushup Depth & Range of Motion
FULL_DEPTH_MAX_ELBOW_ANGLE = 90.0    # Chest to floor level (elbows <= 90°)
DESCENDING_START_ELBOW_ANGLE = 135.0 # Initiating descent
ASCENDING_START_ELBOW_ANGLE = 100.0  # Pushing up from bottom
REP_COMPLETE_ELBOW_ANGLE = 150.0     # Lockout evaluating completion

# 4. Debounce Frame Thresholds
DEBOUNCE_KNEE_GROUND = 3             # Fast trigger for knees touching the floor
DEBOUNCE_SAG = 4
DEBOUNCE_PIKE = 4
DEBOUNCE_NECK = 4
DEBOUNCE_HAND_MISALIGN = 4
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 6
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class PushupTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility

        self.rep_started = False
        self.depth_reached = False
        self.min_elbow_angle = 180.0

        # Posture Tripwires (SIDE / SAGITTAL PROFILE)
        self.knees_off_floor = True        # Catches knee push-up / knees touching down
        self.core_maintained = True
        self.no_pike = True
        self.neck_neutral = True
        self.hands_aligned = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.knee_ground_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.neck_frames = 0
        self.hand_misalign_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.plank_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Get down on the floor into a straight plank, knees off floor"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_pushup.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[PUSHUP ERROR] __init__ model_path={model_path} | error={exc}",
                    flush=True,
                )
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        """Maps MediaPipe 33 landmarks to the NTU-25 skeleton layout."""
        ntu = np.zeros((25, 3), dtype=np.float32)

        def xyz(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        ntu[3] = xyz(0)    # Nose
        ntu[4] = xyz(11)   # Left Shoulder
        ntu[5] = xyz(13)   # Left Elbow
        ntu[6] = xyz(15)   # Left Wrist
        ntu[7] = xyz(19)   # Left Hand/Index
        ntu[8] = xyz(12)   # Right Shoulder
        ntu[9] = xyz(14)   # Right Elbow
        ntu[10] = xyz(16)  # Right Wrist
        ntu[11] = xyz(20)  # Right Hand/Index
        ntu[12] = xyz(23)  # Left Hip
        ntu[13] = xyz(25)  # Left Knee
        ntu[14] = xyz(27)  # Left Ankle
        ntu[15] = xyz(31)  # Left Foot
        ntu[16] = xyz(24)  # Right Hip
        ntu[17] = xyz(26)  # Right Knee
        ntu[18] = xyz(28)  # Right Ankle
        ntu[19] = xyz(32)  # Right Foot

        ntu[0] = (ntu[12] + ntu[16]) / 2.0  # Base Spine (Mid-Hip)
        ntu[2] = (ntu[4] + ntu[8]) / 2.0    # Spine Shoulder / Neck (Mid-Shoulder)
        ntu[1] = (ntu[0] + ntu[2]) / 2.0    # Mid Spine
        ntu[20] = ntu[2]                    # Chest

        ntu[21] = xyz(17)  # Left Pinky
        ntu[22] = xyz(21)  # Left Thumb
        ntu[23] = xyz(18)  # Right Pinky
        ntu[24] = xyz(22)  # Right Thumb
        return ntu

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.depth_reached = False
        self.min_elbow_angle = 180.0
        self.prev_elbow_angle = None

        self.knees_off_floor = True
        self.core_maintained = True
        self.no_pike = True
        self.neck_neutral = True
        self.hands_aligned = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.knee_ground_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.neck_frames = 0
        self.hand_misalign_frames = 0
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Push-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile on floor",
                "elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Orientation check
        dy = abs(active_ankle.y - active_shoulder.y)
        dx = abs(active_ankle.x - active_shoulder.x)
        is_horizontal = dx > (dy * 0.90)

        # Knee on floor / knee-support check:
        # 1. Knee angle bent (not locked out)
        # 2. Knee landmark sagging toward floor near wrist plane level
        is_knee_on_floor = (knee_angle < MIN_KNEE_STRAIGHT_ANGLE) or (active_knee.y > max(active_hip.y, active_ankle.y) + 0.03)

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_plank = (
            (elbow_angle >= SETUP_ELBOW_MIN_ANGLE)
            and (body_angle >= SETUP_BODY_MIN_ANGLE)
            and (knee_angle >= SETUP_KNEE_MIN_ANGLE)
            and not is_knee_on_floor
            and is_horizontal
        )

        if is_plank:
            self.plank_frames += 1
            if self.plank_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.plank_frames = max(0, self.plank_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_horizontal:
                feedback_msg = "Get horizontal on floor"
            elif is_knee_on_floor or knee_angle < SETUP_KNEE_MIN_ANGLE:
                feedback_msg = "Lift knees off the floor! Hold full toe plank"
            else:
                feedback_msg = "Hold straight plank to begin"

            return {
                "exercise": "Push-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                "depth_reached": False
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["PLANK", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[PUSHUP FAULT] AI ST-GCN flagged bad posture: prob={bad_posture_prob:.2f} "
                                f"(min={ML_FAULT_CONFIDENCE_THRESHOLD})",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[PUSHUP ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            # A. Knee Touching Floor / Bending Tripwire
            if is_knee_on_floor:
                self.knee_ground_frames += 1
                if self.knee_ground_frames >= DEBOUNCE_KNEE_GROUND:
                    if self.knees_off_floor:
                        print(
                            f"[PUSHUP FAULT] Knees on floor / bent: knee_angle={knee_angle:.1f}° "
                            f"(min={MIN_KNEE_STRAIGHT_ANGLE}°)",
                            flush=True,
                        )
                    self.knees_off_floor = False
            else:
                self.knee_ground_frames = max(0, self.knee_ground_frames - 1)

            # B. Core Sagging (Banana Back)
            if (active_hip.y > active_shoulder.y + 0.04) and (body_angle < MIN_CORE_SAG_BODY_ANGLE):
                self.sag_frames += 1
                if self.sag_frames >= DEBOUNCE_SAG:
                    if self.core_maintained:
                        print(
                            f"[PUSHUP FAULT] Core sagging: body_angle={body_angle:.1f}° "
                            f"(min={MIN_CORE_SAG_BODY_ANGLE}°)",
                            flush=True,
                        )
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # C. Hip Piking (Butt too high)
            if active_hip.y < active_shoulder.y and body_angle < MIN_CORE_PIKE_BODY_ANGLE:
                self.pike_frames += 1
                if self.pike_frames >= DEBOUNCE_PIKE:
                    if self.no_pike:
                        print(
                            f"[PUSHUP FAULT] Hip piking: body_angle={body_angle:.1f}° "
                            f"(min={MIN_CORE_PIKE_BODY_ANGLE}°)",
                            flush=True,
                        )
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # D. Neck Craning (Chin reaching to fake depth)
            if neck_angle < MIN_NECK_NEUTRAL_ANGLE:
                self.neck_frames += 1
                if self.neck_frames >= DEBOUNCE_NECK:
                    if self.neck_neutral:
                        print(
                            f"[PUSHUP FAULT] Neck craning: neck_angle={neck_angle:.1f}° "
                            f"(min={MIN_NECK_NEUTRAL_ANGLE}°)",
                            flush=True,
                        )
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            # E. Hand Asymmetry / Stagger Check
            if l_wrist.visibility > 0.45 and r_wrist.visibility > 0.45:
                vertical_wrist_diff = abs(l_wrist.y - r_wrist.y)
                horizontal_wrist_stagger = abs(l_wrist.x - r_wrist.x)
                if vertical_wrist_diff > MAX_WRIST_Y_DIFF or horizontal_wrist_stagger > MAX_WRIST_X_STAGGER:
                    self.hand_misalign_frames += 1
                    if self.hand_misalign_frames >= DEBOUNCE_HAND_MISALIGN:
                        if self.hands_aligned:
                            print(
                                f"[PUSHUP FAULT] Hands misaligned: y_diff={vertical_wrist_diff:.3f} "
                                f"(max={MAX_WRIST_Y_DIFF}), x_stagger={horizontal_wrist_stagger:.3f} "
                                f"(max={MAX_WRIST_X_STAGGER})",
                                flush=True,
                            )
                        self.hands_aligned = False
                else:
                    self.hand_misalign_frames = max(0, self.hand_misalign_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_plank:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "PLANK"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep knees locked and lower chest."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Push-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "PLANK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest toward the floor!"

        elif elbow_angle <= DESCENDING_START_ELBOW_ANGLE and self.state in ["PLANK", "ASCENDING"]:
            self.state = "DESCENDING"
            self.rep_started = True
            self.min_elbow_angle = elbow_angle
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest to 90 degrees..."

        # Full ROM: Elbows <= 90°
        elif self.state == "DESCENDING" and elbow_angle <= FULL_DEPTH_MAX_ELBOW_ANGLE:
            self.depth_reached = True
            self.half_rep_frames = 0
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Depth hit! Drive floor away."

        elif self.state == "BOTTOM" and elbow_angle > ASCENDING_START_ELBOW_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Push back to full lockout!"

        # Incomplete rep: Early reversal check with debounce
        elif self.state == "DESCENDING" and (elbow_angle - self.min_elbow_angle) > EARLY_ASCENT_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                self.state = "ASCENDING"
                self.current_feedback = "Incomplete depth! Pushing up early..."
        else:
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Eccentric drop detector
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[PUSHUP FAULT] Uncontrolled descent speed: {drop_speed:.1f} deg/frame "
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
        # 5. REP COMPLETION & EVALUATION (ONLY ON ASCENDING OR BOTTOM)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "BOTTOM"] and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False

            if not self.knees_off_floor:
                self.current_feedback = "Bad rep: Knees touched floor! Push-ups must be on toes. Hold plank to reset."
                is_bad_rep = True
            elif not self.hands_aligned:
                self.current_feedback = "Bad rep: Uneven hand placement! Align hands evenly and hold plank to reset."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "Bad rep: Core collapsed! Do not sag hips. Hold plank to reset."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "Bad rep: Butt too high! Maintain straight plank and hold plank to reset."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "Bad rep: Neutral neck! Do not drop chin. Hold plank to reset."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Do not drop to floor. Hold plank to reset."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected elbow flare/breakdown! Hold plank to reset."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "Bad rep: Incomplete depth! Lower chest to 90° and hold plank to reset."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[PUSHUP REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean push-up! Good rep."
                print(
                    f"[PUSHUP REP #{rep_num} -> GOOD REP] "
                    f"min_elbow={self.min_elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "PLANK"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4))) if self.state != "PLANK" else 100
        if not self.knees_off_floor: form_score -= 40
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 25
        if not self.hands_aligned: form_score -= 20
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Push-Up",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "body_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }