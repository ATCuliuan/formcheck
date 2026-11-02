import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (ADJUST PARAMETERS HERE)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_ELBOW_MIN_ANGLE = 150.0       # Arms must be extended/locked out (> 150°)
SETUP_BODY_MIN_ANGLE = 150.0        # Body must form a straight line (> 150°)
SETUP_FEET_ELEVATION_DELTA = 0.03   # Ankle must be at least this much higher than shoulder (Y axis)
SETUP_MIN_READY_FRAMES = 8          # Frames required holding plank to transition to READY
UNLOCK_RESET_FRAMES = 15            # Frames held in decline plank to clear LOCKED_FAULT (~0.5s)

# 2. Kinematic Angles & Form Thresholds
SAG_BODY_ANGLE_MAX = 150.0          # Angle below which sagging hips are detected
PIKE_BODY_ANGLE_MAX = 140.0         # Angle below which piking hips are detected
NECK_MIN_ANGLE = 130.0              # Angle below which neck craning/bending is detected
MAX_WRIST_Y_DIFF = 0.08             # Vertical height gap allowed between hands on floor
MAX_WRIST_X_STAGGER = 0.12          # Anterior/posterior horizontal stagger gap allowed between hands
MIN_SIDE_VISIBILITY = 3.0           # Cumulative visibility required across key lateral landmarks

# 3. Depth, Descent & Range of Motion
FULL_DEPTH_ELBOW_ANGLE = 90.0       # Elbow angle required to count full bottom depth
DESCENT_START_ELBOW_ANGLE = 135.0   # Angle to transition from PLANKING to DESCENDING
ASCENT_START_ELBOW_ANGLE = 100.0    # Angle to transition from BOTTOM to ASCENDING
EARLY_ASCENT_DELTA = 8.0            # Reversal delta from lowest depth before full depth triggers half-rep
ECCENTRIC_DROP_SPEED_MAX = 22.0     # Maximum drop speed (degrees/frame) before flagging free-fall
REP_COMPLETE_ELBOW_ANGLE = 150.0    # Elbow lockout angle to evaluate rep completion

# 4. Debounce Frame Thresholds (Higher = More Deliberate detection)
DEBOUNCE_FLAT = 4                   # Frames before flagging feet dropped to floor level
DEBOUNCE_SAG = 4                    # Frames before flagging sagging hips (banana back)
DEBOUNCE_PIKE = 4                   # Frames before flagging piking hips (butt too high)
DEBOUNCE_NECK = 4                   # Frames before flagging neck craning
DEBOUNCE_HAND_MISALIGN = 4          # Frames before flagging uneven hand placement
DEBOUNCE_DROP = 2                   # Frames before flagging uncontrolled eccentric free-fall
DEBOUNCE_HALF_REP = 6               # Frames before flagging premature reversal / incomplete depth

# 5. Model & Buffer Settings
SEQUENCE_LENGTH = 30                # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40       # Duration feedback remains visible post-rep

# =====================================================================


class DeclinePushupTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.depth_reached = False
        self.min_elbow_angle = 180.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_decline_elevation = True
        self.core_maintained = True
        self.no_pike = True
        self.neck_neutral = True
        self.controlled_descent = True
        self.hands_aligned = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.sag_frames = 0
        self.pike_frames = 0
        self.flat_frames = 0
        self.neck_frames = 0
        self.drop_frames = 0
        self.hand_misalign_frames = 0
        self.half_rep_frames = 0
        self.plank_frames = 0
        self.unlock_frames = 0

        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Elevate feet on bench/box, hold decline plank"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_decline_pushup.tflite")
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
        self.depth_reached = False
        self.min_elbow_angle = 180.0
        self.prev_elbow_angle = None

        self.is_decline_elevation = True
        self.core_maintained = True
        self.no_pike = True
        self.neck_neutral = True
        self.controlled_descent = True
        self.hands_aligned = True
        self.ml_posture_maintained = True

        self.sag_frames = 0
        self.pike_frames = 0
        self.flat_frames = 0
        self.neck_frames = 0
        self.drop_frames = 0
        self.hand_misalign_frames = 0
        self.half_rep_frames = 0

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
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        # Visibility Guard
        if max(l_vis, r_vis) < MIN_SIDE_VISIBILITY:
            self.frame_buffer.clear()
            return {
                "exercise": "Decline Pushup",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Show full decline side profile",
                "elbow_angle": 0.0,
                "body_angle": 0.0,
                "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematic Angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Arms locked & feet elevated above shoulders)
        # ---------------------------------------------------------
        is_feet_elevated = active_ankle.y < (active_shoulder.y - SETUP_FEET_ELEVATION_DELTA)
        is_plank = (elbow_angle > SETUP_ELBOW_MIN_ANGLE) and (body_angle > SETUP_BODY_MIN_ANGLE) and is_feet_elevated

        if is_plank:
            self.plank_frames += 1
        else:
            self.plank_frames = max(0, self.plank_frames - 1)

        if self.plank_frames > SETUP_MIN_READY_FRAMES:
            self.is_ready = True

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Elevate feet higher on box/bench" if not is_feet_elevated else "Hold decline plank to begin"
            return {
                "exercise": "Decline Pushup",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                predicted_class = np.argmax(output_data, axis=1)[0]
                if predicted_class == 0 and self.state not in ["PLANKING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            # A. Feet on floor cheat
            if active_ankle.y > active_shoulder.y:
                self.flat_frames += 1
                if self.flat_frames >= DEBOUNCE_FLAT:
                    self.is_decline_elevation = False
            else:
                self.flat_frames = max(0, self.flat_frames - 1)

            # B. Core Sagging (Banana Back)
            if active_hip.y > active_shoulder.y and body_angle < SAG_BODY_ANGLE_MAX:
                self.sag_frames += 1
                if self.sag_frames >= DEBOUNCE_SAG:
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # C. Hip Piking
            if active_hip.y < active_shoulder.y and body_angle < PIKE_BODY_ANGLE_MAX:
                self.pike_frames += 1
                if self.pike_frames >= DEBOUNCE_PIKE:
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # D. Neck Craning
            if neck_angle < NECK_MIN_ANGLE:
                self.neck_frames += 1
                if self.neck_frames >= DEBOUNCE_NECK:
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
                    self.state = "PLANKING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Lower chest toward floor."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Decline Pushup",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "PLANKING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest toward floor!"

        elif elbow_angle <= DESCENT_START_ELBOW_ANGLE and self.state in ["PLANKING", "ASCENDING"]:
            self.state = "DESCENDING"
            self.min_elbow_angle = elbow_angle
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest to 90 degrees..."

        # Full ROM: Depth reached
        elif self.state == "DESCENDING" and elbow_angle <= FULL_DEPTH_ELBOW_ANGLE:
            self.depth_reached = True
            self.half_rep_frames = 0
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Depth hit! Drive floor away."

        elif self.state == "BOTTOM" and elbow_angle > ASCENT_START_ELBOW_ANGLE:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock out at the top!"

        # Incomplete rep: Early reversal check with debounce
        elif self.state == "DESCENDING" and (elbow_angle - self.min_elbow_angle) > EARLY_ASCENT_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                self.state = "ASCENDING"
                self.current_feedback = "Perform a full rep! Lower chest all the way to 90°."
        else:
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Eccentric Free-Fall Detector
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (ONLY ON ASCENDING OR BOTTOM)
        # ---------------------------------------------------------
        if self.state in ["ASCENDING", "BOTTOM"] and elbow_angle > REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False

            if not self.is_decline_elevation:
                self.current_feedback = "Bad rep: Elevate feet! Feet dropped to floor level. Hold plank to reset."
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

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean decline push-up! Good rep."
                self.state = "PLANKING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4)))
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 25
        if not self.is_decline_elevation: form_score -= 35
        if not self.hands_aligned: form_score -= 20
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Decline Pushup",
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