import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIDE / SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_HIP_MIN_ANGLE = 162.0          # Hip angle required to confirm standing tall
SETUP_KNEE_MIN_ANGLE = 155.0         # Knee angle required to confirm upright stance
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.42 # Max shoulder width / torso length for valid side view
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames standing upright to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames standing tall to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.5       # Minimum landmark visibility for profile detection

# 2. Kinematic & Form Thresholds
MIN_KNEE_ANGLE = 135.0               # Knees bending past this turns hinge into a squat
MIN_SPINE_NECK_ANGLE = 130.0         # Thoracic/lumbar spine flexion angle threshold (lower = more forgiving)
ECCENTRIC_DROP_SPEED_MAX = 22.0      # Maximum drop speed (degrees/frame) before flagging free-fall
ELBOW_FLARE_DROP_MARGIN = 0.08       # Max distance elbow can drop below shoulder before flagging dropped arms
MAX_SHIN_DISPLACEMENT_RATIO = 0.18   # Max horizontal shin forward drift / shin length (vertical shin check)

# 3. Hinge Depth & Range of Motion
FULL_HINGE_MAX_HIP_ANGLE = 105.0     # Peak hinge depth required for full rep credit
HALF_HINGE_THRESHOLD_ANGLE = 130.0   # Hinging down past 130° but stopping short of 105° flags a half-rep
HINGE_START_HIP_ANGLE = 155.0        # Transition threshold from STANDING/EXTENDING to HINGING
EXTENDING_START_HIP_ANGLE = 115.0    # Moving above this angle transitions from PEAK_HINGE to EXTENDING
EARLY_REVERSAL_DELTA = 8.0           # Upward bounce in degrees before reaching full depth
REP_COMPLETE_HIP_ANGLE = 162.0       # Standing lockout angle to evaluate rep completion

# 4. Debounce Frame Thresholds (Higher = Slower/More Deliberate detection)
DEBOUNCE_PROFILE = 4                 # Frames before flagging lost side profile
DEBOUNCE_ELBOWS = 4                  # Frames before flagging dropped elbows
DEBOUNCE_SHIN = 4                    # Frames before flagging forward shin/knee drift
DEBOUNCE_SQUAT = 4                   # Frames before flagging squatting cheat
DEBOUNCE_SPINE = 4                   # Frames before flagging spinal rounding
DEBOUNCE_DROP = 2                    # Frames before flagging fast/uncontrolled hinge
DEBOUNCE_HALF_REP = 4                # Frames before flagging premature upward reversal

# 5. ST-GCN Machine Learning Model Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 # Required probability (>= 75%) before considering ML breakdown
DEBOUNCE_ML_FAULT = 5                # Consecutive frames required above threshold to trip ML fault

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40        # Feedback display duration post-rep

# =====================================================================


class GoodMorningTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.target_hinge_reached = False
        self.min_hip_angle = 180.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.elbows_raised = True          # Strict check: elbows must stay elevated with hands behind head
        self.vertical_shins = True         # Strict check: shins must stay vertical (knees do not drift over toes)
        self.not_squatting = True
        self.neutral_spine = True
        self.controlled_hinge = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.elbows_frames = 0
        self.shin_frames = 0
        self.squat_frames = 0
        self.spine_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_hip_angle = None
        self.current_feedback = "Raise elbows, place hands behind head, and stand sideways"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_good_morning.tflite")
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
        self.target_hinge_reached = False
        self.min_hip_angle = 180.0
        self.prev_hip_angle = None

        self.is_side_view = True
        self.elbows_raised = True
        self.vertical_shins = True
        self.not_squatting = True
        self.neutral_spine = True
        self.controlled_hinge = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.elbows_frames = 0
        self.shin_frames = 0
        self.squat_frames = 0
        self.spine_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
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
                "exercise": "Good Morning",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Show full side profile",
                "hip_angle": 0.0,
                "knee_angle": 0.0,
                "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal kinematics
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        spine_neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Profile check
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Elbow raised guard: elbow stays elevated near or above shoulder level
        elbows_positioned_properly = active_elbow.y < (active_shoulder.y + ELBOW_FLARE_DROP_MARGIN)

        # Vertical shin / foot-to-knee hinge check (Barbell Row mechanics)
        # Prevents knees from translating forward over toes; shins must remain perpendicular
        shin_len = np.hypot(active_knee.x - active_ankle.x, active_knee.y - active_ankle.y) or 0.35
        horizontal_shin_drift = abs(active_knee.x - active_ankle.x) / shin_len
        shins_vertical = horizontal_shin_drift <= MAX_SHIN_DISPLACEMENT_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_standing = (
            (hip_angle > SETUP_HIP_MIN_ANGLE)
            and (knee_angle > SETUP_KNEE_MIN_ANGLE)
            and is_side
            and elbows_positioned_properly
            and shins_vertical
        )

        if is_standing:
            self.standing_frames += 1
            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif not elbows_positioned_properly:
                feedback_msg = "Raise elbows and place hands behind head"
            elif not shins_vertical:
                feedback_msg = "Keep shins vertical: knees over ankles"
            else:
                feedback_msg = "Stand tall sideways to begin"

            return {
                "exercise": "Good Morning",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1),
                "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                # Softmax normalization over logits
                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["HINGING", "PEAK_HINGE", "EXTENDING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Elbow height check (catches dropping arms during the hinge)
            if not elbows_positioned_properly:
                self.elbows_frames += 1
                if self.elbows_frames >= DEBOUNCE_ELBOWS:
                    self.elbows_raised = False
            else:
                self.elbows_frames = max(0, self.elbows_frames - 1)

            # Foot-to-Knee verticality check (catches forward knee drift during the hinge)
            if not shins_vertical:
                self.shin_frames += 1
                if self.shin_frames >= DEBOUNCE_SHIN:
                    self.vertical_shins = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # Squatting cheat: knees bend past 135°
            if knee_angle < MIN_KNEE_ANGLE:
                self.squat_frames += 1
                if self.squat_frames >= DEBOUNCE_SQUAT:
                    self.not_squatting = False
            else:
                self.squat_frames = max(0, self.squat_frames - 1)

            # Rounded spine/neck check
            if spine_neck_angle < MIN_SPINE_NECK_ANGLE:
                self.spine_frames += 1
                if self.spine_frames >= DEBOUNCE_SPINE:
                    self.neutral_spine = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep shins vertical, hinge hips back."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Good Morning",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "hip_angle": round(hip_angle, 1),
                "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep shins vertical: Hinge hips backward, keep back flat!"

        # Begin hinging forward
        elif self.state in ["STANDING", "EXTENDING"] and hip_angle < HINGE_START_HIP_ANGLE:
            self.state = "HINGING"
            self.rep_started = True
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Push hips back, keep shins straight and vertical..."

        # Full ROM: Hip angle reaches <= 105° (near parallel torso)
        elif self.state == "HINGING" and hip_angle <= FULL_HINGE_MAX_HIP_ANGLE:
            self.target_hinge_reached = True
            self.state = "PEAK_HINGE"
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Great depth! Drive hips forward to stand."

        # Premature reversal: User rises before reaching parallel torso
        elif self.state == "HINGING" and not self.target_hinge_reached and (hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                self.state = "EXTENDING"
                self.current_feedback = "Incomplete hinge! Rising early..."

        elif self.state == "PEAK_HINGE" and hip_angle > EXTENDING_START_HIP_ANGLE:
            self.state = "EXTENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through glutes to stand tall"

        # Decay reversal frames if not reversing
        if self.state == "HINGING" and not ((hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA):
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Track minimum depth angle while in hinging phases
        if self.state in ["HINGING", "PEAK_HINGE"]:
            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

        # Eccentric speed monitor with decaying debounce
        if self.state in ["EXTENDING", "HINGING"]:
            if self.prev_hip_angle is not None:
                drop_speed = self.prev_hip_angle - hip_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_hinge = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_hip_angle = hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["EXTENDING", "PEAK_HINGE", "HINGING"] and self.rep_started and hip_angle >= REP_COMPLETE_HIP_ANGLE:
            is_bad_rep = False

            # Form Cheats Evaluated First
            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Return to start to reset."
                is_bad_rep = True
            elif not self.vertical_shins:
                self.current_feedback = "Bad rep: Keep shins vertical! Do not push knees forward past ankles."
                is_bad_rep = True
            elif not self.elbows_raised:
                self.current_feedback = "Bad rep: Keep elbows raised and hands behind head! Stand tall to reset."
                is_bad_rep = True
            elif not self.not_squatting:
                self.current_feedback = "Bad rep: Do not squat! Keep knees soft, push hips back to reset."
                is_bad_rep = True
            elif not self.neutral_spine:
                self.current_feedback = "Bad rep: Keep back flat! Do not round your spine to reset."
                is_bad_rep = True
            elif not self.controlled_hinge:
                self.current_feedback = "Bad rep: Control your hinge descent! Return to start to reset."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged spinal curvature or kinetic breakdown."
                is_bad_rep = True
            elif not self.target_hinge_reached:
                if self.min_hip_angle <= HALF_HINGE_THRESHOLD_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Hinge down until torso is parallel (under 105°). Stand tall to reset."
                else:
                    self.current_feedback = "Bad rep: Movement was too shallow! Return to start to reset."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Good Morning hinge!"
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((165 - hip_angle) / 60 * 100))) if self.state != "STANDING" else 100
        if not self.vertical_shins: form_score -= 30
        if not self.elbows_raised: form_score -= 25
        if not self.not_squatting: form_score -= 35
        if not self.neutral_spine: form_score -= 35
        if not self.controlled_hinge: form_score -= 20
        if not self.target_hinge_reached and self.rep_started: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Good Morning",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "depth_reached": self.target_hinge_reached
        }