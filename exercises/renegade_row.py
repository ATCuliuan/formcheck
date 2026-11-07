import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (RENEGADE ROW - SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_PLANK_ELBOW_MIN_ANGLE = 150.0  # Arms must be straight (high plank on hands only)
SETUP_PLANK_BODY_MIN_ANGLE = 155.0   # Rigid body line from shoulder to ankle
SETUP_KNEE_EXT_MIN_ANGLE = 150.0     # Legs locked out on toes (drops to ~90-120 on knees)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.45 # Profile ratio
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames holding plank
UNLOCK_RESET_FRAMES = 15             # Frames held in plank to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.5       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds
MIN_KNEE_OFF_FLOOR_ANGLE = 145.0     # Knees bending past this indicates kneeling
MIN_SUPPORT_ARM_ANGLE = 140.0        # Planted arm bending or dropping to forearm
MAX_HIP_TWIST_DELTA = 0.08           # Pelvis rotation height difference
MIN_CORE_SAG_BODY_ANGLE = 138.0      # Hips dropping below shoulders
MIN_CORE_PIKE_BODY_ANGLE = 142.0     # Butt peaking upward
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Slamming dumbbell onto floor cutoff
EARLY_REVERSAL_DELTA = 8.0           # Early downward drop before hitting ribs

# 3. Pull Depth & Range of Motion
FULL_ROW_MAX_ELBOW_ANGLE = 90.0      # Dumbbell pulled to ribs
HALF_ROW_MAX_ELBOW_ANGLE = 115.0     # Pulling past 115° but short of 90° awards half rep
PULLING_START_ELBOW_ANGLE = 140.0    # Initiating pull
LOWERING_START_ELBOW_ANGLE = 100.0   # Lowering weight from rib cage
REP_COMPLETE_ELBOW_ANGLE = 150.0     # Both arms returned to high plank lockout

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_KNEE_FLOOR = 3              # Frames before flagging knees on floor
DEBOUNCE_ELBOW_PLANK = 3             # Frames before flagging dropping to elbows
DEBOUNCE_TWIST = 3
DEBOUNCE_SAG = 5
DEBOUNCE_PIKE = 3
DEBOUNCE_SUPPORT = 3
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class RenegadeRowTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.depth_reached = False
        self.half_depth_reached = False
        self.min_row_elbow = 180.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.knees_off_ground = True       # Enforces plank is on toes, not knees
        self.no_elbow_plank = True         # Enforces high plank on hands, rejects forearms/elbows
        self.no_hip_rotation = True
        self.core_maintained = True
        self.no_pike = True
        self.support_arm_locked = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
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
        self.current_feedback = "Hold a strong pushup plank on hands and toes to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
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
        self.half_depth_reached = False
        self.min_row_elbow = 180.0
        self.reversal_frames = 0
        self.prev_row_elbow = None

        self.is_side_view = True
        self.knees_off_ground = True
        self.no_elbow_plank = True
        self.no_hip_rotation = True
        self.core_maintained = True
        self.no_pike = True
        self.support_arm_locked = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
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
                "form_score": 0, "feedback": "Show full side profile in plank",
                "row_elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        # Bilateral arm tracking
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        if l_elbow_angle < r_elbow_angle:
            rowing_elbow_angle = l_elbow_angle
            support_elbow_angle = r_elbow_angle
        else:
            rowing_elbow_angle = r_elbow_angle
            support_elbow_angle = l_elbow_angle

        # Active side keypoints
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Angles: Body line & Leg lockout
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        knee_extension_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # Profile verification
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < SETUP_MAX_SHOULDER_SPAN_RATIO

        dx_body = abs(active_shoulder.x - active_ankle.x)
        dy_body = abs(active_shoulder.y - active_ankle.y)
        is_horizontal = dx_body > (dy_body * 1.1)

        # Knees on ground detection
        is_kneeling = (knee_extension_angle < MIN_KNEE_OFF_FLOOR_ANGLE) or (active_knee.y >= (active_ankle.y - 0.02))

        # Elbow plank detection: planted support arm is bent down near forearm level (< 135°)
        is_on_elbows = support_elbow_angle < 135.0

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Strict High Plank: On Hands & Toes Only)
        # ---------------------------------------------------------
        is_plank = (
            (rowing_elbow_angle >= SETUP_PLANK_ELBOW_MIN_ANGLE)
            and (support_elbow_angle >= SETUP_PLANK_ELBOW_MIN_ANGLE)
            and (body_angle >= SETUP_PLANK_BODY_MIN_ANGLE)
            and (knee_extension_angle >= SETUP_KNEE_EXT_MIN_ANGLE)
            and (not is_kneeling)
            and (not is_on_elbows)
            and is_side
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
            self.depth_reached = False
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif is_kneeling:
                feedback_msg = "Lift knees off the floor into a full plank"
            elif is_on_elbows:
                feedback_msg = "Hold plank on your hands! Do not rest on elbows"
            else:
                feedback_msg = "Hold straight plank on toes with arms locked to begin"

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
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["PULLING", "TOP", "LOWERING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Knees on floor cheat
            if is_kneeling:
                self.knee_floor_frames += 1
                if self.knee_floor_frames >= DEBOUNCE_KNEE_FLOOR:
                    self.knees_off_ground = False
            else:
                self.knee_floor_frames = max(0, self.knee_floor_frames - 1)

            # Dropping to elbow plank cheat
            if is_on_elbows:
                self.elbow_plank_frames += 1
                if self.elbow_plank_frames >= DEBOUNCE_ELBOW_PLANK:
                    self.no_elbow_plank = False
            else:
                self.elbow_plank_frames = max(0, self.elbow_plank_frames - 1)

            # Pelvis rotation cheat
            hip_level_offset = abs(l_hip.y - r_hip.y)
            if hip_level_offset > MAX_HIP_TWIST_DELTA:
                self.twist_frames += 1
                if self.twist_frames >= DEBOUNCE_TWIST:
                    self.no_hip_rotation = False
            else:
                self.twist_frames = max(0, self.twist_frames - 1)

            # Core sagging
            if active_hip.y > active_shoulder.y and body_angle < MIN_CORE_SAG_BODY_ANGLE:
                self.sag_frames += 1
                if self.sag_frames >= DEBOUNCE_SAG:
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # Core piking
            if active_hip.y < active_shoulder.y and body_angle < MIN_CORE_PIKE_BODY_ANGLE:
                self.pike_frames += 1
                if self.pike_frames >= DEBOUNCE_PIKE:
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # Support arm buckling (must stay tall on hand)
            if support_elbow_angle < MIN_SUPPORT_ARM_ANGLE:
                self.support_frames += 1
                if self.support_frames >= DEBOUNCE_SUPPORT:
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
                    self.current_feedback = "Reset complete! Row dumbbell to hip from high plank."
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
                self.current_feedback = "Row dumbbell to your hip without twisting or dropping to knees/elbows!"

        elif self.state in ["PLANKING", "LOWERING"] and rowing_elbow_angle < PULLING_START_ELBOW_ANGLE:
            self.state = "PULLING"
            self.rep_started = True
            self.min_row_elbow = rowing_elbow_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep hips flat, stay up on hands and toes..."

        # Full ROM: Elbow past 90°
        elif self.state == "PULLING" and rowing_elbow_angle <= FULL_ROW_MAX_ELBOW_ANGLE:
            self.depth_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Top squeeze! Lower weight with control."

        # Early downward reversal (Half rep branch)
        elif self.state == "PULLING" and not self.depth_reached and (rowing_elbow_angle - self.min_row_elbow) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_row_elbow <= HALF_ROW_MAX_ELBOW_ANGLE:
                    self.half_depth_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete row! Lowering weight early..."

        elif self.state == "TOP" and rowing_elbow_angle > LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the weight back to the floor"

        if self.state == "PULLING" and not ((rowing_elbow_angle - self.min_row_elbow) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop detector
        if self.rep_started and self.state in ["LOWERING", "PULLING"]:
            if self.prev_row_elbow is not None:
                drop_speed = rowing_elbow_angle - self.prev_row_elbow
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_row_elbow = rowing_elbow_angle
        else:
            self.prev_row_elbow = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "TOP", "PULLING"] and rowing_elbow_angle >= REP_COMPLETE_ELBOW_ANGLE and support_elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways in plank to camera!"
                is_bad_rep = True
            elif not self.knees_off_ground:
                self.current_feedback = "Bad rep: Knees touched the floor! Full plank on toes only."
                is_bad_rep = True
            elif not self.no_elbow_plank:
                self.current_feedback = "Bad rep: Do not drop to elbows! Perform in a high plank on your hands."
                is_bad_rep = True
            elif not self.no_hip_rotation:
                self.current_feedback = "Bad rep: You twisted your hips! Keep pelvis flat to floor."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "Bad rep: Hips sagged! Squeeze core and glutes."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "Bad rep: Butt too high! Maintain straight plank line."
                is_bad_rep = True
            elif not self.support_arm_locked:
                self.current_feedback = "Bad rep: Supporting arm buckled! Keep it straight on hand."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Do not slam weight down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged torso rotation/breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                if self.min_row_elbow <= HALF_ROW_MAX_ELBOW_ANGLE or self.half_depth_reached:
                    self.current_feedback = "Half rep: Incomplete row! Pull dumbbell fully to ribs."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Row was too shallow! Pull dumbbell all the way up."
                    is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.half_reps += 1
                self.state = "PLANKING"
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean renegade row! Excellent high plank control."
                self.state = "PLANKING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(rowing_elbow_angle - 90) * 0.4))) if self.state != "PLANKING" else 100
        if not self.knees_off_ground: form_score -= 40
        if not self.no_elbow_plank: form_score -= 40
        if not self.no_hip_rotation: form_score -= 35
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 20
        if not self.support_arm_locked: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "LOWERING": form_score -= 25
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