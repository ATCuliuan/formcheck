import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (DUMBBELL KICKBACK - SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_TORSO_MAX_ANGLE = 150.0        # Torso must be hinged below 150° (forgiving hinge)
SETUP_ELBOW_START_MAX_ANGLE = 110.0  # Elbow flexed around 90° (allowing up to 110°)
SETUP_UPPER_ARM_PINNED_ANGLE = 45.0  # Upper arm pinned along ribs (relaxed from 32° to prevent lockup)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.45 # Side profile ratio limit
MAX_FOREARM_VERTICAL_DRIFT = 0.35    # Max horizontal drift between wrist and elbow / forearm length
SETUP_MIN_READY_FRAMES = 5           # Reduced to 5 frames to start easily
UNLOCK_RESET_FRAMES = 15             # Reset frames for LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds
MAX_TORSO_UPRIGHT_ANGLE = 150.0      # Standing up past this flags torso cheat
MAX_UPPER_ARM_DROP_ANGLE = 42.0      # Upper arm dropping forward past this flags dropped elbow
MAX_ELBOW_HIKE_ANGLE = 45.0          # Upper arm hiking above rib cage flags overextension
MAX_TORSO_HEAVE_DELTA = 16.0         # Chest swinging / torso heave degrees from baseline
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Forearm free-fall angular speed cutoff (deg/frame)

# 3. Extension Depth & Range of Motion
FULL_LOCKOUT_MIN_ANGLE = 155.0       # Minimum elbow angle required for full triceps lockout
HALF_EXT_THRESHOLD_ANGLE = 130.0     # Extending past 130° but short of 155° awards half rep
EXTENDING_START_ELBOW_ANGLE = 115.0  # Backward extension threshold triggering EXTENDING
RETURNING_START_ELBOW_ANGLE = 145.0  # Dropping below this from LOCKED transitions to RETURNING
EARLY_REVERSAL_DELTA = 8.0           # Early forward descent delta flagging premature reversal
REP_COMPLETE_ELBOW_ANGLE = 100.0     # Return angle to flexed position evaluating rep completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_TORSO = 4
DEBOUNCE_ELBOW_DROP = 4
DEBOUNCE_OVEREXTENSION = 4
DEBOUNCE_CURL = 2
DEBOUNCE_HEAVE = 3
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 2

# 5. ST-GCN Machine Learning Model Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75
DEBOUNCE_ML_FAULT = 5

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class DumbbellKickbackTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.target_reached = False
        self.half_ext_reached = False
        self.max_extension = 0.0
        self.rep_in_progress = False

        # Posture Tripwires
        self.is_side_view = True
        self.torso_hinged = True
        self.elbow_pinned = True
        self.arm_not_overextended = True
        self.wrist_behind_elbow = True
        self.no_torso_heave = True
        self.controlled_eccentric = True
        self.ml_posture_maintained = True

        self.baseline_torso_angle = None

        # Debounce counters
        self.profile_frames = 0
        self.torso_frames = 0
        self.elbow_drop_frames = 0
        self.overextension_frames = 0
        self.curl_detected_frames = 0
        self.heave_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0

        self.standing_frames = 0
        self.unlock_frames = 0
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Hinge forward, pin elbow to ribs, let hand hang straight down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_kickback.tflite")
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
        self.rep_in_progress = False
        self.target_reached = False
        self.half_ext_reached = False
        self.max_extension = 0.0
        self.prev_elbow_angle = None
        self.reversal_frames = 0

        self.is_side_view = True
        self.torso_hinged = True
        self.elbow_pinned = True
        self.arm_not_overextended = True
        self.wrist_behind_elbow = True
        self.no_torso_heave = True
        self.controlled_eccentric = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.torso_frames = 0
        self.elbow_drop_frames = 0
        self.overextension_frames = 0
        self.curl_detected_frames = 0
        self.heave_frames = 0
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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Kickback",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Show full side profile",
                "elbow_angle": 0.0,
                "torso_angle": 0.0,
                "depth_reached": False
            }

        # Select dominant profile facing camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Sagittal kinematics
        torso_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        # Profile angle verification
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        torso_height = abs(active_shoulder.y - active_hip.y) or 0.40
        is_side = (shoulder_span_x / torso_height) < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Upper arm pinned along ribs check
        is_elbow_pinned_properly = upper_arm_angle <= SETUP_UPPER_ARM_PINNED_ANGLE

        # Forearm vertical check: Hand (wrist) directly under the elbow (gravity line)
        forearm_len = np.hypot(active_wrist.x - active_elbow.x, active_wrist.y - active_elbow.y) or 0.25
        forearm_drift = abs(active_wrist.x - active_elbow.x) / forearm_len
        is_forearm_vertical = (forearm_drift <= MAX_FOREARM_VERTICAL_DRIFT) and (active_wrist.y > active_elbow.y)

        # Setup: Hinge torso, arm pinned, hand vertical under elbow, flexed <= 110°
        is_in_hinge = (
            (torso_angle <= SETUP_TORSO_MAX_ANGLE)
            and is_elbow_pinned_properly
            and is_forearm_vertical
            and is_side
            and (elbow_angle <= SETUP_ELBOW_START_MAX_ANGLE)
        )

        # ------------------------------------------------------------------
        # 1. SETUP GUARD
        # ------------------------------------------------------------------
        if is_in_hinge:
            self.standing_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = torso_angle
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * torso_angle

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif torso_angle > SETUP_TORSO_MAX_ANGLE:
                feedback_msg = "Hinge forward at hips (~45°)"
            elif not is_elbow_pinned_properly:
                feedback_msg = "Pin upper arm up against ribs"
            elif not is_forearm_vertical:
                feedback_msg = "Let hand hang straight down vertically under elbow"
            else:
                feedback_msg = "Hold position to start..."

            return {
                "exercise": "Dumbbell Kickback",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "torso_angle": round(torso_angle, 1),
                "depth_reached": False
            }

        # Orientation & Direction Vectors
        torso_vec_x = active_hip.x - active_shoulder.x
        torso_vec_y = active_hip.y - active_shoulder.y
        torso_mag = np.hypot(torso_vec_x, torso_vec_y) or 1e-4
        unit_torso_x = torso_vec_x / torso_mag
        unit_torso_y = torso_vec_y / torso_mag

        forearm_vec_x = active_wrist.x - active_elbow.x
        forearm_vec_y = active_wrist.y - active_elbow.y
        posterior_projection = (forearm_vec_x * unit_torso_x) + (forearm_vec_y * unit_torso_y)

        dist_wrist_to_shoulder = np.hypot(active_wrist.x - active_shoulder.x, active_wrist.y - active_shoulder.y)
        dist_elbow_to_shoulder = np.hypot(active_elbow.x - active_shoulder.x, active_elbow.y - active_shoulder.y)

        # ------------------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ------------------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_in_progress:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["WAITING", "READY", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ------------------------------------------------------------------
        if self.rep_in_progress and self.state in ["EXTENDING", "LOCKED", "RETURNING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Torso standing up cheat
            if torso_angle > MAX_TORSO_UPRIGHT_ANGLE:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    self.torso_hinged = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # Dropping elbow toward floor
            if upper_arm_angle > MAX_UPPER_ARM_DROP_ANGLE and active_elbow.y > active_shoulder.y:
                self.elbow_drop_frames += 1
                if self.elbow_drop_frames >= DEBOUNCE_ELBOW_DROP:
                    self.elbow_pinned = False
            else:
                self.elbow_drop_frames = max(0, self.elbow_drop_frames - 1)

            # Arm Overextension
            is_elbow_overextended = active_elbow.y < (min(active_shoulder.y, active_hip.y) - 0.05)
            if is_elbow_overextended or (upper_arm_angle > MAX_ELBOW_HIKE_ANGLE and active_elbow.y < active_shoulder.y):
                self.overextension_frames += 1
                if self.overextension_frames >= DEBOUNCE_OVEREXTENSION:
                    self.arm_not_overextended = False
            else:
                self.overextension_frames = max(0, self.overextension_frames - 1)

            # Bicep Curl / Anterior Swing Cheat
            is_curling = (
                (dist_wrist_to_shoulder < (dist_elbow_to_shoulder * 0.95))
                or (posterior_projection < 0.0)
                or (active_wrist.y < active_elbow.y and elbow_angle < 125)
            )
            if is_curling:
                self.curl_detected_frames += 1
                if self.curl_detected_frames >= DEBOUNCE_CURL:
                    self.wrist_behind_elbow = False
            else:
                self.curl_detected_frames = max(0, self.curl_detected_frames - 1)

            # Torso heave / momentum check
            if self.baseline_torso_angle is not None:
                if abs(torso_angle - self.baseline_torso_angle) > MAX_TORSO_HEAVE_DELTA:
                    self.heave_frames += 1
                    if self.heave_frames >= DEBOUNCE_HEAVE:
                        self.no_torso_heave = False
                else:
                    self.heave_frames = max(0, self.heave_frames - 1)

            self.max_extension = max(self.max_extension, elbow_angle)

        # ------------------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ------------------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_in_hinge:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "READY"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Pin elbow to ribs and extend back."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Dumbbell Kickback",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "torso_angle": round(torso_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "READY"
            self.rep_in_progress = False
            if self.post_rep_timer == 0:
                self.current_feedback = "Extend your forearm straight back!"

        # Extension begins
        elif self.state in ["READY", "RETURNING"] and elbow_angle > EXTENDING_START_ELBOW_ANGLE:
            if posterior_projection > 0.02 and dist_wrist_to_shoulder > dist_elbow_to_shoulder:
                self.state = "EXTENDING"
                self.rep_in_progress = True
                self.max_extension = elbow_angle
                self.reversal_frames = 0
                if self.post_rep_timer == 0:
                    self.current_feedback = "Lock arm straight back..."

        # Full Lockout (>= 155°)
        elif self.state == "EXTENDING" and elbow_angle >= FULL_LOCKOUT_MIN_ANGLE:
            if posterior_projection > 0.08 and dist_wrist_to_shoulder > dist_elbow_to_shoulder:
                self.target_reached = True
                self.state = "LOCKED"
                if self.post_rep_timer == 0:
                    self.current_feedback = "Full triceps contraction! Lower under control."
            else:
                self.wrist_behind_elbow = False

        # Incomplete Rep reversal (Half rep branch)
        elif self.state == "EXTENDING" and not self.target_reached and (self.max_extension - elbow_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_extension >= HALF_EXT_THRESHOLD_ANGLE:
                    self.half_ext_reached = True
                self.state = "RETURNING"
                self.current_feedback = "Incomplete extension! Returning forearm early..."

        elif self.state == "LOCKED" and elbow_angle < RETURNING_START_ELBOW_ANGLE:
            self.state = "RETURNING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the weight forward"

        # Decay reversal frames if extending normally
        if self.state == "EXTENDING" and not ((self.max_extension - elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop detector with decaying debounce
        if self.rep_in_progress and self.state in ["RETURNING", "EXTENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_eccentric = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ------------------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ------------------------------------------------------------------
        if self.rep_in_progress and self.state in ["RETURNING", "LOCKED", "EXTENDING"] and elbow_angle <= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False
            is_half_rep = False

            # Form cheats check
            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Reset in hinge."
                is_bad_rep = True
            elif not self.wrist_behind_elbow:
                self.current_feedback = "Bad rep: Do not curl! Kick your arm straight back behind you."
                is_bad_rep = True
            elif not self.torso_hinged:
                self.current_feedback = "Bad rep: Stay hinged! Do not stand up to swing."
                is_bad_rep = True
            elif not self.elbow_pinned:
                self.current_feedback = "Bad rep: Keep upper arm pinned high! Do not swing elbow."
                is_bad_rep = True
            elif not self.arm_not_overextended:
                self.current_feedback = "Bad rep: Do not hike elbow past your back! Keep arm along rib cage."
                is_bad_rep = True
            elif not self.no_torso_heave:
                self.current_feedback = "Bad rep: Do not heave your chest up to swing the weight."
                is_bad_rep = True
            elif not self.controlled_eccentric:
                self.current_feedback = "Bad rep: Control the negative! Do not drop forearm."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor mechanics/body sway."
                is_bad_rep = True
            elif not self.target_reached:
                if self.max_extension >= HALF_EXT_THRESHOLD_ANGLE or self.half_ext_reached:
                    self.current_feedback = "Half rep: Full extension not reached! Extend arm back fully."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Movement was too shallow! Lock arm out."
                    is_bad_rep = True

            # Rep counting logic
            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.half_reps += 1
                self.state = "READY"
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean kickback! Good rep."
                self.state = "READY"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((elbow_angle / 155) * 100)))
        if not self.torso_hinged: form_score -= 30
        if not self.elbow_pinned: form_score -= 35
        if not self.arm_not_overextended: form_score -= 25
        if not self.wrist_behind_elbow: form_score -= 30
        if not self.no_torso_heave: form_score -= 25
        if not self.controlled_eccentric: form_score -= 20
        if not self.target_reached and self.rep_in_progress: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Kickback",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "torso_angle": round(torso_angle, 1),
            "depth_reached": self.target_reached
        }