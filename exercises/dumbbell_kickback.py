import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellKickbackTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility
        
        self.target_reached = False        # Full extension lockout (elbow >= 155°)
        self.max_extension = 0.0
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Profile camera angle check
        self.torso_hinged = True           # Catches standing too upright (> 148°)
        self.elbow_pinned = True           # Catches dropping elbow towards floor / shoulder swinging
        self.no_torso_heave = True         # Catches swinging chest to generate momentum
        self.controlled_eccentric = True   # Catches free-falling the forearm
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Baseline torso angle during static hinge setup
        self.baseline_torso_angle = None

        # Debounce counters
        self.profile_frames = 0
        self.torso_frames = 0
        self.elbow_drop_frames = 0
        self.heave_frames = 0
        self.standing_frames = 0
        self.drop_frames = 0
        
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Stand sideways in a hinge, pin elbow to torso"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
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

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Kickback", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Show full side profile",
                "elbow_angle": 0.0, "torso_angle": 0.0, "depth_reached": False
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
        torso_height = abs(active_shoulder.y - active_hip.y) or 0.4
        is_side = (shoulder_span_x / torso_height) < 0.42

        # Upper arm pinned: angle should stay tight to torso (< 32°)
        is_elbow_pinned_properly = upper_arm_angle <= 32.0

        # Setup: Hinge torso (< 145°) and upper arm pinned back
        is_in_hinge = (torso_angle <= 145) and is_elbow_pinned_properly and is_side
        if is_in_hinge:
            self.standing_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = torso_angle
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * torso_angle
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Hinge at hips (~45°), pin elbow to your side"
            return {
                "exercise": "Dumbbell Kickback", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "torso_angle": round(torso_angle, 1), "depth_reached": False
            }

        # ST-GCN Temporal Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)
        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(out, axis=1)[0] == 0 and self.state != "READY":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # Kinematic Tripwires
        if self.state in ["EXTENDING", "LOCKED", "RETURNING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4: self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Torso standing up cheat (> 148°)
            if torso_angle > 148.0:
                self.torso_frames += 1
                if self.torso_frames >= 4: self.torso_hinged = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # Dropping elbow toward floor (upper arm angle swings forward > 35°)
            if upper_arm_angle > 35.0:
                self.elbow_drop_frames += 1
                if self.elbow_drop_frames >= 4: self.elbow_pinned = False
            else:
                self.elbow_drop_frames = max(0, self.elbow_drop_frames - 1)

            # Torso heave / momentum check
            if self.baseline_torso_angle is not None:
                if abs(torso_angle - self.baseline_torso_angle) > 15.0:
                    self.heave_frames += 1
                    if self.heave_frames >= 3: self.no_torso_heave = False
                else:
                    self.heave_frames = max(0, self.heave_frames - 1)

            self.max_extension = max(self.max_extension, elbow_angle)

        # State Machine (No Half-Reps - Full Lockout Only)
        if self.state == "WAITING" and self.is_ready:
            self.state = "READY"
            if self.post_rep_timer == 0:
                self.current_feedback = "Extend your forearm straight back!"

        elif self.state in ["READY", "RETURNING"] and elbow_angle > 105:
            self.state = "EXTENDING"
            self.max_extension = elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock arm straight back..."

        # Full Lockout (>= 155°)
        elif self.state == "EXTENDING" and elbow_angle >= 155:
            self.target_reached = True
            self.state = "LOCKED"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full triceps contraction! Lower under control."

        # Early downward reversal before reaching lockout (Incomplete Rep)
        elif self.state == "EXTENDING" and (self.max_extension - elbow_angle) > 8:
            self.state = "RETURNING"
            self.current_feedback = "Perform a full rep! Lock elbow out completely."

        elif self.state == "LOCKED" and elbow_angle < 145:
            self.state = "RETURNING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the weight forward"

        # Eccentric drop detector
        if self.state in ["RETURNING", "EXTENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2: self.controlled_eccentric = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Evaluation: Return to ~90° flexion (<= 95°)
        if self.state in ["RETURNING", "LOCKED", "EXTENDING"] and elbow_angle <= 95:
            is_bad_rep = False
            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways to the camera!"
                is_bad_rep = True
            elif not self.torso_hinged:
                self.current_feedback = "No rep: Stay hinged! Do not stand up to swing."
                is_bad_rep = True
            elif not self.elbow_pinned:
                self.current_feedback = "No rep: Keep upper arm pinned high! Do not swing elbow."
                is_bad_rep = True
            elif not self.no_torso_heave:
                self.current_feedback = "No rep: Do not heave your chest up to swing the weight."
                is_bad_rep = True
            elif not self.controlled_eccentric:
                self.current_feedback = "No rep: Control the negative! Do not drop forearm."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor mechanics/body sway."
                is_bad_rep = True
            elif not self.target_reached:
                self.current_feedback = "No rep: Perform a full rep! Lock your arm out."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean kickback! Good rep."

            self.post_rep_timer = 40
            self.state = "READY"
            self.target_reached = False
            self.max_extension = 0.0
            self.is_side_view = True
            self.torso_hinged = True
            self.elbow_pinned = True
            self.no_torso_heave = True
            self.controlled_eccentric = True
            self.ml_posture_maintained = True
            self.prev_elbow_angle = None
            self.profile_frames = 0
            self.torso_frames = 0
            self.elbow_drop_frames = 0
            self.heave_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        form_score = max(0, min(100, int((elbow_angle / 155) * 100)))
        if not self.torso_hinged: form_score -= 30
        if not self.elbow_pinned: form_score -= 35
        if not self.no_torso_heave: form_score -= 25
        if not self.controlled_eccentric: form_score -= 20
        if not self.target_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Kickback", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "torso_angle": round(torso_angle, 1), "depth_reached": self.target_reached
        }