import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class PlankTricepsExtensionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0
        
        self.lockout_reached = False       # Full arm lockout at top (elbow >= 155°)
        self.max_elbow = 0.0
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.plank_rigid = True            # Catches hip sag (< 150°) or hip pike (< 145°)
        self.controlled_eccentric = True   # Catches slamming forearms onto ground
        self.ml_posture_maintained = True

        self.plank_sag_frames = 0
        self.drop_frames = 0
        self.setup_frames = 0
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Hold a solid forearm plank sideways to begin"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_plank_triceps_extension.tflite")
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
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Plank Triceps Extension", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Show full side plank profile",
                "elbow_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        plank_body_line = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Baseline: Forearm plank (elbows ~80°-100° and body line 160°-180°)
        is_plank = (75 <= elbow_angle <= 105) and (plank_body_line >= 155)
        if is_plank:
            self.setup_frames += 1
        else:
            self.setup_frames = 0

        if self.setup_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Plank Triceps Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Hold solid forearm plank to begin",
                "elbow_angle": round(elbow_angle, 1), "hip_angle": round(plank_body_line, 1), "depth_reached": False
            }

        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # Kinematic Tripwires
        if self.state in ["PRESSING_UP", "TOP_SUPPORT", "LOWERING"]:
            # Hip sag or pike
            if plank_body_line < 150:
                self.plank_sag_frames += 1
                if self.plank_sag_frames >= 4: self.plank_rigid = False
            else:
                self.plank_sag_frames = max(0, self.plank_sag_frames - 1)

            self.max_elbow = max(self.max_elbow, elbow_angle)

        # State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "FOREARM_PLANK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Press through palms to lift elbows off ground!"

        elif self.state in ["FOREARM_PLANK", "LOWERING"] and elbow_angle > 110:
            self.state = "PRESSING_UP"
            self.max_elbow = elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Press arms straight..."

        # Full ROM: Elbows locked out in high plank (>= 155°)
        elif self.state == "PRESSING_UP" and elbow_angle >= 155:
            self.lockout_reached = True
            self.state = "TOP_SUPPORT"
            if self.post_rep_timer == 0:
                self.current_feedback = "Locked out! Lower elbows back down slowly."

        # Incomplete Rep reversal
        elif self.state == "PRESSING_UP" and (self.max_elbow - elbow_angle) > 8:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Push all the way up to locked arms."

        elif self.state == "TOP_SUPPORT" and elbow_angle < 145:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent to forearms"

        # Eccentric drop monitor
        if self.state in ["LOWERING", "PRESSING_UP"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2: self.controlled_eccentric = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Completion: Forearms back on floor (elbow <= 95°)
        if self.state in ["LOWERING", "TOP_SUPPORT", "PRESSING_UP"] and elbow_angle <= 95:
            is_bad_rep = False
            if not self.plank_rigid:
                self.current_feedback = "No rep: Core collapsed! Do not sag or pike your hips."
                is_bad_rep = True
            elif not self.controlled_eccentric:
                self.current_feedback = "No rep: Control the descent! Do not slam elbows down."
                is_bad_rep = True
            elif not self.lockout_reached:
                self.current_feedback = "No rep: Incomplete push! Lock elbows out fully at the top."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Solid plank triceps extension!"

            self.post_rep_timer = 40
            self.state = "FOREARM_PLANK"
            self.lockout_reached = False
            self.max_elbow = 0.0
            self.plank_rigid = True
            self.controlled_eccentric = True
            self.plank_sag_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        form_score = max(0, min(100, int((elbow_angle / 155) * 100)))
        if not self.plank_rigid: form_score -= 35
        if not self.controlled_eccentric: form_score -= 25
        if not self.lockout_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Plank Triceps Extension", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "hip_angle": round(plank_body_line, 1), "depth_reached": self.lockout_reached
        }