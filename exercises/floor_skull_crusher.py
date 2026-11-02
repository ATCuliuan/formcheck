import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class FloorSkullCrusherTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility
        
        self.full_depth_reached = False    # Bottom flexion (elbow <= 75°)
        self.min_elbow_angle = 180.0
        
        # Posture Tripwires (SIDE VIEW - LYING DOWN)
        self.is_side_view = True           # Profile camera orientation check
        self.upper_arm_vertical = True     # Upper arm drifts forward toward knees (< 68°)
        self.shoulders_stable = True       # Catches bench press cheat (dropping elbows below shoulders)
        self.controlled_descent = True     # Catches free-falling weight toward forehead
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.arm_drift_frames = 0
        self.press_cheat_frames = 0
        self.drop_frames = 0
        self.setup_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Lie flat sideways, arms straight up above chest"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception:
                self.interpreter = None
        
        model_path = os.path.join("models", "tflite", "st_gcn_floor_skull_crusher.tflite")
        

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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Floor Skull Crusher", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Show full side profile lying down",
                "elbow_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        # Sagittal Joint Kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)

        # Profile & Horizontal Floor Checks
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < 0.42

        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        is_lying_horizontal = dx_torso > (dy_torso * 0.85)

        # Baseline: Lying horizontal with upper arms pointing up (~75°-115°) and elbows locked
        is_at_top = (elbow_angle >= 154) and (75 <= upper_arm_angle <= 115) and is_side and is_lying_horizontal
        if is_at_top:
            self.setup_frames += 1
        else:
            self.setup_frames = 0

        if self.setup_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Lie flat on floor, lock arms straight over chest"
            return {
                "exercise": "Floor Skull Crusher", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "depth_reached": False
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
                if np.argmax(out, axis=1)[0] == 0 and self.state != "TOP":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # Kinematic Tripwires
        if self.state in ["LOWERING", "BOTTOM", "PRESSING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Upper arm drifting forward toward knees (< 68°)
            if upper_arm_angle < 68.0:
                self.arm_drift_frames += 1
                if self.arm_drift_frames >= 4:
                    self.upper_arm_vertical = False
            else:
                self.arm_drift_frames = max(0, self.arm_drift_frames - 1)

            # Bench press cheat: elbows dropping down toward floor/torso
            if active_elbow.y > active_shoulder.y:
                self.press_cheat_frames += 1
                if self.press_cheat_frames >= 4:
                    self.shoulders_stable = False
            else:
                self.press_cheat_frames = max(0, self.press_cheat_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # State Machine (Strict No Half-Reps)
        if self.state == "WAITING" and self.is_ready:
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Bend elbows, lower weight toward forehead"

        elif self.state in ["TOP", "PRESSING"] and elbow_angle < 145:
            self.state = "LOWERING"
            self.min_elbow_angle = elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower to 90 degrees or deeper..."

        # Full Depth: Elbow reaches <= 75°
        elif self.state == "LOWERING" and elbow_angle <= 75:
            self.full_depth_reached = True
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full depth! Extend straight up."

        # Incomplete Rep reversal before hitting depth
        elif self.state == "LOWERING" and (elbow_angle - self.min_elbow_angle) > 8:
            self.state = "PRESSING"
            self.current_feedback = "Perform a full rep! Bend elbows to 90° or deeper."

        elif self.state == "BOTTOM" and elbow_angle > 85:
            self.state = "PRESSING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock out at the top..."

        # Eccentric drop detector
        if self.state in ["LOWERING", "PRESSING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Completion: Returns to top lockout (>= 154°)
        if self.state in ["PRESSING", "BOTTOM", "LOWERING"] and elbow_angle >= 154:
            is_bad_rep = False
            if not self.is_side_view:
                self.current_feedback = "No rep: Lie sideways to camera!"
                is_bad_rep = True
            elif not self.upper_arm_vertical:
                self.current_feedback = "No rep: Keep upper arms fixed! Do not swing elbows forward."
                is_bad_rep = True
            elif not self.shoulders_stable:
                self.current_feedback = "No rep: Do not bench press! Rotate only at the elbow joint."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not drop weights toward face."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor stability/kinetic breakdown."
                is_bad_rep = True
            elif not self.full_depth_reached:
                self.current_feedback = "No rep: Incomplete range! Lower weight fully to forehead level."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean skull crusher rep!"

            self.post_rep_timer = 40
            self.state = "TOP"
            self.full_depth_reached = False
            self.min_elbow_angle = 180.0
            self.is_side_view = True
            self.upper_arm_vertical = True
            self.shoulders_stable = True
            self.controlled_descent = True
            self.ml_posture_maintained = True
            self.profile_frames = 0
            self.arm_drift_frames = 0
            self.press_cheat_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 75) * 0.5)))
        if not self.upper_arm_vertical: form_score -= 35
        if not self.shoulders_stable: form_score -= 30
        if not self.controlled_descent: form_score -= 20
        if not self.full_depth_reached and self.state == "PRESSING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Floor Skull Crusher", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "depth_reached": self.full_depth_reached
        }