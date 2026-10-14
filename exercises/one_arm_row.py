import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class OneArmRowTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.pull_reached = False          # Full pull flag (elbow <= 85 deg)
        self.min_elbow_angle = 180.0       # Tracks deepest point of the pull
        
        # Instructor Tripwires (STRICT SIDE / SAGITTAL VIEW)
        self.torso_stable = True           # Catches standing up during pull
        self.torso_not_twisted = True      # Catches rotating/opening chest upward
        self.neck_neutral = True           # Catches craning the neck upward
        self.controlled_descent = True     # Catches free-falling the weight
        self.pull_to_hip = True            # Catches pulling straight up to armpit (bicep curl row)
        self.forearm_path_correct = True   # Catches curling wrist inward excessively
        self.ml_posture_maintained = True  # AI catches lumbar flexion

        # Debounce counters
        self.drop_frames = 0
        self.forearm_frames = 0
        self.twist_frames = 0
        self.torso_frames = 0
        self.neck_frames = 0
        self.trajectory_frames = 0
        self.ml_bad_frames = 0

        self.prev_elbow_angle = None
        self.ready_frames = 0
        self.is_ready = False
        self.baseline_hip_hinge = None
        self.baseline_shoulder_separation_x = None

        self.current_feedback = "Hinge forward over bench (side view) to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_one_arm_row.tflite")

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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.8:
            self.frame_buffer.clear()
            return {
                "exercise": "One Arm Row",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Show full side profile of working arm",
                "elbow_angle": 0.0,
                "hip_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        support_shoulder = l_shoulder if r_vis > l_vis else r_shoulder

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_hinge = calculate_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        torso_len = abs(active_shoulder.y - active_hip.y)
        if torso_len == 0:
            torso_len = 0.40

        dx_forearm = abs(active_wrist.x - active_elbow.x)
        dy_forearm = abs(active_wrist.y - active_elbow.y)
        forearm_tilt = np.degrees(np.arctan2(dx_forearm, dy_forearm)) if dy_forearm > 0 else 90.0

        shoulder_sep_x = abs(active_shoulder.x - support_shoulder.x) / torso_len

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if elbow_angle > 128 and hip_hinge < 145:
            self.ready_frames += 1
            if self.baseline_hip_hinge is None:
                self.baseline_hip_hinge = hip_hinge
            else:
                self.baseline_hip_hinge = 0.9 * self.baseline_hip_hinge + 0.1 * hip_hinge

            if self.baseline_shoulder_separation_x is None:
                self.baseline_shoulder_separation_x = shoulder_sep_x
            else:
                self.baseline_shoulder_separation_x = 0.9 * self.baseline_shoulder_separation_x + 0.1 * shoulder_sep_x
        else:
            self.ready_frames = 0

        if self.ready_frames > 5:  
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "One Arm Row",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Hinge forward over bench and hang arm down",
                "elbow_angle": round(elbow_angle, 1),
                "hip_angle": round(hip_hinge, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN BUFFER & INFERENCE
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_vals = np.exp(output_data - np.max(output_data))
                probs = exp_vals / np.sum(exp_vals)

                if probs[0] > 0.85 and self.state in ["PULLING", "TOP"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= 4:
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["EXTENDED", "PULLING", "TOP", "LOWERING"]:
            # A. Standing Up Check
            standing_up_detected = False
            if self.baseline_hip_hinge is not None:
                if (hip_hinge - self.baseline_hip_hinge) > 22.0 or hip_hinge > 152.0:
                    standing_up_detected = True
            elif hip_hinge > 152.0:
                standing_up_detected = True

            if standing_up_detected:
                self.torso_frames += 1
                if self.torso_frames >= 4:
                    self.torso_stable = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # B. Chest Rotation Check (Loosened to accommodate perspective drift)
            shoulder_vertical_skew = (support_shoulder.y - active_shoulder.y) / torso_len
            
            delta_shoulder_sep = 0.0
            if self.baseline_shoulder_separation_x is not None:
                delta_shoulder_sep = shoulder_sep_x - self.baseline_shoulder_separation_x

            # Relaxed thresholds: 0.28 vertical skew, 0.35 horizontal delta, 5 debounce frames
            if shoulder_vertical_skew > 0.70 or delta_shoulder_sep > 0.85:
                self.twist_frames += 1
                if self.twist_frames >= 5:
                    self.torso_not_twisted = False
            else:
                self.twist_frames = max(0, self.twist_frames - 1)

            # C. Neck Hyperextension
            if neck_angle < 125:
                self.neck_frames += 1
                if self.neck_frames >= 3:
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            # D. Pull Trajectory
            if self.state == "TOP":
                wrist_to_hip_x = abs(active_wrist.x - active_hip.x) / torso_len
                wrist_to_shoulder_x = abs(active_wrist.x - active_shoulder.x) / torso_len
                if wrist_to_shoulder_x < 0.12 and wrist_to_hip_x > 0.35:
                    self.trajectory_frames += 1
                    if self.trajectory_frames >= 2:
                        self.pull_to_hip = False

            # E. Forearm Path
            if forearm_tilt >= 52.0:
                self.forearm_frames += 1
                if self.forearm_frames >= 3:
                    self.forearm_path_correct = False
            else:
                self.forearm_frames = max(0, self.forearm_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Drive elbow back toward your hip!"

        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < 128:
            self.state = "PULLING"
            self.min_elbow_angle = elbow_angle
            self.current_feedback = "Drive elbow up and back..."

        # Full contraction reached (elbow <= 85 deg)
        elif self.state == "PULLING" and elbow_angle <= 85:
            self.pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze lat! Lower under control."

        # Early downward reversal before full contraction
        elif self.state == "PULLING" and (elbow_angle - self.min_elbow_angle) > 8:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Drive elbow all the way back."

        elif self.state == "TOP" and elbow_angle > 95:
            self.state = "LOWERING"
            self.current_feedback = "Lower weight under control..."

        # Eccentric Free-Fall Monitor
        if self.state in ["LOWERING", "PULLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 30.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "PULLING"] and elbow_angle >= 132:
            is_bad_rep = False

            if not self.torso_stable:
                self.current_feedback = "No rep: Do not stand up! Keep chest facing the bench."
                is_bad_rep = True
            elif not self.torso_not_twisted:
                self.current_feedback = "No rep: Do not twist chest open! Keep shoulders squared."
                is_bad_rep = True
            elif not self.pull_to_hip:
                self.current_feedback = "No rep: Pull back to your hip, not up to your armpit."
                is_bad_rep = True
            elif not self.forearm_path_correct:
                self.current_feedback = "No rep: Do not curl the weight! Drive with your elbow."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Keep neck neutral! Look down at bench, not up."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower the weight slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor spinal mechanics."
                is_bad_rep = True
            elif not self.pull_reached:
                self.current_feedback = "No rep: Perform a full rep! Drive elbow all the way up."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Great one-arm row! Clean repetition."

            # Reset state for next rep
            self.state = "EXTENDED"
            self.pull_reached = False
            self.min_elbow_angle = 180.0

            self.drop_frames = 0
            self.forearm_frames = 0
            self.twist_frames = 0
            self.torso_frames = 0
            self.neck_frames = 0
            self.trajectory_frames = 0
            self.ml_bad_frames = 0

            self.torso_stable = True
            self.torso_not_twisted = True
            self.neck_neutral = True
            self.controlled_descent = True
            self.pull_to_hip = True
            self.forearm_path_correct = True
            self.ml_posture_maintained = True
            self.prev_elbow_angle = None

            if self.baseline_hip_hinge is not None:
                self.baseline_hip_hinge = 0.85 * self.baseline_hip_hinge + 0.15 * hip_hinge

            self.frame_buffer.clear()

        return {
            "exercise": "One Arm Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "hip_angle": round(hip_hinge, 1)
        }