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
        self.pull_reached = False
        
        # Instructor Tripwires
        self.torso_stable = True          # Catches the "Ego Yank" (standing up)
        self.shoulders_square = True      # Catches the "Lawnmower" twist
        self.ml_posture_maintained = True # AI catches bicep-rowing and spine rounding
        
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Lean over a bench or chair to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_one_arm_row.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite One Arm Row model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite One Arm Row weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        # Full 25-joint extraction for ST-GCN consistency
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

        # Dynamic Visibility Check: Find the side facing the camera (the working arm)
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "One Arm Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show your working arm and torso",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_hinge = calculate_angle(active_shoulder, active_hip, active_knee)

        # 1. Starting Guard: Arm must be extended, torso hinged forward
        if elbow_angle > 150 and hip_hinge < 135:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "One Arm Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hinge forward and let the weight hang",
                "knee_angle": round(elbow_angle, 1), "hip_angle": round(hip_hinge, 1)
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "EXTENDED":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # The Ego Yank (Standing up during the pull)
        if hip_hinge > 140 and self.state in ["PULLING", "TOP"]:
            self.torso_stable = False

        # The Lawnmower Twist (Shoulder Rotation)
        # If the working shoulder flies significantly higher than the resting shoulder, they are twisting.
        shoulder_twist = abs(l_shoulder.y - r_shoulder.y)
        if shoulder_twist > 0.20 and self.state in ["PULLING", "TOP"]:
            self.shoulders_square = False

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Pull the weight to your hip!"

        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < 140:
            self.state = "PULLING"
            self.current_feedback = "Drive the elbow back..."

        elif self.state == "PULLING" and elbow_angle <= 85:
            self.pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze the lat! Lower slowly."

        elif self.state == "TOP" and elbow_angle > 100:
            self.state = "LOWERING"
            self.current_feedback = "Control the descent"

        elif self.state == "LOWERING" and elbow_angle >= 150:
            is_bad_rep = False
            
            # Form Evaluation
            if not self.pull_reached:
                self.current_feedback = "No rep: Pull the elbow higher (past your torso)."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: You stood up! Keep your chest facing the floor."
                is_bad_rep = True
            elif not self.shoulders_square:
                self.current_feedback = "No rep: You twisted your spine (Lawnmower pull). Keep shoulders square."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture or a bicep-row."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect row!"
                
            # Reset for next rep
            self.state = "EXTENDED"
            self.pull_reached = False
            self.torso_stable = True
            self.shoulders_square = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(( (160 - elbow_angle) / 75 ) * 100)))
        if not self.pull_reached and self.state == "LOWERING": form_score -= 30
        if not self.torso_stable: form_score -= 35
        if not self.shoulders_square: form_score -= 35
        if not self.ml_posture_maintained: form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "One Arm Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(elbow_angle, 1), # Re-using key for elbow text output in UI
            "hip_angle": round(hip_hinge, 1)
        }