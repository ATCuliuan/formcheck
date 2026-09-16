import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SquatTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.back_posture_maintained = True  
        self.ml_posture_maintained = True  
        self.knees_caved = False          
        self.is_asymmetrical = False      
        self.heels_lifted = False         
        self.descent_frames = 0           
        self.baseline_ankle_y = None      
        
        # Setup Guards
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand upright to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_squat.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Squat model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Squat weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, landmarks[12].visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 2.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: full body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = landmarks[12] if r_vis > l_vis else l_shoulder

        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        if knee_angle > 160:
            self.standing_frames += 1
            self.baseline_ankle_y = active_ankle.y 
        else:
            self.standing_frames = 0

        if self.standing_frames > 20:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand upright to begin",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1), "depth_reached": False
            }

        # Form Tripwires
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)
        if knee_distance < (ankle_distance * 0.55) and self.state in ["DESCENDING", "BOTTOM"]:
            self.knees_caved = True

        if hip_angle < 45 and self.state != "STANDING":
            self.back_posture_maintained = False

        hip_tilt = abs(l_hip.y - r_hip.y)
        if hip_tilt > 0.10 and self.state in ["DESCENDING", "BOTTOM"]:
            self.is_asymmetrical = True

        if self.state == "DESCENDING":
            self.descent_frames += 1
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.05:
                self.heels_lifted = True

        # ST-GCN LiteRT Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0) # Shape: (1, 30, 25, 3)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                predicted_class = np.argmax(output_data, axis=1)[0]
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # State Machine & Coaching Feedback
        if knee_angle > 155:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Squat lower next time"
                    is_bad_rep = True
                elif not self.back_posture_maintained:
                    self.current_feedback = "No rep: Chest fell forward (Danger)"
                    is_bad_rep = True
                elif self.knees_caved:
                    self.current_feedback = "No rep: Knees caved in (Danger)"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected bad posture"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    if self.heels_lifted:
                        self.current_feedback = "Good rep! But keep your heels flat."
                    elif self.is_asymmetrical:
                        self.current_feedback = "Good rep! But watch your hip shift."
                    elif self.descent_frames < 8:
                        self.current_feedback = "Good rep! But slow down the drop."
                    else:
                        self.current_feedback = "Perfect rep!"
                
                self.depth_reached = False
                
            self.state = "STANDING"
            self.back_posture_maintained = True 
            self.ml_posture_maintained = True 
            self.knees_caved = False
            self.is_asymmetrical = False
            self.heels_lifted = False
            self.descent_frames = 0

        elif knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Control the descent..."

            if knee_angle <= 110:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth good — Drive up!"

        elif knee_angle > 110 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Drive up!"
            else:
                self.current_feedback = "Squat lower..."

        # Dynamic Form Score
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))
        if not self.back_posture_maintained or self.knees_caved: form_score -= 50
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        if self.is_asymmetrical: form_score -= 15
        if self.heels_lifted: form_score -= 15
        if self.descent_frames > 0 and self.descent_frames < 8: form_score -= 10
        form_score = max(0, form_score)

        total_reps = self.good_reps + self.bad_reps

        return {
            "exercise": "Bodyweight Squat",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }