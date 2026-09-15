import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class PushupTracker:
    def __init__(self):
        self.state = "STANDING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.core_maintained = True  
        self.ml_posture_maintained = True  
        self.descent_frames = 0           
        
        # Setup Guards
        self.plank_frames = 0
        self.is_ready = False

        self.current_feedback = "Get down on the floor"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # TensorFlow Lite Init
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_pushup.tflite")

        if os.path.exists(model_path):
            try:
                # MUST have the tflite. prefix right here:
                self.interpreter = Interpreter(model_path=model_path)
                
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("✅ TFLite Engine Loaded Successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load TFLite model: {e}")
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Dynamic Profile Tracking
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility])

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        total_reps = self.good_reps + self.bad_reps

        # 1. Orientation Check (Standing vs Horizontal)
        dy = abs(active_ankle.y - active_shoulder.y)
        dx = abs(active_ankle.x - active_shoulder.x)
        is_vertical = dy > dx

        if is_vertical:
            self.is_ready = False
            self.plank_frames = 0
            self.state = "STANDING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Push-Up",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Get down on the floor",
                "knee_angle": round(elbow_angle, 1),
                "hip_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # 2. Hard Reset: Dropping to knees
        if body_angle < 130:
            self.is_ready = False
            self.plank_frames = 0
            self.state = "KNEELING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Push-Up",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Straighten legs into a plank",
                "knee_angle": round(elbow_angle, 1),
                "hip_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # 3. Plank Lock
        if elbow_angle > 150 and body_angle > 155:
            self.plank_frames += 1
        else:
            self.plank_frames = 0

        if self.plank_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "PLANK"

        if not self.is_ready:
            return {
                "exercise": "Push-Up",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Hold plank to start",
                "knee_angle": round(elbow_angle, 1),
                "hip_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # 4. Core Sag Tripwire
        if body_angle < 150 and self.state != "PLANK":
            self.core_maintained = False

        if self.state == "DESCENDING":
            self.descent_frames += 1

        # 5. Buffer & AI
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
                if predicted_class == 0:
                    self.ml_posture_maintained = False
            except Exception as e:
                pass

        # 6. State Machine & Coaching
        if elbow_angle > 150:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Go lower (bend elbows past 100°)"
                    is_bad_rep = True
                elif not self.core_maintained:
                    self.current_feedback = "No rep: Hips sagged"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected broken plank"
                    is_bad_rep = True
                
                # Update specific counters
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    if self.descent_frames < 6:
                        self.current_feedback = "Good rep! But lower yourself slower."
                    else:
                        self.current_feedback = "Perfect push-up!"
                
                self.depth_reached = False
                
            self.state = "PLANK"
            self.core_maintained = True 
            self.ml_posture_maintained = True 
            self.descent_frames = 0

        elif elbow_angle <= 140 and self.state in ["PLANK", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Lower under control..."
            
            if elbow_angle <= 100:  
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth good — Push up!"

        elif elbow_angle > 110 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Push the floor away!"
            else:
                self.current_feedback = "Go lower..."

        # 7. Dynamic Form Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4)))
        if not self.core_maintained: form_score -= 50
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        if self.descent_frames > 0 and self.descent_frames < 6: form_score -= 15
        form_score = max(0, form_score)

        total_reps = self.good_reps + self.bad_reps

        return {
            "exercise": "Push-Up",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(elbow_angle, 1),
            "hip_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }