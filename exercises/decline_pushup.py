import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DeclinePushupTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.core_sagged = False
        self.ml_posture_maintained = True  
        self.is_flat_pushup = False
        
        # Setup Guards
        self.plank_frames = 0
        self.is_ready = False

        self.current_feedback = "Get into a decline plank position"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_decline_pushup.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Decline Pushup model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Decline Pushup weights: {e}")
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
        # Extract required joints
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

        # Calculate visibility to determine camera facing side
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure the user is fully in frame
        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Decline Pushup",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Move back: Full body not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematic Angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Base Initialization Check
        if elbow_angle > 150 and body_angle > 150:
            self.plank_frames += 1
        else:
            self.plank_frames = 0

        if self.plank_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Decline Pushup",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Hold straight plank to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(body_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: Core Collapse / Piking ---
        # A straight body is ~170-180 degrees. If it dips below 150 during the movement, the core has collapsed.
        if body_angle < 150 and self.state != "PLANKING":
            self.core_sagged = True

        # --- Tripwire 2: Elevation Check (Are they actually doing a decline?) ---
        # MediaPipe Y-coordinates start at 0 at the top of the screen. 
        # If the ankle Y is greater than (below) the shoulder Y, their feet are on the floor.
        if active_ankle.y > active_shoulder.y:
            self.is_flat_pushup = True
        else:
            self.is_flat_pushup = False

        # ST-GCN LiteRT Buffer (Catches flaring elbows and subtle asymmetries)
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
                if predicted_class == 0 and self.state != "PLANKING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        if elbow_angle > 150:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                # --- Tripwire 3: Depth Evaluation ---
                if not self.depth_reached:
                    self.current_feedback = "No rep: Go down deeper (Half-rep)"
                    is_bad_rep = True
                elif self.core_sagged:
                    self.current_feedback = "No rep: Core collapsed. Squeeze glutes!"
                    is_bad_rep = True
                elif self.is_flat_pushup:
                    self.current_feedback = "No rep: Elevate your feet on a chair!"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected elbow flare/bad form"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect rep! Lock out strong."
                
                self.depth_reached = False
                
            self.state = "PLANKING"
            # Reset tripwires for the next repetition
            self.core_sagged = False 
            self.ml_posture_maintained = True 

        elif elbow_angle <= 135 and self.state in ["PLANKING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Control the drop..."

            # Beginners often stop at 100-110 degrees. We force them down to 90 for full upper chest activation.
            if elbow_angle <= 90:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth hit — Push up!"

        elif elbow_angle > 90 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Push the floor away!"
            else:
                self.current_feedback = "Lower your chest more..."

        # --- Dynamic Form Score ---
        # Perfect depth is ~85-90 degrees.
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4)))
        if self.core_sagged: form_score -= 50
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        if self.is_flat_pushup: form_score -= 50
        form_score = max(0, form_score)

        return {
            "exercise": "Decline Pushup",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }