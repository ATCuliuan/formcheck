import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class ReverseCurlTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.elbows_drifted = False       # Upper arm moving forward away from ribs
        self.body_swayed = False          # Leaning back or thrusting hips to use momentum
        self.wrong_grip = False           # Palms facing up (Bicep Curl cheat)
        self.ml_posture_maintained = True # ST-GCN catches asymmetrical lifting/shrugging
        
        # Setup Guards
        self.setup_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, arms hanging straight down"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_reverse_curl.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Reverse Curl model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Reverse Curl weights: {e}")
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
        
        # Fingers for grip detection
        l_pinky = landmarks[self.mp_pose.PoseLandmark.LEFT_PINKY.value]
        r_pinky = landmarks[self.mp_pose.PoseLandmark.RIGHT_PINKY.value]
        l_thumb = landmarks[self.mp_pose.PoseLandmark.LEFT_THUMB.value]
        r_thumb = landmarks[self.mp_pose.PoseLandmark.RIGHT_THUMB.value]

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure full side profile is visible
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Reverse Curl",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full side profile not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_pinky = r_pinky if r_vis > l_vis else l_pinky
        active_thumb = r_thumb if r_vis > l_vis else l_thumb

        # Core Kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        shoulder_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Base Initialization Check
        if elbow_angle > 155 and body_angle > 165:
            self.setup_frames += 1
        else:
            self.setup_frames = 0

        if self.setup_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Reverse Curl",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand perfectly straight with arms locked down to start",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(shoulder_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: Elbow Drift (Front Delt Cheat) ---
        if shoulder_angle < 160 and self.state in ["CURLING", "TOP", "LOWERING"]:
            self.elbows_drifted = True

        # --- Tripwire 2: The Hip Thrust (Momentum Swing) ---
        if body_angle < 165 and self.state in ["CURLING", "TOP", "LOWERING"]:
            self.body_swayed = True
            
        # --- Tripwire 3: The Grip Check (Bicep Curl Cheat) ---
        # Z-axis is negative toward the camera. If thumb.z is significantly lower (more negative)
        # than pinky.z, the thumb is on the outside, meaning the palms are facing up.
        if active_thumb.z < active_pinky.z - 0.02 and self.state in ["CURLING", "TOP"]:
            self.wrong_grip = True

        # ST-GCN LiteRT Buffer
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
                if predicted_class == 0 and self.state != "BOTTOM":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        if elbow_angle > 155:
            if self.state == "LOWERING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Curl it higher! Squeeze the forearms."
                    is_bad_rep = True
                elif self.wrong_grip:
                    self.current_feedback = "No rep: Wrong grip! Turn your palms facing down."
                    is_bad_rep = True
                elif self.elbows_drifted:
                    self.current_feedback = "No rep: Pin your elbows to your ribs! Don't let them drift forward."
                    is_bad_rep = True
                elif self.body_swayed:
                    self.current_feedback = "No rep: Stop swinging! Use strict forearm power."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected shoulder shrugging/instability."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect strict curl! Excellent isolation."
                
                self.depth_reached = False
                
            self.state = "BOTTOM"
            # Reset tripwires
            self.elbows_drifted = False 
            self.body_swayed = False
            self.wrong_grip = False
            self.ml_posture_maintained = True 

        elif elbow_angle <= 145 and self.state in ["BOTTOM", "CURLING"]:
            self.state = "CURLING"
            self.current_feedback = "Curl it up, keep elbows pinned..."

            # Full curl depth reached when elbows close tightly
            if elbow_angle <= 70:
                self.depth_reached = True
                self.state = "TOP"
                self.current_feedback = "Squeeze! Now control the drop."

        elif elbow_angle > 70 and self.state in ["TOP", "CURLING"]:
            if self.depth_reached:
                self.state = "LOWERING"
                self.current_feedback = "Lower it slowly to a dead hang..."
            else:
                self.current_feedback = "Curl higher!"

        # --- Dynamic Form Score ---
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 180) * 0.4)))
        if self.wrong_grip: form_score -= 50
        if self.elbows_drifted: form_score -= 40
        if self.body_swayed: form_score -= 50
        if not self.depth_reached and self.state == "LOWERING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Reverse Curl",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(shoulder_angle, 1),
            "depth_reached": self.depth_reached
        }