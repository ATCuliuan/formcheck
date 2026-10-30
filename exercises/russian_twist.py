import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class RussianTwistTracker:
    def __init__(self):
        self.state = "CENTER"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.torso_too_upright = False    # Sitting up too straight (losing core tension)
        self.arms_waved = False           # Moving hands without rotating the shoulders
        self.ml_posture_maintained = True # ST-GCN catches leg flailing and balance loss
        
        # Setup Guards
        self.setup_frames = 0
        self.is_ready = False

        self.current_feedback = "Lean back into a V-sit to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_russian_twist.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Russian Twist model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Russian Twist weights: {e}")
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
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Calculate visibility (Need decent view of shoulders and hips to track rotation)
        l_vis = sum([l_shoulder.visibility, l_wrist.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_wrist.visibility, r_hip.visibility, r_knee.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure upper body and knees are visible
        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Russian Twist",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and knees not fully visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera for the V-sit angle
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Core Kinematics
        # 1. Hip Angle tracks the V-Sit lean (Shoulder -> Hip -> Knee)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        
        # 2. Wrist Displacement tracks how far they twist on the X-axis
        mid_wrist_x = (l_wrist.x + r_wrist.x) / 2
        mid_hip_x = (l_hip.x + r_hip.x) / 2
        twist_offset = mid_wrist_x - mid_hip_x
        
        # 3. Shoulder Z-Delta tracks true thoracic rotation. 
        # If shoulders are rotating, one moves closer to the camera (Z decreases) and one moves further (Z increases).
        shoulder_rotation_z = abs(l_shoulder.z - r_shoulder.z)

        # Base Initialization Check (Sitting back with knees bent)
        # A straight sit is ~90 deg. A good V-sit is ~115-135 degrees.
        if 105 < hip_angle < 145 and abs(twist_offset) < 0.05:
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
                "exercise": "Russian Twist",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lean back into a 45-degree V-sit, hands centered",
                "primary_angle": round(hip_angle, 1), "secondary_angle": round(twist_offset, 2), "depth_reached": False
            }

        # --- Tripwire 1: The Upright Sit ---
        # If the hip angle closes tightly, they sat up and lost core tension.
        if hip_angle < 100 and self.state in ["TWISTING", "SIDE_TAP"]:
            self.torso_too_upright = True

        # --- Tripwire 2: The Arm Wave (No Torso Rotation) ---
        # At the peak of the twist, if their shoulders didn't rotate on the Z-axis, they just swung their arms.
        if self.state == "SIDE_TAP" and shoulder_rotation_z < 0.08:
            self.arms_waved = True

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
                if predicted_class == 0 and self.state != "CENTER":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        # Twist Offset threshold: > 0.15 for Right side, < -0.15 for Left side. 
        # Using abs(twist_offset) allows us to track both sides seamlessly.
        
        if abs(twist_offset) < 0.08:
            if self.state == "CENTER_RETURN":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Twist further to the side!"
                    is_bad_rep = True
                elif self.torso_too_upright:
                    self.current_feedback = "No rep: You sat up! Lean back to engage abs."
                    is_bad_rep = True
                elif self.arms_waved:
                    self.current_feedback = "No rep: Rotate your shoulders, don't just swing your arms!"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected leg flailing/balance loss."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Great twist! Obliques engaged."
                
                self.depth_reached = False
                
            self.state = "CENTER"
            # Reset tripwires
            self.torso_too_upright = False 
            self.arms_waved = False
            self.ml_posture_maintained = True 

        elif 0.08 <= abs(twist_offset) < 0.18 and self.state in ["CENTER", "TWISTING", "CENTER_RETURN"]:
            if self.state == "CENTER":
                self.state = "TWISTING"
            self.current_feedback = "Rotate your shoulders with your hands..."

            # Full twist depth reached
            if abs(twist_offset) >= 0.18:
                self.depth_reached = True
                self.state = "SIDE_TAP"
                self.current_feedback = "Tap! Now return to center."

        elif abs(twist_offset) >= 0.18 and self.state in ["SIDE_TAP", "TWISTING"]:
            if self.depth_reached:
                self.state = "CENTER_RETURN"
                self.current_feedback = "Control it back to the middle..."
            else:
                self.current_feedback = "Twist further!"

        # --- Dynamic Form Score ---
        # Perfect V-sit is ~1