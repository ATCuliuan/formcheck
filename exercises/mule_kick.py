import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class MuleKickTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        
        # Instructor Tripwires
        self.target_height_reached = False
        self.hips_squared = True          # Tracks the "Fire Hydrant" twist
        self.ml_posture_maintained = True # Tracks the "Lumbar Crank" (lower back arch)
        self.is_ready = False
        self.tabletop_frames = 0

        self.current_feedback = "Get on your hands and knees"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_mule_kick.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Mule Kick model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Mule Kick weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_shoulder.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 2.0: 
            self.frame_buffer.clear()
            return {
                "exercise": "Mule Kick", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: full body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Select the active side (the side facing the camera)
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder

        # Calculate standard angles
        # In tabletop, shoulder to hip to knee is ~90 degrees. 
        # When kicking up, it opens up to ~150-180 degrees.
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        
        # 1. Tabletop Initialization
        vertical_drop = abs(active_shoulder.y - active_hip.y)
        is_horizontal = vertical_drop < 0.25 # Torso is parallel to the floor

        if is_horizontal and hip_angle < 110:
            self.tabletop_frames += 1
        else:
            self.tabletop_frames = 0

        if self.tabletop_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Mule Kick", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Get into a tabletop position",
                "knee_angle": 0.0, "hip_angle": round(hip_angle, 1)
            }

        # 2. ML Buffer (Looking for the Lumbar Crank)
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # If the AI detects the lower back sagging
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "TABLETOP":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # 3. Kinematic Tripwires
        
        # The Fire Hydrant Detector: Are the hips level?
        # If the Y distance between the left and right hip is too large, they are twisting.
        hip_twist = abs(l_hip.y - r_hip.y)
        if hip_twist > 0.12 and self.state in ["KICKING", "TOP"]:
            self.hips_squared = False

        # 4. State Machine
        if self.state == "TABLETOP" and hip_angle > 115:
            self.state = "KICKING"
            self.current_feedback = "Drive the heel up!"
            
        elif self.state == "KICKING" and hip_angle > 150:
            self.state = "TOP"
            self.target_height_reached = True
            self.current_feedback = "Squeeze the glute, slowly lower..."

        elif self.state in ["KICKING", "TOP"] and hip_angle < 130:
            self.state = "RETURNING"
            self.current_feedback = "Control the descent..."

        elif self.state == "RETURNING" and hip_angle < 110:
            is_bad_rep = False
            
            # Form Evaluation
            if not self.target_height_reached:
                self.current_feedback = "No rep: Kick higher (thigh parallel to floor)."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: You arched your lower back! Core tight."
                is_bad_rep = True
            elif not self.hips_squared:
                self.current_feedback = "No rep: Keep your hips square to the floor."
                is_bad_rep = True
            
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect kick!"
            
            # Reset for the next leg pump
            self.state = "TABLETOP"
            self.target_height_reached = False
            self.ml_posture_maintained = True
            self.hips_squared = True
            self.frame_buffer.clear()

        # Dynamic Form Score
        form_score = 100
        if not self.target_height_reached and self.state == "RETURNING": form_score -= 40
        if not self.ml_posture_maintained: form_score -= 50
        if not self.hips_squared: form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Mule Kick", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": 0.0, "hip_angle": round(hip_angle, 1)
        }