import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class RenegadeRowTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.hips_twisted = False         # Opening hips to the side
        self.core_sagged = False          # Hips dropping out of plank alignment
        self.support_arm_bent = False     # Bending the non-rowing arm
        self.ml_posture_maintained = True # ST-GCN catches complex twisting
        
        # Setup Guards
        self.plank_frames = 0
        self.is_ready = False

        self.current_feedback = "Hold a strong pushup plank to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_renegade_row.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Renegade Row model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Renegade Row weights: {e}")
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

        # Visibility guard
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_ankle.visibility])
        
        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Renegade Row",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Move back: Full body not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Dynamic Dual-Arm Kinematics
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # Determine which arm is rowing and which is supporting based on flexion
        if l_elbow_angle < r_elbow_angle:
            rowing_elbow_angle = l_elbow_angle
            support_elbow_angle = r_elbow_angle
        else:
            rowing_elbow_angle = r_elbow_angle
            support_elbow_angle = l_elbow_angle

        # General body tracking for the side most visible to camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Base Initialization Check (Strict Plank)
        if rowing_elbow_angle > 155 and support_elbow_angle > 155 and body_angle > 155:
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
                "exercise": "Renegade Row",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Hold a straight plank with arms locked to begin",
                "primary_angle": round(rowing_elbow_angle, 1), "secondary_angle": round(body_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: The "Lawnmower" Twist ---
        # If the Y-coordinates (height) of the left and right hip drift too far apart, the torso is twisting.
        hip_level_offset = abs(l_hip.y - r_hip.y)
        if hip_level_offset > 0.08 and self.state in ["PULLING", "TOP", "LOWERING"]:
            self.hips_twisted = True

        # --- Tripwire 2: Core Collapse ---
        if body_angle < 150 and self.state != "PLANKING":
            self.core_sagged = True

        # --- Tripwire 3: Support Arm Sag ---
        # The arm planted on the ground MUST remain locked. 
        if support_elbow_angle < 140 and self.state in ["PULLING", "TOP"]:
            self.support_arm_bent = True

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
                if predicted_class == 0 and self.state != "PLANKING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        # Driven entirely by the dynamically detected rowing arm
        if rowing_elbow_angle > 150:
            if self.state == "LOWERING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Pull the weight higher to your ribs"
                    is_bad_rep = True
                elif self.hips_twisted:
                    self.current_feedback = "No rep: You twisted your torso! Keep hips flat to the floor."
                    is_bad_rep = True
                elif self.core_sagged:
                    self.current_feedback = "No rep: Hips dropped. Tighten your core."
                    is_bad_rep = True
                elif self.support_arm_bent:
                    self.current_feedback = "No rep: Keep your supporting arm perfectly straight."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected momentum/form break"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Great rep! Core is locked in."
                
                self.depth_reached = False
                
            self.state = "PLANKING"
            # Reset tripwires
            self.hips_twisted = False 
            self.core_sagged = False
            self.support_arm_bent = False
            self.ml_posture_maintained = True 

        elif rowing_elbow_angle <= 135 and self.state in ["PLANKING", "LOWERING"]:
            self.state = "PULLING"
            self.current_feedback = "Keep hips flat. Pull to ribs..."

            # Full row depth reached when the elbow angle closes tightly (< 95 deg)
            if rowing_elbow_angle <= 95:
                self.depth_reached = True
                self.state = "TOP"
                self.current_feedback = "Squeeze back — Now lower slowly!"

        elif rowing_elbow_angle > 95 and self.state in ["TOP", "PULLING"]:
            if self.depth_reached:
                self.state = "LOWERING"
                self.current_feedback = "Control the drop back to plank..."
            else:
                self.current_feedback = "Pull higher!"

        # --- Dynamic Form Score ---
        # Perfect row pulls to roughly 85-90 degrees at the elbow.
        form_score = max(0, min(100, int(100 - abs(rowing_elbow_angle - 90) * 0.4)))
        if self.hips_twisted: form_score -= 50
        if self.core_sagged: form_score -= 40
        if self.support_arm_bent: form_score -= 30
        if not self.depth_reached and self.state == "LOWERING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Renegade Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(rowing_elbow_angle, 1),
            "secondary_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }