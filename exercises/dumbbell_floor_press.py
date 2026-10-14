import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellFloorPressTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.loss_of_verticality = False  # Wrist drifting over face or belly
        self.hips_lifted = False          # Thrusting hips off the floor
        self.ml_posture_maintained = True # ST-GCN catches elbow flare
        self.descent_frames = 0           # Speed tracker to catch bouncing
        
        # Setup Guards
        self.lockout_frames = 0
        self.is_ready = False

        self.current_feedback = "Lie flat and lock dumbbells out over chest"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_floor_press.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Dumbbell Floor Press model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Dumbbell Floor Press weights: {e}")
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure the user's upper body is in frame
        if max(l_vis, r_vis) < 2.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Press",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Move back: Upper body not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Core Kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        
        # Base Initialization Check (Lockout state)
        if elbow_angle > 160:
            self.lockout_frames += 1
        else:
            self.lockout_frames = 0

        if self.lockout_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Press",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lock arms out straight to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(hip_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: Forearm Stacking (Loss of Verticality) ---
        # If the wrist's X-axis drifts too far from the elbow's X-axis, the forearm is tilting dangerously.
        wrist_elbow_offset = abs(active_wrist.x - active_elbow.x)
        if wrist_elbow_offset > 0.08 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.loss_of_verticality = True

        # --- Tripwire 2: Glute Bridge Cheat ---
        if hip_angle > 160 and active_hip.visibility > 0.6:
            self.hips_lifted = True

        # --- Tripwire 3: Bouncing Detection (Tracking descent time) ---
        if self.state == "DESCENDING":
            self.descent_frames += 1

        # ST-GCN LiteRT Buffer (Catches horizontal elbow flare!)
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
                if predicted_class == 0 and self.state != "TOP":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        if elbow_angle > 155:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Triceps didn't touch the floor"
                    is_bad_rep = True
                elif self.loss_of_verticality:
                    self.current_feedback = "No rep: Keep wrists directly over elbows (Danger)"
                    is_bad_rep = True
                elif self.hips_lifted:
                    self.current_feedback = "No rep: Keep your hips flat on the floor"
                    is_bad_rep = True
                elif self.descent_frames > 0 and self.descent_frames < 6:
                    self.current_feedback = "No rep: You bounced off the floor. Control it!"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected elbow flare (Tuck elbows to 45 deg)"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Strong press! Great control."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.loss_of_verticality = False 
            self.hips_lifted = False
            self.descent_frames = 0
            self.ml_posture_maintained = True 

        elif elbow_angle <= 145 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down..."

            # Depth is reached when elbows are at roughly 90 degrees (or lower depending on torso thickness)
            if elbow_angle <= 95:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Pause — Now push up!"

        elif elbow_angle > 95 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Drive the weight up!"
            else:
                self.current_feedback = "Lower until triceps touch floor..."

        # --- Dynamic Form Score ---
        # Perfect lockout is 180, perfect depth is 90. 
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 180) * 0.4)))
        if self.loss_of_verticality: form_score -= 40
        if self.hips_lifted: form_score -= 30
        if self.descent_frames > 0 and self.descent_frames < 6: form_score -= 40
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 50
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Floor Press",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }