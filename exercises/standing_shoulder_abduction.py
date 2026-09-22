import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingShoulderAbductionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.target_height_reached = False
        
        # Posture Tripwires
        self.arms_bent = False            # Did they bend elbows to cheat the lever arm?
        self.is_asymmetrical = False      # Did one arm lift significantly higher than the other?
        self.torso_swayed = False         # Did they use body momentum to swing the arms?
        self.ml_posture_maintained = True # ST-GCN AI flag
        
        # Setup Guards
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, arms at your sides to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_abduction.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Shoulder Abduction model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Shoulder Abduction weights: {e}")
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
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        total_vis = sum([l_shoulder.visibility, r_shoulder.visibility, l_elbow.visibility, r_elbow.visibility, l_wrist.visibility, r_wrist.visibility])
        total_reps = self.good_reps + self.bad_reps

        if total_vis < 4.5: 
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Abduction",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full upper body not visible",
                "abduction_angle": 0.0, "depth_reached": False
            }

        # Calculate Abduction Angles (Arm elevation relative to the torso)
        # We use Hip -> Shoulder -> Elbow to measure how high the arm is raised laterally
        l_abduction_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_abduction = (l_abduction_angle + r_abduction_angle) / 2

        # Calculate Elbow Angles (Shoulder -> Elbow -> Wrist) to catch T-Rex arms
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Baseline: Arms relaxed at the sides
        if avg_abduction < 25:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 10:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Abduction",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Rest arms at your sides to begin",
                "abduction_angle": round(avg_abduction, 1), "depth_reached": False
            }

        # Form Tripwires (Evaluated during the lift)
        if self.state in ["LIFTING", "PEAK"]:
            # 1. T-Rex Arms: Bending elbows to cheat the lever
            if l_elbow_angle < 150 or r_elbow_angle < 150:
                self.arms_bent = True
                
            # 2. See-Saw Lift: One arm lagging significantly behind the other
            if abs(l_abduction_angle - r_abduction_angle) > 25:
                self.is_asymmetrical = True

            # 3. Torso Sway: Using body momentum. 
            # We track the lateral shift of the mid-shoulder point relative to the mid-hip point.
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
            mid_hip_x = (l_hip.x + r_hip.x) / 2
            if abs(mid_shoulder_x - mid_hip_x) > 0.06:  # Strict vertical alignment required
                self.torso_swayed = True

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
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # State Machine
        if avg_abduction < 35:
            if self.state == "LOWERING":
                is_bad_rep = False
                
                if not self.target_height_reached:
                    self.current_feedback = "No rep: Raise arms to shoulder height"
                    is_bad_rep = True
                elif self.arms_bent:
                    self.current_feedback = "No rep: Keep your arms straight"
                    is_bad_rep = True
                elif self.is_asymmetrical:
                    self.current_feedback = "No rep: Both arms must lift equally"
                    is_bad_rep = True
                elif self.torso_swayed:
                    self.current_feedback = "No rep: Stop swinging your torso"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected shoulder shrugging/form breakdown"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect lift!"
                
                # Reset tripwires
                self.target_height_reached = False
                self.arms_bent = False
                self.is_asymmetrical = False
                self.torso_swayed = False
                self.ml_posture_maintained = True
                
            self.state = "STANDING"

        elif avg_abduction >= 35 and self.state in ["STANDING", "LOWERING"]:
            self.state = "LIFTING"
            self.current_feedback = "Raise arms straight out..."

            # Target height logic: Aiming for 80+ degrees (roughly parallel to the floor)
            if l_abduction_angle >= 80 and r_abduction_angle >= 80:
                self.target_height_reached = True
                self.state = "PEAK"
                self.current_feedback = "Great height — Lower slowly!"

        elif avg_abduction < 80 and self.state in ["PEAK", "LIFTING"]:
            if self.target_height_reached:
                self.state = "LOWERING"
                self.current_feedback = "Control the descent..."
            else:
                self.current_feedback = "Lift higher..."

        # Dynamic Form Score
        form_score = max(0, min(100, int(avg_abduction)))
        if self.target_height_reached: form_score = 100
        if self.arms_bent: form_score -= 30
        if self.is_asymmetrical: form_score -= 20
        if self.torso_swayed: form_score -= 30
        if not self.target_height_reached and self.state == "LOWERING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Abduction",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "abduction_angle": round(avg_abduction, 1),
            "depth_reached": self.target_height_reached
        }