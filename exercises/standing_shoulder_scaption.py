import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingShoulderScaptionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.target_height_reached = False
        
        # Posture Tripwires
        self.arms_bent = False            # Did they bend elbows to shorten the lever arm?
        self.torso_leaned = False         # Did they lean backward to cheat gravity?
        self.is_asymmetrical = False      # Did one arm lift significantly higher than the other?
        self.ml_posture_maintained = True # ST-GCN AI flag (catches shrugging/neck tension)
        
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
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_scaption.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Shoulder Scaption model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Shoulder Scaption weights: {e}")
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        total_vis = sum([
            l_shoulder.visibility, r_shoulder.visibility, 
            l_elbow.visibility, r_elbow.visibility, 
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ])
        total_reps = self.good_reps + self.bad_reps

        # Require hips and knees to evaluate torso leaning
        if total_vis < 5.5: 
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Scaption",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and arms not fully visible",
                "elevation_angle": 0.0, "depth_reached": False
            }

        # Calculate Elevation Angles (Arm raising forward/outward relative to the torso)
        l_elevation_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_elevation_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_elevation = (l_elevation_angle + r_elevation_angle) / 2

        # Calculate Elbow Angles to catch T-Rex arms
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Calculate Torso Angle (Shoulder -> Hip -> Knee) to catch leaning backward
        l_torso_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_torso_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_torso_angle = (l_torso_angle + r_torso_angle) / 2

        # Baseline: Arms relaxed at the sides
        if avg_elevation < 30:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Scaption",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Rest arms at your sides to begin",
                "elevation_angle": round(avg_elevation, 1), "depth_reached": False
            }

        # Form Tripwires (Evaluated during the lift)
        if self.state in ["LIFTING", "PEAK"]:
            # 1. T-Rex Arms: Bending elbows to cheat the lever
            if l_elbow_angle < 155 or r_elbow_angle < 155:
                self.arms_bent = True
                
            # 2. Torso Lean (The "Limbo" Cheat): Leaning back to fake arm height
            # A strict vertical torso should be ~170-180 degrees.
            if avg_torso_angle < 165:
                self.torso_leaned = True

            # 3. See-Saw Lift: One arm lagging significantly behind the other
            if abs(l_elevation_angle - r_elevation_angle) > 20:
                self.is_asymmetrical = True

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
        if avg_elevation < 40:
            if self.state == "LOWERING":
                is_bad_rep = False
                
                if not self.target_height_reached:
                    self.current_feedback = "No rep: Raise arms to shoulder height"
                    is_bad_rep = True
                elif self.torso_leaned:
                    self.current_feedback = "No rep: Do not lean backward"
                    is_bad_rep = True
                elif self.arms_bent:
                    self.current_feedback = "No rep: Keep your arms straight"
                    is_bad_rep = True
                elif self.is_asymmetrical:
                    self.current_feedback = "No rep: Both arms must lift equally"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected shoulder shrugging/form breakdown"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect Y-raise!"
                
                # Reset tripwires
                self.target_height_reached = False
                self.arms_bent = False
                self.torso_leaned = False
                self.is_asymmetrical = False
                self.ml_posture_maintained = True
                
            self.state = "STANDING"

        elif avg_elevation >= 40 and self.state in ["STANDING", "LOWERING"]:
            self.state = "LIFTING"
            self.current_feedback = "Raise arms in a Y-shape..."

            # Target height logic: Aiming for 130+ degrees (arms raised diagonally overhead)
            if l_elevation_angle >= 130 and r_elevation_angle >= 130:
                self.target_height_reached = True
                self.state = "PEAK"
                self.current_feedback = "Great height — Lower slowly!"

        elif avg_elevation < 130 and self.state in ["PEAK", "LIFTING"]:
            if self.target_height_reached:
                self.state = "LOWERING"
                self.current_feedback = "Control the descent..."
            else:
                self.current_feedback = "Lift higher..."

        # Dynamic Form Score
        form_score = max(0, min(100, int(avg_elevation * 0.75)))
        if self.target_height_reached: form_score = 100
        if self.torso_leaned: form_score -= 40
        if self.arms_bent: form_score -= 30
        if self.is_asymmetrical: form_score -= 20
        if not self.target_height_reached and self.state == "LOWERING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Scaption",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elevation_angle": round(avg_elevation, 1),
            "depth_reached": self.target_height_reached
        }