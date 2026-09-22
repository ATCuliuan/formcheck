import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class InlineLungeTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture Tripwires
        self.torso_collapsed = False      # Did they lean forward to cheat ankle mobility?
        self.wobbled = False              # Did they lose lateral tightrope balance?
        self.ml_posture_maintained = True # ST-GCN AI flag
        
        # Setup Guards
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, feet in a split stance"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_inline_lunge.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Inline Lunge model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Inline Lunge weights: {e}")
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        total_vis = sum([l_hip.visibility, r_hip.visibility, l_knee.visibility, r_knee.visibility])
        total_reps = self.good_reps + self.bad_reps

        if total_vis < 3.0: 
            self.frame_buffer.clear()
            return {
                "exercise": "Inline Lunge",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full body not visible",
                "knee_angle": 0.0, "depth_reached": False
            }

        # Calculate joint angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        # Calculate torso upright angle (average of both sides)
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2

        # Dynamically identify the back knee (the one dropping toward the floor)
        # MediaPipe Y-axis goes down, so the higher Y value is closer to the floor
        if l_knee.y > r_knee.y:
            back_knee_angle = l_knee_angle
            front_knee_angle = r_knee_angle
        else:
            back_knee_angle = r_knee_angle
            front_knee_angle = l_knee_angle

        if front_knee_angle > 160 and back_knee_angle > 160:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Inline Lunge",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand tall in a split stance to begin",
                "knee_angle": round(front_knee_angle, 1), "depth_reached": False
            }

        # Form Tripwires (Evaluated during descent and bottom)
        if self.state in ["DESCENDING", "BOTTOM"]:
            # 1. Torso Collapse: Beginners lean forward to avoid using glutes
            if avg_hip_angle < 110:
                self.torso_collapsed = True
                
            # 2. Tightrope Wobble: Shoulders shouldn't tilt excessively left or right
            shoulder_tilt = abs(l_shoulder.y - r_shoulder.y)
            if shoulder_tilt > 0.09:
                self.wobbled = True

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
        if front_knee_angle > 150 and back_knee_angle > 150:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Drop back knee lower"
                    is_bad_rep = True
                elif self.torso_collapsed:
                    self.current_feedback = "No rep: Torso leaned too far forward"
                    is_bad_rep = True
                elif self.wobbled:
                    self.current_feedback = "No rep: Lost balance (Keep core tight)"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected form breakdown"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect lunge!"
                
                # Reset tripwires
                self.depth_reached = False
                self.torso_collapsed = False
                self.wobbled = False
                self.ml_posture_maintained = True
                
            self.state = "STANDING"

        elif front_knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Drop straight down..."

            # Depth logic: Both knees should approach 90-100 degrees
            if back_knee_angle <= 105 and front_knee_angle <= 105:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth reached — Push back up!"

        elif front_knee_angle > 105 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Drive through the front heel!"
            else:
                self.current_feedback = "Drop lower..."

        # Dynamic Form Score
        form_score = max(0, min(100, int(100 - abs(front_knee_angle - 90) * 0.3)))
        if self.torso_collapsed: form_score -= 40
        if self.wobbled: form_score -= 30
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Inline Lunge",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(front_knee_angle, 1),
            "depth_reached": self.depth_reached
        }