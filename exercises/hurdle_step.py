import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class HurdleStepTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.hurdle_cleared = False
        self.active_leg = None
        
        # Posture Tripwires
        self.support_leg_bent = False     # Did they bend the standing leg to cheat?
        self.torso_leaned = False         # Did they lean sideways to compensate for mobility?
        self.ml_posture_maintained = True # ST-GCN AI flag
        
        # Setup Guards
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, feet together to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_hurdle_step.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Hurdle Step model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Hurdle Step weights: {e}")
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

        # Check visibility
        total_vis = sum([l_hip.visibility, r_hip.visibility, l_knee.visibility, r_knee.visibility, l_ankle.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps

        if total_vis < 4.0: 
            self.frame_buffer.clear()
            return {
                "exercise": "Hurdle Step",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full lower body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Calculate base standing angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Baseline check: Are they standing still? (MediaPipe Y axis: 0 is top, 1 is bottom)
        ankle_y_diff = l_ankle.y - r_ankle.y

        if l_knee_angle > 160 and r_knee_angle > 160 and abs(ankle_y_diff) < 0.08:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Hurdle Step",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand tall, feet together to begin",
                "knee_angle": round((l_knee_angle + r_knee_angle)/2, 1), "hip_angle": 0.0, "depth_reached": False
            }

        # Determine which leg is moving based on ankle elevation
        if self.state == "STANDING":
            if ankle_y_diff > 0.08: # Left ankle is lower, Right is lifting (smaller Y)
                self.active_leg = "RIGHT"
                self.state = "STEPPING"
                self.current_feedback = "Lift knee high!"
            elif ankle_y_diff < -0.08: # Right ankle is lower, Left is lifting (smaller Y)
                self.active_leg = "LEFT"
                self.state = "STEPPING"
                self.current_feedback = "Lift knee high!"

        # Assign dynamic joints based on the active leg
        if self.active_leg == "RIGHT":
            moving_ankle, moving_knee = r_ankle, r_knee
            support_knee_angle = l_knee_angle
            support_knee = l_knee
        elif self.active_leg == "LEFT":
            moving_ankle, moving_knee = l_ankle, l_knee
            support_knee_angle = r_knee_angle
            support_knee = r_knee
        else:
            # Default to average if standing
            support_knee_angle = (l_knee_angle + r_knee_angle) / 2

        # Form Tripwires (Evaluated while stepping)
        if self.state == "STEPPING":
            # 1. Did the standing leg bend to cheat the hurdle height?
            if support_knee_angle < 150:
                self.support_leg_bent = True
                
            # 2. Did the torso lean sideways to compensate for hip mobility?
            shoulder_tilt = abs(l_shoulder.y - r_shoulder.y)
            if shoulder_tilt > 0.08:
                self.torso_leaned = True

            # 3. Clearance check: Moving ankle must rise near the support knee's height
            # Note: MediaPipe Y-axis goes down, so smaller Y is higher.
            clearance_threshold = support_knee.y + 0.10 # Allow slight leeway for beginners
            if moving_ankle.y < clearance_threshold:
                self.hurdle_cleared = True
                self.current_feedback = "Cleared! Step down."

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
                if predicted_class == 0 and self.state == "STEPPING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # State Machine & Rep Completion Logic
        # If ankles are back together and level, the step is finished
        if self.state == "STEPPING" and abs(ankle_y_diff) < 0.05 and support_knee_angle > 160:
            is_bad_rep = False
            
            if not self.hurdle_cleared:
                self.current_feedback = "No rep: Lift your foot higher (Lazy step)"
                is_bad_rep = True
            elif self.support_leg_bent:
                self.current_feedback = "No rep: Support leg bent (Stand tall)"
                is_bad_rep = True
            elif self.torso_leaned:
                self.current_feedback = "No rep: Torso leaned sideways (Keep balance)"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected form breakdown"
                is_bad_rep = True
            
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect step!"
            
            # Reset state for next rep
            self.state = "STANDING"
            self.active_leg = None
            self.hurdle_cleared = False
            self.support_leg_bent = False
            self.torso_leaned = False
            self.ml_posture_maintained = True

        # Dynamic Form Score
        form_score = 100
        if self.state == "STEPPING":
            if self.support_leg_bent: form_score -= 40
            if self.torso_leaned: form_score -= 30
            if not self.hurdle_cleared: form_score -= 30
            form_score = max(0, form_score)

        return {
            "exercise": "Hurdle Step",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(support_knee_angle, 1),
            "depth_reached": self.hurdle_cleared
        }