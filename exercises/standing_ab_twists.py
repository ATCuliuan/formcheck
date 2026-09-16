import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp

class StandingAbTwistTracker:
    def __init__(self):
        self.state = "CENTER"   
        self.good_reps = 0
        self.bad_reps = 0
        self.twist_reached = False
        
        # Instructor Tripwires
        self.arms_elevated = True
        self.hips_locked = True           # Catches the "Hula Hoop" cheat
        self.ml_posture_maintained = True 
        
        self.baseline_shoulder_width = None
        self.baseline_hip_width = None
        
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Face camera with arms raised in front"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_ab_twists.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Ab Twist model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Ab Twist weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        # Full 25-joint extraction for ST-GCN consistency
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Ab Twist", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face the camera, step back to show full torso",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Calculate live widths (2D Projection)
        current_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        current_hip_width = abs(l_hip.x - r_hip.x)
        
        # Guard: Are BOTH arms elevated near shoulder height?
        arms_are_up = (l_elbow.y < l_shoulder.y + 0.15) and (r_elbow.y < r_shoulder.y + 0.15)

        # 1. Starting Guard: Centered to camera, arms raised
        # If shoulders and hips are wide, they are facing the camera squarely
        if arms_are_up:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready: 
            self.is_ready = True
            self.baseline_shoulder_width = current_shoulder_width
            self.baseline_hip_width = current_hip_width

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Ab Twist", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face forward, raise arms to chest height",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "CENTER":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # The T-Rex Drop (Dropping arms to cheat)
        if not arms_are_up and self.state in ["TWISTING", "MAX_TWIST"]:
            self.arms_elevated = False

        # The Hula Hoop (Hip pivoting)
        # If hip width shrinks below 85% of baseline, they have turned their pelvis away from the camera.
        if current_hip_width < (self.baseline_hip_width * 0.85) and self.state in ["TWISTING", "MAX_TWIST"]:
            self.hips_locked = False

        # 4. State Machine (Driven by shoulder width contraction)
        if self.state == "CENTER":
            self.current_feedback = "Twist left or right!"
            if current_shoulder_width < (self.baseline_shoulder_width * 0.85):
                self.state = "TWISTING"

        elif self.state == "TWISTING":
            self.current_feedback = "Use your obliques..."
            if current_shoulder_width < (self.baseline_shoulder_width * 0.55):
                self.twist_reached = True
                self.state = "MAX_TWIST"
                self.current_feedback = "Hold! Now return to center."
            elif current_shoulder_width > (self.baseline_shoulder_width * 0.90):
                # User gave up and went back to center without hitting max
                self.state = "RETURNING"

        elif self.state == "MAX_TWIST" and current_shoulder_width > (self.baseline_shoulder_width * 0.65):
            self.state = "RETURNING"
            self.current_feedback = "Control the rotation back"

        elif self.state == "RETURNING" and current_shoulder_width >= (self.baseline_shoulder_width * 0.85):
            is_bad_rep = False
            
            if not self.twist_reached:
                self.current_feedback = "No rep: Twist further to the side."
                is_bad_rep = True
            elif not self.arms_elevated:
                self.current_feedback = "No rep: You dropped your arms. Keep them up."
                is_bad_rep = True
            elif not self.hips_locked:
                self.current_feedback = "No rep: You twisted your hips! Keep belt buckle facing forward."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture (arm swinging)."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect core isolation!"
                
            # Reset for next twist
            self.state = "CENTER"
            self.twist_reached = False
            self.arms_elevated = True
            self.hips_locked = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring based on how far they rotated the shoulders
        twist_ratio = current_shoulder_width / self.baseline_shoulder_width if self.baseline_shoulder_width else 1.0
        form_score = max(0, min(100, int(( (1.0 - twist_ratio) / 0.45 ) * 100)))
        
        if not self.twist_reached and self.state == "RETURNING": form_score -= 30
        if not self.hips_locked: form_score -= 40
        if not self.arms_elevated: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Ab Twist",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": 0.0,
            "hip_angle": 0.0
        }