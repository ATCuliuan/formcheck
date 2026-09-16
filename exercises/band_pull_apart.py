import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BandPullTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.pull_reached = False
        
        # Instructor Tripwires
        self.arms_straight = True
        self.shoulders_shrugged = False
        self.ml_posture_maintained = True  
        
        self.is_ready = False
        self.ready_frames = 0
        self.baseline_shoulder_y = None

        self.current_feedback = "Raise arms straight in front"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT (TensorFlow) Model instead of PyTorch
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_band_pull_aparts.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Band Pull model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Band Pull weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        # Extracting all necessary joints to match the ST-GCN training structure
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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]

        # Calculate elbow straightness
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        # Track the horizontal distance between the wrists to determine the "Pull"
        wrist_distance = abs(l_wrist.x - r_wrist.x)
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)

        # 1. Starting Guard: Arms raised in front, hands close together
        arms_level = abs(l_shoulder.y - l_wrist.y) < 0.15 and abs(r_shoulder.y - r_wrist.y) < 0.15
        hands_together = wrist_distance < (shoulder_width * 1.5)

        if arms_level and hands_together and min_elbow_angle > 140:
            self.ready_frames += 1
            self.baseline_shoulder_y = abs(l_ear.y - l_shoulder.y) # Snapshot of a relaxed neck
        else:
            self.ready_frames = 0

        if self.ready_frames > 15:
            self.is_ready = True

        total_reps = self.good_reps + self.bad_reps

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Band Pull Apart", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Raise arms straight in front, hands together",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # 2. ST-GCN Temporal Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0) 
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # T-Rex Arm Tripwire
        if min_elbow_angle < 130 and self.state in ["PULLING", "TOP"]:
            self.arms_straight = False
            
        # The Shrug Tripwire (Neck compression)
        current_shoulder_y = abs(l_ear.y - l_shoulder.y)
        if self.baseline_shoulder_y and current_shoulder_y < (self.baseline_shoulder_y * 0.7) and self.state == "PULLING":
            self.shoulders_shrugged = True

        # 4. State Machine (Driven by wrist horizontal spread)
        if self.state == "WAITING" and wrist_distance > (shoulder_width * 1.5):
            self.state = "PULLING"
            self.current_feedback = "Pull wide, keep arms straight..."

        elif self.state == "PULLING" and wrist_distance > (shoulder_width * 3.0):
            self.pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze the upper back! Now return slowly."

        elif self.state == "TOP" and wrist_distance < (shoulder_width * 2.5):
            self.state = "RETURNING"
            self.current_feedback = "Control it on the way back..."

        elif self.state == "RETURNING" and hands_together:
            is_bad_rep = False
            
            if not self.pull_reached:
                self.current_feedback = "No rep: You didn't pull wide enough (form a 'T')."
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "No rep: You bent your elbows! Keep arms straight."
                is_bad_rep = True
            elif self.shoulders_shrugged:
                self.current_feedback = "No rep: You shrugged your shoulders! Keep them down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Excellent rep!"

            # Reset
            self.state = "WAITING"
            self.pull_reached = False
            self.arms_straight = True
            self.shoulders_shrugged = False
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Score Calculation
        form_score = 100
        if not self.pull_reached and self.state == "RETURNING": form_score -= 40
        if not self.arms_straight: form_score -= 30
        if self.shoulders_shrugged: form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Band Pull Apart",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0
        }