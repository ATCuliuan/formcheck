import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellOverheadShoulderPressTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False

        # Instructor Tripwires
        self.core_stable = True           # Tracks the dangerous back arch
        self.arms_even = True             # Tracks asymmetrical pressing
        self.ml_posture_maintained = True # AI catches the incomplete lockout
        self.is_ready = False
        self.racked_frames = 0

        self.current_feedback = "Bring weights to shoulder level"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_overhead_shoulder_press.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Overhead Press model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Overhead Press weights: {e}")
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
        # Upper body joints are critical here
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Position camera to see torso and arms",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Calculate elbow extension for BOTH arms
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # We track the 'weakest' arm for strict form enforcement
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)
        max_elbow_angle = max(l_elbow_angle, r_elbow_angle)

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip

        # 1. Racked Position Guard (Weights at shoulders, elbows bent)
        if max_elbow_angle < 100:
            self.racked_frames += 1
        else:
            self.racked_frames = 0

        if self.racked_frames > 10:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Rack weights at shoulders to start",
                "knee_angle": round(min_elbow_angle, 1),
                "hip_angle": 0.0,
                "depth_reached": False
            }

        # 2. ST-GCN Temporal Buffer Update
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
                if predicted_class == 0 and self.state != "RACKED":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # 3. Form Tripwires
        
        # The Back Arch Detector: Horizontal distance between shoulder and hip
        # If shoulder moves too far behind the hips, they are leaning back.
        torso_sway = abs(active_shoulder.x - active_hip.x)
        if torso_sway > 0.12 and self.state == "PRESSING":
            self.core_stable = False

        # The Asymmetry Detector: Are the hands moving evenly?
        wrist_imbalance = abs(l_wrist.y - r_wrist.y)
        if wrist_imbalance > 0.15 and self.state == "PRESSING":
            self.arms_even = False

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "RACKED"
            self.depth_reached = True # Starting at the bottom
            self.current_feedback = "Press straight up!"

        elif self.state in ["RACKED", "LOWERING"] and min_elbow_angle > 110:
            self.state = "PRESSING"
            self.current_feedback = "Drive it overhead!"

        elif self.state == "PRESSING" and min_elbow_angle > 155:
            # Reached the top! Wait for them to lower it.
            self.state = "TOP"
            self.current_feedback = "Control the weight down"

        elif self.state == "TOP" and max_elbow_angle < 140:
            self.state = "LOWERING"

        elif self.state == "LOWERING" and max_elbow_angle <= 90:
            self.depth_reached = True
            is_bad_rep = False

            # Form Evaluation is calculated when they return to the start position
            if not self.core_stable:
                self.current_feedback = "No rep: You leaned back! Keep core tight."
                is_bad_rep = True
            elif not self.arms_even:
                self.current_feedback = "No rep: One arm lagged behind. Press evenly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: You didn't lock out at the top."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Great press!"

            # Reset tripwires for the next rep
            self.state = "RACKED"
            self.core_stable = True
            self.arms_even = True
            self.ml_posture_maintained = True
            self.depth_reached = True
            self.frame_buffer.clear()

        # Dynamic Scoring based on weakest elbow extension
        form_score = max(0, min(100, int((min_elbow_angle / 160) * 100)))
        if not self.core_stable: form_score -= 40
        if not self.arms_even: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Overhead Press",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0,
            "depth_reached": self.depth_reached
        }