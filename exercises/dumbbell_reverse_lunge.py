import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellReverseLungeTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False

        # Biomechanical Tripwires
        self.torso_upright = True
        self.knee_over_toes_safe = True
        self.ml_posture_maintained = True
        self.is_ready = False
        self.standing_frames = 0

        self.current_feedback = "Stand upright with weights at sides"
        self.mp_pose = mp.solutions.pose

        # Buffer for 30-frame ST-GCN sequence
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_reverse_lunge.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Reverse Lunge model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Reverse Lunge weights: {e}")
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, r_shoulder.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: full body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Calculate joint angles for both legs to identify front vs back leg
        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # The front leg carries the primary load and undergoes maximum flexion
        if left_knee_angle < right_knee_angle:
            front_knee_angle = left_knee_angle
            active_hip, active_knee, active_ankle = l_hip, l_knee, l_ankle
            active_shoulder = l_shoulder
        else:
            front_knee_angle = right_knee_angle
            active_hip, active_knee, active_ankle = r_hip, r_knee, r_ankle
            active_shoulder = r_shoulder

        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        # 1. Setup Guard (Must stand upright with both knees extended)
        if left_knee_angle > 160 and right_knee_angle > 160:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand upright with feet together",
                "knee_angle": round(front_knee_angle, 1),
                "hip_angle": round(hip_angle, 1),
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
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # 3. Form Tripwires
        # Torso Check: Avoid pitching forward over the front knee
        if hip_angle < 70 and self.state in ["LUNGING", "BOTTOM"]:
            self.torso_upright = False

        # Shin Shear Check: Front knee projecting too far forward past the ankle
        if abs(active_knee.x - active_ankle.x) > 0.16 and self.state in ["LUNGING", "BOTTOM"]:
            self.knee_over_toes_safe = False

        # 4. State Machine
        if self.state == "STANDING" and front_knee_angle <= 140:
            self.state = "LUNGING"
            self.current_feedback = "Step back and lower hips straight down"

        elif self.state == "LUNGING":
            if front_knee_angle <= 100:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Good depth! Drive through the front heel"
            elif front_knee_angle > 145:
                # User stood back up without hitting depth
                self.state = "ASCENDING"

        elif self.state == "BOTTOM" and front_knee_angle > 115:
            self.state = "ASCENDING"
            self.current_feedback = "Push back up to standing"

        elif self.state == "ASCENDING" and left_knee_angle > 155 and right_knee_angle > 155:
            is_bad_rep = False

            if not self.depth_reached:
                self.current_feedback = "No rep: Lunge deeper (front knee to 90°)"
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Keep chest proud, don't lean forward"
                is_bad_rep = True
            elif not self.knee_over_toes_safe:
                self.current_feedback = "No rep: Step further back to protect front knee"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected collapsed upper body"
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean reverse lunge!"

            # Reset tripwires
            self.state = "STANDING"
            self.depth_reached = False
            self.torso_upright = True
            self.knee_over_toes_safe = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(front_knee_angle - 90) * 0.4)))
        if not self.torso_upright: form_score -= 35
        if not self.knee_over_toes_safe: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 40
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Reverse Lunge",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(front_knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }