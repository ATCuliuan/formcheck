import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DeadliftTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.hinge_reached = False
        
        # Instructor Tripwires
        self.squatted_the_weight = False  # Did they bend their knees too much?
        self.bar_drifted = False          # Did the weight swing away from the shins?
        self.ml_posture_maintained = True # AI detects the "Fishing Pole" rounded back
        
        self.standing_frames = 0
        self.is_ready = False
        self.current_feedback = "Stand tall with feet shoulder-width apart"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_deadlift.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Deadlift model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Deadlift weights: {e}")
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show full side profile",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # 1. Starting Guard: Must be standing fully tall
        if hip_angle > 165 and knee_angle > 165:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 20: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Stand up perfectly straight to begin",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
            }

        # 2. ST-GCN Buffer for Lower Back Rounding
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # The "Squatty-Lift" Tripwire
        # Knees should bend slightly to allow hips to go back, but not drop like a squat.
        if knee_angle < 115 and self.state in ["DESCENDING", "BOTTOM"]:
            self.squatted_the_weight = True

        # The "Bar Drift" Tripwire
        # Hands (wrist) should slide down the shins. If they go too far forward over the toes, it's bad.
        drift_distance = abs(active_wrist.x - active_ankle.x)
        if drift_distance > 0.15 and self.state in ["DESCENDING", "BOTTOM"]:
            self.bar_drifted = True

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Hinge hips backward..."

        elif self.state in ["STANDING", "ASCENDING"] and hip_angle <= 150:
            self.state = "DESCENDING"
            self.current_feedback = "Push glutes back to the wall"

        elif self.state == "DESCENDING" and hip_angle <= 100:
            self.hinge_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Squeeze glutes and stand up!"

        elif self.state == "BOTTOM" and hip_angle > 115:
            self.state = "ASCENDING"
            self.current_feedback = "Drive hips forward!"

        elif self.state == "ASCENDING" and hip_angle > 165 and knee_angle > 165:
            is_bad_rep = False
            
            # Form Evaluation
            if not self.hinge_reached:
                self.current_feedback = "No rep: You didn't hinge deep enough (go below the knees)."
                is_bad_rep = True
            elif self.squatted_the_weight:
                self.current_feedback = "No rep: You squatted! Keep hips higher than knees."
                is_bad_rep = True
            elif self.bar_drifted:
                self.current_feedback = "No rep: Weight drifted away from your legs (Danger)."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged a rounded lower back (Danger)."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect deadlift!"
                
            # Reset for next rep
            self.state = "STANDING"
            self.hinge_reached = False
            self.squatted_the_weight = False
            self.bar_drifted = False
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(( (170 - hip_angle) / 70 ) * 100)))
        if not self.hinge_reached and self.state == "ASCENDING": form_score -= 30
        if self.squatted_the_weight: form_score -= 40
        if self.bar_drifted: form_score -= 30
        if not self.ml_posture_maintained: form_score -= 50
        form_score = max(0, form_score)

        return {
            "exercise": "Deadlift",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1)
        }