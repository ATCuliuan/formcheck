import os
import joblib
import mediapipe as mp
from core.kinematics import calculate_angle

class SquatTracker:
    def __init__(self):
        self.state = "STANDING"   # States: STANDING, DESCENDING, BOTTOM, ASCENDING
        self.rep_count = 0
        self.depth_reached = False
        self.current_feedback = "Ready"
        self.mp_pose = mp.solutions.pose
        
        # 1. Load the Machine Learning Model (.pkl)
        self.model = None
        model_path = os.path.join("models", "squat_model.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            print(f"⚠️ Warning: AI Model not found at {model_path}")

    def process_frame(self, landmarks):
        # Extract joints for math
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        # Calculate joint angles
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)

        # 2. Machine Learning Form Check (Runs every frame)
        ml_feedback = None
        if self.model:
            try:
                # Flatten the 33 landmarks into 132 features (just like train_model.py)
                features = []
                for lm in landmarks:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                # Ask the Random Forest for a prediction (0 = Bad, 1 = Good)
                prediction = self.model.predict([features])[0]
                if prediction == 0:
                    ml_feedback = "Warning: Bad Posture Detected!"
            except Exception as e:
                pass

        # 3. Repetition State Machine Logic
        if knee_angle > 155:
            if self.state == "ASCENDING" and self.depth_reached:
                self.rep_count += 1
                self.current_feedback = "Good rep!"
                self.depth_reached = False
            self.state = "STANDING"

        elif knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = ml_feedback if ml_feedback else "Going down..."

            if knee_angle <= 95:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = ml_feedback if ml_feedback else "Depth good — count it"

        elif knee_angle > 100 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = ml_feedback if ml_feedback else "Push up to finish"
            else:
                self.current_feedback = "Squat lower for full depth"

        # Calculate Form Score 
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))
        
        # Penalize the score if the ML model detects bad form
        if ml_feedback:
            form_score = max(0, form_score - 30)

        return {
            "exercise": "Bodyweight Squat",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }