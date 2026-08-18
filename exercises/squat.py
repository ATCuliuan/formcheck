import mediapipe as mp
from core.kinematics import calculate_angle

class SquatTracker:
    def __init__(self):
        self.state = "STANDING"   # States: STANDING, DESCENDING, BOTTOM, ASCENDING
        self.rep_count = 0
        self.depth_reached = False
        self.current_feedback = "Ready"
        self.mp_pose = mp.solutions.pose

    def process_frame(self, landmarks):
        # Extract joints for left and right sides
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        # Calculate joint angles
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)

        # Repetition State Machine Logic
        # 1. Standing position: Knees straight (~160° - 180°)
        if knee_angle > 155:
            if self.state == "ASCENDING" and self.depth_reached:
                self.rep_count += 1
                self.current_feedback = "Good rep!"
                self.depth_reached = False
            self.state = "STANDING"

        # 2. Descending: Knees bending below 140°
        elif knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Going down..."

            # 3. Bottom / Good Depth check (Knee angle <= 90°)
            if knee_angle <= 95:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth good — count it"

        # 4. Ascending: Pushing back up
        elif knee_angle > 100 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Push up to finish"
            else:
                self.current_feedback = "Squat lower for full depth"

        # Calculate Form Score (0 - 100%)
        # Penalize if hips and knees are out of sync
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))

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