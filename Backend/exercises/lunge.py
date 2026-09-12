import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle

class LungeTracker:
    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.depth_reached = False
        self.current_feedback = "Stand ready for lunges"
        self.mp_pose = mp.solutions.pose

    def process_frame(self, landmarks):
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        front_knee = calculate_angle(l_hip, l_knee, l_ankle)
        back_knee = calculate_angle(r_hip, r_knee, r_ankle)

        # Check if standing
        if front_knee > 155 and back_knee > 155:
            if self.state == "ASCENDING" and self.depth_reached:
                self.rep_count += 1
                self.current_feedback = "Lunge completed!"
                self.depth_reached = False
            self.state = "STANDING"

        # Check descent & 90-degree depth
        elif front_knee <= 120 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            if front_knee <= 95:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Good 90° knee angle"

        elif front_knee > 100 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"

        form_score = max(0, min(100, int(100 - abs(front_knee - 90) * 0.4)))

        return {
            "exercise": "Lunge",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "front_knee_angle": round(front_knee, 1),
            "back_knee_angle": round(back_knee, 1),
            "depth_reached": self.depth_reached
        }