import mediapipe as mp
from core.kinematics import calculate_angle

class PushupTracker:
    def __init__(self):
        self.state = "TOP"   # States: TOP, DESCENDING, BOTTOM, ASCENDING
        self.rep_count = 0
        self.depth_reached = False
        self.current_feedback = "Get into plank position"
        self.mp_pose = mp.solutions.pose

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # 1. Elbow angle (measures push-up depth)
        elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        # 2. Body alignment angle (Shoulder - Hip - Ankle -> should be ~180° straight line)
        body_line = calculate_angle(l_shoulder, l_hip, l_ankle)

        # Form check: detect sagging hips or piking
        is_body_straight = 160 <= body_line <= 185

        if not is_body_straight:
            self.current_feedback = "Keep your back and hips straight"

        # State Machine
        if elbow_angle > 155:
            if self.state == "ASCENDING" and self.depth_reached:
                self.rep_count += 1
                self.current_feedback = "Good push-up!"
                self.depth_reached = False
            self.state = "TOP"

        elif elbow_angle <= 130 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            if elbow_angle <= 90:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Good chest depth — push up"

        elif elbow_angle > 100 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"

        form_score = max(0, min(100, int(100 - abs(body_line - 180) * 1.5)))

        return {
            "exercise": "Push-up",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "body_alignment": round(body_line, 1),
            "depth_reached": self.depth_reached
        }