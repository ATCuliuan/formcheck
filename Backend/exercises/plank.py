import time
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle

class PlankTracker:
    def __init__(self):
        self.hold_start_time = None
        self.total_hold_seconds = 0
        self.is_holding = False
        self.current_feedback = "Hold body in a straight line"
        self.mp_pose = mp.solutions.pose

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # Spine & hip alignment angle
        body_line = calculate_angle(l_shoulder, l_hip, l_ankle)

        # Valid plank: straight line between 165° and 185°
        if 165 <= body_line <= 185:
            if not self.is_holding:
                self.hold_start_time = time.time()
                self.is_holding = True
            else:
                self.total_hold_seconds += time.time() - self.hold_start_time
                self.hold_start_time = time.time()
            self.current_feedback = "Good plank form — keep holding"
        else:
            self.is_holding = False
            self.hold_start_time = None
            if body_line < 165:
                self.current_feedback = "Lower your hips (too high)"
            else:
                self.current_feedback = "Raise your hips (sagging)"

        form_score = max(0, min(100, int(100 - abs(body_line - 180) * 2)))

        return {
            "exercise": "Plank",
            "state": "HOLDING" if self.is_holding else "INCORRECT",
            "rep_count": int(self.total_hold_seconds), # Uses seconds held instead of reps
            "form_score": form_score,
            "feedback": self.current_feedback,
            "body_alignment": round(body_line, 1),
            "depth_reached": self.is_holding
        }