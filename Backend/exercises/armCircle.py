import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class ArmCirclesTracker:
    """
    Coaching logic for Dynamic Arm Circles:
      1. STANDING  -> arms resting at sides, shoulders relaxed, standing tall.
      2. RISING    -> arms sweeping upward (frontward or laterally).
      3. TOP       -> arms reach maximum vertical extension near the ears.
      4. FALLING   -> arms sweeping downward on the back half (or front half) of the circle.
      5. BOTTOM    -> arms return to the sides; one full revolution complete.

    Instructor-level faults tracked independently:
      - bent_elbows: Bending the arms to cheat the range of motion (T-Rex arms).
      - shoulder_shrug: Hiking the shoulders up to the ears to compensate for poor joint mobility.
      - half_circle: Reversing direction at the top instead of completing a full 360-degree sweep.
    """

    # Arm elevation is measured as the angle between the Hip, Shoulder, and Wrist
    ELEVATION_BOTTOM_MAX = 35.0      # Arms down at the sides
    ELEVATION_TOP_MIN = 150.0        # Arms fully overhead
    
    ELBOW_STRAIGHT_MIN = 150.0       # Elbows must remain locked to maximize the lever
    SHRUG_TOLERANCE = 0.04           # Max upward displacement of shoulders from baseline

    def __init__(self):
        self.state = "BOTTOM"
        self.rep_count = 0

        # Baselines for catching compensations
        self.baseline_shoulder_y = None

        # Per-rep tripwires
        self.arms_locked = True
        self.shoulders_relaxed = True
        self.reached_top = False
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, arms down, relax your neck to calibrate."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_ARM_CIRCLES_MODEL

        if os.path.exists(model_path):
            try:
                self.model = STGCN(num_classes=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception:
                self.model = None

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # 1. Kinematics
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        
        # Arm elevation (Hip -> Shoulder -> Wrist) tracks vertical range of motion
        l_elevation_angle = calculate_angle(l_hip, l_shoulder, l_wrist)
        r_elevation_angle = calculate_angle(r_hip, r_shoulder, r_wrist)
        avg_elevation = (l_elevation_angle + r_elevation_angle) / 2.0

        current_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0

        # 2. Setup Guard: Wait for a neutral standing posture with arms at the sides
        is_neutral_stance = (avg_elevation < self.ELEVATION_BOTTOM_MAX)

        if is_neutral_stance:
            self.ready_frames += 1
            if self.ready_frames == 15:
                # Capture baseline shoulder height to detect shrugging
                self.baseline_shoulder_y = current_shoulder_y
        else:
            self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "BOTTOM"

        if not self.is_ready:
            return {
                "exercise": "Arm Circles",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Rest your arms at your sides and drop your shoulders to calibrate.",
                "arm_elevation": round(avg_elevation, 1)
            }

        # 3. Fault Tripwires
        if self.state != "BOTTOM":
            # Did they bend their elbows to make the circle easier?
            if avg_elbow_angle < self.ELBOW_STRAIGHT_MIN:
                self.arms_locked = False
            
            # Did they shrug? (In screen coordinates, Y gets smaller as it goes up)
            if self.baseline_shoulder_y - current_shoulder_y > self.SHRUG_TOLERANCE:
                self.shoulders_relaxed = False

            # Did they reach full overhead extension?
            if avg_elevation > self.ELEVATION_TOP_MIN:
                self.reached_top = True

        # 4. Buffer & AI
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.model and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer)
                tensor_input = torch.tensor(np.transpose(seq, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if torch.argmax(self.model(tensor_input), dim=1).item() == 0:
                        self.ml_posture_maintained = False
            except Exception:
                pass

        # 5. State Machine & Coaching
        if avg_elevation < self.ELEVATION_BOTTOM_MAX:
            if self.state in ("RISING", "TOP", "FALLING"):
                if self.reached_top:
                    if not self.arms_locked:
                        self.current_feedback = "No rep: You bent your elbows! Draw a massive circle with straight arms."
                    elif not self.shoulders_relaxed:
                        self.current_feedback = "No rep: You shrugged your shoulders. Pack them down away from your ears."
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected torso swaying. Lock your core!"
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great circle! Excellent shoulder mobility."
                else:
                    self.current_feedback = "No rep: You didn't bring your arms all the way overhead."

            # Reset variables for the next rep
            self.state = "BOTTOM"
            self.arms_locked = True
            self.shoulders_relaxed = True
            self.reached_top = False
            self.ml_posture_maintained = True

        elif avg_elevation > self.ELEVATION_TOP_MIN:
            self.state = "TOP"
            self.current_feedback = "Good overhead reach! Now sweep it down to finish the circle."

        else:
            if self.state == "BOTTOM":
                self.state = "RISING"
                self.current_feedback = "Keep the elbows locked out as you raise your arms."
            elif self.state == "TOP":
                self.state = "FALLING"
                self.current_feedback = "Sweep it down, don't let those shoulders hike up."

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.arms_locked:
            form_score -= 40
        if not self.shoulders_relaxed:
            form_score -= 30
        if not self.reached_top and self.state == "FALLING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Arm Circles",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "arm_elevation": round(avg_elevation, 1),
            "arms_locked": self.arms_locked
        }