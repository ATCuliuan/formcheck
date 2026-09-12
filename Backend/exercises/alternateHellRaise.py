import os
import math
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_HEEL_RAISE_MODEL


class BeginnerAlternatingHeelRaiseTracker:
    """
    Beginner-Friendly coaching logic for Alternating Heel Raises:
      1. FLAT      -> standing comfortably, both feet flat on the floor.
      2. LIFTING   -> shifting weight slightly, lifting one heel off the ground.
      3. TOP       -> modest heel lift achieved (doesn't need to be maximum height).
      4. LOWERING  -> returning the heel to the floor with control.
      5. FLAT      -> rep complete, ready to switch sides.

    Forgiving Fault Tracking:
      - squatting: Bending the knees too much (a "soft knee" is fine, but no squats).
      - double_lift: Lifting both heels at once (encouraging them to keep one planted for balance).
    """

    # Ankle angle (Knee -> Ankle -> Foot Index). 
    # Standing flat is usually 90-105 degrees. 
    HEEL_FLAT_MAX = 105.0            
    HEEL_LIFT_MIN = 118.0            # Very achievable lift target for a beginner
    
    KNEE_SQUAT_FAULT = 130.0         # Highly forgiving: allows up to 50 degrees of knee bend

    def __init__(self):
        self.state = "FLAT"
        self.rep_count = 0
        self.current_side = None     # "LEFT" or "RIGHT" working calf

        # Per-rep tripwires (kept simple for beginners)
        self.avoided_squat = True
        self.kept_balance = True     # True if they didn't try to lift both heels at once
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand comfortably with both feet flat on the floor to start."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init (Optional posture checking)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_HEEL_RAISE_MODEL

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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        l_foot = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        r_foot = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        # 1. Kinematics
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        # Plantar flexion angles
        l_ankle_angle = calculate_angle(l_knee, l_ankle, l_foot)
        r_ankle_angle = calculate_angle(r_knee, r_ankle, r_foot)

        # 2. Setup Guard: Wait for a relaxed, flat-footed stance
        is_neutral_stance = (
            l_ankle_angle < self.HEEL_FLAT_MAX 
            and r_ankle_angle < self.HEEL_FLAT_MAX
            and l_knee_angle > self.KNEE_SQUAT_FAULT
            and r_knee_angle > self.KNEE_SQUAT_FAULT
        )

        if is_neutral_stance and self.state == "FLAT":
            self.ready_frames += 1
        else:
            if self.state == "FLAT":
                self.ready_frames = 0

        if self.ready_frames > 10 and not self.is_ready: # Fast calibration
            self.is_ready = True

        if not self.is_ready:
            return {
                "exercise": "Alternating Heel Raise",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Plant both feet flat on the floor to get started.",
                "l_ankle_angle": round(l_ankle_angle, 1),
                "r_ankle_angle": round(r_ankle_angle, 1)
            }

        # 3. Forgiving Fault Tripwires
        if self.state != "FLAT":
            # Just making sure they don't turn it into a squat
            if l_knee_angle < self.KNEE_SQUAT_FAULT or r_knee_angle < self.KNEE_SQUAT_FAULT:
                self.avoided_squat = False
            
            # If both heels go up significantly at the same time during an alternating drill
            if l_ankle_angle > self.HEEL_LIFT_MIN and r_ankle_angle > self.HEEL_LIFT_MIN:
                self.kept_balance = False

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
        if is_neutral_stance:
            if self.state in ("LIFTING", "TOP", "LOWERING"):
                if not self.avoided_squat:
                    self.current_feedback = "Good try! Next time, keep your legs a bit straighter. No need to squat."
                    self.rep_count += 1
                elif not self.kept_balance:
                    self.current_feedback = "Whoops! Keep one foot flat on the floor to help you balance."
                else:
                    self.rep_count += 1
                    self.current_feedback = "Great lift! Nice and controlled."

            # Reset variables for the next rep
            self.state = "FLAT"
            self.avoided_squat = True
            self.kept_balance = True
            self.ml_posture_maintained = True
            self.current_side = None

        elif l_ankle_angle > self.HEEL_LIFT_MIN and r_ankle_angle < self.HEEL_FLAT_MAX + 5:
            self.state = "TOP"
            self.current_side = "LEFT"
            self.current_feedback = "Good height on the left heel! Now lower it slowly."
            
        elif r_ankle_angle > self.HEEL_LIFT_MIN and l_ankle_angle < self.HEEL_FLAT_MAX + 5:
            self.state = "TOP"
            self.current_side = "RIGHT"
            self.current_feedback = "Good height on the right heel! Now lower it slowly."

        else:
            if self.state == "FLAT":
                self.state = "LIFTING"
                self.current_feedback = "Shift your weight slightly and lift one heel up."
            elif self.state == "TOP":
                self.state = "LOWERING"
                self.current_feedback = "Control it on the way down..."

        # 6. Generous Form Scoring
        form_score = 100
        if not self.avoided_squat:
            form_score -= 15  # Small penalty
        if not self.kept_balance:
            form_score -= 25  # Slightly bigger penalty for losing the form of the exercise
            
        form_score = max(0, form_score)

        active_angle = l_ankle_angle if self.current_side == "LEFT" else (r_ankle_angle if self.current_side == "RIGHT" else None)

        return {
            "exercise": "Alternating Heel Raise",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "active_side": self.current_side,
            "active_ankle_angle": round(active_angle, 1) if active_angle else None
        }