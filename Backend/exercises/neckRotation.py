import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class NeckRotationTracker:
    """
    Coaching logic for a Dynamic Neck Rotation (Cervical Yaw):
      1. NEUTRAL   -> facing dead ahead, shoulders relaxed and level.
      2. TURNING   -> rotating the head to one side smoothly.
      3. END_RANGE -> maximum pain-free cervical rotation achieved.
      4. RETURNING -> bringing the head back to the center.
      5. NEUTRAL   -> rep complete when facing forward again.

    Instructor-level faults tracked independently:
      - shoulder_twist: Twisting the torso to fake neck mobility.
      - head_tilt: Dropping the ear to the shoulder (lateral flexion) instead of pure rotation.
      - chin_tuck: Dropping the chin down toward the chest during the sweep.
    """

    # Yaw is calculated as a ratio of (Nose-to-Left-Ear) / (Nose-to-Right-Ear) distances.
    # Facing forward = ~1.0. 
    YAW_NEUTRAL_MIN = 0.7
    YAW_NEUTRAL_MAX = 1.3
    
    # Thresholds for a successful deep rotation (adjust based on camera focal length)
    YAW_LEFT_THRESH = 0.25   # Nose is very close to the left ear horizontally
    YAW_RIGHT_THRESH = 4.0   # Nose is very close to the right ear horizontally

    SHOULDER_TWIST_TOLERANCE = 0.05  # Max allowed shift in shoulder X-coordinates
    HEAD_TILT_TOLERANCE = 0.04       # Max allowed shift in Nose Y-coordinate relative to ears

    def __init__(self):
        self.state = "NEUTRAL"
        self.rep_count = 0
        self.current_side = None  # "LEFT" or "RIGHT" rotation

        # Baselines for catching compensations
        self.baseline_shoulder_width = None
        self.baseline_nose_y = None

        # Per-rep tripwires
        self.shoulders_stable = True
        self.head_level = True
        self.target_reached = False
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Sit or stand tall, face forward, relax your shoulders."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_NECK_ROTATION_MODEL

        if os.path.exists(model_path):
            try:
                self.model = STGCN(num_classes=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception:
                self.model = None

    def _extract_body_keypoints(self, landmarks):
        # Including head landmarks (0: Nose, 7: L_Ear, 8: R_Ear) and shoulders (11, 12)
        target_indices = [0, 7, 8, 11, 12, 13, 14, 23, 24, 25, 26]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # 1. Kinematics & Geometric Proxies
        # Yaw Ratio: Calculates horizontal head rotation without needing dense FaceMesh
        left_ear_dist = abs(nose.x - l_ear.x)
        right_ear_dist = abs(nose.x - r_ear.x)
        yaw_ratio = left_ear_dist / (right_ear_dist + 1e-6)

        current_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        current_nose_y_relative = nose.y - ((l_ear.y + r_ear.y) / 2.0)

        # 2. Setup Guard: Wait for a neutral, forward-facing posture
        is_neutral_stance = (
            self.YAW_NEUTRAL_MIN <= yaw_ratio <= self.YAW_NEUTRAL_MAX
        )

        if is_neutral_stance:
            self.ready_frames += 1
            if self.ready_frames == 15:
                # Capture baselines once the user is stable and facing forward
                self.baseline_shoulder_width = current_shoulder_width
                self.baseline_nose_y = current_nose_y_relative
        else:
            self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "NEUTRAL"

        if not self.is_ready:
            return {
                "exercise": "Neck Rotation",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Face the camera directly and hold still to calibrate",
                "yaw_ratio": round(yaw_ratio, 2),
                "target_reached": False,
            }

        # 3. Identify Rotation Direction
        if self.state in ("NEUTRAL", "TURNING"):
            if yaw_ratio < self.YAW_NEUTRAL_MIN - 0.15:
                self.current_side = "LEFT"
            elif yaw_ratio > self.YAW_NEUTRAL_MAX + 0.15:
                self.current_side = "RIGHT"

        # 4. Fault Tripwires
        if self.current_side and self.state != "NEUTRAL":
            # Did they twist their shoulders to fake the rotation? (Shoulder width changes in 2D projection)
            if abs(current_shoulder_width - self.baseline_shoulder_width) > self.SHOULDER_TWIST_TOLERANCE:
                self.shoulders_stable = False
            
            # Did they tilt their head (ear to shoulder) or tuck their chin?
            if abs(current_nose_y_relative - self.baseline_nose_y) > self.HEAD_TILT_TOLERANCE:
                self.head_level = False

            # Did they hit the end range?
            if self.current_side == "LEFT" and yaw_ratio <= self.YAW_LEFT_THRESH:
                self.target_reached = True
            elif self.current_side == "RIGHT" and yaw_ratio >= self.YAW_RIGHT_THRESH:
                self.target_reached = True

        # 5. Buffer & AI Posture check (spine integrity)
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

        # 6. State Machine & Coaching
        if is_neutral_stance:
            if self.state in ("TURNING", "END_RANGE", "RETURNING"):
                if self.target_reached:
                    if not self.shoulders_stable:
                        self.current_feedback = "No rep: You twisted your torso! Keep your shoulders locked facing forward."
                    elif not self.head_level:
                        self.current_feedback = "No rep: Don't tilt your head or tuck your chin. Keep it level."
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected slouching. Sit up straight."
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Excellent rotation. Total neck isolation."
                else:
                    self.current_feedback = "No rep: You didn't turn your head far enough."

            # Reset variables for the next rep
            self.state = "NEUTRAL"
            self.shoulders_stable = True
            self.head_level = True
            self.target_reached = False
            self.ml_posture_maintained = True
            self.current_side = None

        elif self.current_side:
            if (self.current_side == "LEFT" and yaw_ratio <= self.YAW_LEFT_THRESH) or \
               (self.current_side == "RIGHT" and yaw_ratio >= self.YAW_RIGHT_THRESH):
                self.state = "END_RANGE"
                self.current_feedback = f"Good range on the {self.current_side.lower()} side! Now smoothly return to center."
            else:
                if self.state == "NEUTRAL":
                    self.state = "TURNING"
                    self.current_feedback = "Turn slowly. Keep the chin level."
                elif self.state == "END_RANGE":
                    self.state = "RETURNING"
                    self.current_feedback = "Bring it back to the middle."

        # 7. Dynamic Form Scoring
        form_score = 100
        if not self.shoulders_stable:
            form_score -= 40
        if not self.head_level:
            form_score -= 35
        if not self.target_reached and self.state == "RETURNING":
            form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Neck Rotation",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "direction": self.current_side,
            "yaw_ratio": round(yaw_ratio, 2),
            "target_reached": self.target_reached
        }