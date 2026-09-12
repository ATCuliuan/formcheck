import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class DynamicChestExpansionTracker:
    """
    Coaching logic for Dynamic Chest Expansions (Arm Pull-backs):
      1. CLOSED    -> standing tall, arms fully extended in front, hands close together.
      2. OPENING   -> arms sweeping horizontally outward and backward.
      3. EXTENDED  -> max wingspan reached, scapulae pinched, chest open.
      4. CLOSING   -> arms sweeping back to the front midline.
      5. CLOSED    -> rep complete when hands meet in the middle again.

    Instructor-level faults tracked independently:
      - dropped_arms: Failing to keep the arms parallel to the floor (shoulder height).
      - bent_elbows: Bending the arms to fake the range of motion.
      - shrugged_shoulders: Hiking the traps to the ears to compensate for tight pecs.
    """

    # Hip -> Shoulder -> Wrist angle. 90 degrees is perfectly parallel to the floor.
    ARM_ELEVATION_MIN = 70.0
    ARM_ELEVATION_MAX = 115.0
    
    ELBOW_STRAIGHT_MIN = 150.0       # Arms must remain straight to maximize the lever

    # We track the wingspan using a ratio of Wrist-to-Wrist distance vs Shoulder Width
    SPAN_CLOSED_THRESH = 1.0         # Wrists are within shoulder-width (in front of body)
    SPAN_OPEN_THRESH = 2.2           # Wrists are stretched wide (at least 2.2x shoulder width)

    SHRUG_TOLERANCE = 0.04           # Max upward displacement of shoulders from baseline

    def __init__(self):
        self.state = "CLOSED"
        self.rep_count = 0

        # Baselines for catching compensations
        self.baseline_shoulder_y = None
        self.baseline_shoulder_width = None

        # Per-rep tripwires
        self.arms_parallel = True
        self.arms_locked = True
        self.shoulders_relaxed = True
        self.full_stretch_achieved = False
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, reach your arms straight out in front to calibrate."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_CHEST_EXPANSION_MODEL

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

    def _calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # 1. Kinematics & Spatial Metrics
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        
        # Arm elevation (Hip -> Shoulder -> Wrist) tracks if arms are parallel to the floor
        l_elevation_angle = calculate_angle(l_hip, l_shoulder, l_wrist)
        r_elevation_angle = calculate_angle(r_hip, r_shoulder, r_wrist)
        avg_elevation = (l_elevation_angle + r_elevation_angle) / 2.0

        current_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        current_shoulder_width = self._calculate_distance(l_shoulder, r_shoulder)
        current_wrist_distance = self._calculate_distance(l_wrist, r_wrist)

        # Dynamic ratio to determine if arms are closed (in front) or open (at sides)
        # We use a ratio rather than hardcoded pixels so it works at any distance from the camera
        span_ratio = current_wrist_distance / (current_shoulder_width + 1e-6)

        # 2. Setup Guard: Wait for arms to be straight out in front, parallel to the floor
        is_neutral_stance = (
            self.ARM_ELEVATION_MIN < avg_elevation < self.ARM_ELEVATION_MAX
            and avg_elbow_angle > self.ELBOW_STRAIGHT_MIN
            and span_ratio < self.SPAN_CLOSED_THRESH
        )

        if is_neutral_stance and self.state == "CLOSED":
            self.ready_frames += 1
            if self.ready_frames == 15:
                # Capture baseline shoulder height to catch shrugging
                self.baseline_shoulder_y = current_shoulder_y
        else:
            if self.state == "CLOSED":
                self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True

        if not self.is_ready:
            return {
                "exercise": "Dynamic Chest Expansion",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Bring hands together straight out in front of your chest to begin.",
                "arm_elevation": round(avg_elevation, 1),
                "span_ratio": round(span_ratio, 2)
            }

        # 3. Fault Tripwires
        if self.state != "CLOSED":
            # Did they drop their arms to swing them?
            if avg_elevation < self.ARM_ELEVATION_MIN or avg_elevation > self.ARM_ELEVATION_MAX:
                self.arms_parallel = False
                
            # Did they bend their elbows?
            if avg_elbow_angle < self.ELBOW_STRAIGHT_MIN:
                self.arms_locked = False
            
            # Did they shrug their shoulders to their ears?
            if self.baseline_shoulder_y - current_shoulder_y > self.SHRUG_TOLERANCE:
                self.shoulders_relaxed = False

            # Did they achieve the full wide stretch?
            if span_ratio > self.SPAN_OPEN_THRESH:
                self.full_stretch_achieved = True

        # 4. Buffer & AI (Catches lumbar hyperextension / torso sway)
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
        if span_ratio < self.SPAN_CLOSED_THRESH:
            if self.state in ("OPENING", "EXTENDED", "CLOSING"):
                if self.full_stretch_achieved:
                    if not self.arms_locked:
                        self.current_feedback = "No rep: You bent your elbows! Keep the levers long."
                    elif not self.arms_parallel:
                        self.current_feedback = "No rep: You dropped your arms. Keep them at shoulder height!"
                    elif not self.shoulders_relaxed:
                        self.current_feedback = "No rep: You shrugged. Keep your shoulders packed down."
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: Don't arch your lower back to fake it. Core tight!"
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great expansion! Perfect chest stretch."
                else:
                    self.current_feedback = "No rep: You didn't open your arms wide enough."

            # Reset variables for the next rep
            self.state = "CLOSED"
            self.arms_parallel = True
            self.arms_locked = True
            self.shoulders_relaxed = True
            self.full_stretch_achieved = False
            self.ml_posture_maintained = True

        elif span_ratio > self.SPAN_OPEN_THRESH:
            self.state = "EXTENDED"
            self.current_feedback = "Pinch the shoulder blades! Now bring it back to the front."

        else:
            if self.state == "CLOSED":
                self.state = "OPENING"
                self.current_feedback = "Sweep the arms wide. Keep them up!"
            elif self.state == "EXTENDED":
                self.state = "CLOSING"
                self.current_feedback = "Bring the hands back together."

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.arms_locked:
            form_score -= 30
        if not self.arms_parallel:
            form_score -= 35
        if not self.shoulders_relaxed:
            form_score -= 20
        if not self.full_stretch_achieved and self.state == "CLOSING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Dynamic Chest Expansion",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "span_ratio": round(span_ratio, 2),
            "arms_parallel": self.arms_parallel
        }