import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class ShoulderRollsTracker:
    """
    Coaching logic for Dynamic Shoulder Rolls:
      1. NEUTRAL   -> arms resting at sides, shoulders relaxed, neck straight.
      2. SHRUGGING -> shoulders elevate toward the ears (Upper Traps).
      3. ROLLING   -> shoulders pull backward (Scapular Retraction) or forward (Protraction).
      4. FALLING   -> shoulders depress back downward.
      5. NEUTRAL   -> rep complete once returned to the relaxed baseline.

    Instructor-level faults tracked independently:
      - arm_rowing: Bending the elbows to swing the shoulders (arms must stay dead/relaxed).
      - shrug_only: Moving straight up and down without actually rolling (no scapular width change).
      - neck_craning: Jutting the chin forward to compensate for locked upper back mobility.
    """

    ELEVATION_THRESH = 0.035         # How much the Y-coordinate must rise to count a shrug
    ROLL_WIDTH_THRESH = 0.025        # How much the shoulder width (X-coord) must shrink to prove a roll
    
    ELBOW_STRAIGHT_MIN = 150.0       # Arms must remain relaxed/straight
    NECK_CRANE_TOLERANCE = 0.04      # Max forward drop of the nose relative to the shoulder baseline

    def __init__(self):
        self.state = "NEUTRAL"
        self.rep_count = 0

        # Baselines for catching compensations and ranges
        self.baseline_shoulder_y = None
        self.baseline_shoulder_width = None
        self.baseline_neck_length = None

        # Per-rep tripwires
        self.arms_relaxed = True
        self.neck_stable = True
        self.roll_achieved = False
        self.ml_posture_maintained = True

        # Intra-rep trackers
        self.min_shoulder_width = 999.0

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, arms dead at your sides, neck neutral to calibrate."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_SHOULDER_ROLLS_MODEL

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
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # 1. Kinematics & Geometric Proxies
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        
        # Screen space metrics (Y is inverted: smaller Y means higher up)
        current_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        current_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        
        # Distance from Nose to the center of the shoulders (proxy for neck posture)
        shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2.0
        shoulder_center_y = current_shoulder_y
        current_neck_length = np.sqrt((nose.x - shoulder_center_x)**2 + (nose.y - shoulder_center_y)**2)

        # 2. Setup Guard: Wait for a neutral, relaxed posture
        is_neutral_stance = (avg_elbow_angle > self.ELBOW_STRAIGHT_MIN)

        if is_neutral_stance and self.state == "NEUTRAL":
            self.ready_frames += 1
            if self.ready_frames == 15:
                # Capture baselines once the user is stable
                self.baseline_shoulder_y = current_shoulder_y
                self.baseline_shoulder_width = current_shoulder_width
                self.baseline_neck_length = current_neck_length
        else:
            if self.state == "NEUTRAL":
                self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True

        if not self.is_ready:
            return {
                "exercise": "Shoulder Rolls",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Relax your arms straight down and look straight ahead to initialize.",
            }

        # 3. Fault Tripwires
        if self.state != "NEUTRAL":
            # Did they bend their elbows to row the shoulders?
            if avg_elbow_angle < self.ELBOW_STRAIGHT_MIN:
                self.arms_relaxed = False
            
            # Did they crane their neck forward? (Neck length shrinks significantly if chin drops)
            if self.baseline_neck_length - current_neck_length > self.NECK_CRANE_TOLERANCE:
                self.neck_stable = False

            # Track the maximum width change to ensure a "roll" happened (scapulae pinched together)
            if current_shoulder_width < self.min_shoulder_width:
                self.min_shoulder_width = current_shoulder_width
                
            if (self.baseline_shoulder_width - self.min_shoulder_width) > self.ROLL_WIDTH_THRESH:
                self.roll_achieved = True

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
        # Shrug detection (Y gets smaller as shoulders go UP)
        is_elevated = current_shoulder_y < (self.baseline_shoulder_y - self.ELEVATION_THRESH)
        is_depressed = current_shoulder_y > (self.baseline_shoulder_y - (self.ELEVATION_THRESH * 0.3))

        if self.state == "NEUTRAL" and is_elevated:
            self.state = "SHRUGGING"
            self.current_feedback = "Good shrug! Now squeeze the shoulder blades and roll them."
            
        elif self.state in ("SHRUGGING", "ROLLING"):
            if self.roll_achieved and is_elevated:
                self.state = "ROLLING"
                self.current_feedback = "Great roll! Keep the arms dead and pull the shoulders down."
                
            if is_depressed:
                self.state = "FALLING"

        elif self.state == "FALLING" and is_depressed:
            if not self.arms_relaxed:
                self.current_feedback = "No rep: You bent your elbows! Let your arms hang dead."
            elif not self.neck_stable:
                self.current_feedback = "No rep: You craned your neck forward. Keep the chin tucked and steady."
            elif not self.roll_achieved:
                self.current_feedback = "No rep: That was just a shrug! You need to roll the shoulders forward or back."
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected spine shifting. Move ONLY the shoulders."
            else:
                self.rep_count += 1
                self.current_feedback = "Perfect shoulder roll! Total scapular isolation."

            # Reset variables for the next rep
            self.state = "NEUTRAL"
            self.arms_relaxed = True
            self.neck_stable = True
            self.roll_achieved = False
            self.ml_posture_maintained = True
            self.min_shoulder_width = 999.0

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.arms_relaxed:
            form_score -= 40
        if not self.neck_stable:
            form_score -= 30
        if not self.roll_achieved and self.state == "FALLING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Rolls",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "arms_locked": self.arms_relaxed,
            "roll_achieved": self.roll_achieved
        }