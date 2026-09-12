import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_CALF_STRETCH_MODEL


class DynamicCalfStretchTracker:
    """
    Coaching logic for a Dynamic Step-Back Calf Stretch:
      1. STANDING  -> feet together, standing tall, knees straight.
      2. STEPPING  -> one foot steps back, front knee bends to accommodate the shift.
      3. STRETCH   -> back heel drives into the floor (dorsiflexion) while the back knee 
                      remains completely straight to target the gastrocnemius.
      4. RISING    -> push off the back foot, returning to the neutral standing position.
      5. STANDING  -> rep complete once both legs are straight again.

    Instructor-level faults tracked independently:
      - bent_back_knee: Bending the back knee shifts the stretch to the soleus or kills it entirely.
      - floating_heel: Not driving the heel down (inadequate dorsiflexion angle).
      - posture_collapse: Hinging too far forward at the hips instead of keeping the chest up.
    """

    STANDING_KNEE_ANGLE = 160.0   # Both knees should be straight to count as standing
    FRONT_KNEE_BENT_MAX = 135.0   # Front knee must bend to allow the back leg to stretch
    STRAIGHT_LEG_MIN = 150.0      # Back leg (stretching leg) must stay straight
    
    # Ankle angle (Knee -> Ankle -> Foot Index). Decreases during dorsiflexion (heel down).
    DORSIFLEXION_TARGET = 85.0    
    STANDING_ANKLE_MIN = 100.0    

    HIP_HINGE_FAULT = 120.0       # Torso leaning too far forward (Shoulder -> Hip -> Knee)

    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.current_side = None  # "LEFT" or "RIGHT" stretching (back) leg

        # Per-rep tripwires
        self.stretch_achieved = False
        self.back_leg_straight = True
        self.heel_grounded = False
        self.posture_maintained = True
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, feet together, to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_CALF_STRETCH_MODEL

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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        l_foot = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        r_foot = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        # Ankle dorsiflexion angles
        l_ankle_angle = calculate_angle(l_knee, l_ankle, l_foot)
        r_ankle_angle = calculate_angle(r_knee, r_ankle, r_foot)

        # Posture angles (Shoulder -> Hip -> Knee)
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        # 1. Setup Guard: hold a neutral standing position before we start counting.
        is_neutral_stance = (
            l_knee_angle > self.STANDING_KNEE_ANGLE
            and r_knee_angle > self.STANDING_KNEE_ANGLE
            and l_ankle_angle > self.STANDING_ANKLE_MIN
            and r_ankle_angle > self.STANDING_ANKLE_MIN
        )

        if is_neutral_stance:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "STANDING"

        if not self.is_ready:
            return {
                "exercise": "Dynamic Calf Stretch",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Stand tall with legs straight to initialize",
                "stretching_ankle_angle": None,
                "side": self.current_side,
                "stretch_achieved": False,
            }

        # 2. Identify the working (stretching) leg. 
        # The leg that remains straight while the other bends is the one stepping back to stretch.
        if l_knee_angle < self.FRONT_KNEE_BENT_MAX and r_knee_angle > self.STRAIGHT_LEG_MIN:
            self.current_side = "RIGHT"
        elif r_knee_angle < self.FRONT_KNEE_BENT_MAX and l_knee_angle > self.STRAIGHT_LEG_MIN:
            self.current_side = "LEFT"

        active_ankle_angle = r_ankle_angle if self.current_side == "RIGHT" else l_ankle_angle
        active_knee_angle = r_knee_angle if self.current_side == "RIGHT" else l_knee_angle

        # 3. Fault Tripwires
        if self.current_side:
            # Did they bend the back knee during the stretch?
            if active_knee_angle < self.STRAIGHT_LEG_MIN:
                self.back_leg_straight = False
            
            # Did they lean too far forward?
            if avg_hip_angle < self.HIP_HINGE_FAULT:
                self.posture_maintained = False
            
            # Did they achieve the deep dorsiflexion indicating a full heel drive?
            if active_ankle_angle < self.DORSIFLEXION_TARGET:
                self.heel_grounded = True

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
            if self.state in ("STEPPING", "STRETCH_BOTTOM", "RISING"):
                if self.state == "STRETCH_BOTTOM" or self.heel_grounded:
                    if not self.back_leg_straight:
                        self.current_feedback = "No rep: You bent your back knee! Keep it locked out."
                    elif not self.heel_grounded:
                        self.current_feedback = "No rep: Drive that back heel all the way into the floor."
                    elif not self.posture_maintained:
                        self.current_feedback = "No rep: Keep your chest up, don't collapse forward."
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected broken form."
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great stretch! Excellent heel drive."
                else:
                    self.current_feedback = "No rep: Didn't step back far enough for a stretch."

            self.state = "STANDING"
            self.heel_grounded = False
            self.back_leg_straight = True
            self.posture_maintained = True
            self.ml_posture_maintained = True
            self.current_side = None

        elif self.current_side and active_ankle_angle <= self.DORSIFLEXION_TARGET:
            self.state = "STRETCH_BOTTOM"
            self.current_feedback = f"Hold it! Good stretch on the {self.current_side.lower()} calf. Now push back up."

        else:
            if self.state in ("STANDING",):
                self.state = "STEPPING"
                self.current_feedback = "Step back, keep the back leg straight, drive the heel down."
            elif self.state in ("STRETCH_BOTTOM", "RISING"):
                self.state = "RISING"
                self.current_feedback = "Push off and return to standing."

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.back_leg_straight:
            form_score -= 40
        if not self.posture_maintained:
            form_score -= 30
        if self.current_side and active_ankle_angle > self.DORSIFLEXION_TARGET and self.state == "RISING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Dynamic Calf Stretch",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "stretching_ankle_angle": round(active_ankle_angle, 1) if self.current_side else None,
            "side": self.current_side,
            "stretch_achieved": self.heel_grounded,
        }