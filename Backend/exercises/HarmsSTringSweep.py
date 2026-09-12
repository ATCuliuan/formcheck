import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class DynamicHamstringSweepTracker:
    """
    Coaching logic for a Dynamic Hamstring Sweep (Scoop Stretch):
      1. STANDING  -> feet together, standing tall.
      2. PLANTING  -> one foot steps slightly forward, heel planted, toes up.
      3. SWEEPING  -> hips hinge back, back knee bends, front leg stays perfectly straight. 
                      Arms sweep down towards the front foot.
      4. BOTTOM    -> maximum safe hip flexion achieved.
      5. RISING    -> driving back up to a neutral standing position.
      6. STANDING  -> rep complete once both legs and hips are extended.

    Instructor-level faults tracked independently:
      - front_knee_bent: The stretching leg must remain straight to load the hamstring.
      - stiff_back_leg: The support leg MUST bend to allow the hips to sit back safely.
      - shallow_hinge: Failing to push the hips back and drop the chest (no real stretch).
    """

    STANDING_KNEE_ANGLE = 160.0    # Both knees straight for neutral stance
    STANDING_HIP_ANGLE = 160.0     # Hips extended for neutral stance
    
    FRONT_LEG_STRAIGHT_MIN = 150.0 # Working (front) leg must stay straight
    BACK_LEG_BENT_MAX = 135.0      # Support (back) leg must bend to allow the hinge
    
    SWEEP_DEPTH_ANGLE = 110.0      # Shoulder-Hip-Knee angle indicating a good hinge
    SHALLOW_HINGE_FAULT = 130.0    # If they never hinge past this, it's a half-rep

    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.current_side = None  # "LEFT" or "RIGHT" working (front) leg

        # Per-rep tripwires
        self.front_leg_locked = True
        self.back_leg_bent = False
        self.good_depth_reached = False
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, feet together, ready to sweep"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_HAMSTRING_SWEEP_MODEL

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

        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Kinematics
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        
        # Average the hip angle for the hinge depth
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        # 1. Setup Guard: Wait for a neutral standing posture
        is_neutral_stance = (
            l_knee_angle > self.STANDING_KNEE_ANGLE
            and r_knee_angle > self.STANDING_KNEE_ANGLE
            and avg_hip_angle > self.STANDING_HIP_ANGLE
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
                "exercise": "Dynamic Hamstring Sweep",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Stand tall with legs and hips fully straight to start",
                "working_leg": self.current_side,
                "hip_angle": round(avg_hip_angle, 1)
            }

        # 2. Identify working leg based on the asymmetrical stance
        # The leg that remains straight while the hips hinge is the hamstring being stretched.
        # The support leg will bend to allow the hips to drop.
        if avg_hip_angle < self.SHALLOW_HINGE_FAULT:
            if l_knee_angle < r_knee_angle - 15.0: # Left is bending more, Right is straight
                self.current_side = "RIGHT"
            elif r_knee_angle < l_knee_angle - 15.0: # Right is bending more, Left is straight
                self.current_side = "LEFT"

        active_knee_angle = r_knee_angle if self.current_side == "RIGHT" else l_knee_angle
        support_knee_angle = l_knee_angle if self.current_side == "RIGHT" else r_knee_angle

        # 3. Fault Tripwires
        if self.current_side:
            # Did they bend the stretching leg?
            if active_knee_angle < self.FRONT_LEG_STRAIGHT_MIN:
                self.front_leg_locked = False
            
            # Did they actually bend the support leg to sit back?
            if support_knee_angle < self.BACK_LEG_BENT_MAX:
                self.back_leg_bent = True
            
            # Did they achieve enough of a hip hinge?
            if avg_hip_angle < self.SWEEP_DEPTH_ANGLE:
                self.good_depth_reached = True

        # 4. Buffer & AI (Posture integrity)
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
            if self.state in ("PLANTING", "SWEEPING", "BOTTOM", "RISING"):
                if self.good_depth_reached:
                    if not self.front_leg_locked:
                        self.current_feedback = "No rep: You bent your front knee! Keep the stretching leg straight."
                    elif not self.back_leg_bent:
                        self.current_feedback = "No rep: Bend your back knee to sit into the stretch, don't just lean over."
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected a rounded lower back. Keep the chest up."
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great sweep! Perfect hamstring isolation."
                else:
                    self.current_feedback = "No rep: You didn't hinge deeply enough to stretch the hamstring."

            # Reset variables for the next rep
            self.state = "STANDING"
            self.front_leg_locked = True
            self.back_leg_bent = False
            self.good_depth_reached = False
            self.ml_posture_maintained = True
            self.current_side = None

        elif self.current_side and avg_hip_angle <= self.SWEEP_DEPTH_ANGLE:
            self.state = "BOTTOM"
            self.current_feedback = f"Great depth on the {self.current_side.lower()} leg! Sweep and drive back up."

        else:
            if self.state in ("STANDING",):
                self.state = "PLANTING"
                self.current_feedback = "Heel out, toes up. Push your hips back."
            elif self.state == "PLANTING" and avg_hip_angle < self.SHALLOW_HINGE_FAULT:
                self.state = "SWEEPING"
                self.current_feedback = "Keep the front leg locked out and sweep the floor!"
            elif self.state in ("BOTTOM", "RISING"):
                self.state = "RISING"
                self.current_feedback = "Squeeze the glutes and return to standing."

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.front_leg_locked:
            form_score -= 40
        if self.current_side and support_knee_angle > self.BACK_LEG_BENT_MAX and self.state == "BOTTOM":
            form_score -= 30
        if not self.good_depth_reached and self.state == "RISING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Dynamic Hamstring Sweep",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "working_leg": self.current_side,
            "hip_angle": round(avg_hip_angle, 1),
            "depth_reached": self.good_depth_reached
        }