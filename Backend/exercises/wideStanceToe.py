import os
import math
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_TOE_TOUCH_MODEL


class WideStanceToeTouchTracker:
    """
    Coaching logic for a Wide-Stance Alternating Toe Touch (Windmill):
      1. STANDING  -> wide stance (feet wider than shoulders), hips extended, legs straight.
      2. HINGING   -> hips push back, torso rotates, one arm reaches for the opposite toe.
      3. BOTTOM    -> maximum rotation/hinge, hand touches (or gets very close to) the foot.
      4. RISING    -> driving back up to the center.
      5. STANDING  -> rep complete ONLY when hips are fully extended again.

    Instructor-level faults tracked independently:
      - bent_knees: Bending the knees to reach the floor (cheats the hamstring stretch).
      - half_rep: Swinging side-to-side without standing fully upright between reps.
      - narrow_stance: Feet not wide enough to stretch the adductors.
    """

    STANDING_HIP_ANGLE = 160.0       # Hips must be fully extended to count a completed rep
    STANDING_KNEE_ANGLE = 150.0      # Both knees must stay locked/straight the entire time
    
    BENT_KNEE_FAULT = 140.0          # If either knee drops below this, they are cheating
    HINGE_ENGAGED_ANGLE = 135.0      # Torso is hinged forward enough to start a rep

    STANCE_MULTIPLIER = 1.4          # Ankles must be at least 1.4x shoulder-width apart
    TOUCH_DISTANCE_THRESH = 0.15     # Normalized distance between wrist and opposite ankle

    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.current_side = None  # "LEFT" (reaching for left toe) or "RIGHT"

        # Per-rep tripwires
        self.legs_straight = True
        self.target_reached = False
        self.full_extension_achieved = True
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Take a wide stance, legs straight, stand tall to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_TOE_TOUCH_MODEL

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
        # 2D distance using x and y for screen-space proximity
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Kinematics
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        # Stance width check
        shoulder_width = self._calculate_distance(l_shoulder, r_shoulder)
        ankle_width = self._calculate_distance(l_ankle, r_ankle)
        is_wide_stance = ankle_width > (shoulder_width * self.STANCE_MULTIPLIER)

        # 1. Setup Guard: Wait for a neutral standing posture with wide feet
        is_neutral_stance = (
            l_knee_angle > self.STANDING_KNEE_ANGLE
            and r_knee_angle > self.STANDING_KNEE_ANGLE
            and avg_hip_angle > self.STANDING_HIP_ANGLE
        )

        if is_neutral_stance and is_wide_stance:
            self.ready_frames += 1
            self.full_extension_achieved = True
        else:
            self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "STANDING"

        if not self.is_ready:
            feedback = "Stand tall with legs straight to initialize"
            if not is_wide_stance:
                feedback = "Step your feet wider! They need to be wider than shoulder-width."
                
            return {
                "exercise": "Wide-Stance Toe Touch",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": feedback,
                "target_reached": False,
            }

        # 2. Identify side and proximity
        # Check cross-body reach: Right Wrist to Left Ankle, Left Wrist to Right Ankle
        r_wrist_to_l_ankle = self._calculate_distance(r_wrist, l_ankle)
        l_wrist_to_r_ankle = self._calculate_distance(l_wrist, r_ankle)

        # Once they hinge past the threshold, determine which side they are reaching for
        if avg_hip_angle < self.HINGE_ENGAGED_ANGLE:
            if r_wrist_to_l_ankle < l_wrist_to_r_ankle:
                self.current_side = "LEFT"   # Reaching toward left foot
            else:
                self.current_side = "RIGHT"  # Reaching toward right foot

        active_distance = r_wrist_to_l_ankle if self.current_side == "LEFT" else l_wrist_to_r_ankle

        # 3. Fault Tripwires
        if self.state != "STANDING":
            # Did they bend either knee to cheat the range of motion?
            if l_knee_angle < self.BENT_KNEE_FAULT or r_knee_angle < self.BENT_KNEE_FAULT:
                self.legs_straight = False
            
            # Did they reach the target?
            if active_distance < self.TOUCH_DISTANCE_THRESH:
                self.target_reached = True

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
            if self.state in ("HINGING", "BOTTOM", "RISING"):
                if self.target_reached:
                    if not self.legs_straight:
                        self.current_feedback = "No rep: You bent your knees! Keep them locked to stretch the hamstrings."
                    elif not is_wide_stance:
                        self.current_feedback = "No rep: Your feet shifted too close together."
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected rounded upper back form."
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great reach! Full extension at the top."
                else:
                    self.current_feedback = "No rep: You didn't reach far enough toward the foot."

            # Reset variables for the next rep
            self.state = "STANDING"
            self.legs_straight = True
            self.target_reached = False
            self.ml_posture_maintained = True
            self.current_side = None

        elif self.current_side and active_distance <= self.TOUCH_DISTANCE_THRESH:
            self.state = "BOTTOM"
            self.current_feedback = f"Great touch on the {self.current_side.lower()} side! Squeeze glutes and stand UP."
            self.full_extension_achieved = False # Force them to stand fully up before next rep

        else:
            if self.state in ("STANDING",):
                self.state = "HINGING"
                self.current_feedback = "Hinge at the hips, reach across the body, keep legs straight."
            elif self.state == "BOTTOM":
                self.state = "RISING"
                self.current_feedback = "Stand all the way back up to the middle!"
            
            # Catching the "Swing" half-rep fault dynamically
            if self.state in ("HINGING", "RISING") and not self.full_extension_achieved and avg_hip_angle < self.HINGE_ENGAGED_ANGLE:
                self.current_feedback = "Don't just swing side to side! Stand fully up between reps."

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.legs_straight:
            form_score -= 40
        if not is_wide_stance:
            form_score -= 20
        if not self.full_extension_achieved and self.state == "HINGING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Wide-Stance Toe Touch",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "working_side": self.current_side,
            "distance_to_target": round(active_distance, 3) if self.current_side else None,
            "legs_locked": self.legs_straight
        }