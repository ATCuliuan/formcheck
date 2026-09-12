import os
import math
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_REACH_TO_TOE_MODEL


class ForgivingOverheadReachTracker:
    """
    Beginner-Friendly logic for Dynamic Overhead Reach to Forward Fold:
      1. OVERHEAD  -> standing up, arms reaching generally upward.
      2. HINGING   -> hips push back, arms sweep downward.
      3. BOTTOM    -> hands reach down toward the shins/ankles (deep stretch not required).
      4. RISING    -> standing back up.
      5. OVERHEAD  -> rep complete when standing with arms raised again.

    Instructor-level faults are now much more forgiving:
      - Deep Squatting: Only triggers a fault if they drop into a literal 90-degree squat.
      - Shin-Level Reach: Reaching past the knees to the shins now counts as a success.
      - Relaxed Overhead: Arms just need to be above the head, not perfectly vertical.
    """

    STANDING_HIP_ANGLE = 145.0       # Relaxed: Allows a slight forward lean at the top
    KNEE_SQUAT_FAULT = 100.0         # Relaxed: Only fail them if they deeply squat (< 100 degrees)
    
    OVERHEAD_ARM_MIN = 120.0         # Relaxed: Arms just need to go up, even if shoulders are tight
    HINGE_ENGAGED_ANGLE = 135.0      # Relaxed: Triggers the downward phase sooner
    
    # Relaxed distance: 0.40 means reaching the mid-shins counts as a successful "toe touch"
    REACH_THRESH = 0.40          

    def __init__(self):
        self.state = "OVERHEAD"
        self.rep_count = 0

        # Per-rep tripwires (now much harder to trip)
        self.avoided_squat = True
        self.reach_achieved = False
        self.overhead_reach_achieved = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand up and reach your arms up to start."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_REACH_TO_TOE_MODEL

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

        # 1. Kinematics
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
        
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        # Arm elevation
        l_elevation = calculate_angle(l_hip, l_shoulder, l_wrist)
        r_elevation = calculate_angle(r_hip, r_shoulder, r_wrist)
        avg_elevation = (l_elevation + r_elevation) / 2.0

        # Distance to toes
        l_reach_dist = self._calculate_distance(l_wrist, l_ankle)
        r_reach_dist = self._calculate_distance(r_wrist, r_ankle)
        avg_reach_dist = (l_reach_dist + r_reach_dist) / 2.0

        # 2. Setup Guard: More forgiving top position
        is_extended_enough = (
            avg_hip_angle > self.STANDING_HIP_ANGLE
            and avg_elevation > self.OVERHEAD_ARM_MIN
        )

        if is_extended_enough:
            self.ready_frames += 1
            self.overhead_reach_achieved = True
        else:
            self.ready_frames = 0

        if self.ready_frames > 10 and not self.is_ready:  # Faster calibration
            self.is_ready = True
            self.state = "OVERHEAD"

        if not self.is_ready:
            return {
                "exercise": "Forgiving Forward Fold",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Stand tall and reach up to begin.",
                "arm_elevation": round(avg_elevation, 1)
            }

        # 3. Forgiving Fault Tripwires
        if self.state != "OVERHEAD":
            # Only flag a fault if they drop into a deep squat
            if avg_knee_angle < self.KNEE_SQUAT_FAULT:
                self.avoided_squat = False
            
            # Shin-level reach is now acceptable
            if avg_reach_dist < self.REACH_THRESH:
                self.reach_achieved = True

        # 4. State Machine & Coaching
        if is_extended_enough:
            if self.state in ("HINGING", "BOTTOM", "RISING"):
                if self.reach_achieved:
                    if not self.avoided_squat:
                        self.current_feedback = "Good rep, but try not to bend your knees quite that much next time!"
                        self.rep_count += 1
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great reach! Nice job."
                else:
                    self.current_feedback = "No rep: Try to reach down just a little further, past your knees."

            # Reset variables
            self.state = "OVERHEAD"
            self.avoided_squat = True
            self.reach_achieved = False
            self.overhead_reach_achieved = True

        elif avg_reach_dist < self.REACH_THRESH:
            self.state = "BOTTOM"
            self.current_feedback = "Good depth! Now stand back up and reach high."
            self.overhead_reach_achieved = False 

        else:
            if self.state == "OVERHEAD" and avg_hip_angle < self.HINGE_ENGAGED_ANGLE:
                self.state = "HINGING"
                self.current_feedback = "Soft knees are okay, reach down for your shins or toes."
            
            elif self.state == "BOTTOM":
                self.state = "RISING"
                self.current_feedback = "Stand tall!"

            if self.state in ("HINGING", "RISING") and not self.overhead_reach_achieved and avg_hip_angle < self.HINGE_ENGAGED_ANGLE:
                self.current_feedback = "Make sure to stand all the way up between reps!"

        # 5. Generous Form Scoring
        form_score = 100
        # Gentle deductions rather than massive penalties
        if not self.avoided_squat:
            form_score -= 20
        if not self.overhead_reach_achieved and self.state == "HINGING":
            form_score -= 20
        if not self.reach_achieved and self.state == "RISING":
            form_score -= 20
            
        # Give them a small boost if they kept their legs mostly straight (positive reinforcement)
        if avg_knee_angle > 140.0 and self.state == "BOTTOM":
            form_score = min(100, form_score + 10)
            
        form_score = max(0, form_score)

        return {
            "exercise": "Forgiving Forward Fold",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "reach_distance": round(avg_reach_dist, 3),
            "squat_avoided": self.avoided_squat
        }