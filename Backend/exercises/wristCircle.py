import os
import math
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_WRIST_CIRCLES_MODEL


class WristCirclesTracker:
    """
    Coaching logic for Dynamic Wrist Circles:
      1. CALIBRATING -> arms raised comfortably in front or at sides, held perfectly still.
      2. TRACKING    -> wrists begin rotating while the elbows and forearms remain anchored in space.
      3. REVOLUTION  -> index finger completes a full 360-degree sweep around the wrist joint.
      4. COMPLETE    -> rep counted, ready for the next rotation.

    Instructor-level faults tracked independently:
      - arm_sway: Moving the elbows or wrists through space to fake the circle.
      - partial_rotation: Wagging the hands back and forth instead of completing a full 360° circle.
    """

    # Tolerances for isolating the wrist. If the elbows or wrists move more than this
    # distance from their anchored baseline, the user is cheating with their arms.
    STABILITY_TOLERANCE = 0.05       
    
    # Degrees required to count a full circle (giving a tiny buffer for real-world kinematics)
    TARGET_ROTATION = 340.0          

    def __init__(self):
        self.state = "CALIBRATING"
        self.rep_count = 0

        # Baselines for isolation tracking
        self.base_l_elbow = None
        self.base_r_elbow = None
        self.base_l_wrist = None
        self.base_r_wrist = None

        # Rotation tracking (Phase unwrapping)
        self.prev_l_angle = None
        self.prev_r_angle = None
        self.cum_l_rotation = 0.0
        self.cum_r_rotation = 0.0

        # Per-rep tripwires
        self.arms_isolated = True
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Hold your arms up and keep them perfectly still to calibrate."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_WRIST_CIRCLES_MODEL

        if os.path.exists(model_path):
            try:
                self.model = STGCN(num_classes=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception:
                self.model = None

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _get_hand_angle(self, wrist, index):
        # Calculates the 2D angle of the hand vector relative to the wrist
        # Returns angle in degrees [-180, 180]
        return math.degrees(math.atan2(index.y - wrist.y, index.x - wrist.x))

    def _get_angle_delta(self, current_angle, previous_angle):
        # Phase unwrapping: Ensures that jumping from 179 to -179 is calculated 
        # as a 2-degree delta, not a 358-degree jump.
        delta = current_angle - previous_angle
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
        return delta

    def process_frame(self, landmarks):
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # We use the index finger to trace the outer edge of the circle
        l_index = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value]
        r_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value]

        # 1. Arm Stability Check (Are they moving their whole arm?)
        if self.is_ready:
            l_elbow_move = self._calculate_distance(l_elbow, self.base_l_elbow)
            r_elbow_move = self._calculate_distance(r_elbow, self.base_r_elbow)
            l_wrist_move = self._calculate_distance(l_wrist, self.base_l_wrist)
            r_wrist_move = self._calculate_distance(r_wrist, self.base_r_wrist)

            max_movement = max(l_elbow_move, r_elbow_move, l_wrist_move, r_wrist_move)
            if max_movement > self.STABILITY_TOLERANCE:
                self.arms_isolated = False

        # 2. Setup Guard: Wait for arms to settle before we start counting math
        if not self.is_ready:
            self.ready_frames += 1
            if self.ready_frames > 20:
                self.is_ready = True
                self.state = "TRACKING"
                
                # Anchor the arms in space
                self.base_l_elbow, self.base_r_elbow = l_elbow, r_elbow
                self.base_l_wrist, self.base_r_wrist = l_wrist, r_wrist
                
                # Initialize angles
                self.prev_l_angle = self._get_hand_angle(l_wrist, l_index)
                self.prev_r_angle = self._get_hand_angle(r_wrist, r_index)
                
            return {
                "exercise": "Wrist Circles",
                "state": "CALIBRATING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Hold your elbows and wrists perfectly still...",
                "rotation_progress": 0
            }

        # 3. Kinematics (Phase Unwrapping the Circle)
        curr_l_angle = self._get_hand_angle(l_wrist, l_index)
        curr_r_angle = self._get_hand_angle(r_wrist, r_index)

        delta_l = self._get_angle_delta(curr_l_angle, self.prev_l_angle)
        delta_r = self._get_angle_delta(curr_r_angle, self.prev_r_angle)

        self.cum_l_rotation += delta_l
        self.cum_r_rotation += delta_r

        self.prev_l_angle = curr_l_angle
        self.prev_r_angle = curr_r_angle

        # Use the slower wrist as the bottleneck to force strict bilateral compliance
        max_progress = min(abs(self.cum_l_rotation), abs(self.cum_r_rotation))
        progress_percentage = min(100, int((max_progress / self.TARGET_ROTATION) * 100))

        # 4. Buffer & AI (Catches leaning/slouching)
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
        if max_progress >= self.TARGET_ROTATION:
            if not self.arms_isolated:
                self.current_feedback = "No rep: You're moving your elbows! Isolate the wrist joint."
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: Sit up straight, don't slouch while rolling."
            else:
                self.rep_count += 1
                self.current_feedback = "Perfect circle! Excellent wrist isolation."

            # Reset accumulators for the next rep. We don't hard reset to 0 so we don't lose 
            # overflow momentum if they are doing fast, continuous circles.
            self.cum_l_rotation = 0.0 if abs(self.cum_l_rotation) < 360 else math.fmod(self.cum_l_rotation, 360)
            self.cum_r_rotation = 0.0 if abs(self.cum_r_rotation) < 360 else math.fmod(self.cum_r_rotation, 360)
            
            # Re-anchor the baseline to prevent drift over long sets
            self.base_l_elbow, self.base_r_elbow = l_elbow, r_elbow
            self.base_l_wrist, self.base_r_wrist = l_wrist, r_wrist
            self.arms_isolated = True
            self.ml_posture_maintained = True

        else:
            if not self.arms_isolated:
                self.current_feedback = "Hold your forearms completely still! Draw the circle with your knuckles."
            elif max_progress > 180:
                self.current_feedback = "Halfway there, push through the tight spots..."
            else:
                self.current_feedback = "Draw a massive circle with your index fingers."

        # 6. Dynamic Form Scoring
        form_score = 100
        if not self.arms_isolated:
            # Penalize linearly based on how much they moved their arms
            penalty = min(60, int((max_movement / self.STABILITY_TOLERANCE) * 20))
            form_score -= penalty
            
        form_score = max(0, form_score)

        return {
            "exercise": "Wrist Circles",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "rotation_progress_pct": progress_percentage,
            "arms_isolated": self.arms_isolated
        }