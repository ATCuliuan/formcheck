import os
import math
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_HIP_CIRCLES_MODEL


class HipCirclesTracker:
    """
    Coaching logic for Dynamic Hip Circles:
      1. CALIBRATING -> standing tall, feet shoulder-width, holding perfectly still to anchor.
      2. TRACKING    -> hips begin orbiting the center line while shoulders and head remain anchored.
      3. REVOLUTION  -> the pelvis completes a full 360-degree sweep around the baseline center.
      4. COMPLETE    -> rep counted, ready for continuous rotation.

    Instructor-level faults tracked independently:
      - torso_sway: Moving the shoulders and head instead of isolating the hips.
      - bent_knees: Squatting through the movement (knees must stay mostly straight).
      - small_radius: Just wiggling in place instead of drawing a massive circle.
    """

    # Tolerances for isolation
    SHOULDER_SWAY_MAX = 0.04         # Shoulders cannot drift from their calibrated center
    KNEE_BENT_MIN = 150.0            # Allow a soft knee, but no squatting (<150 is a fault)
    
    # Circle dimensions and requirements
    TARGET_ROTATION = 340.0          # Degrees required to count a full circle
    MIN_HIP_RADIUS = 0.03            # Hips must deviate from center by this much to count as a "circle"

    def __init__(self):
        self.state = "CALIBRATING"
        self.rep_count = 0

        # Baselines for isolation and orbital tracking
        self.base_mid_shoulder = None
        self.base_mid_hip = None

        # Phase unwrapping variables
        self.prev_angle = None
        self.cum_rotation = 0.0

        # Per-rep tripwires
        self.shoulders_anchored = True
        self.knees_straight = True
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand with feet shoulder-width, hold still to calibrate."
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_HIP_CIRCLES_MODEL

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

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_orbital_angle(self, center, current):
        # Calculates the 2D angle of the hip relative to its baseline center
        # Returns angle in degrees [-180, 180]
        return math.degrees(math.atan2(current[1] - center[1], current[0] - center[0]))

    def _get_angle_delta(self, current_angle, previous_angle):
        # Phase unwrapping: Ensures continuous rotation accumulation
        delta = current_angle - previous_angle
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
        return delta

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 1. Calculate Midpoints & Kinematics
        curr_mid_shoulder = ((l_shoulder.x + r_shoulder.x) / 2.0, (l_shoulder.y + r_shoulder.y) / 2.0)
        curr_mid_hip = ((l_hip.x + r_hip.x) / 2.0, (l_hip.y + r_hip.y) / 2.0)

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # 2. Setup Guard: Wait for the user to be perfectly still to anchor the tracking points
        if not self.is_ready:
            self.ready_frames += 1
            if self.ready_frames > 20:
                self.is_ready = True
                self.state = "TRACKING"
                
                # Anchor the baseline center points
                self.base_mid_shoulder = curr_mid_shoulder
                self.base_mid_hip = curr_mid_hip
                
                # Initialize phase tracking
                self.prev_angle = self._get_orbital_angle(self.base_mid_hip, curr_mid_hip)
                
            return {
                "exercise": "Hip Circles",
                "state": "CALIBRATING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Stand tall and hold perfectly still to anchor your center line...",
                "rotation_progress": 0
            }

        # 3. Fault Tripwires (Upper Body Isolation & Knee Straightness)
        shoulder_sway = self._calculate_distance(curr_mid_shoulder, self.base_mid_shoulder)
        if shoulder_sway > self.SHOULDER_SWAY_MAX:
            self.shoulders_anchored = False

        if l_knee_angle < self.KNEE_BENT_MIN or r_knee_angle < self.KNEE_BENT_MIN:
            self.knees_straight = False

        # 4. Kinematics (Phase Unwrapping the Pelvis Orbit)
        hip_radius = self._calculate_distance(curr_mid_hip, self.base_mid_hip)
        curr_angle = self._get_orbital_angle(self.base_mid_hip, curr_mid_hip)
        delta_angle = self._get_angle_delta(curr_angle, self.prev_angle)
        
        # Only accumulate rotation if they are actually drawing a circle wide enough
        if hip_radius > self.MIN_HIP_RADIUS:
            self.cum_rotation += delta_angle
        
        self.prev_angle = curr_angle

        progress_percentage = min(100, int((abs(self.cum_rotation) / self.TARGET_ROTATION) * 100))

        # 5. Buffer & AI 
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
        if abs(self.cum_rotation) >= self.TARGET_ROTATION:
            if not self.shoulders_anchored:
                self.current_feedback = "No rep: You're swaying your shoulders! Keep your upper body anchored."
            elif not self.knees_straight:
                self.current_feedback = "No rep: You're squatting! Keep your legs relatively straight."
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected broken form or slouching."
            else:
                self.rep_count += 1
                self.current_feedback = "Excellent circle! Great pelvic isolation."

            # Reset accumulators, wrapping the math so we don't lose continuous momentum
            self.cum_rotation = 0.0 if abs(self.cum_rotation) < 360 else math.fmod(self.cum_rotation, 360)
            
            # Reset tripwires
            self.shoulders_anchored = True
            self.knees_straight = True
            self.ml_posture_maintained = True

        else:
            if not self.shoulders_anchored:
                self.current_feedback = "Keep your shoulders frozen! Move ONLY your hips."
            elif not self.knees_straight:
                self.current_feedback = "Lock out those knees, don't squat."
            elif hip_radius <= self.MIN_HIP_RADIUS and progress_percentage > 0:
                self.current_feedback = "Push your hips out wider! That circle is too small."
            elif progress_percentage > 180:
                self.current_feedback = "Keep sweeping, finish the circle..."
            else:
                self.current_feedback = "Draw a massive circle with your belt buckle."

        # 7. Dynamic Form Scoring
        form_score = 100
        if not self.shoulders_anchored:
            penalty = min(50, int((shoulder_sway / self.SHOULDER_SWAY_MAX) * 25))
            form_score -= penalty
        if not self.knees_straight:
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Hip Circles",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "rotation_progress_pct": progress_percentage,
            "shoulders_anchored": self.shoulders_anchored,
            "hip_radius": round(hip_radius, 3)
        }