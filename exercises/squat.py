import os
from collections import deque
import numpy as np
import torch
import mediapipe as mp
from core.kinematics import calculate_angle
from st_gcn_model import STGCN

class SquatTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.rep_count = 0
        self.depth_reached = False
        
        # Posture tripwires for Strict Tracking
        self.back_posture_maintained = True  
        self.ml_posture_maintained = True  
        self.knees_caved = False          # NEW: Knee valgus collapse tracking
        self.is_asymmetrical = False      # NEW: Hip shift/uneven squat tracking
        
        # Setup Guards
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand upright to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load ST-GCN weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = os.path.join("models", "st_gcn_squat.pt")

        if os.path.exists(model_path):
            try:
                self.model = STGCN(num_classes=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ ST-GCN model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading ST-GCN weights: {e}")
                self.model = None
        else:
            print(f"⚠️ ST-GCN Model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        """Fixes the leg-blindness bug by explicitly grabbing body landmarks instead of the face."""
        # Grab torso, arms, and legs (ignoring face landmarks 1-10 to fit into 25 joints)
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            lm = landmarks[idx]
            coords.append([lm.x, lm.y, lm.z])
            
        # Pad the rest with zeros to match the 25-joint ST-GCN input requirement
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        # 1. Grab all critical joints (Left and Right)
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        # 2. Dynamic Profile Tracking & Visibility Guard
        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, landmarks[12].visibility])

        if max(l_vis, r_vis) < 2.5: # 4 joints * 0.65 min visibility ~ 2.6
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Step back: full body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Track the active side for depth math
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = landmarks[12] if r_vis > l_vis else l_shoulder

        # 3. Kinematic Angle Calculations
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        # 4. The "Ready Stance" Lock
        if knee_angle > 160:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 20:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": self.state,
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Stand upright to begin",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1), "depth_reached": False
            }

        # 5. NEW: Strict Form Tripwires (Knee Valgus & Hip Shift)
        # Check if knees cave inward relative to ankle width
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)
        if knee_distance < (ankle_distance * 0.65) and self.state in ["DESCENDING", "BOTTOM"]:
            self.knees_caved = True

        # Check for hip shift (one side dropping lower than the other)
        hip_tilt = abs(l_hip.y - r_hip.y)
        if hip_tilt > 0.06 and self.state in ["DESCENDING", "BOTTOM"]:
            self.is_asymmetrical = True

        # Check for forward lean (Good Morning)
        if hip_angle < 50 and self.state != "STANDING":
            self.back_posture_maintained = False

        # 6. Update temporal buffer & Run ST-GCN
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        ml_feedback = None
        if self.model and len(self.frame_buffer) == self.sequence_length:
            try:
                sequence = np.array(self.frame_buffer)
                tensor_input = np.transpose(sequence, (2, 0, 1))
                tensor_input = torch.tensor(tensor_input, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(tensor_input)
                    prediction = torch.argmax(outputs, dim=1).item()

                if prediction == 0:
                    ml_feedback = "Warning: Form breakdown detected"
                    if self.state != "STANDING":
                        self.ml_posture_maintained = False
            except Exception:
                pass

        # 7. State Machine & Strict Rep Gatekeeper
        if knee_angle > 155:
            if self.state == "ASCENDING":
                if self.depth_reached:
                    # ONLY count if ALL safety tripwires passed
                    if self.back_posture_maintained and not self.knees_caved and not self.is_asymmetrical and self.ml_posture_maintained:
                        self.rep_count += 1
                        self.current_feedback = "Good rep!"
                    elif not self.back_posture_maintained:
                        self.current_feedback = "No rep: Chest fell forward"
                    elif self.knees_caved:
                        self.current_feedback = "No rep: Knees caved in"
                    elif self.is_asymmetrical:
                        self.current_feedback = "No rep: Hips shifted/Uneven"
                    else:
                        self.current_feedback = "No rep: AI detected bad posture"
                
                self.depth_reached = False
                
            self.state = "STANDING"
            # Reset all tripwires for the next rep
            self.back_posture_maintained = True 
            self.ml_posture_maintained = True 
            self.knees_caved = False
            self.is_asymmetrical = False

        elif knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Going down..."

            if knee_angle <= 95:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth good — Push up"

        elif knee_angle > 100 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Drive through heels!"
            else:
                self.current_feedback = "Squat lower for full depth"

        # 8. Punishing Form Score
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))
        if not self.back_posture_maintained: form_score -= 30
        if self.knees_caved: form_score -= 20
        if self.is_asymmetrical: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Bodyweight Squat",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }