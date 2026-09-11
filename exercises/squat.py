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
        
        # Posture tripwires
        self.back_posture_maintained = True  
        self.ml_posture_maintained = True  
        self.knees_caved = False          
        self.is_asymmetrical = False      
        self.heels_lifted = False         
        self.descent_frames = 0           
        self.baseline_ankle_y = None      
        
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
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            lm = landmarks[idx]
            coords.append([lm.x, lm.y, lm.z])
            
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, landmarks[12].visibility])

        if max(l_vis, r_vis) < 2.5: 
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

        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = landmarks[12] if r_vis > l_vis else l_shoulder

        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        if knee_angle > 160:
            self.standing_frames += 1
            self.baseline_ankle_y = active_ankle.y 
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

        # 5. Beginner-Friendly Form Tripwires
        # DANGER (Invalidates Rep)
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)
        if knee_distance < (ankle_distance * 0.55) and self.state in ["DESCENDING", "BOTTOM"]:
            self.knees_caved = True

        if hip_angle < 45 and self.state != "STANDING":
            self.back_posture_maintained = False

        # WARNINGS (Lowers score but doesn't kill the rep)
        hip_tilt = abs(l_hip.y - r_hip.y)
        if hip_tilt > 0.10 and self.state in ["DESCENDING", "BOTTOM"]:
            self.is_asymmetrical = True

        if self.state == "DESCENDING":
            self.descent_frames += 1
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.05:
                self.heels_lifted = True

        # 6. ST-GCN Buffer
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

        # 7. State Machine & Coaching Feedback
        if knee_angle > 155:
            if self.state == "ASCENDING":
                if not self.depth_reached:
                    self.current_feedback = "No rep: Squat lower next time"
                elif not self.back_posture_maintained:
                    self.current_feedback = "No rep: Chest fell forward (Danger)"
                elif self.knees_caved:
                    self.current_feedback = "No rep: Knees caved in (Danger)"
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected bad posture"
                else:
                    # It's a rep! Give coaching tips for sloppy form
                    self.rep_count += 1
                    if self.heels_lifted:
                        self.current_feedback = "Good rep! But keep your heels flat."
                    elif self.is_asymmetrical:
                        self.current_feedback = "Good rep! But watch your hip shift."
                    elif self.descent_frames < 8:
                        self.current_feedback = "Good rep! But slow down the drop."
                    else:
                        self.current_feedback = "Perfect rep!"
                
                self.depth_reached = False
                
            self.state = "STANDING"
            self.back_posture_maintained = True 
            self.ml_posture_maintained = True 
            self.knees_caved = False
            self.is_asymmetrical = False
            self.heels_lifted = False
            self.descent_frames = 0

        elif knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Control the descent..."

            if knee_angle <= 110: # Relaxed depth threshold for beginners
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth good — Drive up!"

        elif knee_angle > 110 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Drive up!"
            else:
                self.current_feedback = "Squat lower..."

        # 8. Adaptive Form Score
        # Restored the dynamic "ticking" baseline score based on real-time knee angle
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))
        
        # Apply the beginner-friendly coaching penalties
        if not self.back_posture_maintained or self.knees_caved: form_score -= 50
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        if self.is_asymmetrical: form_score -= 15
        if self.heels_lifted: form_score -= 15
        if self.descent_frames > 0 and self.descent_frames < 8: form_score -= 10
        
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