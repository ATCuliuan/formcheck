import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellBicepsCurlsTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.peak_contraction_reached = False
        
        # Instructor Tripwires
        self.elbows_pinned = True         # Catches the "Shoulder Swing"
        self.torso_stable = True          # Catches the "Hip Thrust" momentum cheat
        self.ml_posture_maintained = True # AI catches the half-rep
        
        self.extended_frames = 0
        self.is_ready = False
        self.current_feedback = "Let arms hang straight down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_biceps_curls.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Biceps Curl model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Biceps Curl weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        # Full 25-joint extraction for ST-GCN consistency
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            lm = landmarks[idx]
            coords.append([lm.x, lm.y, lm.z])
        while len(coords) < 25: 
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Biceps Curls", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show your full arm and torso",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Select the active side facing the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        # 1. Starting Guard: Arm must be extended
        if elbow_angle > 150:
            self.extended_frames += 1
        else:
            self.extended_frames = 0

        if self.extended_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Biceps Curls", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arm hang completely straight to start",
                "knee_angle": round(elbow_angle, 1), "hip_angle": 0.0
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "EXTENDED":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # Elbow Drift Detector (Shoulder Swing)
        elbow_drift = abs(active_shoulder.x - active_elbow.x)
        if elbow_drift > 0.12 and self.state in ["CURLING", "TOP"]:
            self.elbows_pinned = False

        # Body Rocking Detector (Hip Thrust)
        torso_sway = abs(active_shoulder.x - active_hip.x)
        if torso_sway > 0.10 and self.state in ["CURLING", "LOWERING"]:
            self.torso_stable = False

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Curl it up!"

        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < 130:
            self.state = "CURLING"
            self.current_feedback = "Squeeze the bicep..."

        elif self.state == "CURLING" and elbow_angle < 50:
            self.state = "TOP"
            self.peak_contraction_reached = True
            self.current_feedback = "Control the drop"

        elif self.state == "TOP" and elbow_angle > 70:
            self.state = "LOWERING"

        elif self.state == "LOWERING" and elbow_angle >= 150:
            is_bad_rep = False
            
            # Form Evaluation
            if not self.peak_contraction_reached:
                self.current_feedback = "No rep: Curl all the way up."
                is_bad_rep = True
            elif not self.elbows_pinned:
                self.current_feedback = "No rep: You swung your elbows! Pin them to your ribs."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: Stop throwing your back into it."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged a half-rep."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect curl!"
                
            # Reset for next rep
            self.state = "EXTENDED"
            self.peak_contraction_reached = False
            self.elbows_pinned = True
            self.torso_stable = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(( (160 - elbow_angle) / 110 ) * 100)))
        if not self.elbows_pinned: form_score -= 30
        if not self.torso_stable: form_score -= 30
        if not self.ml_posture_maintained: form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Biceps Curls",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(elbow_angle, 1), # Re-using key for elbow text output
            "hip_angle": 0.0
        }