import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class LateralRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.target_height_reached = False
        
        # Instructor Tripwires
        self.arms_too_bent = False  
        self.torso_stable = True          # Catches the "Momentum Swing"
        self.shoulders_shrugged = False   # Catches the "Flying Bird" trap cheat
        self.ml_posture_maintained = True 
        
        self.standing_frames = 0
        self.is_ready = False
        self.baseline_shoulder_y = None
        self.current_feedback = "Let arms hang straight down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_side_lateral_raise.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Lateral Raise model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Lateral Raise weights: {e}")
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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Lateral Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show your torso and arms",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Calculate kinematics for both arms, but use the most visible side for the state machine
        l_abduction = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction = calculate_angle(r_hip, r_shoulder, r_elbow)
        
        l_elbow_bend = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_bend = calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # Enforce straight arms on whichever arms are visible
        min_elbow_bend = min(l_elbow_bend, r_elbow_bend)

        active_abduction = r_abduction if r_vis > l_vis else l_abduction
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip

        # 1. Starting Guard: Arms down by sides
        if active_abduction < 25:
            self.standing_frames += 1
            # Capture baseline neck length when relaxed
            self.baseline_shoulder_y = abs(l_ear.y - l_shoulder.y) 
        else:
            self.standing_frames = 0

        if self.standing_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Lateral Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arms hang straight down by your sides",
                "knee_angle": round(min_elbow_bend, 1), "hip_angle": round(active_abduction, 1)
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
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # The T-Rex Wing (Severe Elbow Bend)
        if min_elbow_bend < 130 and self.state in ["ASCENDING", "TOP"]:
            self.arms_too_bent = True

        # The Momentum Swing (Body Rocking)
        torso_sway = abs(active_shoulder.x - active_hip.x)
        if torso_sway > 0.12 and self.state in ["ASCENDING", "TOP"]:
            self.torso_stable = False
            
        # The Flying Bird (Trap Shrug)
        current_shoulder_y = abs(l_ear.y - l_shoulder.y)
        if self.baseline_shoulder_y and current_shoulder_y < (self.baseline_shoulder_y * 0.75) and self.state == "ASCENDING":
            self.shoulders_shrugged = True

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Raise arms out to the sides!"

        elif self.state in ["RESTING", "DESCENDING"] and active_abduction > 30:
            self.state = "ASCENDING"
            self.current_feedback = "Raise to shoulder height..."

        elif self.state == "ASCENDING" and active_abduction >= 80:
            self.target_height_reached = True
            self.state = "TOP"
            self.current_feedback = "Hold, then lower slowly"

        elif self.state == "TOP" and active_abduction < 70:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down"

        elif self.state == "DESCENDING" and active_abduction <= 30:
            is_bad_rep = False
            
            if not self.target_height_reached:
                self.current_feedback = "No rep: Raise arms until parallel with the floor."
                is_bad_rep = True
            elif self.arms_too_bent:
                self.current_feedback = "No rep: You bent your elbows too much (T-Rex arms)."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: You swung your body. Keep torso completely still."
                is_bad_rep = True
            elif self.shoulders_shrugged:
                self.current_feedback = "No rep: You shrugged your shoulders! Keep them pressed down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect lateral raise!"
                
            # Reset for next rep
            self.state = "RESTING"
            self.target_height_reached = False
            self.arms_too_bent = False
            self.torso_stable = True
            self.shoulders_shrugged = False
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(( active_abduction / 85 ) * 100)))
        if not self.target_height_reached and self.state == "DESCENDING": form_score -= 30
        if self.arms_too_bent: form_score -= 30
        if not self.torso_stable: form_score -= 40
        if self.shoulders_shrugged: form_score -= 35
        form_score = max(0, form_score)

        return {
            "exercise": "Lateral Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(min_elbow_bend, 1), # Repurposed for UI
            "hip_angle": round(active_abduction, 1) # Repurposed for UI
        }