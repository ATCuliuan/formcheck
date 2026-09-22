import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingActiveStraightLegRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.target_height_reached = False
        
        # Posture Tripwires
        self.lifted_knee_bent = False     # Did they bend the lifting leg to cheat hamstring flexibility?
        self.support_knee_bent = False    # Did they bend the standing leg to relieve pelvic tension?
        self.torso_leaned = False         # Did they lean backward to counter-weight the leg?
        self.ml_posture_maintained = True # ST-GCN AI flag
        
        # Tracking Variables
        self.standing_frames = 0
        self.is_ready = False
        self.active_leg = None

        self.current_feedback = "Stand tall, feet together to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_active_straight_leg_raise.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Standing Leg Raise model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Standing Leg Raise weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

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
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        total_vis = sum([l_hip.visibility, r_hip.visibility, l_knee.visibility, r_knee.visibility, l_ankle.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps

        if total_vis < 5.0: 
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Leg Raise",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full body not visible",
                "hip_angle": 0.0, "depth_reached": False
            }

        # Calculate joint angles
        # Hip angle determines the height of the leg raise (180 is standing straight, 90 is parallel to floor)
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        
        # Knee angles check if the legs are straight
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Baseline: Both hips open (> 165) and knees straight (> 160)
        if l_hip_angle > 165 and r_hip_angle > 165 and l_knee_angle > 160 and r_knee_angle > 160:
            self.standing_frames += 1
            self.active_leg = None
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Leg Raise",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand tall, legs straight to begin",
                "hip_angle": round(min(l_hip_angle, r_hip_angle), 1), "depth_reached": False
            }

        # Dynamically determine the active (lifting) leg
        # The leg with the significantly smaller hip angle is the one lifting forward
        if l_hip_angle < r_hip_angle - 15:
            self.active_leg = "LEFT"
        elif r_hip_angle < l_hip_angle - 15:
            self.active_leg = "RIGHT"

        if self.active_leg == "LEFT":
            active_hip_angle = l_hip_angle
            support_hip_angle = r_hip_angle
            active_knee_angle = l_knee_angle
            support_knee_angle = r_knee_angle
        elif self.active_leg == "RIGHT":
            active_hip_angle = r_hip_angle
            support_hip_angle = l_hip_angle
            active_knee_angle = r_knee_angle
            support_knee_angle = l_knee_angle
        else:
            # Default to tracking whichever hip is closing the most
            active_hip_angle = min(l_hip_angle, r_hip_angle)
            support_hip_angle = max(l_hip_angle, r_hip_angle)
            active_knee_angle = l_knee_angle if l_hip_angle < r_hip_angle else r_knee_angle
            support_knee_angle = r_knee_angle if l_hip_angle < r_hip_angle else l_knee_angle

        # Form Tripwires (Evaluated during the lift)
        if self.state in ["LIFTING", "PEAK"]:
            # 1. Hamstring Cheat: Bending the lifting leg
            if active_knee_angle < 155:
                self.lifted_knee_bent = True
                
            # 2. Support Leg Crutch: Bending the standing leg to drop hips
            if support_knee_angle < 155:
                self.support_knee_bent = True

            # 3. Torso Counter-weight: Leaning backward (support hip angle opens up or collapses)
            # A strict support leg hip angle should stay between 165 and 185
            if support_hip_angle < 160 or support_hip_angle > 190:
                self.torso_leaned = True

        # ST-GCN LiteRT Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0) 
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                predicted_class = np.argmax(output_data, axis=1)[0]
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # State Machine
        if active_hip_angle > 165:
            if self.state == "LOWERING":
                is_bad_rep = False
                
                if not self.target_height_reached:
                    self.current_feedback = "No rep: Lift leg higher (aim for waist)"
                    is_bad_rep = True
                elif self.lifted_knee_bent:
                    self.current_feedback = "No rep: Keep lifting leg perfectly straight"
                    is_bad_rep = True
                elif self.support_knee_bent:
                    self.current_feedback = "No rep: Standing leg bent (Lock it out)"
                    is_bad_rep = True
                elif self.torso_leaned:
                    self.current_feedback = "No rep: Leaning torso (Brace your core)"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected form breakdown"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect straight leg raise!"
                
                # Reset tripwires
                self.target_height_reached = False
                self.lifted_knee_bent = False
                self.support_knee_bent = False
                self.torso_leaned = False
                self.ml_posture_maintained = True
                
            self.state = "STANDING"

        elif active_hip_angle <= 155 and self.state in ["STANDING", "LOWERING"]:
            self.state = "LIFTING"
            self.current_feedback = "Keep both legs straight..."

            # Target height logic: Aiming for 115 degrees or lower (closer to parallel)
            if active_hip_angle <= 115:
                self.target_height_reached = True
                self.state = "PEAK"
                self.current_feedback = "Great height — Lower under control!"

        elif active_hip_angle > 115 and self.state in ["PEAK", "LIFTING"]:
            if self.target_height_reached:
                self.state = "LOWERING"
                self.current_feedback = "Control the drop..."
            else:
                self.current_feedback = "Lift higher..."

        # Dynamic Form Score
        form_score = max(0, min(100, int(100 - abs(active_hip_angle - 90) * 0.4)))
        if self.lifted_knee_bent: form_score -= 30
        if self.support_knee_bent: form_score -= 30
        if self.torso_leaned: form_score -= 20
        if not self.target_height_reached and self.state == "LOWERING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Leg Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "hip_angle": round(active_hip_angle, 1),
            "depth_reached": self.target_height_reached
        }