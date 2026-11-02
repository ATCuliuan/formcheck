import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SideLungeTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture Tripwires
        self.trailing_leg_bent = False    # Did they bend the straight leg to cheat the stretch?
        self.torso_collapsed = False      # Did they throw their chest to the floor?
        self.heel_lifted = False          # Did the active heel pop off the ground?
        self.ml_posture_maintained = True # ST-GCN AI flag
        
        # Tracking Variables
        self.standing_frames = 0
        self.is_ready = False
        self.baseline_ankle_y = None
        self.active_leg = None

        self.current_feedback = "Stand tall, feet wide or together to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_side_lunge.tflite")

        
        if os.path.exists(model_path):

        
            try:

        
                self.interpreter = Interpreter(model_path=model_path)

        
                self.interpreter.allocate_tensors()

        
                self.input_details = self.interpreter.get_input_details()

        
                self.output_details = self.interpreter.get_output_details()

        
            except Exception:

        
                self.interpreter = None

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

        total_vis = sum([l_hip.visibility, r_hip.visibility, l_knee.visibility, r_knee.visibility])
        total_reps = self.good_reps + self.bad_reps

        if total_vis < 3.0: 
            self.frame_buffer.clear()
            return {
                "exercise": "Side Lunge",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full lower body not visible",
                "knee_angle": 0.0, "depth_reached": False
            }

        # Calculate joint angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        # Calculate torso upright angle
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

        # Baseline: Both legs straight
        if l_knee_angle > 160 and r_knee_angle > 160:
            self.standing_frames += 1
            # Reset active leg when standing perfectly straight
            self.active_leg = None
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True
            self.baseline_ankle_y = (l_ankle.y + r_ankle.y) / 2  # Capture standing heel height

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Side Lunge",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand tall to begin",
                "knee_angle": round(min(l_knee_angle, r_knee_angle), 1), "depth_reached": False
            }

        # Dynamically determine the active (lunging) leg and trailing (straight) leg
        # The active leg is the one bending the most.
        if l_knee_angle < r_knee_angle - 10:
            self.active_leg = "LEFT"
        elif r_knee_angle < l_knee_angle - 10:
            self.active_leg = "RIGHT"

        if self.active_leg == "LEFT":
            active_knee_angle = l_knee_angle
            trailing_knee_angle = r_knee_angle
            active_hip_angle = l_hip_angle
            active_ankle = l_ankle
        elif self.active_leg == "RIGHT":
            active_knee_angle = r_knee_angle
            trailing_knee_angle = l_knee_angle
            active_hip_angle = r_hip_angle
            active_ankle = r_ankle
        else:
            active_knee_angle = min(l_knee_angle, r_knee_angle)
            trailing_knee_angle = max(l_knee_angle, r_knee_angle)
            active_hip_angle = min(l_hip_angle, r_hip_angle)
            active_ankle = l_ankle if l_knee_angle < r_knee_angle else r_ankle

        # Form Tripwires (Evaluated during descent and bottom)
        if self.state in ["DESCENDING", "BOTTOM"]:
            # 1. Trailing Leg Bend: The "kickstand" leg must stay relatively straight
            if trailing_knee_angle < 150:
                self.trailing_leg_bent = True
                
            # 2. Torso Collapse: Chest dropping too far forward
            if active_hip_angle < 50:
                self.torso_collapsed = True

            # 3. Heel Lift: Active heel popping off the floor
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.05:
                self.heel_lifted = True

        # ST-GCN inference
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
        if active_knee_angle > 155:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Lunge deeper"
                    is_bad_rep = True
                elif self.trailing_leg_bent:
                    self.current_feedback = "No rep: Keep trailing leg straight"
                    is_bad_rep = True
                elif self.torso_collapsed:
                    self.current_feedback = "No rep: Chest collapsed forward"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected form breakdown"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    if self.heel_lifted:
                        self.current_feedback = "Good rep! But keep your heel flat."
                    else:
                        self.current_feedback = "Perfect side lunge!"
                
                # Reset tripwires
                self.depth_reached = False
                self.trailing_leg_bent = False
                self.torso_collapsed = False
                self.heel_lifted = False
                self.ml_posture_maintained = True
                
            self.state = "STANDING"

        elif active_knee_angle <= 140 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Push hips back..."

            # Depth logic: Active knee drops while trailing stays straight
            if active_knee_angle <= 110:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Depth reached — Push back to center!"

        elif active_knee_angle > 110 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Push off that leg!"
            else:
                self.current_feedback = "Drop lower..."

        # Dynamic Form Score
        form_score = max(0, min(100, int(100 - abs(active_knee_angle - 90) * 0.3)))
        if self.trailing_leg_bent: form_score -= 40
        if self.torso_collapsed: form_score -= 30
        if self.heel_lifted: form_score -= 20
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Side Lunge",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(active_knee_angle, 1),
            "depth_reached": self.depth_reached
        }