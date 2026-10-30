import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class TricepDipsTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.shoulders_shrugged = False   # Shoulders collapsing up toward the ears
        self.ml_posture_maintained = True # ST-GCN catches asymmetrical dipping or hip thrusting
        
        # Setup Guards
        self.setup_frames = 0
        self.is_ready = False
        self.baseline_shrug_distance = None # Tracks healthy neck posture at the start of the set

        self.current_feedback = "Sit on the edge, plant hands, and lock arms straight"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_tricep_dips.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Tricep Dips model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Tricep Dips weights: {e}")
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
        # Extract required joints
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_ear.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_ear.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure side profile is visible (arms, torso, and head)
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Tricep Dips",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso, arms, and head must be visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_ear = r_ear if r_vis > l_vis else l_ear

        # Core Kinematics
        # 1. Elbow angle tracks depth (Shoulder -> Elbow -> Wrist)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        
        # 2. Shrug distance (Vertical Y distance between Ear and Shoulder)
        # MediaPipe Y-axis goes top-down (0 at top of screen). 
        # A smaller distance means the shoulder is creeping up to the ear.
        current_shrug_distance = abs(active_shoulder.y - active_ear.y)

        # Base Initialization Check (Supported on straight arms, hips clear of the bench)
        if elbow_angle > 160:
            self.setup_frames += 1
            if self.baseline_shrug_distance is None or self.setup_frames == 10:
                self.baseline_shrug_distance = current_shrug_distance
        else:
            self.setup_frames = 0

        if self.setup_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Tricep Dips",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lock your arms straight and keep chest up to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": 0.0, "depth_reached": False
            }

        # --- Tripwire 1: The Shoulder Shrug ---
        # If the ear-to-shoulder distance shrinks by more than 35% from their straight-arm baseline, they collapsed.
        if self.baseline_shrug_distance and current_shrug_distance < (self.baseline_shrug_distance * 0.65):
            if self.state in ["LOWERING", "BOTTOM", "PRESSING"]:
                self.shoulders_shrugged = True

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
                if predicted_class == 0 and self.state != "TOP":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        if elbow_angle > 155:
            if self.state == "PRESSING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: You didn't dip low enough! Hit 90 degrees."
                    is_bad_rep = True
                elif self.shoulders_shrugged:
                    self.current_feedback = "No rep: You shrugged your shoulders! Keep chest up and shoulders down."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected asymmetrical pressing/hip throwing."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect dip! Great tricep isolation."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.shoulders_shrugged = False
            self.ml_posture_maintained = True 

        elif elbow_angle <= 145 and self.state in ["TOP", "LOWERING"]:
            self.state = "LOWERING"
            self.current_feedback = "Lower slowly, keep shoulders pressed down..."

            # True depth requires elbows to hit roughly 90 degrees
            if elbow_angle <= 95:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Perfect depth! Press up strongly."

        elif elbow_angle > 95 and self.state in ["BOTTOM", "LOWERING"]:
            if self.depth_reached:
                self.state = "PRESSING"
                self.current_feedback = "Lock it out at the top..."
            else:
                self.current_feedback = "Dip lower!"

        # --- Dynamic Form Score ---
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 180) * 0.4)))
        if self.shoulders_shrugged: form_score -= 50
        if not self.depth_reached and self.state == "PRESSING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Tricep Dips",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(current_shrug_distance, 3), # Tracking the shrug margin as the secondary metric
            "depth_reached": self.depth_reached
        }