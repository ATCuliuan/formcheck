import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellFloorFlyeTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.is_pressing = False      # Bending elbows too much (< 120 deg)
        self.is_locked_out = False    # Locking elbows completely straight (> 175 deg)
        self.hips_lifted = False      # Thrusting hips off the floor
        self.ml_posture_maintained = True  
        
        # Setup Guards
        self.top_hold_frames = 0
        self.is_ready = False

        self.current_feedback = "Lie flat and hold dumbbells above chest"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_floor_flye.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Dumbbell Floor Flye model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Dumbbell Floor Flye weights: {e}")
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
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure the user's upper body is in frame
        if max(l_vis, r_vis) < 2.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Flye",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Move back: Upper body not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Core Kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        
        # Vertical displacement: How high is the wrist compared to the shoulder? 
        # MediaPipe Y=0 is the top of the screen. 
        wrist_elevation = active_shoulder.y - active_wrist.y 

        # Base Initialization Check (Holding weight up with a slight bend)
        if wrist_elevation > 0.15 and 130 < elbow_angle < 175:
            self.top_hold_frames += 1
        else:
            self.top_hold_frames = 0

        if self.top_hold_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Flye",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lie flat, weights up, slight elbow bend",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(hip_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: The "Press" Cheat ---
        if elbow_angle < 125 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.is_pressing = True

        # --- Tripwire 2: The Tendon Strain (Lockout) ---
        if elbow_angle > 175 and self.state in ["DESCENDING", "BOTTOM"]:
            self.is_locked_out = True

        # --- Tripwire 3: Glute Bridge Cheat ---
        # Assuming knees are bent on the floor, lifting hips straightens the body angle towards 180
        if hip_angle > 160 and active_hip.visibility > 0.6:
            self.hips_lifted = True

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
        if wrist_elevation > 0.15:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Open arms wider/lower"
                    is_bad_rep = True
                elif self.is_pressing:
                    self.current_feedback = "No rep: You bent your elbows (Pressing, not flying)"
                    is_bad_rep = True
                elif self.is_locked_out:
                    self.current_feedback = "No rep: Do not lock elbows perfectly straight!"
                    is_bad_rep = True
                elif self.hips_lifted:
                    self.current_feedback = "No rep: Keep your hips glued to the floor"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected unstable movement"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Great stretch! Excellent chest isolation."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.is_pressing = False 
            self.is_locked_out = False
            self.hips_lifted = False
            self.ml_posture_maintained = True 

        elif wrist_elevation <= 0.12 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Keep a slight bend. Open wide..."

            # Depth reached when wrists are roughly level with the shoulders (triceps hit the floor)
            if wrist_elevation <= 0.05:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Triceps on floor — Squeeze chest up!"

        elif wrist_elevation > 0.05 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Squeeze the chest!"
            else:
                self.current_feedback = "Lower until upper arms touch floor..."

        # --- Dynamic Form Score ---
        # Perfect elbow angle for a flye is ~150 degrees. 
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 150) * 0.8)))
        if self.is_pressing: form_score -= 50
        if self.is_locked_out: form_score -= 50
        if self.hips_lifted: form_score -= 30
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Floor Flye",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }