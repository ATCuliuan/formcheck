import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SitUpTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.neck_yanked = False          # Yanking head forward
        self.feet_lifted = False          # Lifting feet to use momentum
        self.ml_posture_maintained = True # ST-GCN catches asymmetrical twisting
        
        # Setup Guards
        self.lying_frames = 0
        self.is_ready = False
        self.baseline_ankle_y = None      # Anchor point for the feet

        self.current_feedback = "Lie flat with knees bent to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_sit_up.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Sit-Up model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Sit-Up weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️️ ST-GCN TFLite model not found at {model_path}")

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
        # Extract required joints (Adding the Ear for cervical tracking)
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate visibility 
        l_vis = sum([l_ear.visibility, l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_ear.visibility, r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure full profile is visible
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Sit-Up",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Adjust camera: Full body profile not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematics
        # Hip angle tracks torso rising (Shoulder -> Hip -> Knee)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        # Neck angle tracks cervical spine neutral alignment (Ear -> Shoulder -> Hip)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Base Initialization Check (Lying flat with bent knees)
        # When lying flat, the hip angle is open (> 105 degrees)
        if hip_angle > 105:
            self.lying_frames += 1
            # Capture the baseline Y-coordinate of the ankle to track feet lifting
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
        else:
            self.lying_frames = 0

        if self.lying_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Sit-Up",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lie flat, knees bent, shoulders on floor to begin",
                "primary_angle": round(hip_angle, 1), "secondary_angle": round(neck_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: The Neck Yank ---
        # A neutral neck has an angle of ~160-180. Dropping heavily means chin is tucked violently.
        if neck_angle < 135 and self.state in ["ASCENDING", "TOP"]:
            self.neck_yanked = True

        # --- Tripwire 2: The Leg Swing (Feet Lifting) ---
        # If the ankle's Y-coordinate moves significantly upwards (decreases in MediaPipe) from the baseline
        if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.08:
            self.feet_lifted = True

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
                if predicted_class == 0 and self.state != "BOTTOM":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        if hip_angle > 105:
            if self.state == "DESCENDING":
                is_bad_rep = False
                
                # --- Tripwire 3: Depth Evaluation ---
                if not self.depth_reached:
                    self.current_feedback = "No rep: Sit all the way up to your knees!"
                    is_bad_rep = True
                elif self.neck_yanked:
                    self.current_feedback = "No rep: Stop yanking your neck! Keep chin up."
                    is_bad_rep = True
                elif self.feet_lifted:
                    self.current_feedback = "No rep: You swung your legs. Keep feet flat!"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected twisting/momentum cheat"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect rep! Great core isolation."
                
                self.depth_reached = False
                
            self.state = "BOTTOM"
            # Reset tripwires
            self.neck_yanked = False 
            self.feet_lifted = False
            self.ml_posture_maintained = True 

        elif hip_angle <= 95 and self.state in ["BOTTOM", "ASCENDING"]:
            self.state = "ASCENDING"
            self.current_feedback = "Squeeze the abs, pull up..."

            # Depth hit when torso compresses tightly against the thighs (< 65 degrees)
            if hip_angle <= 65:
                self.depth_reached = True
                self.state = "TOP"
                self.current_feedback = "Full compression — Control it back down!"

        elif hip_angle > 65 and self.state in ["TOP", "ASCENDING"]:
            if self.depth_reached:
                self.state = "DESCENDING"
                self.current_feedback = "Roll back slowly, don't drop..."
            else:
                self.current_feedback = "Sit up higher!"

        # --- Dynamic Form Score ---
        # Perfect sit-up depth compresses the hip angle to roughly 60 degrees.
        form_score = max(0, min(100, int(100 - abs(hip_angle - 60) * 0.5)))
        if self.neck_yanked: form_score -= 40
        if self.feet_lifted: form_score -= 50
        if not self.depth_reached and self.state == "DESCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Sit-Up",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(hip_angle, 1),
            "secondary_angle": round(neck_angle, 1),
            "depth_reached": self.depth_reached
        }