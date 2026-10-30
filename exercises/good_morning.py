import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class GoodMorningTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.spine_rounded = False        # Neck/Upper back caving forward
        self.knees_locked = False         # Legs perfectly straight (danger)
        self.squatted_rep = False         # Bending knees too much (squatting)
        self.ml_posture_maintained = True # ST-GCN catches complex asymmetrical shifting
        
        # Setup Guards
        self.stand_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall with a soft knee bend to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_good_morning.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Good Morning model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Good Morning weights: {e}")
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
        # Extract required joints (Using Ear for spine alignment tracking)
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

        # Guard: Ensure the user's full side profile is visible
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Good Morning",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full side profile not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematics
        # Hinge angle tracks the torso bending forward (Shoulder -> Hip -> Knee)
        hinge_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        # Knee angle tracks the hamstring stretch/squat cheat (Hip -> Knee -> Ankle)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        # Spine angle tracks neck and upper back rounding (Ear -> Shoulder -> Hip)
        spine_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Base Initialization Check (Standing upright)
        if hinge_angle > 165:
            self.stand_frames += 1
        else:
            self.stand_frames = 0

        if self.stand_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Good Morning",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand upright with a slight knee bend to begin",
                "primary_angle": round(hinge_angle, 1), "secondary_angle": round(knee_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: Spinal Rounding ---
        # A rigid, neutral spine keeps the ear, shoulder, and hip relatively aligned. 
        # If this angle drops heavily, the head is jutting forward or the upper back is rounding.
        if spine_angle < 145 and self.state in ["DESCENDING", "BOTTOM"]:
            self.spine_rounded = True

        # --- Tripwire 2: The "Squat" Cheat ---
        # A proper good morning maintains a soft knee bend (approx 150-165 degrees).
        # Dropping below 135 means the hips are dropping, not just pushing back.
        if knee_angle < 135 and self.state in ["DESCENDING", "BOTTOM"]:
            self.squatted_rep = True

        # --- Tripwire 3: Locked Knees ---
        if knee_angle > 175 and self.state in ["DESCENDING", "BOTTOM"]:
            self.knees_locked = True

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

        # --- State Machine & Coaching Feedback ---
        if hinge_angle > 160:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Push hips further back! Hinge deeper."
                    is_bad_rep = True
                elif self.spine_rounded:
                    self.current_feedback = "No rep: You rounded your back (Spinal Danger)."
                    is_bad_rep = True
                elif self.squatted_rep:
                    self.current_feedback = "No rep: You squatted the weight. Keep knees at a soft bend."
                    is_bad_rep = True
                elif self.knees_locked:
                    self.current_feedback = "No rep: Do not lock knees straight! (Hamstring Danger)."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected structural imbalance."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect hinge! Excellent glute engagement."
                
                self.depth_reached = False
                
            self.state = "STANDING"
            # Reset tripwires
            self.spine_rounded = False 
            self.squatted_rep = False
            self.knees_locked = False
            self.ml_posture_maintained = True 

        elif hinge_angle <= 150 and self.state in ["STANDING", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Push hips back, keep chest proud..."

            # Depth is reached when the torso is roughly parallel to the floor (~90-100 deg hinge angle)
            # We set the trigger at 105 to account for varying hamstring flexibility.
            if hinge_angle <= 105:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Deep stretch hit — Squeeze glutes to stand!"

        elif hinge_angle > 105 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Drive hips forward to stand!"
            else:
                self.current_feedback = "Hinge deeper, push hips back..."

        # --- Dynamic Form Score ---
        # Perfect depth is roughly 90-100 degrees for the hinge.
        form_score = max(0, min(100, int(100 - abs(hinge_angle - 95) * 0.4)))
        if self.spine_rounded: form_score -= 60
        if self.squatted_rep: form_score -= 40
        if self.knees_locked: form_score -= 40
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Good Morning",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(hinge_angle, 1),
            "secondary_angle": round(knee_angle, 1),
            "depth_reached": self.depth_reached
        }