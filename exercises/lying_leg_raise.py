import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class LyingLegRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.knees_bent = False           # Bending knees to shorten the lever
        self.feet_rested = False          # Dropping feet completely to the floor to rest
        self.ml_posture_maintained = True # ST-GCN catches the lower back arching off the floor
        
        # Setup Guards
        self.setup_frames = 0
        self.is_ready = False

        self.current_feedback = "Lie flat and raise straight legs to the ceiling"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_lying_leg_raise.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Lying Leg Raise model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Lying Leg Raise weights: {e}")
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure full body profile is visible (critical for leg lever tracking)
        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Lying Leg Raise",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full leg and torso profile not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematics
        # 1. Hip angle tracks the range of motion of the raise (Shoulder -> Hip -> Ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        # 2. Knee angle tracks if they are cheating the lever (Hip -> Knee -> Ankle)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # Base Initialization Check (Legs pointing straight up at 90 degrees to the torso)
        if hip_angle < 110 and knee_angle > 150:
            self.setup_frames += 1
        else:
            self.setup_frames = 0

        if self.setup_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Lying Leg Raise",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lie flat and point straight legs to the ceiling to start",
                "primary_angle": round(hip_angle, 1), "secondary_angle": round(knee_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: The Lever Shortener (Bending Knees) ---
        # Legs must remain relatively straight. Dropping below 140 means they are doing a knee tuck.
        if knee_angle < 140 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.knees_bent = True

        # --- Tripwire 2: The Dead Rest ---
        # If the hip angle opens up past 175, their legs are completely flat on the floor.
        if hip_angle > 175 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.feet_rested = True

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
        if hip_angle < 110:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Lower your legs further! (Half rep)"
                    is_bad_rep = True
                elif self.knees_bent:
                    self.current_feedback = "No rep: Keep your legs straight! Don't bend knees."
                    is_bad_rep = True
                elif self.feet_rested:
                    self.current_feedback = "No rep: Don't let your heels touch the floor. Hover!"
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected lower back arching. Press lower back down!"
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect raise! Great lower ab control."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.knees_bent = False 
            self.feet_rested = False
            self.ml_posture_maintained = True 

        elif hip_angle >= 120 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Lower legs slowly, keep lower back glued to floor..."

            # Full depth requires bringing legs down to hover just above the floor (approx 150-165 degrees)
            if hip_angle >= 155:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Hover — Now lift straight up!"

        elif hip_angle < 155 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Squeeze lower abs to lift!"
            else:
                self.current_feedback = "Lower legs closer to the floor..."

        # --- Dynamic Form Score ---
        # Perfect form requires a straight knee (180).
        form_score = max(0, min(100, int(100 - abs(knee_angle - 180) * 0.6)))
        if self.knees_bent: form_score -= 50
        if self.feet_rested: form_score -= 40
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Lying Leg Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(hip_angle, 1),
            "secondary_angle": round(knee_angle, 1),
            "depth_reached": self.depth_reached
        }