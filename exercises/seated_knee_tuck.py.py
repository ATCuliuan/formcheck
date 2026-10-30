import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SeatedKneeTuckTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.feet_dropped = False         # Torso angle opens too wide, indicating legs dropped to floor
        self.torso_rocked = False         # Failed to close the hip angle tightly (fake crunch)
        self.ml_posture_maintained = True # ST-GCN catches asymmetrical leaning or twisting
        
        # Setup Guards
        self.setup_frames = 0
        self.is_ready = False

        self.current_feedback = "Sit, lean back slightly, and hover straight legs to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_seated_knee_tuck.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Seated Knee Tuck model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Seated Knee Tuck weights: {e}")
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

        # Guard: Ensure full side profile is visible (critical for tracking the crunch angle)
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Seated Knee Tuck",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full side profile not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematics
        # 1. Hip Angle tracks the "Crunch" (Shoulder -> Hip -> Knee)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        # 2. Knee Angle tracks the leg extension (Hip -> Knee -> Ankle)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # Base Initialization Check (Seated V-hold: torso leaning back, legs straight out)
        # Proper extension creates a wide hip angle (110-145) and locked knees (>150)
        if 110 < hip_angle < 150 and knee_angle > 150:
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
                "exercise": "Seated Knee Tuck",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lean back slightly and extend legs straight out to hover",
                "primary_angle": round(hip_angle, 1), "secondary_angle": round(knee_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: The Dead Drop ---
        # If the hip angle opens beyond 155 degrees, they have either laid their back completely 
        # flat on the floor, or dropped their heels completely to the floor to rest.
        if hip_angle > 155 and self.state in ["TUCKING", "TUCKED", "EXTENDING"]:
            self.feet_dropped = True

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
                if predicted_class == 0 and self.state != "EXTENDED":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # --- State Machine & Coaching Feedback ---
        if knee_angle > 140:
            if self.state == "EXTENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: You didn't tuck your knees to your chest!"
                    is_bad_rep = True
                elif self.feet_dropped:
                    self.current_feedback = "No rep: Keep your heels hovering! Don't rest them."
                    is_bad_rep = True
                elif self.torso_rocked:
                    self.current_feedback = "No rep: Crunch your chest forward! Don't just rock back and forth."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected loss of core stability."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect extension! Keep the core tight."
                
                self.depth_reached = False
                
            self.state = "EXTENDED"
            # Reset tripwires
            self.feet_dropped = False 
            self.torso_rocked = False
            self.ml_posture_maintained = True 

        elif knee_angle <= 120 and self.state in ["EXTENDED", "TUCKING"]:
            self.state = "TUCKING"
            self.current_feedback = "Crunch knees to chest..."

            # True depth requires the knees to bend tightly AND the torso to crunch forward
            # Knee angle < 80 ensures a full leg tuck. 
            # Hip angle < 75 ensures they actually crunched their chest to their knees.
            if knee_angle <= 80:
                if hip_angle <= 80:
                    self.depth_reached = True
                    self.state = "TUCKED"
                    self.current_feedback = "Peak crunch! Extend legs slowly."
                else:
                    # They tucked their knees but didn't crunch their chest forward
                    self.torso_rocked = True

        elif knee_angle > 80 and self.state in ["TUCKED", "TUCKING"]:
            if self.depth_reached:
                self.state = "EXTENDING"
                self.current_feedback = "Extend legs fully, hover the heels..."
            else:
                self.current_feedback = "Tuck knees closer to chest!"

        # --- Dynamic Form Score ---
        # Perfect leg extension is 180 degrees. Perfect tuck depth is < 65.
        form_score = max(0, min(100, int(100 - abs(knee_angle - 180) * 0.4)))
        if self.feet_dropped: form_score -= 50
        if self.torso_rocked: form_score -= 40
        if not self.depth_reached and self.state == "EXTENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Seated Knee Tuck",
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