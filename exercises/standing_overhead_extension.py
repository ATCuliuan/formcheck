import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingOverheadExtensionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.elbows_drifted = False       # Elbows falling forward away from ears
        self.back_arched = False          # Leaning back / hyperextending lumbar spine
        self.ml_posture_maintained = True # ST-GCN catches asymmetrical flaring
        
        # Setup Guards
        self.lockout_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, weight pressed overhead"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_overhead_extension.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Standing Overhead Ext model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Standing Overhead Ext weights: {e}")
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
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure the user's upper body and hips are visible
        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Overhead Extension",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and arms not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Core Kinematics
        # 1. Elbow angle tracks the actual triceps extension
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        # 2. Shoulder flexion angle tracks elbow drift (Hip -> Shoulder -> Elbow)
        shoulder_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        # 3. Core/Hip angle tracks the lumbar spine (Shoulder -> Hip -> Knee)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        # Base Initialization Check (Arms locked out straight overhead)
        # When arms are straight up, shoulder_angle is > 160 and elbow_angle > 150
        if elbow_angle > 150 and shoulder_angle > 150:
            self.lockout_frames += 1
        else:
            self.lockout_frames = 0

        if self.lockout_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Overhead Extension",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Press weight straight overhead to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(shoulder_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: Elbow Drift (Shoulder Cheat) ---
        # If the upper arm drops forward, the Hip-Shoulder-Elbow angle decreases heavily.
        if shoulder_angle < 145 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.elbows_drifted = True

        # --- Tripwire 2: The Banana Back ---
        # A straight standing core is ~175-180. Dropping below 165 means heavy leaning/arching.
        if hip_angle < 165 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.back_arched = True

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
        if elbow_angle > 150:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Lower the weight deeper behind your head!"
                    is_bad_rep = True
                elif self.elbows_drifted:
                    self.current_feedback = "No rep: Keep elbows pointing up! Don't drop them."
                    is_bad_rep = True
                elif self.back_arched:
                    self.current_feedback = "No rep: You arched your lower back. Squeeze your core."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected elbow flaring/instability."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect lockout! Great triceps isolation."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.elbows_drifted = False 
            self.back_arched = False
            self.ml_posture_maintained = True 

        elif elbow_angle <= 135 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Lower weight behind head..."

            # Full stretch depth is reached when elbows bend tightly (usually < 85 degrees)
            if elbow_angle <= 85:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Deep stretch — Press up!"

        elif elbow_angle > 85 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Lock it out overhead!"
            else:
                self.current_feedback = "Lower it further..."

        # --- Dynamic Form Score ---
        # Perfect depth is roughly 80 degrees for maximum tricep stretch.
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 80) * 0.4)))
        if self.elbows_drifted: form_score -= 50
        if self.back_arched: form_score -= 40
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Overhead Extension",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(shoulder_angle, 1),
            "depth_reached": self.depth_reached
        }