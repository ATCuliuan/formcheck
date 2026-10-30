import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellKickbackTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.arm_swung = False            # Elbow dropping and swinging from the shoulder
        self.torso_too_high = False       # Standing up too straight (losing gravity resistance)
        self.ml_posture_maintained = True # ST-GCN catches horizontal flaring/twisting
        
        # Setup Guards
        self.setup_frames = 0
        self.is_ready = False

        self.current_feedback = "Hinge forward and pin your elbow up to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_kickback.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Dumbbell Kickback model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Dumbbell Kickback weights: {e}")
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
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_knee.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure full torso and working arm are visible
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Kickback",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and arm not fully visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera (the working arm side)
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Core Kinematics
        # 1. Elbow angle tracks the actual kickback extension
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        # 2. Torso angle tracks the hinge depth (Shoulder -> Hip -> Knee)
        torso_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        # Base Initialization Check (Hinged forward, elbow bent at 90 deg, pinned up)
        # We want a bent torso (< 120) and a bent elbow (< 110)
        if torso_angle < 120 and elbow_angle < 110:
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
                "exercise": "Dumbbell Kickback",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Hinge forward, pin elbows up to ribs to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(torso_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1: The Pendulum Swing ---
        # If the elbow drops significantly below the shoulder line on the Y-axis, they unpinned it.
        # MediaPipe Y increases downward. If elbow.y is much greater than shoulder.y, it has dropped.
        if (active_elbow.y - active_shoulder.y) > 0.15 and self.state in ["KICKING", "TOP", "LOWERING"]:
            self.arm_swung = True

        # --- Tripwire 2: Standing Too Upright ---
        # Torso must remain hinged to fight gravity. Opening up past 130 ruins the physics.
        if torso_angle > 130 and self.state in ["KICKING", "TOP", "LOWERING"]:
            self.torso_too_high = True

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
        if elbow_angle < 110:
            if self.state == "LOWERING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: You didn't lock your arm out! (Half rep)"
                    is_bad_rep = True
                elif self.arm_swung:
                    self.current_feedback = "No rep: Pin your elbow high! Don't swing."
                    is_bad_rep = True
                elif self.torso_too_high:
                    self.current_feedback = "No rep: You stood up. Hinge over to fight gravity."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected twisting or instability."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect lockout! Great triceps isolation."
                
                self.depth_reached = False
                
            self.state = "BOTTOM"
            # Reset tripwires
            self.arm_swung = False 
            self.torso_too_high = False
            self.ml_posture_maintained = True 

        elif elbow_angle >= 110 and self.state in ["BOTTOM", "KICKING"]:
            self.state = "KICKING"
            self.current_feedback = "Kick the weight back, keep elbow pinned..."

            # True kickback depth (lockout) is achieved when the arm is completely straight
            if elbow_angle >= 165:
                self.depth_reached = True
                self.state = "TOP"
                self.current_feedback = "Squeeze triceps! Now lower slowly."

        elif elbow_angle < 165 and self.state in ["TOP", "KICKING"]:
            if self.depth_reached:
                self.state = "LOWERING"
                self.current_feedback = "Control it down..."
            else:
                self.current_feedback = "Kick it all the way back!"

        # --- Dynamic Form Score ---
        # Perfect rep hits 175-180 degree elbow extension.
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 180) * 0.5)))
        if self.arm_swung: form_score -= 50
        if self.torso_too_high: form_score -= 40
        if not self.depth_reached and self.state == "LOWERING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Kickback",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(torso_angle, 1),
            "depth_reached": self.depth_reached
        }