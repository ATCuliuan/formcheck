import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class FloorSkullCrusherTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.elbows_dropped = False       # Upper arm drifting forward toward the chest
        self.rep_bounced = False          # Dropping the weight too fast without control
        self.ml_posture_maintained = True # ST-GCN catches elbow flare (elbows pointing outward)
        
        # Setup Guards
        self.setup_frames = 0
        self.descent_frames = 0
        self.is_ready = False

        self.current_feedback = "Lie flat and lock dumbbells straight up"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_floor_skull_crusher.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Floor Skull Crusher model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Floor Skull Crusher weights: {e}")
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

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure the upper body is fully visible
        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Floor Skull Crusher",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and arms not fully visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        # Core Kinematics
        # 1. Elbow angle tracks the triceps extension (Shoulder -> Elbow -> Wrist)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        # 2. Arm verticality angle tracks the upper arm position relative to the torso (Hip -> Shoulder -> Elbow)
        arm_verticality = calculate_angle(active_hip, active_shoulder, active_elbow)

        # Base Initialization Check (Lying flat, arms locked out vertically)
        # An arm pointed straight up to the ceiling creates a ~90 degree angle with the flat torso.
        if elbow_angle > 160 and 75 < arm_verticality < 115:
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
                "exercise": "Floor Skull Crusher",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lie flat, press dumbbells straight up to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(arm_verticality, 1), "depth_reached": False
            }

        # --- Tripwire 1: The Pullover-Press (Elbow Drop) ---
        # If the arm verticality angle drops below 75, they are pulling their elbows down toward their chest.
        # (Over 90 is fine, it just means elbows are pointing back toward the head, which is actually better for tension).
        if arm_verticality < 75 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.elbows_dropped = True

        # --- Tripwire 2: The Gravity Bomb (Bouncing) ---
        if self.state == "DESCENDING":
            self.descent_frames += 1

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
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Bend elbows deeper! Dumbbells to the floor."
                    is_bad_rep = True
                elif self.elbows_dropped:
                    self.current_feedback = "No rep: You pulled your elbows forward. Keep them pointed up!"
                    is_bad_rep = True
                elif self.descent_frames > 0 and self.descent_frames < 6:
                    self.current_feedback = "No rep: You dropped the weight too fast. Control it!"
                    self.rep_bounced = True
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected elbow flare (Tuck elbows in)."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect rep! Excellent triceps isolation."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.elbows_dropped = False 
            self.rep_bounced = False
            self.descent_frames = 0
            self.ml_posture_maintained = True 

        elif elbow_angle <= 140 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down to your ears..."

            # Full stretch depth requires closing the elbow to at least 70 degrees
            if elbow_angle <= 70:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Deep stretch — Press it up!"

        elif elbow_angle > 70 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Lock it out!"
            else:
                self.current_feedback = "Lower until dumbbells touch the floor..."

        # --- Dynamic Form Score ---
        # Perfect extension is 180. Perfect depth is < 65.
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 180) * 0.4)))
        if self.elbows_dropped: form_score -= 50
        if self.rep_bounced: form_score -= 40
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Floor Skull Crusher",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,