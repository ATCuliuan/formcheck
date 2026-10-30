import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class PlankTricepsExtensionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.depth_reached = False
        
        # Posture tripwires
        self.core_broken = False          # Catches both the "Cobra" sag and the "Pike"
        self.ml_posture_maintained = True # ST-GCN catches asymmetrical pushing (one arm at a time)
        
        # Setup Guards
        self.plank_frames = 0
        self.is_ready = False

        self.current_feedback = "Get into a straight forearm plank to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_plank_triceps_extension.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Plank Triceps Extension model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Plank Triceps Extension weights: {e}")
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
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate visibility 
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps

        # Guard: Ensure full body profile is visible (critical for tracking the plank line)
        if max(l_vis, r_vis) < 3.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Plank Triceps Extension",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full body profile not visible",
                "primary_angle": 0.0, "secondary_angle": 0.0, "depth_reached": False
            }

        # Track the side most visible to the camera
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematics
        # 1. Elbow angle tracks the actual triceps press
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        # 2. Body angle tracks the rigidity of the plank (Shoulder -> Hip -> Ankle)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Base Initialization Check (Forearm plank or top pushup plank)
        # As long as the body is relatively straight, we can start counting.
        if body_angle > 155:
            self.plank_frames += 1
        else:
            self.plank_frames = 0

        if self.plank_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Plank Triceps Extension",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lock your core straight like a board to begin",
                "primary_angle": round(elbow_angle, 1), "secondary_angle": round(body_angle, 1), "depth_reached": False
            }

        # --- Tripwire 1 & 2: The Cobra Sag & The Pike ---
        # A perfect plank is 170-180 degrees. 
        # Dropping below 150 means the hips have heavily deviated from the shoulder-to-ankle axis.
        if body_angle < 150 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            self.core_broken = True

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
        if elbow_angle > 145:
            if self.state == "ASCENDING":
                is_bad_rep = False
                
                if not self.depth_reached:
                    self.current_feedback = "No rep: Forearms didn't touch the floor! Go deeper."
                    is_bad_rep = True
                elif self.core_broken:
                    self.current_feedback = "No rep: You lost your plank! Squeeze glutes and abs."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI detected uneven pressing or form break."
                    is_bad_rep = True
                
                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Perfect press! Core is locked in tight."
                
                self.depth_reached = False
                
            self.state = "TOP"
            # Reset tripwires
            self.core_broken = False 
            self.ml_posture_maintained = True 

        elif elbow_angle <= 130 and self.state in ["TOP", "DESCENDING"]:
            self.state = "DESCENDING"
            self.current_feedback = "Lower elbows to the floor simultaneously..."

            # Full depth is reached when forearms are resting flat on the ground (~90-95 degrees)
            if elbow_angle <= 100:
                self.depth_reached = True
                self.state = "BOTTOM"
                self.current_feedback = "Forearms down — Press the floor away!"

        elif elbow_angle > 100 and self.state in ["BOTTOM", "DESCENDING"]:
            if self.depth_reached:
                self.state = "ASCENDING"
                self.current_feedback = "Push!"
            else:
                self.current_feedback = "Lower until elbows touch the floor..."

        # --- Dynamic Form Score ---
        # Perfect depth is roughly 90 degrees.
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4)))
        if self.core_broken: form_score -= 50
        if not self.depth_reached and self.state == "ASCENDING": form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Plank Triceps Extension",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "primary_angle": round(elbow_angle, 1),
            "secondary_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }