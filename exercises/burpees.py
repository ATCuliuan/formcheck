import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BurpeesTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        
        # Posture & Movement Tripwires
        self.plank_reached = False
        self.core_maintained = True       # NEW: Tracks hip sag in the plank
        self.ml_posture_maintained = True # Tracks the "lazy hunch" at the top
        self.is_ready = False
        self.standing_frames = 0

        self.current_feedback = "Stand upright to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_burpees.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Burpees model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Burpees weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, r_shoulder.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 2.5: 
            self.frame_buffer.clear()
            return {
                "exercise": "Burpees", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: full body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder

        # Calculate standard angles
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        
        # NEW: The ultimate core-sag detector (Shoulder -> Hip -> Ankle)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        if knee_angle > 160 and hip_angle > 150:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Burpees", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Stand upright to begin",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
            }

        # ML Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # THE INSTRUCTOR-APPROVED STATE MACHINE
        # ---------------------------------------------------------
        
        # Plank Detection: Allow for step-backs by looking at overall body extension and horizontal plane
        vertical_drop = abs(active_shoulder.y - active_ankle.y)
        is_horizontal = vertical_drop < 0.45 
        
        # State Transitions
        # Allow drop via squat (knee < 120) OR stiff-leg hinge (hip < 100)
        if self.state == "STANDING" and (knee_angle < 120 or hip_angle < 100):
            self.state = "DROPPING"
            self.current_feedback = "Drop to the floor!"
            
        elif self.state == "DROPPING" and is_horizontal:
            self.state = "PLANK"
            self.plank_reached = True
            self.current_feedback = "Hold tight, don't let hips sag!"

        # Core Sag Tripwire: If hips drop while horizontal, invalidate
        if self.state == "PLANK" and body_angle < 145:
            self.core_maintained = False

        # Ascending: Shoulders rise significantly relative to ankles
        elif self.state == "PLANK" and vertical_drop > 0.45:
            self.state = "ASCENDING"
            self.current_feedback = "Explode up!"

        elif self.state == "ASCENDING" and knee_angle > 155 and hip_angle > 150:
            is_bad_rep = False
            
            # Form Evaluation
            if not self.plank_reached:
                self.current_feedback = "No rep: You didn't reach a full plank."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "No rep: Hips sagged to the floor (Danger)."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: Lazy extension at the top."
                is_bad_rep = True
            
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect burpee!"
            
            # Reset
            self.state = "STANDING"
            self.plank_reached = False
            self.core_maintained = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Score Engine
        form_score = 100
        if not self.plank_reached and self.state == "ASCENDING": form_score -= 50
        if not self.core_maintained: form_score -= 40
        if not self.ml_posture_maintained: form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Burpees", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
        }