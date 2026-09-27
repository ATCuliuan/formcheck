import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BarbellRowTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Tracks partial repetitions
        
        self.pull_reached = False          # Full pull flag (elbow <= 80 deg)
        self.half_pull_reached = False     # Partial pull flag (elbow <= 115 deg)
        self.min_elbow_angle = 180.0       # Tracks highest point of the row
        self.reversal_frames = 0           # Debounce counter for half-rep detection
        
        # Instructor Tripwires (Sagittal / Side View)
        self.torso_stable = True           # Catches the "Ego Yank" (standing up)
        self.knees_static = True           # Catches squatting / leg-drive bounce
        self.neck_neutral = True           # Catches craning the neck up at the mirror
        self.controlled_descent = True     # Catches dropping the bar with no eccentric control
        self.ml_posture_maintained = True  # ST-GCN catches rounded back / lumbar flexion
        
        self.initial_knee_angle = None
        self.prev_elbow_angle = None
        
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Hinge forward at the hips to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_barbell_row.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Barbell Row model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Barbell Row weights: {e}")
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
        # Joints needed for side-profile tracking
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
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
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_knee.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show full body profile",
                "elbow_angle": 0.0, "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Active side determination
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Biomechanical Angle Calculations
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_hinge = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 1. Starting Guard: Arms straight down, deep hip hinge
        if elbow_angle > 150 and hip_hinge < 120:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hinge forward (back flat) & let arms hang",
                "elbow_angle": round(elbow_angle, 1),
                "knee_angle": round(knee_angle, 1),
                "hip_angle": round(hip_hinge, 1)
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Form Tripwires (Side-View Biomechanics)
        
        # A. Ego Yank: Standing upright during the pull
        if hip_hinge > 125 and self.state in ["ASCENDING", "TOP"]:
            self.torso_stable = False

        # B. Knee Bounce / Squat Cheat: Bouncing legs to generate momentum
        if self.initial_knee_angle is not None and self.state in ["ASCENDING", "TOP"]:
            if abs(knee_angle - self.initial_knee_angle) > 15:
                self.knees_static = False

        # C. Cervical Hyperextension: Craning neck up at mirror (normal is flat/neutral: 140°-180°)
        if neck_angle < 135 and self.state in ["ASCENDING", "TOP"]:
            self.neck_neutral = False

        # D. Track peak pull depth
        if self.state in ["ASCENDING", "TOP"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.initial_knee_angle = knee_angle
            self.current_feedback = "Pull to your stomach!"

        elif self.state in ["EXTENDED", "DESCENDING"] and elbow_angle < 140:
            self.state = "ASCENDING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Pull elbows back..."

        # Full contraction reached (elbow past ribs)
        elif self.state == "ASCENDING" and elbow_angle <= 80:
            self.pull_reached = True
            self.half_pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze back! Lower slowly."

        # Half rep branch: Pull reached at least 115° and reversed downward by > 8°
        elif self.state == "ASCENDING" and elbow_angle <= 115 and (elbow_angle - self.min_elbow_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_pull_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering weight..."

        elif self.state == "TOP" and elbow_angle > 100:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down"

        # E. Eccentric Drop Detector: Monitor lowering speed while in DESCENDING
        if self.state == "DESCENDING":
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 20.0:  # Dropped > 20° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Completion and Evaluation
        if self.state == "DESCENDING" and elbow_angle > 150:
            is_bad_rep = False
            
            # Form Cheats Evaluation
            if not self.torso_stable:
                self.current_feedback = "No rep: You stood up! Keep torso bent over."
                is_bad_rep = True
            elif not self.knees_static:
                self.current_feedback = "No rep: Keep knees fixed! Do not bounce with your legs."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Neutral spine! Look down at the floor, not up."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower the bar slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture (rounded back)."
                is_bad_rep = True
                
            # Range of Motion Evaluation
            if is_bad_rep:
                self.bad_reps += 1
            elif self.pull_reached:
                self.good_reps += 1
                self.current_feedback = "Great row! Full range of motion."
            elif self.half_pull_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Pull elbows past your ribs next time!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Pull was too shallow to count."
                
            # Reset flags for the next repetition
            self.state = "EXTENDED"
            self.pull_reached = False
            self.half_pull_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            self.torso_stable = True
            self.knees_static = True
            self.neck_neutral = True
            self.controlled_descent = True
            self.prev_elbow_angle = None
            self.ml_posture_maintained = True
            self.initial_knee_angle = knee_angle  # Re-baseline knee bend for next rep
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(((160 - elbow_angle) / 80) * 100)))
        if not self.torso_stable: form_score -= 30
        if not self.knees_static: form_score -= 20
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_pull_reached and not self.pull_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Barbell Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_hinge, 1)
        }