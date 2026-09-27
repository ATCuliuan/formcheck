import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class OneArmRowTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Partial pull rep counter
        
        self.pull_reached = False          # Full pull flag (elbow <= 85 deg)
        self.half_pull_reached = False     # Partial pull flag (elbow <= 115 deg)
        self.min_elbow_angle = 180.0       # Tracks deepest point of the row
        self.reversal_frames = 0           # Debounce counter
        
        # Instructor Tripwires (STRICT SIDE / SAGITTAL VIEW)
        self.torso_stable = True           # Catches standing up during the pull
        self.torso_not_twisted = True      # Catches rotating chest toward the camera (Z-depth)
        self.neck_neutral = True           # Catches craning the neck up at the mirror
        self.controlled_descent = True     # Catches free-falling the weight
        self.pull_to_hip = True            # Catches pulling straight up to armpit (bicep row)
        self.ml_posture_maintained = True  # AI catches lumbar rounding
        
        self.prev_elbow_angle = None
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Lean over bench (side profile to camera) to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_one_arm_row.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite One Arm Row model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite One Arm Row weights: {e}")
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

        # In side view, the active/working arm is the one facing the camera
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 2.8:
            self.frame_buffer.clear()
            return {
                "exercise": "One Arm Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of working arm",
                "elbow_angle": 0.0, "hip_angle": 0.0
            }

        # Active side is closest to camera
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        support_shoulder = l_shoulder if r_vis > l_vis else r_shoulder

        # Sagittal Biomechanical Angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_hinge = calculate_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 1. Starting Guard: Arm straight down, back flat and hinged (< 135 deg)
        if elbow_angle > 150 and hip_hinge < 135:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "One Arm Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hinge forward flat and hang arm straight down",
                "elbow_angle": round(elbow_angle, 1), "hip_angle": round(hip_hinge, 1)
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
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "EXTENDED":
                    self.ml_posture_maintained = False
            except Exception: pass

        # ---------------------------------------------------------
        # 3. SIDE-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        
        # A. Ego Yank: Standing up or opening hip hinge during pull
        if hip_hinge > 140 and self.state in ["PULLING", "TOP"]:
            self.torso_stable = False

        # B. Lawnmower Chest Twist: Uses MediaPipe Z-depth from side view
        # When chest twists open toward the camera, shoulder depth difference spikes
        shoulder_depth_drift = abs(active_shoulder.z - support_shoulder.z)
        if shoulder_depth_drift > 0.28 and self.state in ["PULLING", "TOP"]:
            self.torso_not_twisted = False

        # C. Cervical Hyperextension: Craning neck up to stare into mirror
        if neck_angle < 130 and self.state in ["PULLING", "TOP"]:
            self.neck_neutral = False

        # D. Pull Trajectory (Pull to Hip vs Pull to Armpit)
        # Wrist should travel back toward the hip, not stay forward under the shoulder
        if self.state == "TOP":
            wrist_to_hip_x = abs(active_wrist.x - active_hip.x)
            wrist_to_shoulder_x = abs(active_wrist.x - active_shoulder.x)
            # If wrist stayed directly under the shoulder instead of moving toward the hip:
            if wrist_to_shoulder_x < 0.05 and wrist_to_hip_x > 0.18:
                self.pull_to_hip = False

        # Continuously record the peak pull contraction
        if self.state in ["PULLING", "TOP"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Drive elbow back toward your hip!"

        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < 140:
            self.state = "PULLING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Pull elbow back..."

        # Full contraction achieved (elbow <= 85 deg)
        elif self.state == "PULLING" and elbow_angle <= 85:
            self.pull_reached = True
            self.half_pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze lat! Lower under control."

        # Half rep reversal: Reached at least 115°, but began dropping early
        elif self.state == "PULLING" and elbow_angle <= 115 and (elbow_angle - self.min_elbow_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_pull_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Half rep: Lowering weight..."

        elif self.state == "TOP" and elbow_angle > 100:
            self.state = "LOWERING"
            self.current_feedback = "Control the descent"

        # E. Eccentric Drop Detector: Track lowering speed
        if self.state == "LOWERING":
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 20.0:  # Dropped > 20° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Evaluation: Weight returned to full arm extension
        if self.state == "LOWERING" and elbow_angle >= 150:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.torso_stable:
                self.current_feedback = "No rep: You stood up! Keep your chest facing the bench."
                is_bad_rep = True
            elif not self.torso_not_twisted:
                self.current_feedback = "No rep: Don't twist your chest open! Keep shoulders level."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Neutral spine! Look down at the bench, not up."
                is_bad_rep = True
            elif not self.pull_to_hip:
                self.current_feedback = "No rep: Pull back toward your hip, not straight up to your armpit."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower the weight slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged rounded back or improper mechanics."
                is_bad_rep = True
                
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.pull_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect row! Full range of motion."
            elif self.half_pull_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Drive elbow higher past your ribs next time!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Pull was too shallow to count."
                
            # Reset flags for next rep
            self.state = "EXTENDED"
            self.pull_reached = False
            self.half_pull_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            self.torso_stable = True
            self.torso_not_twisted = True
            self.neck_neutral = True
            self.controlled_descent = True
            self.pull_to_hip = True
            self.prev_elbow_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(((160 - elbow_angle) / 75) * 100)))
        if not self.torso_stable: form_score -= 30
        if not self.torso_not_twisted: form_score -= 25
        if not self.pull_to_hip: form_score -= 20
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_pull_reached and not self.pull_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "One Arm Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "hip_angle": round(hip_hinge, 1)
        }