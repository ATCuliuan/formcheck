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
        self.half_reps = 0                 
        
        self.pull_reached = False          # Full pull (elbow <= 95 deg, relaxed from 80)
        self.half_pull_reached = False     # Partial pull (elbow <= 125 deg)
        self.min_elbow_angle = 180.0       
        self.reversal_frames = 0           
        
        # Debounce Fault Counters (requires multi-frame persistence to avoid lighting jitter)
        self.torso_yank_frames = 0
        self.knee_bounce_frames = 0
        self.neck_crane_frames = 0
        self.descent_drop_frames = 0
        
        # Evaluated Tripwire Flags
        self.torso_stable = True           
        self.knees_static = True           
        self.neck_neutral = True           
        self.controlled_descent = True     
        self.ml_posture_maintained = True  
        
        self.baseline_hip_hinge = None
        self.initial_knee_angle = None
        self.prev_elbow_angle = None
        
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Hinge forward at hips to begin"
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

        # More forgiving visibility threshold for poor lighting (lowered from 3.0 to 2.2)
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_knee.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 2.2:
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step into camera view",
                "elbow_angle": 0.0, "knee_angle": 0.0, "hip_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Calculate joint kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        hip_hinge = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 1. Relaxed Starting Guard:
        # Accommodates angles between 45°-70° torso inclination (hip_hinge < 140 instead of 120)
        # Elbows only need to be unlocked/hanging (> 140 instead of 150)
        if elbow_angle > 140 and hip_hinge < 140:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        # Ready after 8 stable frames (~0.25s) instead of 15
        if self.ready_frames > 8: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hinge forward and hang arms down",
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
                
                # Check ST-GCN prediction
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception: pass

        # ---------------------------------------------------------
        # 3. TOLERANT KINEMATIC TRIPWIRES (Multi-frame debounced)
        # ---------------------------------------------------------
        if self.state in ["ASCENDING", "TOP"]:
            # A. Ego Yank: Compares against user's OWN starting hinge, not a hardcoded 125°.
            # Allows up to 25° of natural torso swing under heavy pull.
            if self.baseline_hip_hinge is not None:
                if (hip_hinge - self.baseline_hip_hinge) > 25:
                    self.torso_yank_frames += 1
                    if self.torso_yank_frames >= 4:  # Must persist 4+ frames (~130ms)
                        self.torso_stable = False

            # B. Leg Bounce: Increased buffer from 15° to 25° deviation to allow natural shock absorption
            if self.initial_knee_angle is not None:
                if abs(knee_angle - self.initial_knee_angle) > 25:
                    self.knee_bounce_frames += 1
                    if self.knee_bounce_frames >= 4:
                        self.knees_static = False

            # C. Neck Craning: Threshold lowered from 135° to 115° to absorb off-angle perspective warping
            if neck_angle < 115:
                self.neck_crane_frames += 1
                if self.neck_crane_frames >= 5:
                    self.neck_neutral = False

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Accommodates angled cameras & clothing)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.initial_knee_angle = knee_angle
            self.baseline_hip_hinge = hip_hinge
            self.current_feedback = "Pull to your stomach!"

        elif self.state in ["EXTENDED", "DESCENDING"] and elbow_angle < 135:
            self.state = "ASCENDING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Pull elbows back..."

        # Full contraction: Changed from strict 80° to realistic 95°
        elif self.state == "ASCENDING" and elbow_angle <= 95:
            self.pull_reached = True
            self.half_pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Good squeeze! Lower under control."

        # Half-rep branch: Reached at least 125° and reversed back downward by > 10°
        elif self.state == "ASCENDING" and elbow_angle <= 125 and (elbow_angle - self.min_elbow_angle) > 10:
            self.reversal_frames += 1
            if self.reversal_frames >= 4:
                self.half_pull_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering weight..."

        elif self.state == "TOP" and elbow_angle > 105:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down"

        # D. Eccentric Drop Detector: Raised threshold from 20° to 30°/frame to prevent false alarms on noisy frames
        if self.state == "DESCENDING":
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 30.0:
                    self.descent_drop_frames += 1
                    if self.descent_drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Completion: Lowered return threshold from 150° to 142° for soft-lockout lifters
        if self.state == "DESCENDING" and elbow_angle >= 142:
            is_bad_rep = False
            
            # Form Cheats Evaluation
            if not self.torso_stable:
                self.current_feedback = "No rep: You stood up excessively during the pull."
                is_bad_rep = True
            elif not self.knees_static:
                self.current_feedback = "No rep: Too much leg bounce! Keep knees stable."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Keep neck neutral with your back."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Lower the bar under control."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged excessive back rounding."
                is_bad_rep = True
                
            # Range of Motion Evaluation
            if is_bad_rep:
                self.bad_reps += 1
            elif self.pull_reached:
                self.good_reps += 1
                self.current_feedback = "Great row! Full range of motion."
            elif self.half_pull_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Pull elbows slightly higher next time!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Pull was too shallow to count."
                
            # Reset for next repetition
            self.state = "EXTENDED"
            self.pull_reached = False
            self.half_pull_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            
            self.torso_yank_frames = 0
            self.knee_bounce_frames = 0
            self.neck_crane_frames = 0
            self.descent_drop_frames = 0
            
            self.torso_stable = True
            self.knees_static = True
            self.neck_neutral = True
            self.controlled_descent = True
            self.prev_elbow_angle = None
            self.ml_posture_maintained = True
            
            # Running baseline update to adapt to user posture adjustments between reps
            self.initial_knee_angle = 0.8 * self.initial_knee_angle + 0.2 * knee_angle
            self.baseline_hip_hinge = 0.8 * self.baseline_hip_hinge + 0.2 * hip_hinge
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(((155 - elbow_angle) / 60) * 100)))
        if not self.torso_stable: form_score -= 25
        if not self.knees_static: form_score -= 15
        if not self.neck_neutral: form_score -= 10
        if not self.controlled_descent: form_score -= 15
        if not self.ml_posture_maintained: form_score -= 25
        if self.half_pull_reached and not self.pull_reached: form_score -= 10
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