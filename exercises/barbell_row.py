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
<<<<<<< HEAD
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
=======
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.pull_reached = False          # Full pull flag (elbow <= 80 deg)
        self.min_elbow_angle = 180.0       # Tracks highest point of the row
        
        # Biomechanical Tripwires (Sagittal / Side View)
        self.shin_vertical = True          # Knee-to-ankle remains relatively straight/perpendicular
        self.knee_bend_proper = True       # Knee-to-hip angle stays within 130° - 150°
        self.torso_angle_proper = True     # Torso inclination between 10° and 45° from horizontal
        self.forearm_path_correct = True   # Catches curling wrist/horizontal forearm pull
        self.controlled_descent = True     # Catches dropping the bar with no eccentric control
        self.ml_posture_maintained = True  # ST-GCN posture detection
        
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
        self.prev_elbow_angle = None
        self.drop_frames = 0
        
        # Debounce counters
        self.shin_frames = 0
        self.knee_frames = 0
        self.torso_frames = 0
        self.forearm_frames = 0

        self.ready_frames = 0
        self.is_ready = False
<<<<<<< HEAD
        self.current_feedback = "Hinge forward at hips to begin"
=======
        self.current_feedback = "Hinge forward deep (~10°-45° torso angle), bend knees slightly"
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
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
            except Exception:
                self.interpreter = None

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

<<<<<<< HEAD
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
=======
        # Relaxed visibility threshold so bent-over frames don't drop tracking
        if max(l_vis, r_vis) < 1.8:
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Step back to show full body profile",
                "elbow_angle": 0.0,
                "knee_angle": 0.0,
                "hip_angle": 0.0
            }

>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

<<<<<<< HEAD
        # Calculate joint kinematics
=======
        # Joint angle calculations
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

<<<<<<< HEAD
        # 1. Relaxed Starting Guard:
        # Accommodates angles between 45°-70° torso inclination (hip_hinge < 140 instead of 120)
        # Elbows only need to be unlocked/hanging (> 140 instead of 150)
        if elbow_angle > 140 and hip_hinge < 140:
=======
        # 1. Torso Angle from Horizontal (Hips to Shoulder plane):
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        torso_angle_horizontal = np.degrees(np.arctan2(dy_torso, dx_torso)) if dx_torso > 0 else 90.0

        # 2. Shin Verticality (Knee to Ankle drift):
        dx_shin = abs(active_knee.x - active_ankle.x)
        dy_shin = abs(active_knee.y - active_ankle.y)
        shin_tilt_angle = np.degrees(np.arctan2(dx_shin, dy_shin)) if dy_shin > 0 else 0.0

        # 3. Forearm Angle (Wrist relative to Elbow off vertical):
        dx_forearm = abs(active_wrist.x - active_elbow.x)
        dy_forearm = abs(active_wrist.y - active_elbow.y)
        forearm_tilt = np.degrees(np.arctan2(dx_forearm, dy_forearm)) if dy_forearm > 0 else 90.0

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        # Torso hinged between 10° and 45°, arms extended down (> 120°)
        is_in_hinge = (10.0 <= torso_angle_horizontal <= 45.0) and (elbow_angle > 120.0)

        if is_in_hinge:
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
            self.ready_frames += 1
        else:
            self.ready_frames = 0

<<<<<<< HEAD
        # Ready after 8 stable frames (~0.25s) instead of 15
        if self.ready_frames > 8: 
=======
        if self.ready_frames > 5:
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
<<<<<<< HEAD
                "exercise": "Barbell Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hinge forward and hang arms down",
=======
                "exercise": "Barbell Row",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Hinge forward deep (~10°-45°) & let arms hang down",
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
                "elbow_angle": round(elbow_angle, 1),
                "knee_angle": round(knee_angle, 1),
                "hip_angle": round(torso_angle_horizontal, 1)
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
            except Exception:
                pass

        # ---------------------------------------------------------
<<<<<<< HEAD
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
=======
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["EXTENDED", "ASCENDING", "TOP", "DESCENDING"]:
            # A. Knee-to-Ankle Vertical Check:
            if shin_tilt_angle > 18.0:
                self.shin_frames += 1
                if self.shin_frames >= 5:
                    self.shin_vertical = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # B. Knee Flexion Range Check:
            if (knee_angle < 110.0) or (knee_angle > 168.0):
                self.knee_frames += 1
                if self.knee_frames >= 5:
                    self.knee_bend_proper = False
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

            # C. Torso Inclination Check (Locked between 15.0° and 60.0°):
            if (torso_angle_horizontal < 15.0) or (torso_angle_horizontal > 60.0):
                self.torso_frames += 1
                if self.torso_frames >= 12:
                    self.torso_angle_proper = False
            else:
                self.torso_frames = max(0, self.torso_frames - 3)

            # D. Forearm Path (Prevent Bicep Curl Cheating):
            if forearm_tilt >= 52.0:
                self.forearm_frames += 1
                if self.forearm_frames >= 4:
                    self.forearm_path_correct = False
            else:
                self.forearm_frames = max(0, self.forearm_frames - 1)
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
<<<<<<< HEAD
        # 4. STATE MACHINE (Accommodates angled cameras & clothing)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.initial_knee_angle = knee_angle
            self.baseline_hip_hinge = hip_hinge
            self.current_feedback = "Pull to your stomach!"

        elif self.state in ["EXTENDED", "DESCENDING"] and elbow_angle < 135:
=======
        # 4. STATE MACHINE (Full Range Enforcement)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Pull bar toward your stomach!"

        elif self.state in ["EXTENDED", "DESCENDING"] and elbow_angle < 130:
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
            self.state = "ASCENDING"
            self.min_elbow_angle = elbow_angle
            self.current_feedback = "Drive elbows back past ribs..."

<<<<<<< HEAD
        # Full contraction: Changed from strict 80° to realistic 95°
        elif self.state == "ASCENDING" and elbow_angle <= 95:
=======
        # Full contraction reached
        elif self.state == "ASCENDING" and elbow_angle <= 80:
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
            self.pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Good squeeze! Lower under control."

<<<<<<< HEAD
        # Half-rep branch: Reached at least 125° and reversed back downward by > 10°
        elif self.state == "ASCENDING" and elbow_angle <= 125 and (elbow_angle - self.min_elbow_angle) > 10:
            self.reversal_frames += 1
            if self.reversal_frames >= 4:
                self.half_pull_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering weight..."

        elif self.state == "TOP" and elbow_angle > 105:
=======
        # Returning downward from full contraction
        elif self.state == "TOP" and elbow_angle > 95:
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
            self.state = "DESCENDING"
            self.current_feedback = "Control the bar down..."

<<<<<<< HEAD
        # D. Eccentric Drop Detector: Raised threshold from 20° to 30°/frame to prevent false alarms on noisy frames
        if self.state == "DESCENDING":
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 30.0:
                    self.descent_drop_frames += 1
                    if self.descent_drop_frames >= 2:
=======
        # Early return prior to reaching full row height
        elif self.state == "ASCENDING" and (elbow_angle - self.min_elbow_angle) > 8:
            self.state = "DESCENDING"
            self.current_feedback = "Perform a full rep! Pull elbows past your torso."

        # Eccentric Free-Fall Monitor
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 30.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
                        self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

<<<<<<< HEAD
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
=======
        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "TOP", "ASCENDING"] and elbow_angle >= 135:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.torso_angle_proper:
                self.current_feedback = "No rep: Keep torso hinged low between 15°-60°!"
                is_bad_rep = True
            elif not self.knee_bend_proper:
                self.current_feedback = "No rep: Keep knees softly bent (110°-168°)."
                is_bad_rep = True
            elif not self.shin_vertical:
                self.current_feedback = "No rep: Keep shins vertical to floor! Do not push knees forward."
                is_bad_rep = True
            elif not self.forearm_path_correct:
                self.current_feedback = "No rep: Do not curl the bar! Drive through your elbows."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not drop the weight."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor spinal alignment."
                is_bad_rep = True
            elif not self.pull_reached:
                self.current_feedback = "No rep: Perform a full rep! Pull elbows past your torso."
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
                is_bad_rep = True
                
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
<<<<<<< HEAD
            elif self.pull_reached:
                self.good_reps += 1
                self.current_feedback = "Great row! Full range of motion."
            elif self.half_pull_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Pull elbows slightly higher next time!"
=======
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d
            else:
                self.good_reps += 1
                self.current_feedback = "Great row! Clean rep."
                
            # Reset for next repetition
            self.state = "EXTENDED"
            self.pull_reached = False
            self.min_elbow_angle = 180.0
<<<<<<< HEAD
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
=======
            
            self.shin_frames = 0
            self.knee_frames = 0
            self.torso_frames = 0
            self.forearm_frames = 0
            self.drop_frames = 0

            self.shin_vertical = True
            self.knee_bend_proper = True
            self.torso_angle_proper = True
            self.forearm_path_correct = True
            self.controlled_descent = True
            self.ml_posture_maintained = True
            self.prev_elbow_angle = None
            self.frame_buffer.clear()
>>>>>>> e48b1d0aebbcb6a610dd42d5be265aba3823a35d

        return {
            "exercise": "Barbell Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(torso_angle_horizontal, 1)
        }