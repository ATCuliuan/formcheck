import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellHammerCurlsTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 

        self.peak_contraction_reached = False  # Full curl (elbow <= 75 deg)
        self.half_curl_reached = False         # Half curl (elbow <= 105 deg)
        self.min_elbow_angle = 180.0          
        self.reversal_frames = 0               

        # Upper-Body Strict Tripwires (SIDE / SAGITTAL VIEW)
        self.elbow_stayed_down = True          # Catches lifting/swinging upper arm forward or up
        self.shoulder_not_shrugged = True      # Catches heavy trap shrugging
        self.torso_stable = True               # Catches excessive backward torso swing
        self.ml_posture_maintained = True      # ST-GCN posture detection

        # Advisory / Warning Flags (Does not invalidate a good rep)
        self.fast_descent_warning = False  
        self.knee_bounce_warning = False       # Advisory warning for using leg momentum

        self.baseline_ear_shoulder_dist = None
        self.baseline_shoulder_hip_dist = None
        self.baseline_knee_angle = None

        # Debounce counters
        self.elevation_frames = 0
        self.shrug_frames = 0
        self.torso_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0

        self.prev_elbow_angle = None
        self.extended_frames = 0
        self.is_ready = False
        self.current_feedback = "Show body sideways, let arm hang"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT Model for Hammer Curls
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_hammer_curls.tflite")

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

        # Visibility across profile
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.8:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Show your arm and torso (side view)",
                "elbow_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Joint angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        
        upper_arm_len = np.sqrt((active_shoulder.x - active_elbow.x)**2 + (active_shoulder.y - active_elbow.y)**2)
        if upper_arm_len == 0:
            upper_arm_len = 0.25

        ear_shoulder_dist = abs(active_shoulder.y - active_ear.y)
        shoulder_hip_dist = abs(active_hip.y - active_shoulder.y)

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if elbow_angle > 130:
            self.extended_frames += 1
        else:
            self.extended_frames = 0

        if self.extended_frames > 6:
            self.is_ready = True
            if self.baseline_ear_shoulder_dist is None:
                self.baseline_ear_shoulder_dist = ear_shoulder_dist
            if self.baseline_shoulder_hip_dist is None:
                self.baseline_shoulder_hip_dist = shoulder_hip_dist
            if self.baseline_knee_angle is None:
                self.baseline_knee_angle = knee_angle
            else:
                self.baseline_knee_angle = 0.9 * self.baseline_knee_angle + 0.1 * knee_angle

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Let arm hang down to start",
                "elbow_angle": round(elbow_angle, 1)
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["CURLING", "TOP"]:
            # A. Shoulder-to-Elbow Elevation & Forward Swing
            vertical_drop_ratio = (active_elbow.y - active_shoulder.y) / upper_arm_len
            is_elevating = (upper_arm_angle > 15.0) or (vertical_drop_ratio < 0.30)
            
            if is_elevating:
                self.elevation_frames += 1
                if self.elevation_frames >= 2:
                    self.elbow_stayed_down = False
            else:
                self.elevation_frames = 0

            # B. Shoulder Shrug Check
            if self.baseline_ear_shoulder_dist is not None:
                ear_compressed = ear_shoulder_dist < (self.baseline_ear_shoulder_dist * 0.85)
                if ear_compressed:
                    self.shrug_frames += 1
                    if self.shrug_frames >= 3:
                        self.shoulder_not_shrugged = False
                else:
                    self.shrug_frames = max(0, self.shrug_frames - 1)

            # C. Torso Lean-Back / Spine Heave
            torso_sway = abs(active_shoulder.x - active_hip.x) / upper_arm_len
            if torso_sway > 0.75:
                self.torso_frames += 1
                if self.torso_frames >= 5:
                    self.torso_stable = False
            else:
                self.torso_frames = 0

            # D. Knee Bounce / Leg-Drive Momentum (Sets advisory warning)
            if self.baseline_knee_angle is not None:
                knee_deflection = abs(knee_angle - self.baseline_knee_angle)
                if knee_deflection > 14.0:
                    self.knee_frames += 1
                    if self.knee_frames >= 3:
                        self.knee_bounce_warning = True
                else:
                    self.knee_frames = max(0, self.knee_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Wrist to Chest Flow)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Curl the weight up!"

        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < 125:
            self.state = "CURLING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Curl up to your chest..."

        elif self.state == "CURLING" and elbow_angle <= 75:
            self.state = "TOP"
            self.peak_contraction_reached = True
            self.half_curl_reached = True
            self.current_feedback = "Good squeeze! Lower the weight."

        elif self.state == "CURLING" and (80 <= self.min_elbow_angle <= 95) and (elbow_angle - self.min_elbow_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_curl_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Half rep: Lowering arm early..."

        elif self.state == "TOP" and elbow_angle > 80:
            self.state = "LOWERING"
            self.current_feedback = "Lower arm down..."

        # Eccentric Free-Fall Monitor (Advisory only)
        if self.state in ["LOWERING", "CURLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 30.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.fast_descent_warning = True
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "CURLING"] and (self.peak_contraction_reached or self.half_curl_reached) and elbow_angle >= 130:
            is_bad_rep = False
            
            # Form Cheats First (Invalidates the rep)
            if not self.elbow_stayed_down:
                self.current_feedback = "No rep: Keep elbows pinned down! Do not lift arm forward or up."
                is_bad_rep = True
            elif not self.shoulder_not_shrugged:
                self.current_feedback = "No rep: Relax your traps! Do not shrug shoulders up."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: Do not heave or lean back with your spine."
                is_bad_rep = True
            
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            elif self.peak_contraction_reached:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Keep knees steady, avoid leg drive."
                elif self.fast_descent_warning:
                    self.current_feedback = "Good rep! Reminder: Lower it slower for better muscle growth."
                else:
                    self.current_feedback = "Great hammer curl! Clean rep."
            elif self.half_curl_reached:
                self.half_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Half Rep! Reminder: Curl to chest and keep knees steady."
                elif self.fast_descent_warning:
                    self.current_feedback = "Half Rep! Reminder: Lower it slower & curl to chest."
                else:
                    self.current_feedback = "Counted as Half Rep: Curl all the way to chest!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Movement was too shallow."
            
            # Reset flags for next repetition
            self.state = "EXTENDED"
            self.peak_contraction_reached = False
            self.half_curl_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            
            self.elevation_frames = 0
            self.shrug_frames = 0
            self.torso_frames = 0
            self.knee_frames = 0
            self.drop_frames = 0
            
            self.elbow_stayed_down = True
            self.shoulder_not_shrugged = True
            self.torso_stable = True
            self.knee_bounce_warning = False
            self.fast_descent_warning = False
            self.prev_elbow_angle = None
            
            if self.baseline_ear_shoulder_dist is not None:
                self.baseline_ear_shoulder_dist = 0.85 * self.baseline_ear_shoulder_dist + 0.15 * ear_shoulder_dist
            if self.baseline_shoulder_hip_dist is not None:
                self.baseline_shoulder_hip_dist = 0.85 * self.baseline_shoulder_hip_dist + 0.15 * shoulder_hip_dist
                
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(((150 - elbow_angle) / 80) * 100)))
        if not self.elbow_stayed_down:
            form_score -= 30
        if not self.shoulder_not_shrugged:
            form_score -= 20
        if not self.torso_stable:
            form_score -= 20
        if self.knee_bounce_warning:
            form_score -= 10
        if self.fast_descent_warning:
            form_score -= 8
        if self.half_curl_reached and not self.peak_contraction_reached:
            form_score -= 10
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Hammer Curls",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1)
        }