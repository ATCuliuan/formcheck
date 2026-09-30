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

        self.pull_reached = False          # Full pull flag (elbow <= 80 deg)
        self.half_pull_reached = False     # Partial pull flag
        self.min_elbow_angle = 180.0       # Tracks highest point of the row
        self.reversal_frames = 0           # Debounce counter for half-rep detection
        
        # Biomechanical Tripwires (Sagittal / Side View)
        self.shin_vertical = True          # Knee-to-ankle remains relatively straight/perpendicular
        self.knee_bend_proper = True       # Knee-to-hip angle stays within 130° - 150°
        self.torso_angle_proper = True     # Torso inclination near ~30° from horizontal
        self.forearm_path_correct = True   # Catches curling wrist/horizontal forearm pull
        self.controlled_descent = True     # Catches dropping the bar with no eccentric control
        self.ml_posture_maintained = True  # ST-GCN posture detection
        
        self.prev_elbow_angle = None
        self.drop_frames = 0
        
        # Debounce counters
        self.shin_frames = 0
        self.knee_frames = 0
        self.torso_frames = 0
        self.forearm_frames = 0

        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Hinge forward to ~30° torso angle, bend knees slightly"
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

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_knee.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

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
                "form_score": 0,
                "feedback": "Step back to show full body profile",
                "elbow_angle": 0.0,
                "knee_angle": 0.0,
                "hip_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Joint angle calculations
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # 1. Torso Angle from Horizontal (Hips to Head/Shoulder plane):
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        torso_angle_horizontal = np.degrees(np.arctan2(dy_torso, dx_torso)) if dx_torso > 0 else 90.0

        # 2. Shin Verticality (Knee to Ankle drift):
        dx_shin = abs(active_knee.x - active_ankle.x)
        dy_shin = abs(active_knee.y - active_ankle.y)
        shin_tilt_angle = np.degrees(np.arctan2(dx_shin, dy_shin)) if dy_shin > 0 else 0.0

        # 3. Forearm Angle (Wrist relative to Elbow off vertical):
        # 0° = dead vertical hang down, 90° = fully horizontal forearm
        dx_forearm = abs(active_wrist.x - active_elbow.x)
        dy_forearm = abs(active_wrist.y - active_elbow.y)
        forearm_tilt = np.degrees(np.arctan2(dx_forearm, dy_forearm)) if dy_forearm > 0 else 90.0

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        # Torso roughly hinged (~15° to ~55°), arms hanging down (> 130°)
        is_in_hinge = (15.0 <= torso_angle_horizontal <= 55.0) and (elbow_angle > 130.0)

        if is_in_hinge:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 6:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Hinge forward to ~30° & let arms hang down",
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
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["EXTENDED", "ASCENDING", "TOP", "DESCENDING"]:
            # A. Knee-to-Ankle/Toe Straight/Vertical Check:
            if shin_tilt_angle > 16.0:
                self.shin_frames += 1
                if self.shin_frames >= 4:
                    self.shin_vertical = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # B. Knee-to-Hip Angle Check (Must remain 130° - 150°):
            if (knee_angle < 125.0) or (knee_angle > 155.0):
                self.knee_frames += 1
                if self.knee_frames >= 4:
                    self.knee_bend_proper = False
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

            # C. Torso Inclination (~30° from horizontal, allowed window 18° to 45°):
            if (torso_angle_horizontal < 16.0) or (torso_angle_horizontal > 48.0):
                self.torso_frames += 1
                if self.torso_frames >= 4:
                    self.torso_angle_proper = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # D. Forearm Path (Diagonal Pull Check):
            # Allows natural diagonal pull trajectory (~15° to ~45°). If forearm tilts >= 50° toward horizontal, flag curl
            if forearm_tilt >= 50.0:
                self.forearm_frames += 1
                if self.forearm_frames >= 3:
                    self.forearm_path_correct = False
            else:
                self.forearm_frames = max(0, self.forearm_frames - 1)

            # Track peak pull depth during the lift
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Full Pull vs Half Rep)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Pull bar toward your stomach!"

        elif self.state in ["EXTENDED", "DESCENDING"] and elbow_angle < 130:
            self.state = "ASCENDING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Drive elbows back past ribs..."

        # Full contraction reached (elbow <= 80 deg)
        elif self.state == "ASCENDING" and elbow_angle <= 80:
            self.pull_reached = True
            self.state = "TOP"
            self.current_feedback = "Good squeeze! Lower under control."

        # Half rep branch: Pull reached within 90°-120° and reversed downward by > 8°
        elif self.state == "ASCENDING" and (90 <= self.min_elbow_angle <= 120) and (elbow_angle - self.min_elbow_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_pull_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering weight early..."

        elif self.state == "TOP" and elbow_angle > 95:
            self.state = "DESCENDING"
            self.current_feedback = "Control the bar down..."

        # Eccentric Free-Fall Monitor
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 28.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION (Return to bottom >= 135 deg)
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "TOP", "ASCENDING"] and (self.pull_reached or self.half_pull_reached) and elbow_angle >= 135:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.torso_angle_proper:
                self.current_feedback = "No rep: Maintain ~30° torso angle! Do not stand upright."
                is_bad_rep = True
            elif not self.knee_bend_proper:
                self.current_feedback = "No rep: Keep knees bent between 130°-150°."
                is_bad_rep = True
            elif not self.shin_vertical:
                self.current_feedback = "No rep: Keep shins vertical to floor! Do not push knees forward."
                is_bad_rep = True
            elif not self.forearm_path_correct:
                self.current_feedback = "No rep: Do not curl the bar! Drive with elbows, keep forearms angled down."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower the bar with control."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged spinal curvature."
                is_bad_rep = True
                
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            elif self.pull_reached:
                self.good_reps += 1
                self.current_feedback = "Great row! Clean rep."
            elif self.half_pull_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Pull elbows all the way past your torso!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Pull was too shallow."
                
            # Reset flags for the next repetition
            self.state = "EXTENDED"
            self.pull_reached = False
            self.half_pull_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            
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

        # Dynamic Scoring
        form_score = max(0, min(100, int(((160 - elbow_angle) / 80) * 100)))
        if not self.torso_angle_proper: form_score -= 25
        if not self.knee_bend_proper: form_score -= 25
        if not self.shin_vertical: form_score -= 20
        if not self.forearm_path_correct: form_score -= 20
        if not self.controlled_descent: form_score -= 15
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
            "hip_angle": round(torso_angle_horizontal, 1)
        }