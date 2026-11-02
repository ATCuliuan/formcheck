import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class OverheadTrapRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.overhead_reached = False
        self.half_raise_reached = False
        self.max_raise_angle = 0.0
        self.reversal_frames = 0
        
        # Instructor Tripwires (SAGITTAL / SIDE VIEW)
        self.arms_straight = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Advisory / Warning Flags
        self.knee_bounce_warning = False
        self.baseline_knee_angle = None

        # Debounce counters
        self.straight_frames = 0
        self.torso_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0
        self.ready_frames = 0
        
        self.prev_raise_angle = None
        self.is_ready = False
        self.current_feedback = "Stand sideways to the camera, arms hanging down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_overhead_trap_raises.tflite")

        
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

        # In side view, determine the active side facing the camera
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.8:
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of working arm and torso",
                "elbow_angle": 0.0, "raise_angle": 0.0
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal Kinematics
        raise_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        
        # Torso inclination angle off vertical (relative to hips)
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        torso_tilt_angle = np.degrees(np.arctan2(dx_torso, dy_torso)) if dy_torso > 0 else 0.0

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if raise_angle < 50 and elbow_angle > 125:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 6:
            self.is_ready = True
            if self.baseline_knee_angle is None:
                self.baseline_knee_angle = knee_angle
            else:
                self.baseline_knee_angle = 0.9 * self.baseline_knee_angle + 0.1 * knee_angle

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arms hang down to start",
                "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
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
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "RESTING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. SIDE-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["RESTING", "ASCENDING", "TOP", "DESCENDING"]:
            # A. Excessive Elbow Bend
            if elbow_angle < 125:
                self.straight_frames += 1
                if self.straight_frames >= 6:
                    self.arms_straight = False
            else:
                self.straight_frames = max(0, self.straight_frames - 1)

            # B. Backward Lean / Torso Hyperextension
            if torso_tilt_angle > 25.0 and self.state in ["ASCENDING", "TOP"]:
                self.torso_frames += 1
                if self.torso_frames >= 6:
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # C. Knee Bounce / Leg-Drive Momentum (Bicep Curl Logic)
            if self.baseline_knee_angle is not None and self.state == "ASCENDING":
                knee_deflection = abs(knee_angle - self.baseline_knee_angle)
                if knee_deflection > 14.0:
                    self.knee_frames += 1
                    if self.knee_frames >= 3:
                        self.knee_bounce_warning = True
                else:
                    self.knee_frames = max(0, self.knee_frames - 1)

            self.max_raise_angle = max(self.max_raise_angle, raise_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Full Lockout vs Half Rep)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Raise arms up overhead!"

        elif self.state in ["RESTING", "DESCENDING"] and raise_angle > 50:
            self.state = "ASCENDING"
            self.max_raise_angle = raise_angle
            self.reversal_frames = 0
            self.current_feedback = "Raise arms up overhead..."

        elif self.state == "ASCENDING" and raise_angle >= 150:
            self.overhead_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze overhead! Control the negative."

        elif self.state == "ASCENDING" and (80 <= self.max_raise_angle < 145) and (self.max_raise_angle - raise_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_raise_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering arms early..."

        elif self.state == "TOP" and raise_angle < 135:
            self.state = "DESCENDING"
            self.current_feedback = "Control the descent"

        # Eccentric Free-Fall Monitor
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_raise_angle is not None:
                drop_speed = self.prev_raise_angle - raise_angle
                if drop_speed > 28.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_raise_angle = raise_angle
        else:
            self.prev_raise_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "TOP", "ASCENDING"] and (self.overhead_reached or self.half_raise_reached) and raise_angle <= 55:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.arms_straight:
                self.current_feedback = "No rep: Keep arms extended! Do not curl the weights."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Do not lean backward to heave the weight up."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower weights slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
                
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            elif self.overhead_reached:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Keep knees steady, avoid leg drive."
                else:
                    self.current_feedback = "Excellent trap raise! Full overhead range."
            elif self.half_raise_reached:
                self.half_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Half Rep! Reminder: Raise fully overhead and keep knees steady."
                else:
                    self.current_feedback = "Counted as Half Rep: Raise arms all the way overhead past shoulder height!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Raise was too shallow to count."
                
            # Reset flags for next rep
            self.state = "RESTING"
            self.overhead_reached = False
            self.half_raise_reached = False
            self.max_raise_angle = 0.0
            self.reversal_frames = 0
            
            self.straight_frames = 0
            self.torso_frames = 0
            self.knee_frames = 0
            self.drop_frames = 0

            self.arms_straight = True
            self.torso_upright = True
            self.controlled_descent = True
            self.knee_bounce_warning = False
            self.prev_raise_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((raise_angle / 150) * 100)))
        if not self.arms_straight: form_score -= 25
        if not self.torso_upright: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if self.knee_bounce_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_raise_reached and not self.overhead_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Overhead Trap Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "raise_angle": round(raise_angle, 1)
        }