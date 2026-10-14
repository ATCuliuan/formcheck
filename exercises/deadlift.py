import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DeadliftTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.depth_reached = False
        self.half_depth_reached = False
        self.reversal_during_ascent = False
        
        self.prev_hinge_angle = None
        self.lockout_frames = 0
        self.ascent_reversal_frames = 0
        
        # Biomechanical Tripwires (SAGITTAL / SIDE VIEW)
        self.back_neutral = True
        self.bar_path_close = True
        self.heels_flat = True
        self.feet_aligned = True
        self.ml_posture_maintained = True
        
        # Debounce counters
        self.back_frames = 0
        self.drift_frames = 0
        self.heel_frames = 0
        self.foot_align_frames = 0
        self.ml_bad_frames = 0
        self.standing_frames = 0

        self.baseline_ankle_y = None
        self.is_ready = False
        self.current_feedback = "Stand sideways to the camera holding weights"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_deadlift.tflite")

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
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25: 
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Determine side closest to camera
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.8:
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of working body",
                "bar_height_rel": 0.0
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        # Sagittal Kinematics
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        torso_len = np.sqrt((active_shoulder.x - active_hip.x)**2 + (active_shoulder.y - active_hip.y)**2)
        if torso_len == 0:
            torso_len = 0.40

        # Stable Lockout Check
        is_standing = (knee_angle > 158.0) and (hip_angle > 155.0) and (active_wrist.y < (active_knee.y - 0.06))

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if is_standing:
            self.standing_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.9 * self.baseline_ankle_y + 0.1 * active_ankle.y
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Stand tall holding weights to begin",
                "bar_height_rel": 0.0
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_vals = np.exp(output_data - np.max(output_data))
                probs = exp_vals / np.sum(exp_vals)

                if probs[0] > 0.85 and self.state in ["DESCENDING", "BOTTOM"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= 6:
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. SIDE-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            # A. Spinal Flexion / Rounding
            if hip_angle < 50.0:
                self.back_frames += 1
                if self.back_frames >= 5:
                    self.back_neutral = False
            else:
                self.back_frames = max(0, self.back_frames - 1)

            # B. Bar Path Drift
            wrist_drift = abs(active_wrist.x - active_ankle.x) / torso_len
            if wrist_drift > 0.65:
                self.drift_frames += 1
                if self.drift_frames >= 5:
                    self.bar_path_close = False
            else:
                self.drift_frames = max(0, self.drift_frames - 1)

            # C. Heel Lift
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.045:
                self.heel_frames += 1
                if self.heel_frames >= 4:
                    self.heels_flat = False
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

            # D. Foot Alignment
            ankle_x_offset = abs(l_ankle.x - r_ankle.x)
            if ankle_x_offset > 0.08:
                self.foot_align_frames += 1
                if self.foot_align_frames >= 5:
                    self.feet_aligned = False
            else:
                self.foot_align_frames = max(0, self.foot_align_frames - 1)

        # ---------------------------------------------------------
        # 4. PURE ANGLE-DRIVEN STATE MACHINE
        # ---------------------------------------------------------
        hinge_angle = (hip_angle + knee_angle) / 2.0

        if self.prev_hinge_angle is None:
            self.prev_hinge_angle = hinge_angle

        angle_delta = hinge_angle - self.prev_hinge_angle
        self.prev_hinge_angle = hinge_angle

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Hinge hips back, lower past knees"

        # STANDING: Lockout posture
        elif is_standing and hinge_angle > 155.0:
            valid_motion = self.depth_reached or self.half_depth_reached or self.reversal_during_ascent

            if self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and valid_motion:
                self.lockout_frames += 1

                if self.lockout_frames >= 4:
                    # ---------------------------------------------------------
                    # 5. REP COMPLETION
                    # ---------------------------------------------------------
                    is_bad_rep = False

                    if not self.back_neutral:
                        self.current_feedback = "No rep: Back rounded! Keep your spine flat."
                        is_bad_rep = True
                    elif not self.feet_aligned:
                        self.current_feedback = "No rep: Keep feet aligned parallel."
                        is_bad_rep = True
                    elif not self.heels_flat:
                        self.current_feedback = "No rep: Keep heels planted flat on floor."
                        is_bad_rep = True
                    elif not self.bar_path_close:
                        self.current_feedback = "No rep: Keep bar close to your body."
                        is_bad_rep = True
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI flagged spinal curvature."
                        is_bad_rep = True

                    if is_bad_rep:
                        self.bad_reps += 1
                    elif self.reversal_during_ascent:
                        self.half_reps += 1
                        self.current_feedback = "Counted as Half Rep: You stopped before completing lockout!"
                    elif self.half_depth_reached and not self.depth_reached:
                        self.half_reps += 1
                        self.current_feedback = "Counted as Half Rep: Lower weights past your knees!"
                    elif self.depth_reached:
                        self.good_reps += 1
                        self.current_feedback = "Great deadlift! Clean rep."
                    else:
                        self.bad_reps += 1
                        self.current_feedback = "No rep: Incomplete movement."

                    # Reset rep tracking flags
                    self.state = "STANDING"
                    self.depth_reached = False
                    self.half_depth_reached = False
                    self.reversal_during_ascent = False
                    self.lockout_frames = 0
                    self.ascent_reversal_frames = 0

                    self.back_frames = 0
                    self.drift_frames = 0
                    self.heel_frames = 0
                    self.foot_align_frames = 0
                    self.ml_bad_frames = 0

                    self.back_neutral = True
                    self.bar_path_close = True
                    self.heels_flat = True
                    self.feet_aligned = True
                    self.ml_posture_maintained = True
                    self.frame_buffer.clear()
            else:
                self.state = "STANDING"
                self.lockout_frames = 0
                self.current_feedback = "Hinge hips back, lower past knees"

        # DESCENDING: Joints closing / angle decreasing
        elif angle_delta < -1.2 and hinge_angle < 145.0:
            self.lockout_frames = 0

            # If moving downwards from ASCENDING before reaching lockout, mark ascent reversal
            if self.state == "ASCENDING":
                self.ascent_reversal_frames += 1
                if self.ascent_reversal_frames >= 3:
                    self.reversal_during_ascent = True
                    self.current_feedback = "Reversed during ascent: moving back down..."
            else:
                self.ascent_reversal_frames = 0

            self.state = "DESCENDING"
            self.current_feedback = "Push hips back, keep bar close..."

            # Depth evaluated cleanly through hip and knee hinge levels
            if hinge_angle <= 118.0 or active_wrist.y >= (active_knee.y - 0.01):
                self.depth_reached = True
                self.half_depth_reached = True
            elif hinge_angle <= 134.0:
                self.half_depth_reached = True

        # BOTTOM: Peak hinge inflection zone
        elif self.state == "DESCENDING" and (hinge_angle <= 118.0 or active_wrist.y >= (active_knee.y - 0.01)):
            self.state = "BOTTOM"
            self.depth_reached = True
            self.half_depth_reached = True
            self.current_feedback = "Drive hips forward to stand!"

        # ASCENDING: Joints opening / angle increasing
        elif angle_delta > 1.2 and hinge_angle < 155.0:
            self.lockout_frames = 0
            self.state = "ASCENDING"
            self.current_feedback = "Drive hips forward to lockout!"

        # Dynamic Scoring
        form_score = 100
        if not self.back_neutral: form_score -= 30
        if not self.feet_aligned: form_score -= 25
        if not self.heels_flat: form_score -= 20
        if not self.bar_path_close: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if (self.half_depth_reached or self.reversal_during_ascent) and not self.depth_reached:
            form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Deadlift",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "bar_height_rel": round(active_wrist.y - active_knee.y, 2)
        }