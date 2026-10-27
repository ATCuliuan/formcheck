import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SquatTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.depth_reached = False
        self.half_depth_reached = False
        self.min_knee_angle = 180.0
        self.reversal_frames = 0
        
        # Instructor Tripwires (SAGITTAL / SIDE VIEW)
        self.torso_upright = True
        self.heels_flat = True
        self.feet_aligned = True
        self.controlled_descent = True
        self.ml_posture_maintained = True
        
        # Debounce counters
        self.torso_frames = 0
        self.heel_frames = 0
        self.foot_align_frames = 0
        self.drop_frames = 0
        self.ml_bad_frames = 0
        self.standing_frames = 0

        self.prev_knee_angle = None
        self.baseline_ankle_y = None
        self.is_ready = False

        self.current_feedback = "Stand sideways to the camera to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT model
        self.interpreter = None
        model_path = os.path.join("models", "pt", "st_gcn_squat.pt")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"ST-GCN Pt Squat model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading Pt Squat weights: {e}")
                self.interpreter = None
        else:
            print(f"ST-GCN Pt model not found at {model_path}")

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
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # In side view, determine the side closest to the camera
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.8: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: show full side profile of body",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal Kinematics
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        # Torso inclination relative to vertical (0° = perfectly upright)
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = abs(active_shoulder.y - active_hip.y)
        torso_forward_tilt = np.degrees(np.arctan2(dx_torso, dy_torso)) if dy_torso > 0 else 0.0

        # Hip crease vs knee level
        hip_crease_at_knee = active_hip.y >= (active_knee.y - 0.02)

        # ---------------------------------------------------------
        # 1. SETUP GUARD: Standing upright with knees extended
        # ---------------------------------------------------------
        if knee_angle > 155:
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
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand upright with hips and knees extended",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1), "depth_reached": False
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
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                
                exp_vals = np.exp(output_data - np.max(output_data))
                probs = exp_vals / np.sum(exp_vals)
                
                if probs[0] > 0.75 and self.state in ["DESCENDING", "BOTTOM"]:
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
            # A. Good Morning Squat / Excessive Torso Pitch
            if torso_forward_tilt > 50.0 and self.state in ["DESCENDING", "BOTTOM"]:
                self.torso_frames += 1
                if self.torso_frames >= 4:
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # B. Heel Lift
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.045:
                self.heel_frames += 1
                if self.heel_frames >= 3:
                    self.heels_flat = False
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

            # C. Foot Alignment
            ankle_x_offset = abs(l_ankle.x - r_ankle.x)
            if ankle_x_offset > 0.08:
                self.foot_align_frames += 1
                if self.foot_align_frames >= 4:
                    self.feet_aligned = False
            else:
                self.foot_align_frames = max(0, self.foot_align_frames - 1)

            # Track lowest knee flexion
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Parallel Depth vs Half Rep)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Squat down by pushing hips back!"

        elif self.state in ["STANDING", "ASCENDING"] and knee_angle <= 145:
            self.state = "DESCENDING"
            self.min_knee_angle = knee_angle
            self.reversal_frames = 0
            self.current_feedback = "Lower under control..."

        # Full Depth: Knee angle <= 100° or hip crease at/below knee level
        elif self.state == "DESCENDING" and (knee_angle <= 100 or hip_crease_at_knee):
            self.depth_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Good depth! Drive through mid-foot to stand."

        # Half Rep Reversal
        elif self.state == "DESCENDING" and (105 <= self.min_knee_angle <= 135) and (knee_angle - self.min_knee_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Pushing up early..."

        elif self.state == "BOTTOM" and knee_angle > 105:
            self.state = "ASCENDING"
            self.current_feedback = "Drive up to standing!"

        # Eccentric Free-Fall Monitor
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_knee_angle is not None:
                drop_speed = self.prev_knee_angle - knee_angle
                if drop_speed > 20.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_knee_angle = knee_angle
        else:
            self.prev_knee_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to standing (knee >= 155°)
        # ---------------------------------------------------------
        if self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and (self.depth_reached or self.half_depth_reached) and knee_angle >= 155:
            is_bad_rep = False
            
            # Form Cheats First (Strict - Invalidates rep)
            if not self.torso_upright:
                self.current_feedback = "No rep: Torso pitched too far forward! Keep chest up."
                is_bad_rep = True
            elif not self.feet_aligned:
                self.current_feedback = "No rep: Feet are not aligned! Keep both feet parallel and even."
                is_bad_rep = True
            elif not self.heels_flat:
                self.current_feedback = "No rep: Keep your heels glued to the floor!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not drop into the hole."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected collapsed squat mechanics."
                is_bad_rep = True
            
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            elif self.depth_reached:
                self.good_reps += 1
                self.current_feedback = "Great squat! Full depth."
            elif self.half_depth_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Squat lower until thighs are parallel!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Squat was too shallow."
            
            # Reset flags for next rep
            self.state = "STANDING"
            self.depth_reached = False
            self.half_depth_reached = False
            self.min_knee_angle = 180.0
            self.reversal_frames = 0
            
            self.torso_frames = 0
            self.heel_frames = 0
            self.foot_align_frames = 0
            self.drop_frames = 0
            self.ml_bad_frames = 0

            self.torso_upright = True
            self.heels_flat = True
            self.feet_aligned = True
            self.controlled_descent = True
            self.prev_knee_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Form Scoring
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))
        if not self.torso_upright: form_score -= 30
        if not self.feet_aligned: form_score -= 25
        if not self.heels_flat: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_depth_reached and not self.depth_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Bodyweight Squat",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }