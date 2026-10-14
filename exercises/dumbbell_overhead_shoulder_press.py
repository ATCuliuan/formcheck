import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellOverheadShoulderPressTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.lockout_reached = False       # Full overhead lockout (elbow angle >= 155 deg)
        self.max_reach_angle = 0.0         # Tracks peak extension during upward phase
        
        # Front View Form Tripwires
        self.arm_path_aligned = True       # Catches flaring elbows excessively away from body
        self.torso_upright = True          # Catches lateral side bending
        self.arms_even = True              # Catches asymmetric pressing
        self.started_from_rack = True      # Catches front raises or starting below shoulder level
        self.ml_posture_maintained = True

        # Advisory / Warning Flags (Does not invalidate a good rep)
        self.knee_bounce_warning = False   # Catches using leg-drive / push press momentum
        self.baseline_knee_angle = None
        self.baseline_hip_y = None         # Baseline for vertical hip dip detection

        # Debounce counters
        self.flare_frames = 0
        self.torso_frames = 0
        self.even_frames = 0
        self.knee_frames = 0
        self.rack_violation_frames = 0

        self.is_ready = False
        self.racked_frames = 0

        self.current_feedback = "Bring weights to shoulder level"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # LiteRT ST-GCN Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_overhead_shoulder_press.tflite")

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

        # Visibility check
        vis_all = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.35:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "feedback": "Face camera directly: show full torso and arms",
                "elbow_angle": 0.0, "hip_angle": 0.0
            }

        # Elbow extension angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)
        max_elbow_angle = max(l_elbow_angle, r_elbow_angle)

        # Bilateral knee angles (checks for knee bounce / leg drive)
        has_legs_visible = (l_knee.visibility > 0.40 and l_ankle.visibility > 0.40) or (r_knee.visibility > 0.40 and r_ankle.visibility > 0.40)
        current_knee_angle = None
        if has_legs_visible:
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            current_knee_angle = min(l_knee_angle, r_knee_angle)

        # Scale reference: Shoulder Width
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        if shoulder_width == 0:
            shoulder_width = 0.30

        # Midpoint references
        current_hip_y = (l_hip.y + r_hip.y) / 2.0
        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        max_wrist_y = max(l_wrist.y, r_wrist.y)  # Lower hand on screen

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Weights strictly racked near shoulder level)
        # ---------------------------------------------------------
        # Both wrists must be above chest/mid-rib level (smaller Y = higher up)
        # and elbows must be bent (< 115 deg)
        wrists_at_shoulders = max_wrist_y < (avg_shoulder_y + 0.12) and (max_wrist_y < current_hip_y - 0.10)
        is_racked = (max_elbow_angle < 115) and wrists_at_shoulders

        if is_racked:
            self.racked_frames += 1
        else:
            self.racked_frames = 0

        if self.racked_frames > 6:
            self.is_ready = True
            
            # Calibrate baseline hip Y
            if self.baseline_hip_y is None:
                self.baseline_hip_y = current_hip_y
            else:
                self.baseline_hip_y = 0.9 * self.baseline_hip_y + 0.1 * current_hip_y

            # Calibrate baseline knee angle
            if current_knee_angle is not None:
                if self.baseline_knee_angle is None:
                    self.baseline_knee_angle = current_knee_angle
                else:
                    self.baseline_knee_angle = 0.9 * self.baseline_knee_angle + 0.1 * current_knee_angle

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "feedback": "Rack dumbbells at shoulder level to begin",
                "elbow_angle": round(min_elbow_angle, 1), "hip_angle": 0.0
            }

        # 2. ST-GCN Buffer & Inference
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

                if probs[0] > 0.85 and self.state in ["PRESSING", "TOP"]:
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["RACKED", "PRESSING", "TOP", "LOWERING"]:
            # A. Front Raise Trap: Wrists dropped significantly below shoulders during movement
            # Catches swinging up from waist/hips
            if max_wrist_y > (avg_shoulder_y + 0.20):
                self.rack_violation_frames += 1
                if self.rack_violation_frames >= 4:
                    self.started_from_rack = False
            else:
                self.rack_violation_frames = max(0, self.rack_violation_frames - 1)

            # B. Elbow Flare Check:
            l_elbow_flare = abs(l_elbow.x - l_shoulder.x) / shoulder_width
            r_elbow_flare = abs(r_elbow.x - r_shoulder.x) / shoulder_width

            is_flaring = (l_elbow_flare > 0.48) or (r_elbow_flare > 0.48)
            
            if is_flaring:
                self.flare_frames += 1
                if self.flare_frames >= 3:
                    self.arm_path_aligned = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # C. Lateral Spine Sway:
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            if abs(mid_shoulder_x - mid_hip_x) > 0.09:
                self.torso_frames += 1
                if self.torso_frames >= 3:
                    self.torso_upright = False
            else:
                self.torso_frames = 0

            # D. Asymmetric Lag:
            if abs(l_wrist.y - r_wrist.y) > 0.14:
                self.even_frames += 1
                if self.even_frames >= 3:
                    self.arms_even = False
            else:
                self.even_frames = 0

            # E. Knee Bounce / Leg Drive Detection:
            hip_drop_normalized = (current_hip_y - self.baseline_hip_y) / shoulder_width if self.baseline_hip_y else 0.0

            knee_deflection = 0.0
            if self.baseline_knee_angle is not None and current_knee_angle is not None:
                knee_deflection = abs(current_knee_angle - self.baseline_knee_angle)

            is_bouncing = (hip_drop_normalized > 0.06) or (knee_deflection > 6.0)

            if is_bouncing and self.state in ["RACKED", "PRESSING"]:
                self.knee_frames += 1
                if self.knee_frames >= 2:
                    self.knee_bounce_warning = True
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

            self.max_reach_angle = max(self.max_reach_angle, min_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Full Range Enforcement)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RACKED"
            self.current_feedback = "Press dumbbells overhead!"

        # Start pressing: arms extend past 100 deg, wrists must be moving above shoulders
        elif self.state in ["RACKED", "LOWERING"] and min_elbow_angle > 100 and max_wrist_y < avg_shoulder_y:
            self.state = "PRESSING"
            self.max_reach_angle = min_elbow_angle
            self.current_feedback = "Drive weights straight up..."

        # Full Lockout (Both arms overhead >= 155 deg)
        elif self.state == "PRESSING" and min_elbow_angle >= 155:
            self.state = "TOP"
            self.lockout_reached = True
            self.current_feedback = "Full lockout! Lower under control."

        # Early return prior to reaching full lockout
        elif self.state == "PRESSING" and (self.max_reach_angle - min_elbow_angle) > 8:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Press all the way to full lockout."

        elif self.state == "TOP" and max_elbow_angle < 140:
            self.state = "LOWERING"
            self.current_feedback = "Lower weights down..."

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "PRESSING"] and max_elbow_angle <= 115:
            is_bad_rep = False

            # Form Cheats First (Invalidates the rep)
            if not self.started_from_rack:
                self.current_feedback = "No rep: Start and end at shoulder level! Do not swing from hips."
                is_bad_rep = True
            elif not self.arm_path_aligned:
                self.current_feedback = "No rep: Keep elbows in! Do not flare arms wide outward."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Do not lean sideways! Keep spine upright."
                is_bad_rep = True
            elif not self.arms_even:
                self.current_feedback = "No rep: Press evenly! One arm lagged behind."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor overhead posture."
                is_bad_rep = True
            elif not self.lockout_reached:
                self.current_feedback = "No rep: Perform a full rep! Press all the way to full lockout."
                is_bad_rep = True

            # Rep Evaluation
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Strict press only, avoid leg bounce."
                else:
                    self.current_feedback = "Great press! Full lockout."

            # Reset flags for next rep
            self.state = "RACKED"
            self.lockout_reached = False
            self.max_reach_angle = 0.0
            
            self.flare_frames = 0
            self.torso_frames = 0
            self.even_frames = 0
            self.knee_frames = 0
            self.rack_violation_frames = 0

            self.arm_path_aligned = True
            self.torso_upright = True
            self.arms_even = True
            self.started_from_rack = True
            self.ml_posture_maintained = True
            self.knee_bounce_warning = False

            if self.baseline_hip_y is not None:
                self.baseline_hip_y = 0.85 * self.baseline_hip_y + 0.15 * current_hip_y

            self.frame_buffer.clear()

        return {
            "exercise": "Dumbbell Overhead Press",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "feedback": self.current_feedback,
            "elbow_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0
        }