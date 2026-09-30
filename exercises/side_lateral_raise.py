import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class LateralRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Counts partial ROM raises
        
        self.target_height_reached = False # Full ROM: parallel reach
        self.half_raise_reached = False    # Half ROM
        self.max_abduction = 0.0           # Tracks peak raise height achieved
        self.reversal_frames = 0           # Debounce counter for jitter
        
        # Instructor Tripwires (FRONT VIEW - Calibrated to Realistic Lifter Limits)
        self.arms_straight_enough = True   # Soft bend allowed (~115°+), only catches excessive curling
        self.torso_stable = True           # Catches large momentum swings / side lurches
        self.shoulders_down = True         # Catches heavy trap shrugging
        self.elbows_lead_wrists = True     # Catches wrist-hiking over elbows
        self.arms_balanced = True          # Catches asymmetric arm height
        self.controlled_descent = True     # Catches dropping weights completely free-fall
        self.not_over_raised = True        # Catches raising way above shoulder level (>105°)
        self.ml_posture_maintained = True

        # Advisory / Warning Flags
        self.knee_bounce_warning = False   # Leg drive / bounce warning
        self.baseline_knee_angle = None
        self.baseline_hip_y = None

        # Debounce counters
        self.standing_frames = 0
        self.knee_frames = 0
        self.over_raise_frames = 0
        self.straight_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.wrist_frames = 0
        self.balance_frames = 0
        self.ml_bad_frames = 0

        self.is_ready = False
        self.baseline_neck_ratio = None
        self.prev_abduction = None
        
        self.current_feedback = "Face camera, let arms hang straight down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_side_lateral_raise.tflite")

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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        vis_all = [l_shoulder.visibility, r_shoulder.visibility, l_elbow.visibility, 
                   r_elbow.visibility, l_wrist.visibility, r_wrist.visibility,
                   l_hip.visibility, r_hip.visibility]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.30:
            self.frame_buffer.clear()
            return {
                "exercise": "Lateral Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show full torso and arms",
                "elbow_bend": 0.0, "abduction_angle": 0.0
            }

        # Bilateral Kinematics
        l_abduction = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction = calculate_angle(r_hip, r_shoulder, r_elbow)
        min_abduction = min(l_abduction, r_abduction)
        max_abduction = max(l_abduction, r_abduction)

        l_elbow_bend = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_bend = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_bend = min(l_elbow_bend, r_elbow_bend)

        # Knee visibility & angles
        has_legs = (l_knee.visibility > 0.35 and l_ankle.visibility > 0.35) or (r_knee.visibility > 0.35 and r_ankle.visibility > 0.35)
        current_knee_angle = None
        if has_legs:
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            current_knee_angle = min(l_knee_angle, r_knee_angle)

        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), 0.30)
        current_hip_y = (l_hip.y + r_hip.y) / 2.0

        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0))
        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height if torso_height > 0 else 0.0

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if max_abduction < 32:
            self.standing_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            else:
                self.baseline_neck_ratio = 0.9 * self.baseline_neck_ratio + 0.1 * neck_ratio

            if self.baseline_hip_y is None:
                self.baseline_hip_y = current_hip_y
            else:
                self.baseline_hip_y = 0.9 * self.baseline_hip_y + 0.1 * current_hip_y

            if current_knee_angle is not None:
                if self.baseline_knee_angle is None:
                    self.baseline_knee_angle = current_knee_angle
                else:
                    self.baseline_knee_angle = 0.9 * self.baseline_knee_angle + 0.1 * current_knee_angle
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Lateral Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arms hang by sides to begin",
                "elbow_bend": round(min_elbow_bend, 1), "abduction_angle": round(min_abduction, 1)
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
                
                if probs[0] > 0.85 and self.state in ["ASCENDING", "TOP"]:
                    self.ml_bad_frames += 1
                    if self.ml_bad_frames >= 8:
                        self.ml_posture_maintained = False
                else:
                    self.ml_bad_frames = max(0, self.ml_bad_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["RESTING", "ASCENDING", "TOP", "DESCENDING"]:
            # A. Elbow Bend
            if min_elbow_bend < 115:
                self.straight_frames += 1
                if self.straight_frames >= 5:
                    self.arms_straight_enough = False
            else:
                self.straight_frames = max(0, self.straight_frames - 1)

            # B. Momentum Swing
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            if abs(mid_shoulder_x - mid_hip_x) > 0.13:
                self.torso_frames += 1
                if self.torso_frames >= 5:
                    self.torso_stable = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)
                
            # C. Shoulder Shrug
            if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * 0.65):
                self.shrug_frames += 1
                if self.shrug_frames >= 5:
                    self.shoulders_down = False
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # D. Wrist Hiking
            if (l_wrist.y < (l_elbow.y - 0.08) or r_wrist.y < (r_elbow.y - 0.08)):
                self.wrist_frames += 1
                if self.wrist_frames >= 4:
                    self.elbows_lead_wrists = False
            else:
                self.wrist_frames = max(0, self.wrist_frames - 1)

            # E. Bilateral Asymmetry
            if abs(l_abduction - r_abduction) > 26.0:
                self.balance_frames += 1
                if self.balance_frames >= 5:
                    self.arms_balanced = False
            else:
                self.balance_frames = max(0, self.balance_frames - 1)

            # F. Over-Raise
            # F. Over-Raise (> 100° abduction or hands clearly over shoulders)
            wrist_over_shoulder = (l_wrist.y < l_shoulder.y - 0.08) or (r_wrist.y < r_shoulder.y - 0.08)
            if max_abduction > 100.0 or wrist_over_shoulder:
                self.over_raise_frames += 1
                if self.over_raise_frames >= 3:
                    self.not_over_raised = False
            else:
                self.over_raise_frames = max(0, self.over_raise_frames - 1)
                
            # G. Strict Front-View Knee Bounce
            hip_drop_normalized = (current_hip_y - self.baseline_hip_y) / shoulder_width if self.baseline_hip_y else 0.0
            knee_deflection = 0.0
            if self.baseline_knee_angle is not None and current_knee_angle is not None:
                knee_deflection = abs(current_knee_angle - self.baseline_knee_angle)

            is_bouncing = (hip_drop_normalized > 0.06) or (knee_deflection > 6.0)
            
            if is_bouncing and self.state in ["RESTING", "ASCENDING"]:
                self.knee_frames += 1
                if self.knee_frames >= 2:
                    self.knee_bounce_warning = True
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

            self.max_abduction = max(self.max_abduction, min_abduction)

        # ---------------------------------------------------------
        # 4. STATE MACHINE
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Raise arms out to shoulder height!"

        # Start ascending when arms pass 30°
        elif self.state in ["RESTING", "DESCENDING"] and min_abduction > 30:
            self.state = "ASCENDING"
            self.max_abduction = min_abduction
            self.reversal_frames = 0
            self.current_feedback = "Raise dumbbells to shoulder height..."

        # Full ROM: 60° to 95° enters TOP
        elif self.state == "ASCENDING" and min_abduction >= 60:
            self.target_height_reached = True
            self.half_raise_reached = True
            self.state = "TOP"
            self.current_feedback = "Good height! Lower under control."

        # Half Rep Reversal: Reached between 30° and 60°, then reversed downward by > 8°
        elif self.state == "ASCENDING" and (30 <= self.max_abduction < 60) and (self.max_abduction - min_abduction) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_raise_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering arms early..."

        # Lowering out of TOP once arms drop below 55°
        elif self.state == "TOP" and max_abduction < 55:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down"

        # Eccentric Free-Fall Monitor
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_abduction is not None:
                drop_speed = self.prev_abduction - min_abduction
                if drop_speed > 28.0:
                    self.controlled_descent = False
            self.prev_abduction = min_abduction
        else:
            self.prev_abduction = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return down (max abduction <= 35°)
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "TOP", "ASCENDING"] and (self.target_height_reached or self.half_raise_reached) and max_abduction <= 35:
            is_bad_rep = False

            # Form Cheats First (Invalidates the rep)
            if not self.not_over_raised:
                self.current_feedback = "No rep: Raised too high! Stop at shoulder height, avoid going over 90-100°."
                is_bad_rep = True
            elif not self.arms_straight_enough:
                self.current_feedback = "No rep: Arms bent too much! Maintain a soft, natural bend, don't curl it."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: You swung your body! Keep torso steady."
                is_bad_rep = True
            elif not self.shoulders_down:
                self.current_feedback = "No rep: Keep traps relaxed! Don't shrug shoulders up."
                is_bad_rep = True
            elif not self.elbows_lead_wrists:
                self.current_feedback = "No rep: Lead with elbows, don't hike wrists above elbows."
                is_bad_rep = True
            elif not self.arms_balanced:
                self.current_feedback = "No rep: Raise both arms evenly! One side lagged."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower weights with control."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
                
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            elif self.target_height_reached:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Strict raise only, avoid leg bounce."
                else:
                    self.current_feedback = "Clean lateral raise! Good rep."
            elif self.half_raise_reached:
                self.half_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Half Rep! Reminder: Lift to shoulder height and avoid leg bounce."
                else:
                    self.current_feedback = "Counted as Half Rep: Lift arms all the way to shoulder height!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Raise was too shallow to count."
                
            # Reset flags for next rep
            self.state = "RESTING"
            self.target_height_reached = False
            self.half_raise_reached = False
            self.max_abduction = 0.0
            self.reversal_frames = 0
            
            self.knee_frames = 0
            self.over_raise_frames = 0
            self.straight_frames = 0
            self.torso_frames = 0
            self.shrug_frames = 0
            self.wrist_frames = 0
            self.balance_frames = 0
            self.ml_bad_frames = 0

            self.arms_straight_enough = True
            self.torso_stable = True
            self.shoulders_down = True
            self.elbows_lead_wrists = True
            self.arms_balanced = True
            self.controlled_descent = True
            self.not_over_raised = True
            self.knee_bounce_warning = False
            self.prev_abduction = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((min_abduction / 75) * 100)))
        if not self.arms_straight_enough: form_score -= 20
        if not self.torso_stable: form_score -= 20
        if not self.shoulders_down: form_score -= 20
        if not self.elbows_lead_wrists: form_score -= 15
        if not self.arms_balanced: form_score -= 15
        if not self.controlled_descent: form_score -= 15
        if not self.not_over_raised: form_score -= 20
        if self.knee_bounce_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 25
        if self.half_raise_reached and not self.target_height_reached: form_score -= 10
        form_score = max(0, form_score)

        return {
            "exercise": "Lateral Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps, 
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_bend": round(min_elbow_bend, 1),
            "abduction_angle": round(min_abduction, 1)
        }