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
        self.half_reps = 0                 # Partial lockout rep counter
        
        self.lockout_reached = False       # Full ROM (> 155 deg)
        self.half_press_reached = False    # Half ROM (>= 130 deg)
        self.max_reach_angle = 0.0         # Deepest lockout tracked
        self.reversal_frames = 0           # Debounce counter
        
        # Instructor Tripwires (FRONT VIEW)
        self.torso_upright = True          # Catches lateral body tilt / side-bending
        self.arms_even = True              # Catches asymmetric hand heights
        self.forearms_vertical = True      # Catches flared/collapsed elbows
        self.legs_strict = True            # Catches knee dip / push press bounce
        self.ml_posture_maintained = True
        
        self.is_ready = False
        self.racked_frames = 0

        self.current_feedback = "Bring weights to shoulder level"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_overhead_shoulder_press.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Overhead Press model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Overhead Press weights: {e}")
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
        # Front View requires full bilateral upper body
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

        # Check visibility on both sides
        vis_all = [l_shoulder.visibility, r_shoulder.visibility, l_elbow.visibility, 
                   r_elbow.visibility, l_wrist.visibility, r_wrist.visibility,
                   l_hip.visibility, r_hip.visibility]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.35:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face camera directly: show full torso and arms",
                "elbow_angle": 0.0, "hip_angle": 0.0
            }

        # Elbow extensions
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)
        max_elbow_angle = max(l_elbow_angle, r_elbow_angle)

        # Knee extensions (to detect leg push-press cheat)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        min_knee_angle = min(l_knee_angle, r_knee_angle)

        # 1. Racked Guard (Weights at shoulders, elbows bent)
        if max_elbow_angle < 100:
            self.racked_frames += 1
        else:
            self.racked_frames = 0

        if self.racked_frames > 12:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Overhead Press",
                "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Rack dumbbells at shoulders to begin",
                "elbow_angle": round(min_elbow_angle, 1), "hip_angle": 0.0
            }

        # 2. ST-GCN Temporal Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)

                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                predicted_class = np.argmax(output_data, axis=1)[0]
                if predicted_class == 0 and self.state != "RACKED":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. FRONT-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        
        # A. Lateral Body Tilt: Check if spine/shoulders tilt sideways off hip center
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        if abs(mid_shoulder_x - mid_hip_x) > 0.08 and self.state in ["PRESSING", "TOP"]:
            self.torso_upright = False

        # B. Asymmetric Press: Hand height difference
        wrist_height_diff = abs(l_wrist.y - r_wrist.y)
        if wrist_height_diff > 0.12 and self.state in ["PRESSING", "TOP"]:
            self.arms_even = False

        # C. Forearm Stacking: In a front view, wrists should remain relatively stacked over elbows
        l_forearm_tilt = abs(l_wrist.x - l_elbow.x)
        r_forearm_tilt = abs(r_wrist.x - r_elbow.x)
        if max(l_forearm_tilt, r_forearm_tilt) > 0.15 and self.state in ["PRESSING", "TOP"]:
            self.forearms_vertical = False

        # D. Leg Drive / Push Press: Knees bending below 165° during upward press
        if min_knee_angle < 165 and self.state == "PRESSING":
            self.legs_strict = False

        # Continuously monitor highest arm extension achieved in this press
        if self.state in ["PRESSING", "TOP"]:
            self.max_reach_angle = max(self.max_reach_angle, min_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RACKED"
            self.current_feedback = "Press dumbbells overhead!"

        elif self.state in ["RACKED", "LOWERING"] and min_elbow_angle > 110:
            self.state = "PRESSING"
            self.max_reach_angle = min_elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Drive weights straight up!"

        # Full Lockout achieved
        elif self.state == "PRESSING" and min_elbow_angle > 155:
            self.state = "TOP"
            self.lockout_reached = True
            self.half_press_reached = True
            self.current_feedback = "Full lockout! Control the descent."

        # Half Rep Reversal: Reached at least 130°, then dropped back down by > 10°
        elif self.state == "PRESSING" and min_elbow_angle >= 130 and (self.max_reach_angle - min_elbow_angle) > 10:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_press_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Half rep: Lowering weights..."

        elif self.state == "TOP" and max_elbow_angle < 140:
            self.state = "LOWERING"

        # Rep Evaluation: Reached bottom starting position
        elif self.state == "LOWERING" and max_elbow_angle <= 95:
            is_bad_rep = False

            # Form Evaluation (Priority Checks)
            if not self.torso_upright:
                self.current_feedback = "No rep: Do not lean sideways! Keep spine upright."
                is_bad_rep = True
            elif not self.arms_even:
                self.current_feedback = "No rep: Press evenly! One arm lagged behind."
                is_bad_rep = True
            elif not self.forearms_vertical:
                self.current_feedback = "No rep: Keep forearms vertical, do not flare out elbows."
                is_bad_rep = True
            elif not self.legs_strict:
                self.current_feedback = "No rep: No leg bounce! Strict shoulder press only."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor mechanics."
                is_bad_rep = True

            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.lockout_reached:
                self.good_reps += 1
                self.current_feedback = "Great press! Full lockout overhead."
            elif self.half_press_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Lock out your arms fully overhead!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Range of motion too shallow."

            # Reset flags
            self.state = "RACKED"
            self.lockout_reached = False
            self.half_press_reached = False
            self.max_reach_angle = 0.0
            self.reversal_frames = 0
            self.torso_upright = True
            self.arms_even = True
            self.forearms_vertical = True
            self.legs_strict = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((min_elbow_angle / 160) * 100)))
        if not self.torso_upright: form_score -= 25
        if not self.arms_even: form_score -= 20
        if not self.forearms_vertical: form_score -= 20
        if not self.legs_strict: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_press_reached and not self.lockout_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Overhead Press",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0
        }