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
        
        self.target_height_reached = False # Full ROM: abduction >= 78 deg
        self.half_raise_reached = False    # Half ROM: abduction >= 55 deg
        self.max_abduction = 0.0           # Tracks peak raise height achieved
        self.reversal_frames = 0           # Debounce counter for jitter
        
        # Instructor Tripwires (FRONT VIEW)
        self.arms_straight_enough = True   # Catches excessive T-Rex elbow bending (< 130 deg)
        self.torso_stable = True           # Catches lateral momentum sway / torso rocking
        self.shoulders_down = True         # Catches trap shrugging
        self.elbows_lead_wrists = True     # Catches wrist-hiking over elbows
        self.arms_balanced = True          # Catches asymmetric arm height
        self.controlled_descent = True     # Catches dropping weights on eccentric
        self.ml_posture_maintained = True 
        
        self.standing_frames = 0
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
                print(f"✅ ST-GCN TFLite Lateral Raise model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Lateral Raise weights: {e}")
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
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        # Bilateral visibility check for frontal perspective
        vis_all = [l_shoulder.visibility, r_shoulder.visibility, l_elbow.visibility, 
                   r_elbow.visibility, l_wrist.visibility, r_wrist.visibility,
                   l_hip.visibility, r_hip.visibility]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.35:
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

        # Torso height normalization factor
        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0))
        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height if torso_height > 0 else 0.0

        # 1. Starting Guard: Arms down by sides (< 25 deg)
        if max_abduction < 25:
            self.standing_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            else:
                # Running average to lock baseline neck length
                self.baseline_neck_ratio = 0.9 * self.baseline_neck_ratio + 0.1 * neck_ratio
        else:
            self.standing_frames = 0

        if self.standing_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Lateral Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arms hang straight down by sides to begin",
                "elbow_bend": round(min_elbow_bend, 1), "abduction_angle": round(min_abduction, 1)
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
        # 3. FRONT-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        
        # A. T-Rex Wing: Excessive elbow bend (> 50 deg bend)
        if min_elbow_bend < 130 and self.state in ["ASCENDING", "TOP"]:
            self.arms_straight_enough = False

        # B. Momentum Swing: Lateral torso rocking off hip midpoint
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        if abs(mid_shoulder_x - mid_hip_x) > 0.08 and self.state in ["ASCENDING", "TOP"]:
            self.torso_stable = False
            
        # C. Flying Bird Shrug: Neck space collapses by > 25% relative to torso
        if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * 0.75):
            if self.state in ["ASCENDING", "TOP"]:
                self.shoulders_down = False

        # D. Wrist Hiking: Wrists lifted higher than elbows (internal rotation cheat)
        # Note: In screen coords, smaller Y is higher
        if (l_wrist.y < (l_elbow.y - 0.04) or r_wrist.y < (r_elbow.y - 0.04)):
            if self.state in ["ASCENDING", "TOP"]:
                self.elbows_lead_wrists = False

        # E. Bilateral Asymmetry: Arms must rise evenly
        if abs(l_abduction - r_abduction) > 18.0 and self.state in ["ASCENDING", "TOP"]:
            self.arms_balanced = False

        # Continuously monitor highest elevation reached by the weaker arm
        if self.state in ["ASCENDING", "TOP"]:
            self.max_abduction = max(self.max_abduction, min_abduction)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Raise arms outward to the sides!"

        elif self.state in ["RESTING", "DESCENDING"] and min_abduction > 30:
            self.state = "ASCENDING"
            self.max_abduction = min_abduction
            self.reversal_frames = 0
            self.current_feedback = "Raise dumbbells to shoulder height..."

        # Full ROM: Parallel achieved (elbow reaches near shoulder level, >= 78 deg)
        elif self.state == "ASCENDING" and min_abduction >= 78:
            self.target_height_reached = True
            self.half_raise_reached = True
            self.state = "TOP"
            self.current_feedback = "Parallel reached! Lower under control."

        # Half Rep Reversal: Reached at least 55 deg, then reversed downward by > 8 deg
        elif self.state == "ASCENDING" and min_abduction >= 55 and (self.max_abduction - min_abduction) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_raise_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering weights..."

        elif self.state == "TOP" and max_abduction < 70:
            self.state = "DESCENDING"
            self.current_feedback = "Control the weight down"
            
        elif self.state == "DESCENDING" and max_abduction <= 30:
                    is_bad_rep = False

        # F. Eccentric Drop Detector: Monitor rate of arm lowering
        if self.state == "DESCENDING":
            if self.prev_abduction is not None:
                drop_speed = self.prev_abduction - min_abduction
                if drop_speed > 18.0:  # Dropped > 18° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_abduction = min_abduction
        else:
            self.prev_abduction = None

        # Rep Evaluation: Returned arms to resting position (< 30 deg)
   
            
            # Form Cheats First
            if not self.arms_straight_enough:
                self.current_feedback = "No rep: Arms too bent! Keep a slight bend, not 90°."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: You swung your body! Keep torso completely still."
                is_bad_rep = True
            elif not self.shoulders_down:
                self.current_feedback = "No rep: Do not shrug your traps! Keep shoulders pressed down."
                is_bad_rep = True
            elif not self.elbows_lead_wrists:
                self.current_feedback = "No rep: Lead with your elbows, do not hike your hands above elbows."
                is_bad_rep = True
            elif not self.arms_balanced:
                self.current_feedback = "No rep: Raise both arms evenly! One side lagged."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower weights slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
                
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.target_height_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect lateral raise! Full height."
            elif self.half_raise_reached:
                self.half_reps += 1
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
            self.arms_straight_enough = True
            self.torso_stable = True
            self.shoulders_down = True
            self.elbows_lead_wrists = True
            self.arms_balanced = True
            self.controlled_descent = True
            self.prev_abduction = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((min_abduction / 80) * 100)))
        if not self.arms_straight_enough: form_score -= 25
        if not self.torso_stable: form_score -= 30
        if not self.shoulders_down: form_score -= 25
        if not self.elbows_lead_wrists: form_score -= 20
        if not self.arms_balanced: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_raise_reached and not self.target_height_reached: form_score -= 15
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