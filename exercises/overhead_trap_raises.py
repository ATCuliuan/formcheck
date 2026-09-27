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
        self.half_reps = 0                 # Counts partial ROM raises
        
        self.overhead_reached = False      # Full ROM (abduction >= 150 deg)
        self.half_raise_reached = False    # Half ROM (abduction >= 95 deg)
        self.max_abduction_angle = 0.0     # Tracks peak elevation achieved
        self.reversal_frames = 0           # Debounce counter for jitter
        
        # Instructor Tripwires (FRONT VIEW)
        self.arms_straight = True          # Catches the "T-Rex" bent-elbow cheat
        self.torso_upright = True          # Catches lateral side-bending / swaying
        self.arms_balanced = True          # Catches one arm lagging behind the other
        self.controlled_descent = True     # Catches dropping arms with zero eccentric control
        self.ml_posture_maintained = True 
        
        self.prev_abduction_angle = None
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Stand tall facing the camera, arms down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_overhead_trap_raises.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Overhead Trap Raise model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Overhead Trap Raise weights: {e}")
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

        # Bilateral visibility check for front view
        vis_all = [l_shoulder.visibility, r_shoulder.visibility, l_elbow.visibility, 
                   r_elbow.visibility, l_wrist.visibility, r_wrist.visibility,
                   l_hip.visibility, r_hip.visibility]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.35:
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face the camera: show full upper body and arms",
                "elbow_angle": 0.0, "abduction_angle": 0.0
            }

        # Bilateral kinematics
        l_shoulder_abduction = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_shoulder_abduction = calculate_angle(r_hip, r_shoulder, r_elbow)
        min_abduction = min(l_shoulder_abduction, r_shoulder_abduction)
        max_abduction = max(l_shoulder_abduction, r_shoulder_abduction)

        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow = min(l_elbow_angle, r_elbow_angle)

        # 1. Starting Guard: Arms hanging at sides, elbows extended
        if min_abduction < 45 and min_elbow > 150:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arms hang completely straight to start",
                "elbow_angle": round(min_elbow, 1), "abduction_angle": round(min_abduction, 1)
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
            except Exception: pass

        # ---------------------------------------------------------
        # 3. FRONT-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        
        # A. T-Rex Arms: Bending elbows to shorten the lever
        if min_elbow < 145 and self.state in ["ASCENDING", "TOP"]:
            self.arms_straight = False

        # B. Lateral Spine Tilt: Shifting torso left/right off-center
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        if abs(mid_shoulder_x - mid_hip_x) > 0.08 and self.state in ["ASCENDING", "TOP"]:
            self.torso_upright = False

        # C. Bilateral Asymmetry: Arms must rise evenly together
        wrist_height_diff = abs(l_wrist.y - r_wrist.y)
        abduction_diff = abs(l_shoulder_abduction - r_shoulder_abduction)
        if (wrist_height_diff > 0.12 or abduction_diff > 20.0) and self.state in ["ASCENDING", "TOP"]:
            self.arms_balanced = False

        # Track peak abduction angle reached
        if self.state in ["ASCENDING", "TOP"]:
            self.max_abduction_angle = max(self.max_abduction_angle, min_abduction)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Raise arms overhead in a Y-formation!"

        elif self.state in ["RESTING", "DESCENDING"] and min_abduction > 55:
            self.state = "ASCENDING"
            self.max_abduction_angle = min_abduction
            self.reversal_frames = 0
            self.current_feedback = "Keep elbows locked, raise straight up..."

        # Full Overhead Lockout (Arms elevated >= 150 deg)
        elif self.state == "ASCENDING" and min_abduction >= 150:
            self.overhead_reached = True
            self.half_raise_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze traps overhead! Control the negative."

        # Half Rep Reversal: Reached past parallel (>= 95 deg), but began descending early (> 10 deg drop)
        elif self.state == "ASCENDING" and min_abduction >= 95 and (self.max_abduction_angle - min_abduction) > 10:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_raise_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Half rep: Lowering arms..."

        elif self.state == "TOP" and max_abduction < 135:
            self.state = "DESCENDING"
            self.current_feedback = "Control the descent"

        # D. Eccentric Drop Detector: Monitor rate of lowering
        if self.state == "DESCENDING":
            if self.prev_abduction_angle is not None:
                drop_speed = self.prev_abduction_angle - min_abduction
                if drop_speed > 22.0:  # Dropped > 22° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_abduction_angle = min_abduction
        else:
            self.prev_abduction_angle = None

        # Rep Evaluation: Arms returned down to resting position
        if self.state == "DESCENDING" and min_abduction <= 50:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.arms_straight:
                self.current_feedback = "No rep: Keep elbows locked! Do not bend your arms."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Do not lean sideways to hoist the weights."
                is_bad_rep = True
            elif not self.arms_balanced:
                self.current_feedback = "No rep: Raise both arms evenly! One side lagged behind."
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
            elif self.overhead_reached:
                self.good_reps += 1
                self.current_feedback = "Excellent trap raise! Full overhead range."
            elif self.half_raise_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Raise all the way overhead into a Y-shape!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Raise was too shallow to count."
                
            # Reset flags for next rep
            self.state = "RESTING"
            self.overhead_reached = False
            self.half_raise_reached = False
            self.max_abduction_angle = 0.0
            self.reversal_frames = 0
            self.arms_straight = True
            self.torso_upright = True
            self.arms_balanced = True
            self.controlled_descent = True
            self.prev_abduction_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((min_abduction / 150) * 100)))
        if not self.arms_straight: form_score -= 30
        if not self.torso_upright: form_score -= 25
        if not self.arms_balanced: form_score -= 20
        if not self.controlled_descent: form_score -= 20
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
            "elbow_angle": round(min_elbow, 1),
            "abduction_angle": round(min_abduction, 1)
        }