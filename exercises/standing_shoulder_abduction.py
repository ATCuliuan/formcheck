import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingShoulderAbductionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Partial ROM counter
        
        self.target_height_reached = False # Full ROM: both arms >= 80 deg
        self.half_raise_reached = False    # Half ROM: both arms >= 55 deg
        self.max_abduction = 0.0           # Peak abduction reached by weakest arm
        self.reversal_frames = 0           # Debounce counter
        
        # Posture Tripwires (FRONT VIEW)
        self.arms_straight = True          # Catches T-Rex bent elbows (< 145 deg)
        self.is_symmetrical = True        # Catches see-saw asymmetric lifting
        self.torso_upright = True          # Catches lateral momentum sway
        self.elbows_lead_wrists = True     # Catches wrist-hiking over elbows
        self.controlled_descent = True     # Catches free-falling the arms down
        self.ml_posture_maintained = True  # ST-GCN AI flag
        
        self.prev_abduction = None
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, arms at your sides to begin"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_abduction.tflite")

        
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

        vis_joints = [l_shoulder.visibility, r_shoulder.visibility, 
                      l_elbow.visibility, r_elbow.visibility, 
                      l_wrist.visibility, r_wrist.visibility,
                      l_hip.visibility, r_hip.visibility]
        
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_joints) < 0.35: 
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Abduction",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full upper body not visible",
                "abduction_angle": 0.0, "depth_reached": False
            }

        # Bilateral abduction angles (measured from hip to shoulder to elbow)
        l_abduction_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        min_abduction = min(l_abduction_angle, r_abduction_angle)
        max_abduction = max(l_abduction_angle, r_abduction_angle)

        # Bilateral elbow angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow = min(l_elbow_angle, r_elbow_angle)

        # Torso height to normalize lateral shift
        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0))

        # 1. Starting Guard: Both arms relaxed at sides (< 25 deg)
        if max_abduction < 25:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 12:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Abduction",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Rest arms down at sides to begin",
                "abduction_angle": round(min_abduction, 1), "depth_reached": False
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
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. FRONT-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["LIFTING", "PEAK"]:
            # A. T-Rex Cheat: Bending elbows to shorten the moment arm
            if min_elbow < 145:
                self.arms_straight = False
                
            # B. Asymmetric See-Saw Lift
            if abs(l_abduction_angle - r_abduction_angle) > 20:
                self.is_symmetrical = False

            # C. Torso Sway: Lateral displacement of spine center normalized by torso height
            if torso_height > 0:
                mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
                mid_hip_x = (l_hip.x + r_hip.x) / 2.0
                sway_ratio = abs(mid_shoulder_x - mid_hip_x) / torso_height
                if sway_ratio > 0.16:
                    self.torso_upright = False

            # D. Wrist Hiking: Wrists elevated significantly higher than elbows
            if (l_wrist.y < (l_elbow.y - 0.04) or r_wrist.y < (r_elbow.y - 0.04)):
                self.elbows_lead_wrists = False

            self.max_abduction = max(self.max_abduction, min_abduction)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Raise arms out laterally!"

        elif self.state in ["STANDING", "LOWERING"] and min_abduction >= 30:
            self.state = "LIFTING"
            self.max_abduction = min_abduction
            self.reversal_frames = 0
            self.current_feedback = "Raise arms straight out to sides..."

        # Full ROM: Both arms reach parallel (>= 80 deg)
        elif self.state == "LIFTING" and min_abduction >= 80:
            self.target_height_reached = True
            self.half_raise_reached = True
            self.state = "PEAK"
            self.current_feedback = "Parallel reached! Lower under control."

        # Half Rep Reversal: Reached at least 55 deg, then reversed downward by > 8 deg
        elif self.state == "LIFTING" and min_abduction >= 55 and (self.max_abduction - min_abduction) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_raise_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Half rep: Lowering arms early..."

        elif self.state == "PEAK" and max_abduction < 72:
            self.state = "LOWERING"
            self.current_feedback = "Control the descent"

        # E. Eccentric Drop Detector: Monitor lowering speed
        if self.state == "LOWERING":
            if self.prev_abduction is not None:
                drop_speed = self.prev_abduction - min_abduction
                if drop_speed > 18.0:  # Dropped > 18° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_abduction = min_abduction
        else:
            self.prev_abduction = None

        # Rep Evaluation: Arms return down to resting position (< 30 deg)
        if self.state == "LOWERING" and max_abduction < 30:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.arms_straight:
                self.current_feedback = "No rep: Keep your arms straight! Do not bend elbows."
                is_bad_rep = True
            elif not self.is_symmetrical:
                self.current_feedback = "No rep: Both arms must lift equally together."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Keep torso still! Do not sway sideways."
                is_bad_rep = True
            elif not self.elbows_lead_wrists:
                self.current_feedback = "No rep: Lead with elbows, do not hike wrists upward."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower arms slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
            
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.target_height_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect shoulder abduction!"
            elif self.half_raise_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Lift arms all the way to shoulder height!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Raise was too shallow to count."
            
            # Reset flags for next rep
            self.state = "STANDING"
            self.target_height_reached = False
            self.half_raise_reached = False
            self.max_abduction = 0.0
            self.reversal_frames = 0
            self.arms_straight = True
            self.is_symmetrical = True
            self.torso_upright = True
            self.elbows_lead_wrists = True
            self.controlled_descent = True
            self.prev_abduction = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Form Score
        form_score = max(0, min(100, int((min_abduction / 80) * 100)))
        if not self.arms_straight: form_score -= 25
        if not self.is_symmetrical: form_score -= 20
        if not self.torso_upright: form_score -= 25
        if not self.elbows_lead_wrists: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_raise_reached and not self.target_height_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Abduction",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "abduction_angle": round(min_abduction, 1),
            "elbow_angle": round(min_elbow, 1),
            "depth_reached": self.target_height_reached
        }