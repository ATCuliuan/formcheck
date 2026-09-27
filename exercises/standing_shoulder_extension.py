import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingShoulderExtensionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Partial ROM counter
        
        self.target_extension_reached = False # Full ROM: extension >= 40 deg
        self.half_extension_reached = False   # Half ROM: extension >= 28 deg
        self.max_extension = 0.0              # Peak extension angle achieved
        self.reversal_frames = 0              # Debounce counter
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.arms_straight = True          # Catches tricep kickback cheats (elbow flex < 155 deg)
        self.torso_upright = True          # Catches forward hip hinge / bowing cheat
        self.neck_neutral = True           # Catches poking chin/head forward
        self.controlled_return = True      # Catches dropping arms with zero eccentric control
        self.ml_posture_maintained = True  # ST-GCN AI flag
        
        self.prev_extension = None
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand sideways to the camera, arms relaxed at sides"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_extension.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Shoulder Extension model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Shoulder Extension weights: {e}")
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

        # In side view, evaluate tracking based on the dominant profile facing the camera
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 2.5: 
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Extension",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full side profile not visible",
                "extension_angle": 0.0, "depth_reached": False
            }

        # Select dominant side facing camera
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee

        # Sagittal plane kinematics
        extension_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        torso_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 1. Starting Guard: Arm resting naturally at side (< 18 deg) and body upright
        if extension_angle < 18 and torso_angle > 165:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 12:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Extension",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand tall sideways, arms resting at your sides",
                "extension_angle": round(extension_angle, 1), "depth_reached": False
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
        # 3. SIDE-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["EXTENDING", "PEAK"]:
            # A. Tricep Kickback Cheat: Bending elbow to push forearm back instead of upper arm
            if elbow_angle < 155:
                self.arms_straight = False
                
            # B. Forward Bow Cheat: Hinging at hips to artificially tilt arms up
            if torso_angle < 165:
                self.torso_upright = False

            # C. Forward Head Thrust: Poking chin forward to balance
            if neck_angle < 135:
                self.neck_neutral = False

            self.max_extension = max(self.max_extension, extension_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Push arms straight back behind you!"

        elif self.state in ["STANDING", "RETURNING"] and extension_angle >= 22:
            self.state = "EXTENDING"
            self.max_extension = extension_angle
            self.reversal_frames = 0
            self.current_feedback = "Push arm straight back..."

        # Full ROM: Extension reaches standard physiological limit (>= 40 deg)
        elif self.state == "EXTENDING" and extension_angle >= 40:
            self.target_extension_reached = True
            self.half_extension_reached = True
            self.state = "PEAK"
            self.current_feedback = "Peak contraction reached! Lower slowly."

        # Half Rep Reversal: Reached at least 28 deg, then dropped back (> 6 deg reversal)
        elif self.state == "EXTENDING" and extension_angle >= 28 and (self.max_extension - extension_angle) > 6:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_extension_reached = True
                self.state = "RETURNING"
                self.current_feedback = "Half rep: Returning arm early..."

        elif self.state == "PEAK" and extension_angle < 34:
            self.state = "RETURNING"
            self.current_feedback = "Control the return"

        # D. Eccentric Return Speed Detector
        if self.state == "RETURNING":
            if self.prev_extension is not None:
                drop_speed = self.prev_extension - extension_angle
                if drop_speed > 16.0:  # Dropped > 16° in a single frame (~33ms)
                    self.controlled_return = False
            self.prev_extension = extension_angle
        else:
            self.prev_extension = None

        # Rep Evaluation: Arm returns to starting rest position (< 20 deg)
        if self.state == "RETURNING" and extension_angle < 20:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.arms_straight:
                self.current_feedback = "No rep: Lock your elbows! Do not do a tricep kickback."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Stand tall! Do not lean forward at the hips."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Neutral spine! Do not poke your chin forward."
                is_bad_rep = True
            elif not self.controlled_return:
                self.current_feedback = "No rep: Control the return! Do not swing arms forward."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
            
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.target_extension_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect shoulder extension!"
            elif self.half_extension_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Push arm further behind the torso next time!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Extension was too shallow to count."
            
            # Reset flags for next rep
            self.state = "STANDING"
            self.target_extension_reached = False
            self.half_extension_reached = False
            self.max_extension = 0.0
            self.reversal_frames = 0
            self.arms_straight = True
            self.torso_upright = True
            self.neck_neutral = True
            self.controlled_return = True
            self.prev_extension = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Form Score
        form_score = max(0, min(100, int((extension_angle / 40) * 100)))
        if not self.arms_straight: form_score -= 30
        if not self.torso_upright: form_score -= 35
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_return: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_extension_reached and not self.target_extension_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Extension",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "extension_angle": round(extension_angle, 1),
            "elbow_angle": round(elbow_angle, 1),
            "depth_reached": self.target_extension_reached
        }