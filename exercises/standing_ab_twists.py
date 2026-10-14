import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp

class StandingAbTwistTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0
        self.twist_reached = False
        
        # Instructor Tripwires (FRONT VIEW)
        self.arms_elevated = True
        self.wrists_at_chest = True       # Enforces symmetric chest-level positioning
        self.hips_locked = True           # Catches pelvic rotation cheat
        self.ml_posture_maintained = True 
        
        self.baseline_shoulder_width = None
        self.baseline_hip_width = None
        
        # Debounce & Post-Rep Feedback Persistence
        self.ready_frames = 0
        self.arm_drop_frames = 0
        self.wrist_misplaced_frames = 0
        self.hip_twist_frames = 0
        self.post_rep_feedback_timer = 0  # Latches the result text on screen after rep finishes
        self.is_ready = False
        self.current_feedback = "Face camera with arms raised in front"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_ab_twists.tflite")

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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        # Count down post-rep persistent message
        if self.post_rep_feedback_timer > 0:
            self.post_rep_feedback_timer -= 1

        if max(l_vis, r_vis) < 2.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Ab Twist", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face the camera, step back to show full torso",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # 2D Projected widths in Front View
        current_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        current_hip_width = abs(l_hip.x - r_hip.x)
        
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_len = abs(mid_shoulder_y - mid_hip_y)
        if torso_len == 0:
            torso_len = 0.40

        # General elevation: Elbows or wrists held above lower-torso
        arms_are_up = (
            (l_elbow.y < mid_hip_y - 0.05 or l_wrist.y < mid_hip_y) and
            (r_elbow.y < mid_hip_y - 0.05 or r_wrist.y < mid_hip_y)
        )

        # SYMMETRIC CHEST LEVEL & DIAGONAL WRIST CHECK (Fixes left-turn bias):
        # Instead of strict isolated per-wrist thresholds that break when turning left,
        # we check the combined center of the hands and ensure neither drops below the navel.
        hands_mid_y = (l_wrist.y + r_wrist.y) / 2.0
        hands_height_ok = (mid_shoulder_y - 0.15) <= hands_mid_y <= (mid_hip_y - 0.08)
        neither_dropped = (l_wrist.y < mid_hip_y) and (r_wrist.y < mid_hip_y)
        
        wrist_properly_positioned = hands_height_ok and neither_dropped

        shoulders_level = abs(l_shoulder.y - r_shoulder.y) < 0.15
        hips_level = abs(l_hip.y - r_hip.y) < 0.15

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_facing_front = arms_are_up and shoulders_level and hips_level

        if is_facing_front:
            self.ready_frames += 1
            if self.baseline_shoulder_width is None:
                self.baseline_shoulder_width = current_shoulder_width
            else:
                self.baseline_shoulder_width = 0.9 * self.baseline_shoulder_width + 0.1 * current_shoulder_width

            if self.baseline_hip_width is None:
                self.baseline_hip_width = current_hip_width
            else:
                self.baseline_hip_width = 0.9 * self.baseline_hip_width + 0.1 * current_hip_width
        else:
            self.ready_frames = 0

        if self.ready_frames > 6:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Ab Twist", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face camera squarely, raise arms to chest height",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # ---------------------------------------------------------
        # 2. ST-GCN BUFFER & INFERENCE
        # ---------------------------------------------------------
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

                if probs[0] > 0.85 and self.state in ["TWISTING", "MAX_TWIST"]:
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["TWISTING", "MAX_TWIST"]:
            # A. Dropping arms completely
            if not arms_are_up:
                self.arm_drop_frames += 1
                if self.arm_drop_frames >= 5:
                    self.arms_elevated = False
            else:
                self.arm_drop_frames = max(0, self.arm_drop_frames - 1)

            # B. Wrists dropped below chest line during turn (Left or Right)
            if not wrist_properly_positioned:
                self.wrist_misplaced_frames += 1
                if self.wrist_misplaced_frames >= 5:
                    self.wrists_at_chest = False
            else:
                self.wrist_misplaced_frames = max(0, self.wrist_misplaced_frames - 1)

            # C. The Hula Hoop cheat (Pelvis turning away from camera)
            if self.baseline_hip_width and current_hip_width < (self.baseline_hip_width * 0.72):
                self.hip_twist_frames += 1
                if self.hip_twist_frames >= 4:
                    self.hips_locked = False
            else:
                self.hip_twist_frames = max(0, self.hip_twist_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (NO HALF REPS - FULL TWISTS ONLY)
        # ---------------------------------------------------------
        if self.state in ["WAITING", "CENTER"] and self.is_ready:
            if self.state == "WAITING":
                self.state = "CENTER"
            
            # Show live guidance only after post-rep message finishes
            if self.post_rep_feedback_timer == 0:
                self.current_feedback = "Twist torso left or right!"

            # Rotation initiates when 2D shoulder width narrows to < 82% of baseline
            if current_shoulder_width < (self.baseline_shoulder_width * 0.82):
                self.state = "TWISTING"
                self.post_rep_feedback_timer = 0
                self.current_feedback = "Twist using core..."

        elif self.state == "TWISTING":
            # Peak twist reached when shoulder line narrows below 58% of baseline
            if current_shoulder_width < (self.baseline_shoulder_width * 0.58):
                self.twist_reached = True
                self.state = "MAX_TWIST"
                self.current_feedback = "Good twist! Return to center."
            # Early return before reaching full rotation
            elif current_shoulder_width > (self.baseline_shoulder_width * 0.88):
                self.state = "RETURNING"

        elif self.state == "MAX_TWIST":
            if current_shoulder_width > (self.baseline_shoulder_width * 0.68):
                self.state = "RETURNING"
                self.current_feedback = "Return smoothly to center"

        elif self.state == "RETURNING" and current_shoulder_width >= (self.baseline_shoulder_width * 0.85):
            is_bad_rep = False
            
            # Evaluate after the set/rep finishes
            if not self.twist_reached:
                self.current_feedback = "No rep: Twist further to the side."
                is_bad_rep = True
            elif not self.wrists_at_chest:
                self.current_feedback = "No rep: Keep hands at chest level."
                is_bad_rep = True
            elif not self.arms_elevated:
                self.current_feedback = "No rep: Arms dropped below chest."
                is_bad_rep = True
            elif not self.hips_locked:
                self.current_feedback = "No rep: Keep hips forward, rotate only shoulders."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected form breakdown."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Good rep! Clean torso twist."
                
            # Lock the feedback for 40 frames (~1.3s) so you have time to read it
            self.post_rep_feedback_timer = 40

            # Reset cleanly for next repetition
            self.state = "CENTER"
            self.twist_reached = False
            self.arms_elevated = True
            self.wrists_at_chest = True
            self.hips_locked = True
            self.ml_posture_maintained = True
            self.arm_drop_frames = 0
            self.wrist_misplaced_frames = 0
            self.hip_twist_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        twist_ratio = current_shoulder_width / self.baseline_shoulder_width if self.baseline_shoulder_width else 1.0
        form_score = max(0, min(100, int(((1.0 - twist_ratio) / 0.42) * 100)))
        
        if not self.twist_reached and self.state == "RETURNING": form_score -= 30
        if not self.hips_locked: form_score -= 40
        if not self.arms_elevated or not self.wrists_at_chest: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Ab Twist",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps, 
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": 0.0,
            "hip_angle": 0.0
        }