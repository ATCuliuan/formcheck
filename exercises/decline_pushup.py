import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DeclinePushupTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 
        
        self.depth_reached = False         
        self.min_elbow_angle = 180.0       
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_decline_elevation = True   
        self.core_maintained = True        
        self.no_pike = True                
        self.neck_neutral = True           
        self.controlled_descent = True     
        self.hands_aligned = True          # <--- NEW: Hand alignment tripwire
        self.ml_posture_maintained = True  

        # Debounce counters
        self.sag_frames = 0
        self.pike_frames = 0
        self.flat_frames = 0
        self.neck_frames = 0
        self.drop_frames = 0
        self.hand_misalign_frames = 0      # <--- NEW: Hand debounce counter
        self.plank_frames = 0
        
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Elevate feet on bench/box, hold decline plank"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_decline_pushup.tflite")

        
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
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        # Visibility Guard
        if max(l_vis, r_vis) < 3.0: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Decline Pushup",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Show full decline side profile",
                "elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Core Kinematic Angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Arms locked & feet elevated above shoulders)
        # ---------------------------------------------------------
        is_feet_elevated = active_ankle.y < (active_shoulder.y - 0.03)
        is_plank = (elbow_angle > 150) and (body_angle > 150) and is_feet_elevated

        if is_plank:
            self.plank_frames += 1
        else:
            self.plank_frames = 0

        if self.plank_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            feedback_msg = "Elevate feet higher on box/bench" if not is_feet_elevated else "Hold decline plank to begin"
            return {
                "exercise": "Decline Pushup",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
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
                if predicted_class == 0 and self.state != "PLANKING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            # A. Feet on floor cheat
            if active_ankle.y > active_shoulder.y:
                self.flat_frames += 1
                if self.flat_frames >= 4:
                    self.is_decline_elevation = False
            else:
                self.flat_frames = max(0, self.flat_frames - 1)

            # B. Core Sagging (Banana Back)
            if active_hip.y > active_shoulder.y and body_angle < 150:
                self.sag_frames += 1
                if self.sag_frames >= 4:
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # C. Hip Piking
            if active_hip.y < active_shoulder.y and body_angle < 140:
                self.pike_frames += 1
                if self.pike_frames >= 4:
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # D. Neck Craning
            if neck_angle < 130:
                self.neck_frames += 1
                if self.neck_frames >= 4:
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            # E. Hand Asymmetry / Stagger Check (NEW)
            # Checks if both hands are visible, then compares vertical level (Y) and anterior/posterior stagger (X)
            if l_wrist.visibility > 0.45 and r_wrist.visibility > 0.45:
                vertical_wrist_diff = abs(l_wrist.y - r_wrist.y)
                horizontal_wrist_stagger = abs(l_wrist.x - r_wrist.x)

                # Staggered hands along floor (X) or uneven height (Y)
                if vertical_wrist_diff > 0.08 or horizontal_wrist_stagger > 0.12:
                    self.hand_misalign_frames += 1
                    if self.hand_misalign_frames >= 4:
                        self.hands_aligned = False
                else:
                    self.hand_misalign_frames = max(0, self.hand_misalign_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FULL ROM)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "PLANKING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest toward floor!"

        elif elbow_angle <= 135 and self.state in ["PLANKING", "ASCENDING"]:
            self.state = "DESCENDING"
            self.min_elbow_angle = elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest to 90 degrees..."

        elif self.state == "DESCENDING" and elbow_angle <= 90:
            self.depth_reached = True
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Depth hit! Drive floor away."

        elif self.state == "DESCENDING" and (elbow_angle - self.min_elbow_angle) > 8:
            self.state = "ASCENDING"
            self.current_feedback = "Perform a full rep! Lower chest all the way to 90°."

        elif self.state == "BOTTOM" and elbow_angle > 100:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock out at the top!"

        # Eccentric Free-Fall Detector
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > 22.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to top lockout (elbow > 150 deg)
        # ---------------------------------------------------------
        if self.state in ["ASCENDING", "BOTTOM", "DESCENDING"] and elbow_angle > 150:
            is_bad_rep = False
            
            # Form Cheats Evaluated First
            if not self.is_decline_elevation:
                self.current_feedback = "No rep: Elevate your feet! Feet dropped to floor level."
                is_bad_rep = True
            elif not self.hands_aligned:
                self.current_feedback = "No rep: Uneven hand placement! Align both hands evenly."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "No rep: Core collapsed! Do not sag hips."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "No rep: Butt too high! Maintain straight decline plank."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Neutral neck! Lower your chest, not your chin."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not drop to floor."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected elbow flare/breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "No rep: Perform a full rep! Lower chest to 90 degrees."
                is_bad_rep = True

            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean decline push-up! Good rep."

            self.post_rep_timer = 40

            # Reset flags for next rep
            self.state = "PLANKING"
            self.depth_reached = False
            self.min_elbow_angle = 180.0
            self.is_decline_elevation = True
            self.hands_aligned = True          # Reset hand flag
            self.core_maintained = True
            self.no_pike = True
            self.neck_neutral = True
            self.controlled_descent = True
            self.prev_elbow_angle = None
            self.ml_posture_maintained = True
            self.sag_frames = 0
            self.pike_frames = 0
            self.flat_frames = 0
            self.neck_frames = 0
            self.drop_frames = 0
            self.hand_misalign_frames = 0      # Reset hand counter
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4)))
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 25
        if not self.is_decline_elevation: form_score -= 35
        if not self.hands_aligned: form_score -= 20
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Decline Pushup",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "body_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }