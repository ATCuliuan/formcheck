import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellFloorFlyeTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility
        
        self.depth_reached = False         # Full stretch: lateral wrist span expanded to floor
        self.max_wrist_span = 0.0          # Peak expansion tracker for no-half-rep logic
        
        # Posture Tripwires (TOP-DOWN / HEAD-FACING VIEW)
        self.is_pressing = False           # Bending elbows too much (< 130 deg, turning into press)
        self.is_locked_out = False         # Locking elbows dead straight (> 175 deg)
        self.is_asymmetrical = False       # One arm dropping or lifting unevenly
        self.ml_posture_maintained = True  # ST-GCN AI flag
        
        # Debounce counters
        self.press_frames = 0
        self.lock_frames = 0
        self.asym_frames = 0
        self.top_hold_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Lie on back with head toward camera, dumbbells over chest"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_floor_flye.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Dumbbell Floor Flye model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Dumbbell Floor Flye weights: {e}")
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

        # In top-of-head perspective, both arms and shoulders must be clearly visible
        vis_arms = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(vis_arms) < 0.35: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Flye",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Ensure shoulders, elbows, and hands are visible",
                "wrist_span": 0.0, "elbow_angle": 0.0, "depth_reached": False
            }

        # Anatomical scale reference
        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), 0.22)
        body_midline_x = (l_shoulder.x + r_shoulder.x) / 2.0

        # Bilateral Elbow Angles (maintain soft bend: 135° - 165°)
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)
        max_elbow_angle = max(l_elbow_angle, r_elbow_angle)

        # Lateral Wrist Opening Metrics (Head View: lateral spread on X axis)
        wrist_spread_x = abs(l_wrist.x - r_wrist.x)
        wrist_span_ratio = wrist_spread_x / shoulder_width

        # Symmetry: Lateral distance of each wrist from body midline
        l_dist_midline = abs(l_wrist.x - body_midline_x)
        r_dist_midline = abs(r_wrist.x - body_midline_x)
        asymmetry_diff = abs(l_dist_midline - r_dist_midline) / shoulder_width

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Arms extended up over chest, wrists together)
        # ---------------------------------------------------------
        # At top position, wrists are close together (span ratio <= 1.25) with soft elbow bend
        is_at_top = (wrist_span_ratio <= 1.25) and (130 <= avg_elbow_angle <= 175)

        if is_at_top:
            self.top_hold_frames += 1
        else:
            self.top_hold_frames = 0

        if self.top_hold_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Flye",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Hold weights together over chest to begin",
                "wrist_span": round(wrist_span_ratio, 2),
                "elbow_angle": round(avg_elbow_angle, 1),
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
                if predicted_class == 0 and self.state != "TOP":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["OPENING", "BOTTOM_STRETCH", "CLOSING"]:
            # A. The Press Cheat: Bending elbows excessively (< 130°) to press weights
            if min_elbow_angle < 130:
                self.press_frames += 1
                if self.press_frames >= 4:
                    self.is_pressing = True
            else:
                self.press_frames = max(0, self.press_frames - 1)

            # B. The Tendon Strain: Locking elbows completely straight (> 175°)
            if max_elbow_angle > 175:
                self.lock_frames += 1
                if self.lock_frames >= 4:
                    self.is_locked_out = True
            else:
                self.lock_frames = max(0, self.lock_frames - 1)

            # C. Asymmetrical Opening: One arm dropping lower or spreading wider than the other
            if asymmetry_diff > 0.40:
                self.asym_frames += 1
                if self.asym_frames >= 4:
                    self.is_asymmetrical = True
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            self.max_wrist_span = max(self.max_wrist_span, wrist_span_ratio)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (NO HALF REPS - FULL LATERAL FLYE ONLY)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Open arms wide in a smooth arc!"

        # Arms opening outwards: wrist span begins expanding (> 1.35)
        elif self.state in ["TOP", "CLOSING"] and wrist_span_ratio > 1.35:
            self.state = "OPENING"
            self.max_wrist_span = wrist_span_ratio
            if self.post_rep_timer == 0:
                self.current_feedback = "Maintain slight elbow bend, open arms wide..."

        # Full Depth Reach: Arms fully opened laterally to floor (span ratio >= 2.40)
        elif self.state == "OPENING" and wrist_span_ratio >= 2.40:
            self.depth_reached = True
            self.state = "BOTTOM_STRETCH"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full chest stretch reached! Squeeze arms back up."

        # Early upward reversal before reaching full floor stretch
        elif self.state == "OPENING" and (self.max_wrist_span - wrist_span_ratio) > 0.25:
            self.state = "CLOSING"
            self.current_feedback = "Perform a full rep! Open arms all the way to floor level."

        # Closing back upwards toward center
        elif self.state == "BOTTOM_STRETCH" and wrist_span_ratio < 2.25:
            self.state = "CLOSING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Squeeze chest, bring weights together over chest..."

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Wrists meet over chest (span ratio <= 1.25)
        # ---------------------------------------------------------
        if self.state in ["CLOSING", "BOTTOM_STRETCH", "OPENING"] and wrist_span_ratio <= 1.25:
            is_bad_rep = False
            
            # Form Cheats Evaluated First
            if self.is_pressing:
                self.current_feedback = "No rep: You bent your elbows too much! Fly the weights, don't press."
                is_bad_rep = True
            elif self.is_locked_out:
                self.current_feedback = "No rep: Maintain a soft elbow bend! Do not lock joints straight."
                is_bad_rep = True
            elif self.is_asymmetrical:
                self.current_feedback = "No rep: Move both arms evenly together."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged unstable shoulder/elbow tracking."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "No rep: Incomplete range! Open arms all the way to the floor."
                is_bad_rep = True

            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean dumbbell floor flye! Great chest squeeze."

            self.post_rep_timer = 40

            # Reset flags for next rep
            self.state = "TOP"
            self.depth_reached = False
            self.max_wrist_span = 0.0
            self.is_pressing = False
            self.is_locked_out = False
            self.is_asymmetrical = False
            self.ml_posture_maintained = True
            self.press_frames = 0
            self.lock_frames = 0
            self.asym_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((wrist_span_ratio / 2.40) * 100))) if self.state != "TOP" else 100
        if self.is_pressing: form_score -= 35
        if self.is_locked_out: form_score -= 30
        if self.is_asymmetrical: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.depth_reached and self.state == "CLOSING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Floor Flye",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "wrist_span": round(wrist_span_ratio, 2),
            "elbow_angle": round(avg_elbow_angle, 1),
            "depth_reached": self.depth_reached
        }