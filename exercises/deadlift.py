import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DeadliftTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Counts partial depth hinges
        
        self.depth_reached = False         # Full depth: Hands below knee level
        self.half_depth_reached = False    # Half depth: Hands pass mid-thigh
        self.lowest_wrist_y = 0.0          # Tracks lowest point reached (higher Y = lower to floor)
        self.reversal_frames = 0           # Debounce counter
        
        # Instructor Tripwires (FRONT VIEW)
        self.knees_tracking_well = True    # Catches Knee Cave (Knee Valgus)
        self.bar_level = True              # Catches asymmetrical / tilted bar
        self.hips_level = True             # Catches lateral hip shift / twisting
        self.ml_posture_maintained = True  # ST-GCN posture detection
        
        self.standing_frames = 0
        self.is_ready = False
        self.current_feedback = "Stand tall facing the camera"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_deadlift.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Deadlift model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Deadlift weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25: 
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        # Front view requires BOTH sides of the body
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # In front view, both sides must be clearly visible
        vis_all = [l_shoulder.visibility, r_shoulder.visibility, l_hip.visibility, 
                   r_hip.visibility, l_knee.visibility, r_knee.visibility, 
                   l_ankle.visibility, r_ankle.visibility]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.40:
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Face camera directly: full body must be visible",
                "bar_drop_pct": 0.0
            }

        # Calculate midpoints for bilateral tracking
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2.0
        avg_hip_y = (l_hip.y + r_hip.y) / 2.0
        avg_knee_y = (l_knee.y + r_knee.y) / 2.0
        
        # Normalized stance widths
        ankle_dist = abs(l_ankle.x - r_ankle.x)
        knee_dist = abs(l_knee.x - r_knee.x)

        # 1. Starting Guard: Hands resting near hip level (standing tall)
        is_standing = avg_wrist_y < (avg_hip_y + 0.10)

        if is_standing:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Stand tall holding weights at thighs to begin",
                "bar_drop_pct": 0.0
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
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. FRONT-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        
        # A. Knee Valgus (Knee Cave): Knees collapse closer together than feet
        if ankle_dist > 0.10 and self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            if (knee_dist / ankle_dist) < 0.70:
                self.knees_tracking_well = False

        # B. Tilted Bar: One hand dragging or pulling higher than the other
        wrist_tilt = abs(l_wrist.y - r_wrist.y)
        if wrist_tilt > 0.08 and self.state in ["DESCENDING", "ASCENDING"]:
            self.bar_level = False

        # C. Lateral Shift: Hips shifting off-center
        hip_center_x = (l_hip.x + r_hip.x) / 2.0
        ankle_center_x = (l_ankle.x + r_ankle.x) / 2.0
        if abs(hip_center_x - ankle_center_x) > 0.07 and self.state in ["DESCENDING", "ASCENDING"]:
            self.hips_level = False

        # Track the deepest descent (in screen coordinates, larger Y is lower)
        if self.state in ["DESCENDING", "BOTTOM"]:
            self.lowest_wrist_y = max(self.lowest_wrist_y, avg_wrist_y)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Driven by vertical barbell displacement)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Hinge down past your knees"

        # Descending: Hands travel downwards below upper thigh
        elif self.state in ["STANDING", "ASCENDING"] and avg_wrist_y > (avg_hip_y + 0.12):
            self.state = "DESCENDING"
            self.lowest_wrist_y = avg_wrist_y
            self.reversal_frames = 0
            self.current_feedback = "Push hips back, lower weights..."

        # Full Depth: Hands reach or cross below knee level
        elif self.state == "DESCENDING" and avg_wrist_y >= (avg_knee_y - 0.02):
            self.depth_reached = True
            self.half_depth_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Drive through heels to stand!"

        # Half Rep Branch: Hands lowered past mid-thigh, but reversed upward early (> 0.04 above lowest point)
        elif self.state == "DESCENDING" and avg_wrist_y > (avg_hip_y + 0.15) and (self.lowest_wrist_y - avg_wrist_y) > 0.04:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Ascending early..."

        elif self.state == "BOTTOM" and avg_wrist_y < (self.lowest_wrist_y - 0.04):
            self.state = "ASCENDING"
            self.current_feedback = "Drive hips forward!"

        # Rep Completion: Standing upright at the top
        elif self.state == "ASCENDING" and is_standing:
            is_bad_rep = False
            
            # Priority Biomechanical Checks
            if not self.knees_tracking_well:
                self.current_feedback = "No rep: Knees caved inward! Push knees out over toes."
                is_bad_rep = True
            elif not self.bar_level:
                self.current_feedback = "No rep: Keep the bar level! One side lagged behind."
                is_bad_rep = True
            elif not self.hips_level:
                self.current_feedback = "No rep: You shifted your hips sideways. Pull symmetrically."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor mechanics / rounding."
                is_bad_rep = True
            
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.depth_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect deadlift!"
            elif self.half_depth_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Lower weights past your knees!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Movement was too shallow."

            # Reset flags
            self.state = "STANDING"
            self.depth_reached = False
            self.half_depth_reached = False
            self.lowest_wrist_y = 0.0
            self.reversal_frames = 0
            self.knees_tracking_well = True
            self.bar_level = True
            self.hips_level = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = 100
        if not self.knees_tracking_well: form_score -= 30
        if not self.bar_level: form_score -= 20
        if not self.hips_level: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_depth_reached and not self.depth_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Deadlift",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "bar_height_rel": round(avg_wrist_y - avg_knee_y, 2)
        }