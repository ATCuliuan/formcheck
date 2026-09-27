import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SquatTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Partial depth rep counter
        
        self.depth_reached = False         # Full depth: parallel/crease below knee
        self.half_depth_reached = False    # Half depth: knee flexion >= 125 deg
        self.min_knee_angle = 180.0        # Tracks lowest depth achieved
        self.reversal_frames = 0           # Debounce counter
        
        # Instructor Tripwires (FRONT VIEW)
        self.knees_tracking_well = True    # Catches Knee Cave (Valgus collapse)
        self.hips_level = True             # Catches lateral hip tilt / pelvic drop
        self.torso_centered = True         # Catches shifting weight onto one leg
        self.heels_flat = True             # Catches lifting heels off floor
        self.controlled_descent = True     # Catches dive-bombing the drop
        self.ml_posture_maintained = True 
        
        self.prev_knee_angle = None
        self.baseline_ankle_y = None      
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand upright facing the camera"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_squat.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Squat model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Squat weights: {e}")
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Bilateral visibility check for front view
        vis_all = [l_shoulder.visibility, r_shoulder.visibility,
                   l_hip.visibility, r_hip.visibility,
                   l_knee.visibility, r_knee.visibility,
                   l_ankle.visibility, r_ankle.visibility]

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if min(vis_all) < 0.35: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: full body not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Bilateral Kinematics
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        knee_angle = min(l_knee_angle, r_knee_angle)

        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        hip_angle = min(l_hip_angle, r_hip_angle)

        avg_ankle_y = (l_ankle.y + r_ankle.y) / 2.0
        knee_dist = abs(l_knee.x - r_knee.x)
        ankle_dist = abs(l_ankle.x - r_ankle.x)

        # 1. Starting Guard: Full extension on both knees
        if l_knee_angle > 160 and r_knee_angle > 160:
            self.standing_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = avg_ankle_y
            else:
                self.baseline_ankle_y = 0.9 * self.baseline_ankle_y + 0.1 * avg_ankle_y
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Bodyweight Squat",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand upright with knees locked to begin",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1), "depth_reached": False
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
                
                predicted_class = np.argmax(output_data, axis=1)[0]
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. FRONT-VIEW KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        
        # A. Knee Valgus / Knee Cave: Knees collapse inward narrower than stance
        if ankle_dist > 0.12 and self.state in ["DESCENDING", "BOTTOM"]:
            if knee_dist < (ankle_dist * 0.65):
                self.knees_tracking_well = False

        # B. Pelvic Tilt / Lateral Asymmetry: Hips tilting unevenly
        hip_tilt = abs(l_hip.y - r_hip.y)
        if hip_tilt > 0.08 and self.state in ["DESCENDING", "BOTTOM"]:
            self.hips_level = False

        # C. Lateral Torso Sway: Shifting center of mass over one foot
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_ankle_x = (l_ankle.x + r_ankle.x) / 2.0
        if abs(mid_shoulder_x - mid_ankle_x) > 0.08 and self.state in ["DESCENDING", "BOTTOM"]:
            self.torso_centered = False

        # D. Heel Lift: Ankles rising significantly above calibrated floor baseline
        if self.baseline_ankle_y and (self.baseline_ankle_y - avg_ankle_y) > 0.04:
            if self.state in ["DESCENDING", "BOTTOM"]:
                self.heels_flat = False

        # Track the deepest knee flexion achieved
        if self.state in ["DESCENDING", "BOTTOM"]:
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        
        # FIX: Explicit transition out of WAITING
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Squat down by pushing hips back!"

        elif self.state in ["STANDING", "ASCENDING"] and knee_angle <= 140:
            self.state = "DESCENDING"
            self.min_knee_angle = knee_angle
            self.reversal_frames = 0
            self.current_feedback = "Lower under control..."

        # Full Depth: Hips reach parallel with knees (knee angle <= 100 deg)
        elif self.state == "DESCENDING" and knee_angle <= 100:
            self.depth_reached = True
            self.half_depth_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Great depth! Drive through mid-foot to stand."

        # Half Rep Reversal: Reached at least 125 deg, then pushed back up (> 8 deg reversal)
        elif self.state == "DESCENDING" and knee_angle <= 125 and (knee_angle - self.min_knee_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Pushing up early..."

        elif self.state == "BOTTOM" and knee_angle > 110:
            self.state = "ASCENDING"
            self.current_feedback = "Drive up to standing!"

        # E. Eccentric Drop Detector: Monitor drop velocity
        if self.state == "DESCENDING":
            if self.prev_knee_angle is not None:
                drop_speed = self.prev_knee_angle - knee_angle
                if drop_speed > 18.0:  # Dropped > 18° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_knee_angle = knee_angle
        else:
            self.prev_knee_angle = None

        # Rep Evaluation: Fully returned to top standing position
        if self.state == "ASCENDING" and l_knee_angle > 155 and r_knee_angle > 155:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.knees_tracking_well:
                self.current_feedback = "No rep: Knees caved inward! Push knees outward over toes."
                is_bad_rep = True
            elif not self.hips_level:
                self.current_feedback = "No rep: Asymmetric squat! Keep hips level."
                is_bad_rep = True
            elif not self.torso_centered:
                self.current_feedback = "No rep: Do not shift your body weight to one side."
                is_bad_rep = True
            elif not self.heels_flat:
                self.current_feedback = "No rep: Keep your heels glued to the floor!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower yourself slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected collapsed squat mechanics."
                is_bad_rep = True
            
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.depth_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect squat! Full depth."
            elif self.half_depth_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Squat lower until thighs are parallel!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Squat was too shallow."
            
            # Reset flags for next rep
            self.state = "STANDING"
            self.depth_reached = False
            self.half_depth_reached = False
            self.min_knee_angle = 180.0
            self.reversal_frames = 0
            self.knees_tracking_well = True
            self.hips_level = True
            self.torso_centered = True
            self.heels_flat = True
            self.controlled_descent = True
            self.prev_knee_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Form Scoring
        form_score = max(0, min(100, int(100 - abs(knee_angle - 90) * 0.3)))
        if not self.knees_tracking_well: form_score -= 35
        if not self.hips_level: form_score -= 20
        if not self.torso_centered: form_score -= 20
        if not self.heels_flat: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_depth_reached and not self.depth_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Bodyweight Squat",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }