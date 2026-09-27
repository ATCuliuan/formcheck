import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellReverseLungeTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Partial depth rep counter
        
        self.depth_reached = False         # Full depth: Front knee <= 95 deg
        self.half_depth_reached = False    # Half depth: Front knee <= 115 deg
        self.min_knee_angle = 180.0        # Tracks peak depth reached
        self.reversal_frames = 0           # Debounce counter

        # Biomechanical Tripwires (SIDE / SAGITTAL VIEW)
        self.torso_upright = True          # Catches forward pitching over thigh
        self.knee_over_toes_safe = True    # Catches front knee shearing past toes
        self.controlled_descent = True     # Catches free-falling into the bottom
        self.no_ground_slam = True         # Catches slamming the back knee onto floor
        self.ml_posture_maintained = True
        
        self.prev_knee_angle = None
        self.is_ready = False
        self.standing_frames = 0

        self.current_feedback = "Stand sideways to camera, weights at sides"
        self.mp_pose = mp.solutions.pose

        # Buffer for 30-frame ST-GCN sequence
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_reverse_lunge.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Reverse Lunge model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Reverse Lunge weights: {e}")
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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, r_shoulder.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: full profile not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Calculate joint angles for both legs
        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Distinguish front vs rear leg (the front leg undergoes deeper flexion)
        if left_knee_angle < right_knee_angle:
            front_knee_angle = left_knee_angle
            front_hip, front_knee, front_ankle = l_hip, l_knee, l_ankle
            rear_knee = r_knee
            active_shoulder = l_shoulder
        else:
            front_knee_angle = right_knee_angle
            front_hip, front_knee, front_ankle = r_hip, r_knee, r_ankle
            rear_knee = l_knee
            active_shoulder = r_shoulder

        hip_angle = calculate_angle(active_shoulder, front_hip, front_knee)

        # Lower leg length to normalize knee-past-toes check against camera distance
        shin_length = np.sqrt((front_knee.x - front_ankle.x)**2 + (front_knee.y - front_ankle.y)**2)

        # 1. Setup Guard (Must stand upright with both knees extended)
        if left_knee_angle > 160 and right_knee_angle > 160:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Reverse Lunge",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand upright with feet together",
                "knee_angle": round(front_knee_angle, 1),
                "hip_angle": round(hip_angle, 1),
                "depth_reached": False
            }

        # 2. ST-GCN Temporal Buffer Update
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
        
        # A. Torso Pitch: Avoid leaning chest heavily over front thigh
        if hip_angle < 75 and self.state in ["LUNGING", "BOTTOM"]:
            self.torso_upright = False

        # B. Shin Shear: Front knee pushing excessively forward relative to shin length
        if shin_length > 0:
            shear_ratio = abs(front_knee.x - front_ankle.x) / shin_length
            if shear_ratio > 0.60 and self.state in ["LUNGING", "BOTTOM"]:
                self.knee_over_toes_safe = False

        # C. Back-Knee Slam: Rear knee crashing all the way down onto the floor plane
        if self.state in ["LUNGING", "BOTTOM"]:
            if (rear_knee.y - front_ankle.y) > 0.05:
                self.no_ground_slam = False

        # Track the lowest knee angle reached
        if self.state in ["LUNGING", "BOTTOM"]:
            self.min_knee_angle = min(self.min_knee_angle, front_knee_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        
        # Transition out of WAITING
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Step back into your lunge"

        elif self.state == "STANDING" and front_knee_angle <= 140:
            self.state = "LUNGING"
            self.min_knee_angle = front_knee_angle
            self.reversal_frames = 0
            self.current_feedback = "Lower hips straight down..."

        # Full Depth achieved (thigh parallel to floor, knee <= 95 deg)
        elif self.state == "LUNGING" and front_knee_angle <= 95:
            self.depth_reached = True
            self.half_depth_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Great depth! Drive through the front heel"

        # Half Rep Reversal: Reached at least 115°, but began rising early (> 8° above min point)
        elif self.state == "LUNGING" and front_knee_angle <= 115 and (front_knee_angle - self.min_knee_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Pushing back up early..."

        # Descents that completely miss half depth and return
        elif self.state == "LUNGING" and front_knee_angle > 145:
            self.state = "ASCENDING"

        elif self.state == "BOTTOM" and front_knee_angle > 110:
            self.state = "ASCENDING"
            self.current_feedback = "Drive back to standing"

        # D. Eccentric Drop Detector: Measure downward speed during LUNGING
        if self.state == "LUNGING":
            if self.prev_knee_angle is not None:
                drop_speed = self.prev_knee_angle - front_knee_angle
                if drop_speed > 18.0:  # Dropped > 18° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_knee_angle = front_knee_angle
        else:
            self.prev_knee_angle = None

        # Rep Evaluation: Both legs straightened back to standing
        if self.state == "ASCENDING" and left_knee_angle > 155 and right_knee_angle > 155:
            is_bad_rep = False

            # Form Evaluation (Priority Checks)
            if not self.torso_upright:
                self.current_feedback = "No rep: Keep chest proud! Avoid leaning forward."
                is_bad_rep = True
            elif not self.knee_over_toes_safe:
                self.current_feedback = "No rep: Step further back to prevent front knee shear."
                is_bad_rep = True
            elif not self.no_ground_slam:
                self.current_feedback = "No rep: Don't slam your rear knee into the floor!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the drop! Lower yourself slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected collapsed upper body."
                is_bad_rep = True

            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.depth_reached:
                self.good_reps += 1
                self.current_feedback = "Clean reverse lunge! Full range of motion."
            elif self.half_depth_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Lower hips deeper until front thigh is parallel!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Lunge depth was too shallow."

            # Reset tripwires and counters for next rep
            self.state = "STANDING"
            self.depth_reached = False
            self.half_depth_reached = False
            self.min_knee_angle = 180.0
            self.reversal_frames = 0
            self.torso_upright = True
            self.knee_over_toes_safe = True
            self.controlled_descent = True
            self.no_ground_slam = True
            self.prev_knee_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(front_knee_angle - 90) * 0.4)))
        if not self.torso_upright: form_score -= 30
        if not self.knee_over_toes_safe: form_score -= 25
        if not self.no_ground_slam: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_depth_reached and not self.depth_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Reverse Lunge",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(front_knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "depth_reached": self.depth_reached
        }