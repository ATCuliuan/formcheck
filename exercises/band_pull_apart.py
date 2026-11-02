import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BandPullTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Preserved for camera test harness compatibility
        self.pull_reached = False
        self.max_wrist_dist = 0.0
        
        # Instructor Tripwires
        self.arms_straight = True
        self.shoulders_shrugged = False
        self.arm_height_valid = True
        self.ml_posture_maintained = True
        
        self.is_ready = False
        self.ready_frames = 0
        
        # Debounce counters
        self.elbow_bend_frames = 0
        self.shrug_frames = 0
        self.height_fault_frames = 0
        self.baseline_neck_ratio = None

        self.current_feedback = "Bring hands together in front of your chest"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_band_pull_apart.tflite")

        
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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        # Elbow straightness
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        wrist_distance = abs(l_wrist.x - r_wrist.x)
        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), 0.25)
        
        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        avg_hip_y = (l_hip.y + r_hip.y) / 2.0
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2.0

        # Torso-normalized neck metric
        torso_height = abs(avg_shoulder_y - avg_hip_y)
        if torso_height == 0:
            torso_height = 0.40
            
        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        hands_inside_body = wrist_distance < (shoulder_width * 1.35)
        hands_aligned = abs(l_wrist.y - r_wrist.y) < 0.12
        hands_at_chest_level = (avg_wrist_y < (avg_hip_y - 0.05)) and (avg_wrist_y > (avg_shoulder_y - 0.20))

        if hands_inside_body and hands_aligned and hands_at_chest_level:
            self.ready_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            elif self.ready_frames < 10:
                self.baseline_neck_ratio = 0.85 * self.baseline_neck_ratio + 0.15 * neck_ratio
        else:
            self.ready_frames = 0

        if self.ready_frames > 4:
            self.is_ready = True

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Band Pull Apart", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hold hands together in front of your chest",
                "knee_angle": round(min_elbow_angle, 1), "hip_angle": 0.0
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
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception: 
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES & ARM HEIGHT LIMITATION
        # ---------------------------------------------------------
        if self.state in ["PULLING", "TOP", "RETURNING"]:
            # A. Elbow Bend Check
            if min_elbow_angle < 95:
                self.elbow_bend_frames += 1
                if self.elbow_bend_frames >= 3:
                    self.arms_straight = False
            else:
                self.elbow_bend_frames = max(0, self.elbow_bend_frames - 1)

            # B. Responsive Shrug Check
            if self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * 0.80):
                self.shrug_frames += 1
                if self.shrug_frames >= 3:
                    self.shoulders_shrugged = True
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # C. Arm Height Limitation
            too_low = avg_wrist_y > (avg_shoulder_y + 0.1)
            too_high = avg_wrist_y < (avg_shoulder_y - 0.1)

            if too_low or too_high:
                self.height_fault_frames += 1
                if self.height_fault_frames >= 2:
                    self.arm_height_valid = False
                    if too_low:
                        self.current_feedback = "Hands too low! Raise arms higher"
                    else:
                        self.current_feedback = "Hands too high! Lower hands slightly"
            else:
                self.height_fault_frames = max(0, self.height_fault_frames - 1)

            self.max_wrist_dist = max(self.max_wrist_dist, wrist_distance)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (Full Range Enforcement)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Pull band apart outward!"

        # Pull initiated (wrist distance separates past 1.30x shoulder width)
        elif self.state in ["RESTING", "RETURNING"] and wrist_distance > (shoulder_width * 1.30):
            self.state = "PULLING"
            self.max_wrist_dist = wrist_distance
            if self.arm_height_valid:
                self.current_feedback = "Pull wide, keep arms extended..."

        # Full ROM: Reached full extension
        elif self.state == "PULLING" and wrist_distance >= (shoulder_width * 1.7):
            self.pull_reached = True
            self.state = "TOP"
            if self.arm_height_valid:
                self.current_feedback = "Squeeze upper back! Return under control."

        # Returning to starting position
        elif self.state == "TOP" and wrist_distance < (shoulder_width * 1.5):
            self.state = "RETURNING"
            if self.arm_height_valid:
                self.current_feedback = "Control the band back to center..."

        # Early return prior to reaching full pull width
        elif self.state == "PULLING" and (self.max_wrist_dist - wrist_distance) > (shoulder_width * 0.15):
            self.state = "RETURNING"
            if self.arm_height_valid:
                self.current_feedback = "Perform a full rep! Pull wide until band touches chest."
       
        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["RETURNING", "TOP", "PULLING"] and hands_inside_body and hands_aligned:
            is_bad_rep = False
            
            if not self.arm_height_valid:
                self.current_feedback = "No rep: Keep arms level with chest throughout the pull!"
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "No rep: You bent your elbows too much! Keep arms extended."
                is_bad_rep = True
            elif self.shoulders_shrugged:
                self.current_feedback = "No rep: You shrugged your shoulders! Keep traps relaxed."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture."
                is_bad_rep = True
            elif not self.pull_reached:
                self.current_feedback = "No rep: Perform a full rep! Pull wide until band touches chest."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Excellent rep! Perfect arm height."

            # Reset rep state
            self.state = "RESTING"
            self.pull_reached = False
            self.max_wrist_dist = 0.0

            self.arms_straight = True
            self.shoulders_shrugged = False
            self.arm_height_valid = True
            self.elbow_bend_frames = 0
            self.shrug_frames = 0
            self.height_fault_frames = 0
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Score Calculation
        form_score = 100
        if not self.arm_height_valid: form_score -= 25
        if not self.pull_reached: form_score -= 25
        if not self.arms_straight: form_score -= 30
        if self.shoulders_shrugged: form_score -= 30
        if not self.ml_posture_maintained: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Band Pull Apart",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(min_elbow_angle, 1),
            "hip_angle": 0.0
        }