import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class HurdleStepTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera harness compatibility
        self.hurdle_cleared = False
        self.active_leg = None
        
        # Posture & Foot Mechanics Tripwires
        self.support_leg_bent = False     # Did they bend the standing leg to cheat?
        self.torso_leaned = False         # Did the spine pitch forward/backward?
        self.hands_behind_head = True     # Strict: Arms must stay up behind head, never at hips
        self.body_stayed_still = True     # Did the pelvis/torso shift or sway horizontally?
        self.foot_slid_cheated = False    # Did they slide the foot across the floor without lifting high?
        self.foot_pointed_correctly = False # Foot must point between 10 and 35 degrees during step
        self.step_displaced_forward = False # Must actually step across rather than lift in place
        self.ml_posture_maintained = True # ST-GCN AI flag
        
        # Debounce counters
        self.sway_frames = 0
        self.slide_frames = 0
        self.pointed_frames = 0
        self.arm_drop_frames = 0
        
        # Setup Guards
        self.standing_frames = 0
        self.is_ready = False
        self.baseline_hip_x = None

        self.current_feedback = "Side profile: Stand tall, hands behind head"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_hurdle_step.tflite")

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

    def _calculate_foot_pitch(self, ankle, toe):
        """
        Calculates angle of foot (ankle -> toe) relative to horizontal ground.
        0 deg is completely flat/parallel to ground.
        Positive degrees represent toe pointing downward/forward.
        """
        dy = toe.y - ankle.y
        dx = abs(toe.x - ankle.x)
        if dx == 0:
            return 90.0
        angle_rad = np.arctan2(dy, dx)
        return float(np.degrees(angle_rad))

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        r_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        # Determine dominant camera-facing side
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.8:
            self.frame_buffer.clear()
            return {
                "exercise": "Hurdle Step",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Step back: Show full body side profile",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Select primary visible joints
        if l_vis > r_vis:
            v_shoulder, v_elbow, v_hip, v_knee, v_ankle = l_shoulder, l_elbow, l_hip, l_knee, l_ankle
        else:
            v_shoulder, v_elbow, v_hip, v_knee, v_ankle = r_shoulder, r_elbow, r_hip, r_knee, r_ankle

        # Joint angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        primary_knee_angle = calculate_angle(v_hip, v_knee, v_ankle)

        # Torso length normalization metric
        torso_len = abs(v_shoulder.y - v_hip.y)
        if torso_len == 0:
            torso_len = 0.45

        # Torso angle relative to vertical (sagittal plane)
        dx_torso = abs(v_shoulder.x - v_hip.x)
        dy_torso = abs(v_shoulder.y - v_hip.y)
        torso_lean_angle = np.degrees(np.arctan2(dx_torso, dy_torso)) if dy_torso > 0 else 0.0

        ankle_y_diff = l_ankle.y - r_ankle.y
        ankle_x_diff = abs(l_ankle.x - r_ankle.x)

        # Strict elbow check: Elbow must stay at or above the shoulder level
        # In MediaPipe, smaller Y is higher up on screen
        is_elbow_raised = v_elbow.y <= (v_shoulder.y + 0.05)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Strict Standing Form)
        # ---------------------------------------------------------
        is_standing_tall = (
            primary_knee_angle > 162.0 and
            abs(ankle_y_diff) < 0.05 and
            ankle_x_diff < 0.06 and
            torso_lean_angle < 10.0 and
            is_elbow_raised
        )

        if is_standing_tall:
            self.standing_frames += 1
            if self.baseline_hip_x is None:
                self.baseline_hip_x = v_hip.x
            else:
                self.baseline_hip_x = 0.9 * self.baseline_hip_x + 0.1 * v_hip.x
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Hurdle Step",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "feedback": "Stand tall, hands behind head (keep elbows at/above shoulders)",
                "knee_angle": round(primary_knee_angle, 1),
                "hip_angle": 0.0,
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. STATE MACHINE & ACTIVE LEG ASSIGNMENT
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Lift knee high, keep hands behind head!"

        elif self.state == "STANDING":
            if ankle_y_diff > 0.08:
                self.active_leg = "RIGHT"
                self.state = "STEPPING"
                self.current_feedback = "Lift knee high, point foot 10-35 deg!"
            elif ankle_y_diff < -0.08:
                self.active_leg = "LEFT"
                self.state = "STEPPING"
                self.current_feedback = "Lift knee high, point foot 10-35 deg!"
            elif abs(l_ankle.x - r_ankle.x) > 0.15:
                self.active_leg = "SLIDING"
                self.state = "STEPPING"
                self.foot_slid_cheated = True
                self.current_feedback = "Do not slide your foot! Lift the knee high."

        # Assign moving and support limbs
        if self.active_leg == "RIGHT":
            moving_ankle, moving_knee = r_ankle, r_knee
            moving_toe = r_foot_index
            support_knee_angle = l_knee_angle
            support_knee, support_ankle = l_knee, l_ankle
        elif self.active_leg == "LEFT":
            moving_ankle, moving_knee = l_ankle, l_knee
            moving_toe = l_foot_index
            support_knee_angle = r_knee_angle
            support_knee, support_ankle = r_knee, r_ankle
        else:
            moving_ankle, moving_knee = v_ankle, v_knee
            moving_toe = l_foot_index if l_vis > r_vis else r_foot_index
            support_knee_angle = primary_knee_angle
            support_knee, support_ankle = v_knee, v_ankle

        # Calculate foot pitch angle (10 to 35 deg requirement)
        foot_pitch = self._calculate_foot_pitch(moving_ankle, moving_toe)

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state == "STEPPING":
            # A. Standing support leg must remain locked
            if support_knee_angle < 150:
                self.support_leg_bent = True

            # B. Excessive torso lean in sagittal view
            if torso_lean_angle > 18.0:
                self.torso_leaned = True

            # C. STRICT HAND/ELBOW POSTURE TRIPWIRE:
            # If elbow dips below the shoulder level (e.g. resting at ribs or hips)
            if not is_elbow_raised:
                self.arm_drop_frames += 1
                if self.arm_drop_frames >= 3:
                    self.hands_behind_head = False
            else:
                self.arm_drop_frames = max(0, self.arm_drop_frames - 1)

            # D. Body Stillness: Pelvis/torso shifting horizontally
            if self.baseline_hip_x is not None:
                hip_horizontal_sway = abs(v_hip.x - self.baseline_hip_x) / torso_len
                if hip_horizontal_sway > 0.20:
                    self.sway_frames += 1
                    if self.sway_frames >= 4:
                        self.body_stayed_still = False
                else:
                    self.sway_frames = max(0, self.sway_frames - 1)

            # E. Foot Slide Tripwire: Foot travels horizontally without clearing hurdle height
            horizontal_step_dist = abs(moving_ankle.x - support_ankle.x)
            is_skimming_floor = moving_ankle.y > (support_knee.y + 0.16)
            if horizontal_step_dist > 0.16 and is_skimming_floor and not self.hurdle_cleared:
                self.slide_frames += 1
                if self.slide_frames >= 3:
                    self.foot_slid_cheated = True
            else:
                self.slide_frames = max(0, self.slide_frames - 1)

            # F. Hurdle clearance: Moving ankle must rise near support knee height
            clearance_threshold = support_knee.y + 0.12
            if moving_ankle.y < clearance_threshold:
                self.hurdle_cleared = True

            # G. Pointing Foot Check
            if 10.0 <= foot_pitch <= 35.0:
                self.pointed_frames += 1
                if self.pointed_frames >= 3:
                    self.foot_pointed_correctly = True
            
            # Forward stride requirement
            if horizontal_step_dist > 0.18:
                self.step_displaced_forward = True

            # Immediate feedback priority
            if not self.hands_behind_head:
                self.current_feedback = "Form error: Keep hands behind head, elbows up!"
            elif self.hurdle_cleared and self.foot_pointed_correctly and self.step_displaced_forward:
                self.current_feedback = "Good clearance & foot angle! Return foot to floor."
            elif not self.foot_pointed_correctly:
                self.current_feedback = f"Point foot downward (10°-35°)! Current: {round(foot_pitch, 1)}°"

        # ---------------------------------------------------------
        # 4. ST-GCN BUFFER & INFERENCE
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
                if predicted_class == 0 and self.state == "STEPPING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 5. REP COMPLETION LOGIC (ENFORCING HAND POSTURE ON EXIT)
        # ---------------------------------------------------------
        feet_together = abs(ankle_y_diff) < 0.06 and abs(l_ankle.x - r_ankle.x) < 0.08
        return_knee_locked = support_knee_angle > 155

        if self.state == "STEPPING" and feet_together and return_knee_locked:
            is_bad_rep = False
            
            # Check posture faults first
            if not self.hands_behind_head or not is_elbow_raised:
                self.current_feedback = "No rep: Hands dropped! Keep hands behind head with elbows up."
                is_bad_rep = True
            elif self.foot_slid_cheated:
                self.current_feedback = "No rep: Do not slide foot! Lift knee high."
                is_bad_rep = True
            elif not self.step_displaced_forward:
                self.current_feedback = "No rep: Step forward across the hurdle! Do not lift in place."
                is_bad_rep = True
            elif not self.foot_pointed_correctly:
                self.current_feedback = "No rep: Foot must point between 10° and 35° during the step."
                is_bad_rep = True
            elif not self.hurdle_cleared:
                self.current_feedback = "No rep: Perform a full rep! Lift foot over hurdle height."
                is_bad_rep = True
            elif not self.body_stayed_still:
                self.current_feedback = "No rep: Keep body still! Only move the leg."
                is_bad_rep = True
            elif self.support_leg_bent:
                self.current_feedback = "No rep: Support leg bent! Keep standing leg locked."
                is_bad_rep = True
            elif self.torso_leaned:
                self.current_feedback = "No rep: Torso leaned excessively! Keep spine upright."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected form breakdown"
                is_bad_rep = True
            
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Great hurdle step! Clean rep."
            
            # Reset state for next rep
            self.state = "STANDING"
            self.active_leg = None
            self.hurdle_cleared = False
            self.support_leg_bent = False
            self.torso_leaned = False
            self.hands_behind_head = True
            self.body_stayed_still = True
            self.foot_slid_cheated = False
            self.foot_pointed_correctly = False
            self.step_displaced_forward = False
            self.ml_posture_maintained = True
            self.sway_frames = 0
            self.slide_frames = 0
            self.pointed_frames = 0
            self.arm_drop_frames = 0
            self.frame_buffer.clear()

        return {
            "exercise": "Hurdle Step",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "feedback": self.current_feedback,
            "knee_angle": round(support_knee_angle, 1),
            "depth_reached": self.hurdle_cleared
        }