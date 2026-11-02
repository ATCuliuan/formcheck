import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellFloorPressTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.depth_reached = False         # Full floor contact: elbows flex <= 95 deg
        self.min_elbow_angle = 180.0       # Tracks deepest flexion reached

        # Posture Tripwires (HEAD-FACING / TOP-DOWN VIEW)
        self.forearms_stacked = True       # Wrists drifting away from vertical line above elbows
        self.elbows_not_flared = True      # Catches T-flare out to 90 degrees
        self.is_symmetrical = True        # Catches uneven arm extension
        self.controlled_descent = True     # Catches dropping/slamming triceps to floor
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Debounce counters
        self.stack_frames = 0
        self.flare_frames = 0
        self.asym_frames = 0
        self.drop_frames = 0
        self.lockout_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Lie flat with head toward camera, lock dumbbells out over chest"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_floor_press.tflite")

        
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

        # Bilateral visibility check from top of head
        vis_joints = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(vis_joints) < 0.35:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Press",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Ensure both arms and shoulders are in view",
                "elbow_angle": 0.0, "depth_reached": False
            }

        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), 0.22)

        # Bilateral Kinematic Angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        # Upper arm flare angle relative to torso (Shoulder to Hip vs Shoulder to Elbow)
        l_arm_flare = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_arm_flare = calculate_angle(r_hip, r_shoulder, r_elbow)
        max_flare_angle = max(l_arm_flare, r_arm_flare)

        # Forearm Stacking: In crown view, wrist X offset relative to elbow X
        l_stack_drift = abs(l_wrist.x - l_elbow.x) / shoulder_width
        r_stack_drift = abs(r_wrist.x - r_elbow.x) / shoulder_width
        max_stack_drift = max(l_stack_drift, r_stack_drift)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Lockout above chest)
        # ---------------------------------------------------------
        is_locked_out = min_elbow_angle >= 155.0

        if is_locked_out:
            self.lockout_frames += 1
        else:
            self.lockout_frames = 0

        if self.lockout_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Floor Press",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lock arms out straight over chest to begin",
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
        if self.state in ["LOWERING", "BOTTOM", "PRESSING"]:
            # A. Forearm Stacking: Wrist collapsing inward or outward (> 0.40 shoulder width)
            if max_stack_drift > 0.40:
                self.stack_frames += 1
                if self.stack_frames >= 4:
                    self.forearms_stacked = False
            else:
                self.stack_frames = max(0, self.stack_frames - 1)

            # B. Excessive Elbow Flare: Flaring arms flat out to 90 degrees (shoulder impingement)
            if max_flare_angle > 85.0:
                self.flare_frames += 1
                if self.flare_frames >= 4:
                    self.elbows_not_flared = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # C. Asymmetrical Pressing
            if abs(l_elbow_angle - r_elbow_angle) > 22.0:
                self.asym_frames += 1
                if self.asym_frames >= 4:
                    self.is_symmetrical = False
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, avg_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (NO HALF REPS - FULL LOCKOUT & FLOOR TOUCH)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower dumbbells until triceps rest on the floor!"

        elif self.state in ["TOP", "PRESSING"] and avg_elbow_angle < 145.0:
            self.state = "LOWERING"
            self.min_elbow_angle = avg_elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent to the floor..."

        # Full ROM: Elbows flex to <= 95 deg (triceps on floor)
        elif self.state == "LOWERING" and avg_elbow_angle <= 95.0:
            self.depth_reached = True
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Triceps on floor — Drive dumbbells up!"

        # Early upward reversal before reaching floor depth
        elif self.state == "LOWERING" and (avg_elbow_angle - self.min_elbow_angle) > 8.0:
            self.state = "PRESSING"
            self.current_feedback = "Perform a full rep! Lower until triceps touch the floor."

        elif self.state == "BOTTOM" and avg_elbow_angle > 105.0:
            self.state = "PRESSING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive up to lockout!"

        # Eccentric drop detector (bouncing elbows off floor)
        if self.state in ["LOWERING", "PRESSING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - avg_elbow_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = avg_elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to top lockout (min_elbow_angle >= 155 deg)
        # ---------------------------------------------------------
        if self.state in ["PRESSING", "BOTTOM", "LOWERING"] and min_elbow_angle >= 155.0:
            is_bad_rep = False

            if not self.forearms_stacked:
                self.current_feedback = "No rep: Keep wrists stacked directly over your elbows!"
                is_bad_rep = True
            elif not self.elbows_not_flared:
                self.current_feedback = "No rep: Tuck elbows to 45-70 degrees! Do not flare flat."
                is_bad_rep = True
            elif not self.is_symmetrical:
                self.current_feedback = "No rep: Press both arms evenly together."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not slam elbows onto floor."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged unstable pressing mechanics."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "No rep: Incomplete depth! Lower triceps all the way to floor."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Strong dumbbell floor press! Clean rep."

            self.post_rep_timer = 40

            # Reset flags for next rep
            self.state = "TOP"
            self.depth_reached = False
            self.min_elbow_angle = 180.0
            self.forearms_stacked = True
            self.elbows_not_flared = True
            self.is_symmetrical = True
            self.controlled_descent = True
            self.ml_posture_maintained = True
            
            self.stack_frames = 0
            self.flare_frames = 0
            self.asym_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(avg_elbow_angle - 180) * 0.4))) if self.state != "TOP" else 100
        if not self.forearms_stacked: form_score -= 30
        if not self.elbows_not_flared: form_score -= 30
        if not self.is_symmetrical: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "PRESSING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Floor Press",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(avg_elbow_angle, 1),
            "depth_reached": self.depth_reached
        }