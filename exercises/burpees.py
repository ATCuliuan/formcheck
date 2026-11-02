import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BurpeesTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility
        
        # Posture & Movement Tripwires (Side View)
        self.is_side_view = True           # Catches facing the camera instead of profile
        self.plank_reached = False         # Full kickback into straight horizontal plank
        self.core_maintained = True        # Catches hip sag / banana back
        self.no_pike = True                # Catches butt shooting up into a pike
        self.jump_finished = False         # Catches skipping the jump/reach at top
        self.ml_posture_maintained = True
        
        # Debounce counters
        self.profile_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.jump_frames = 0
        self.standing_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False
        
        self.baseline_ankle_y = None       # Tracks floor level for jump height

        self.current_feedback = "Stand sideways to the camera to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception:
                self.interpreter = None
        
        model_path = os.path.join("models", "tflite", "st_gcn_burpees.tflite")
        

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
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
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Side view dominant side detection
        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, r_shoulder.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5: 
            self.frame_buffer.clear()
            return {
                "exercise": "Burpees", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: full profile not visible",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        # Side-view Biomechanical Angles
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Profile validation: Horizontal shoulder spread vs vertical torso
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        torso_height = abs(active_shoulder.y - active_hip.y) or 0.40
        is_turned_sideways = (shoulder_span_x / torso_height) < 0.42

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if knee_angle > 160 and hip_angle > 155 and is_turned_sideways:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera (side profile)" if not is_turned_sideways else "Stand tall sideways to start"
            return {
                "exercise": "Burpees", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1),
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
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC GEOMETRY & STATE MACHINE
        # ---------------------------------------------------------
        vertical_drop = abs(active_shoulder.y - active_ankle.y)
        horizontal_span = abs(active_shoulder.x - active_ankle.x)
        is_horizontal = vertical_drop < 0.28 and horizontal_span > 0.35

        # Initialize to STANDING once ready
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.baseline_ankle_y = active_ankle.y
            if self.post_rep_timer == 0:
                self.current_feedback = "Drop down into a plank!"

        # Drop: Bending down to plant hands
        elif self.state == "STANDING" and (knee_angle < 120 or hip_angle < 100):
            self.state = "DROPPING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Kick legs straight back into full plank!"

        # Full Plank Reached (Horizontal on floor with flat body line >= 150°)
        elif self.state == "DROPPING" and is_horizontal:
            self.state = "PLANK"
            if body_angle >= 150:
                self.plank_reached = True
                if self.post_rep_timer == 0:
                    self.current_feedback = "Good plank! Jump feet forward and explode up."
            else:
                self.current_feedback = "Kick all the way back into a flat plank!"

        # ---------------------------------------------------------
        # 4. PLANK TRIPWIRES
        # ---------------------------------------------------------
        if self.state == "PLANK":
            # Orientation check
            if not is_turned_sideways:
                self.profile_frames += 1
                if self.profile_frames >= 4:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Hip Sag (Hips sink below shoulder and ankle line)
            if active_hip.y > active_shoulder.y and active_hip.y > active_ankle.y and body_angle < 145:
                self.sag_frames += 1
                if self.sag_frames >= 3:
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # Hip Pike (Butt peaks higher than shoulder and ankle line)
            if active_hip.y < active_shoulder.y and active_hip.y < active_ankle.y and body_angle < 135:
                self.pike_frames += 1
                if self.pike_frames >= 3:
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

        # Transition: Rising from floor (Torso shifts toward vertical)
        elif self.state == "PLANK" and vertical_drop > 0.45:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Jump up and reach overhead!"

        # Jump & Overhead Extension Verification
        if self.state in ["ASCENDING", "STANDING"]:
            jumped = (self.baseline_ankle_y is not None) and ((self.baseline_ankle_y - active_ankle.y) > 0.035)
            arms_up = active_wrist.y < active_shoulder.y
            if jumped or arms_up:
                self.jump_frames += 1
                if self.jump_frames >= 2:
                    self.jump_finished = True

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Re-extending to standing
        # ---------------------------------------------------------
        if self.state == "ASCENDING" and knee_angle > 160 and hip_angle > 155:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways to the camera!"
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "No rep: Hips sagged! Squeeze core in the plank."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "No rep: Butt too high! Flatten into a real plank."
                is_bad_rep = True
            elif not self.plank_reached:
                self.current_feedback = "No rep: Incomplete kickback! Kick legs out to a full plank."
                is_bad_rep = True
            elif not self.jump_finished:
                self.current_feedback = "No rep: Finish with an explosive jump or overhead reach!"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected collapsed posture."
                is_bad_rep = True
            
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Great explosive burpee!"

            self.post_rep_timer = 40

            # Reset flags for next rep
            self.state = "STANDING"
            self.plank_reached = False
            self.core_maintained = True
            self.no_pike = True
            self.jump_finished = False
            self.is_side_view = True
            self.ml_posture_maintained = True
            self.baseline_ankle_y = active_ankle.y
            self.profile_frames = 0
            self.sag_frames = 0
            self.pike_frames = 0
            self.jump_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = 100
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 20
        if not self.jump_finished: form_score -= 25
        if not self.plank_reached: form_score -= 30
        if not self.ml_posture_maintained: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Burpees", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1),
            "depth_reached": self.plank_reached
        }