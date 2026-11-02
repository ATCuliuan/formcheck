import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class GoodMorningTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera harness compatibility
        
        self.target_hinge_reached = False  # Full depth hinge: hip angle <= 105°
        self.min_hip_angle = 180.0         # Tracks peak hinge depth
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Profile camera alignment check
        self.not_squatting = True          # Catches bending knees too much (knee angle < 135°)
        self.neutral_spine = True          # Catches rounding thoracic/lumbar spine
        self.controlled_hinge = True       # Catches collapsing/dropping forward too fast
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Debounce counters
        self.profile_frames = 0
        self.squat_frames = 0
        self.spine_frames = 0
        self.drop_frames = 0
        self.standing_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_hip_angle = None
        self.current_feedback = "Stand sideways to camera, tall posture"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_good_morning.tflite")

        
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
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Determine dominant side facing camera
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Good Morning", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Step back: Show full side profile",
                "hip_angle": 0.0, "knee_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_ear = r_ear if r_vis > l_vis else l_ear
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal kinematics
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        spine_neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Profile check
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < 0.42

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_standing = (hip_angle > 162.0) and (knee_angle > 155.0) and is_side
        if is_standing:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Stand tall sideways to begin"
            return {
                "exercise": "Good Morning", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1), "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)

                self.interpreter.invoke()

                out = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(out, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["HINGING", "PEAK_HINGE", "EXTENDING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4: self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Squatting cheat: knees bend past 135° (turning hip hinge into a squat)
            if knee_angle < 135.0:
                self.squat_frames += 1
                if self.squat_frames >= 4: self.not_squatting = False
            else:
                self.squat_frames = max(0, self.squat_frames - 1)

            # Rounded spine/neck check (< 130°)
            if spine_neck_angle < 130.0:
                self.spine_frames += 1
                if self.spine_frames >= 4: self.neutral_spine = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (STRICT NO HALF-REPS)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Hinge hips backward, keep back flat!"

        # Begin hinging forward
        elif self.state in ["STANDING", "EXTENDING"] and hip_angle < 155.0:
            self.state = "HINGING"
            self.min_hip_angle = hip_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Push hips back, chest forward..."

        # Full ROM: Hip angle reaches <= 105° (near parallel torso)
        elif self.state == "HINGING" and hip_angle <= 105.0:
            self.target_hinge_reached = True
            self.state = "PEAK_HINGE"
            if self.post_rep_timer == 0:
                self.current_feedback = "Great depth! Drive hips forward to stand."

        # Incomplete Rep reversal: User rises before reaching parallel torso
        elif self.state == "HINGING" and (hip_angle - self.min_hip_angle) > 8.0:
            self.state = "EXTENDING"
            self.current_feedback = "Perform a full rep! Hinge down until torso is near parallel."

        elif self.state == "PEAK_HINGE" and hip_angle > 115.0:
            self.state = "EXTENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through glutes to stand tall"

        # Eccentric speed monitor
        if self.state in ["EXTENDING", "HINGING"]:
            if self.prev_hip_angle is not None:
                drop_speed = self.prev_hip_angle - hip_angle
                if drop_speed > 22.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2: self.controlled_hinge = False
            self.prev_hip_angle = hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to standing lockout (>= 162°)
        # ---------------------------------------------------------
        if self.state in ["EXTENDING", "PEAK_HINGE", "HINGING"] and hip_angle >= 162.0:
            is_bad_rep = False
            
            # Form Cheats Evaluated First
            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways to the camera!"
                is_bad_rep = True
            elif not self.not_squatting:
                self.current_feedback = "No rep: Do not squat! Keep knees soft, push hips back."
                is_bad_rep = True
            elif not self.neutral_spine:
                self.current_feedback = "No rep: Keep back flat! Do not round your spine."
                is_bad_rep = True
            elif not self.controlled_hinge:
                self.current_feedback = "No rep: Control your hinge descent!"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged spinal curvature or kinetic breakdown."
                is_bad_rep = True
            elif not self.target_hinge_reached:
                self.current_feedback = "No rep: Incomplete hinge! Lower until torso is parallel."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Good Morning hinge!"

            self.post_rep_timer = 40
            self.state = "STANDING"
            self.target_hinge_reached = False
            self.min_hip_angle = 180.0
            self.is_side_view = True
            self.not_squatting = True
            self.neutral_spine = True
            self.controlled_hinge = True
            self.ml_posture_maintained = True
            self.profile_frames = 0
            self.squat_frames = 0
            self.spine_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((165 - hip_angle) / 60 * 100))) if self.state != "STANDING" else 100
        if not self.not_squatting: form_score -= 35
        if not self.neutral_spine: form_score -= 35
        if not self.controlled_hinge: form_score -= 20
        if not self.target_hinge_reached and self.state == "EXTENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Good Morning", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1), "depth_reached": self.target_hinge_reached
        }