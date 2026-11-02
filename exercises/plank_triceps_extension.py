import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class PlankTricepsExtensionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.lockout_reached = False       # Full arm lockout at top (elbow >= 155°)
        self.max_elbow = 0.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Profile camera orientation check
        self.core_maintained = True        # Catches hip sag / banana back
        self.no_pike = True                # Catches piking hips upward to cheat
        self.controlled_eccentric = True   # Catches slamming forearms onto ground
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Debounce counters
        self.profile_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.drop_frames = 0
        self.setup_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Hold a solid forearm plank sideways to begin"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model

        self.interpreter = None

        model_path = os.path.join("models", "tflite", "st_gcn_plank_triceps_extension.tflite")

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
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Determine dominant side closest to camera
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Plank Triceps Extension", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Show full side plank profile",
                "elbow_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal kinematics
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        plank_body_line = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Profile and horizontal checks
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < 0.42

        dx_body = abs(active_shoulder.x - active_ankle.x)
        dy_body = abs(active_shoulder.y - active_ankle.y)
        is_horizontal = dx_body > (dy_body * 1.1)

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        # Baseline: Forearm plank (elbows ~75°-105° and body line >= 155°)
        is_plank = (75 <= elbow_angle <= 105) and (plank_body_line >= 155) and is_side and is_horizontal
        if is_plank:
            self.setup_frames += 1
        else:
            self.setup_frames = 0

        if self.setup_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Hold solid forearm plank to begin"
            return {
                "exercise": "Plank Triceps Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "hip_angle": round(plank_body_line, 1), "depth_reached": False
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
                if np.argmax(out, axis=1)[0] == 0 and self.state != "FOREARM_PLANK":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["PRESSING_UP", "TOP_SUPPORT", "LOWERING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4: self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Hip Sag (Hips drop below spinal axis)
            if active_hip.y > active_shoulder.y and plank_body_line < 150.0:
                self.sag_frames += 1
                if self.sag_frames >= 3: self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # Hip Pike (Hips shot upward into an A-frame)
            if active_hip.y < active_shoulder.y and plank_body_line < 142.0:
                self.pike_frames += 1
                if self.pike_frames >= 3: self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            self.max_elbow = max(self.max_elbow, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (STRICT NO HALF-REPS)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "FOREARM_PLANK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Press through palms to lift elbows off ground!"

        elif self.state in ["FOREARM_PLANK", "LOWERING"] and elbow_angle > 110.0:
            self.state = "PRESSING_UP"
            self.max_elbow = elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Press arms straight..."

        # Full ROM: Elbows locked out in high plank (>= 155°)
        elif self.state == "PRESSING_UP" and elbow_angle >= 155.0:
            self.lockout_reached = True
            self.state = "TOP_SUPPORT"
            if self.post_rep_timer == 0:
                self.current_feedback = "Locked out! Lower elbows back down slowly."

        # Incomplete Rep reversal: user drops back down before hitting lockout
        elif self.state == "PRESSING_UP" and (self.max_elbow - elbow_angle) > 8.0:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Push all the way up to locked arms."

        elif self.state == "TOP_SUPPORT" and elbow_angle < 145.0:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent to forearms"

        # Eccentric drop detector
        if self.state in ["LOWERING", "PRESSING_UP"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2: self.controlled_eccentric = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to forearms (elbow <= 95°)
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP_SUPPORT", "PRESSING_UP"] and elbow_angle <= 95.0:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways to the camera!"
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "No rep: Core collapsed! Do not sag your hips."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "No rep: Butt too high! Maintain straight plank line."
                is_bad_rep = True
            elif not self.controlled_eccentric:
                self.current_feedback = "No rep: Control the descent! Do not slam elbows down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged unstable plank mechanics."
                is_bad_rep = True
            elif not self.lockout_reached:
                self.current_feedback = "No rep: Incomplete push! Lock elbows out fully at the top."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Solid plank triceps extension!"

            self.post_rep_timer = 40
            self.state = "FOREARM_PLANK"
            self.lockout_reached = False
            self.max_elbow = 0.0
            self.is_side_view = True
            self.core_maintained = True
            self.no_pike = True
            self.controlled_eccentric = True
            self.ml_posture_maintained = True
            self.profile_frames = 0
            self.sag_frames = 0
            self.pike_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((elbow_angle / 155) * 100)))
        if not self.core_maintained: form_score -= 35
        if not self.no_pike: form_score -= 25
        if not self.controlled_eccentric: form_score -= 20
        if not self.lockout_reached and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Plank Triceps Extension", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "hip_angle": round(plank_body_line, 1), "depth_reached": self.lockout_reached
        }