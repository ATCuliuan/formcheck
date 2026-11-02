import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class PullupTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera harness compatibility

        self.depth_reached = False         # Full concentric chin-over-bar pull
        self.min_elbow_angle = 180.0       # Tracks peak concentric pull

        # Posture Tripwires (FRONT / CORONAL VIEW)
        self.is_front_view = True          # Frontal camera alignment check
        self.no_kipping = True             # Catches knee kicking / kipping momentum
        self.is_symmetrical = True        # Catches uneven arm pulling
        self.controlled_descent = True     # Catches free-falling out of top lockout
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Debounce counters
        self.front_frames = 0
        self.kip_frames = 0
        self.asym_frames = 0
        self.drop_frames = 0
        self.hang_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Hang from the bar facing the camera to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model

        self.interpreter = None

        model_path = os.path.join("models", "tflite", "st_gcn_pullup.tflite")

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
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Bilateral visibility check for frontal perspective
        upper_joints = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(upper_joints) < 0.40:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Pull-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Ensure arms and bar are visible",
                "elbow_angle": 0.0, "depth_reached": False
            }

        # Bilateral Joint Kinematics
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        # Bilateral Knee angles to detect kipping
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        min_knee_angle = min(l_knee_angle, r_knee_angle)

        # Frontal Orientation Check (Shoulders wide and level)
        torso_len = np.hypot(
            (l_shoulder.x + r_shoulder.x) / 2.0 - (l_hip.x + r_hip.x) / 2.0,
            (l_shoulder.y + r_shoulder.y) / 2.0 - (l_hip.y + r_hip.y) / 2.0
        ) or 0.40
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        is_facing_front = (shoulder_width / torso_len) > 0.42 and abs(l_shoulder.y - r_shoulder.y) < 0.08

        # Height comparison: Average wrist position (bar level) vs nose level
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2.0
        is_chin_over_bar = nose.y >= (avg_wrist_y - 0.02)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Full Dead Hang from Bar)
        # ---------------------------------------------------------
        is_overhead_hang = (l_wrist.y < l_shoulder.y) and (r_wrist.y < r_shoulder.y)
        is_dead_hang = is_overhead_hang and (avg_elbow_angle >= 155.0) and is_facing_front

        if is_dead_hang:
            self.hang_frames += 1
        else:
            self.hang_frames = 0

        if self.hang_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            feedback_msg = "Face camera directly" if not is_facing_front else "Hang with arms straight to begin"
            return {
                "exercise": "Pull-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(avg_elbow_angle, 1), "depth_reached": False
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
                if np.argmax(out, axis=1)[0] == 0 and self.state != "HANGING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["PULLING", "TOP", "LOWERING"]:
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= 4:
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # A. Kipping Cheat: Bending knees deeply (< 130°) to swing body
            if min_knee_angle < 130.0:
                self.kip_frames += 1
                if self.kip_frames >= 3:
                    self.no_kipping = False
            else:
                self.kip_frames = max(0, self.kip_frames - 1)

            # B. Asymmetric Pull: One arm pulling significantly harder (> 22° difference)
            if abs(l_elbow_angle - r_elbow_angle) > 22.0:
                self.asym_frames += 1
                if self.asym_frames >= 4:
                    self.is_symmetrical = False
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, avg_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (STRICT NO HALF-REPS)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "HANGING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull chin over the bar!"

        elif self.state in ["HANGING", "LOWERING"] and avg_elbow_angle < 145.0:
            self.state = "PULLING"
            self.min_elbow_angle = avg_elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive elbows down and back..."

        # Full ROM: Elbows flexed <= 85° AND chin clears wrists/bar
        elif self.state == "PULLING" and (avg_elbow_angle <= 85.0 or is_chin_over_bar):
            self.depth_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Chin over bar! Lower with control."

        # Incomplete Rep reversal before reaching top ROM
        elif self.state == "PULLING" and (avg_elbow_angle - self.min_elbow_angle) > 8.0:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Pull chin completely over bar."

        elif self.state == "TOP" and avg_elbow_angle > 95.0:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower all the way back to dead hang"

        # Eccentric drop detector (free-falling out of the pull)
        if self.state in ["LOWERING", "PULLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = avg_elbow_angle - self.prev_elbow_angle
                if drop_speed > 25.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_elbow_angle = avg_elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to Dead Hang (elbows >= 155°)
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "PULLING"] and min_elbow_angle >= 155.0:
            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "No rep: Face the camera directly!"
                is_bad_rep = True
            elif not self.no_kipping:
                self.current_feedback = "No rep: Do not kip or swing legs! Keep body still."
                is_bad_rep = True
            elif not self.is_symmetrical:
                self.current_feedback = "No rep: Pull evenly with both arms!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not drop abruptly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged excessive body swing/breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "No rep: Incomplete pull! Clear your chin over the bar."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean pull-up! Full extension."

            self.post_rep_timer = 40
            self.state = "HANGING"
            self.depth_reached = False
            self.min_elbow_angle = 180.0
            self.is_front_view = True
            self.no_kipping = True
            self.is_symmetrical = True
            self.controlled_descent = True
            self.ml_posture_maintained = True
            self.front_frames = 0
            self.kip_frames = 0
            self.asym_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((180 - avg_elbow_angle) / 95 * 100))) if self.state != "HANGING" else 100
        if not self.no_kipping: form_score -= 35
        if not self.is_symmetrical: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "LOWERING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Pull-Up",
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