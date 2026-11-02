import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class RenegadeRowTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera harness compatibility

        self.depth_reached = False         # Full row depth reached (rowing elbow <= 90°)
        self.min_row_elbow = 180.0         # Tracks peak concentric pull

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Profile camera alignment check
        self.no_hip_rotation = True       # Catches hip twisting / opening to the side
        self.core_maintained = True        # Catches hip sag / banana back
        self.no_pike = True                # Catches piking butt into air
        self.support_arm_locked = True     # Catches planted arm buckling/bending
        self.controlled_descent = True     # Catches dropping weight back to ground
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Debounce counters
        self.profile_frames = 0
        self.twist_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.support_frames = 0
        self.drop_frames = 0
        self.plank_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_row_elbow = None
        self.current_feedback = "Hold a strong pushup plank sideways to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model

        self.interpreter = None

        model_path = os.path.join("models", "tflite", "st_gcn_renegade_row.tflite")

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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5:
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Renegade Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile in plank",
                "row_elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        # Bilateral Arm Angles
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Dynamic selection: the arm flexing more is rowing, the other supports
        if l_elbow_angle < r_elbow_angle:
            rowing_elbow_angle = l_elbow_angle
            support_elbow_angle = r_elbow_angle
        else:
            rowing_elbow_angle = r_elbow_angle
            support_elbow_angle = l_elbow_angle

        # Sagittal side profile selection
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Orientation & horizontal plank checks
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < 0.45

        dx_body = abs(active_shoulder.x - active_ankle.x)
        dy_body = abs(active_shoulder.y - active_ankle.y)
        is_horizontal = dx_body > (dy_body * 1.1)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Plank hold with both arms straight)
        # ---------------------------------------------------------
        is_plank = (rowing_elbow_angle >= 150) and (support_elbow_angle >= 150) and (body_angle >= 155) and is_side and is_horizontal
        if is_plank:
            self.plank_frames += 1
        else:
            self.plank_frames = 0

        if self.plank_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Hold straight plank with arms locked to begin"
            return {
                "exercise": "Renegade Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "row_elbow_angle": round(rowing_elbow_angle, 1),
                "body_angle": round(body_angle, 1),
                "depth_reached": False
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

                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "PLANKING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["PULLING", "TOP", "LOWERING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # A. Hip Twist / Lawnmower rotation
            hip_level_offset = abs(l_hip.y - r_hip.y)
            if hip_level_offset > 0.08:
                self.twist_frames += 1
                if self.twist_frames >= 3:
                    self.no_hip_rotation = False
            else:
                self.twist_frames = max(0, self.twist_frames - 1)

            # B. Core Sagging (Banana back)
            if active_hip.y > active_shoulder.y and body_angle < 150.0:
                self.sag_frames += 1
                if self.sag_frames >= 3:
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # C. Hip Piking (Glutes shooting up)
            if active_hip.y < active_shoulder.y and body_angle < 142.0:
                self.pike_frames += 1
                if self.pike_frames >= 3:
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # D. Supporting arm buckling (< 140°)
            if support_elbow_angle < 140.0:
                self.support_frames += 1
                if self.support_frames >= 3:
                    self.support_arm_locked = False
            else:
                self.support_frames = max(0, self.support_frames - 1)

            self.min_row_elbow = min(self.min_row_elbow, rowing_elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (STRICT NO HALF-REPS)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "PLANKING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Row dumbbell to your hip without twisting!"

        elif self.state in ["PLANKING", "LOWERING"] and rowing_elbow_angle < 140.0:
            self.state = "PULLING"
            self.min_row_elbow = rowing_elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep hips flat, drive elbow back..."

        # Full ROM: Elbow pulls past 90 degrees
        elif self.state == "PULLING" and rowing_elbow_angle <= 90.0:
            self.depth_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Top squeeze! Lower weight with control."

        # Incomplete Rep reversal before reaching ribs/90°
        elif self.state == "PULLING" and (rowing_elbow_angle - self.min_row_elbow) > 8.0:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Row dumbbell all the way to ribs."

        elif self.state == "TOP" and rowing_elbow_angle > 100.0:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the weight back to the floor"

        # Eccentric drop detector
        if self.state in ["LOWERING", "PULLING"]:
            if self.prev_row_elbow is not None:
                drop_speed = rowing_elbow_angle - self.prev_row_elbow
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_row_elbow = rowing_elbow_angle
        else:
            self.prev_row_elbow = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to locked plank (both elbows >= 150°)
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "PULLING"] and rowing_elbow_angle >= 150.0 and support_elbow_angle >= 150.0:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways in plank to camera!"
                is_bad_rep = True
            elif not self.no_hip_rotation:
                self.current_feedback = "No rep: You twisted your hips! Keep pelvis flat to floor."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "No rep: Hips sagged! Squeeze core and glutes."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "No rep: Butt too high! Maintain straight plank line."
                is_bad_rep = True
            elif not self.support_arm_locked:
                self.current_feedback = "No rep: Supporting arm buckled! Keep it straight."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not slam weight down."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged torso rotation/breakdown."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "No rep: Incomplete row! Pull dumbbell fully to ribs."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean renegade row! Excellent anti-rotation."

            self.post_rep_timer = 40
            self.state = "PLANKING"
            self.depth_reached = False
            self.min_row_elbow = 180.0
            self.is_side_view = True
            self.no_hip_rotation = True
            self.core_maintained = True
            self.no_pike = True
            self.support_arm_locked = True
            self.controlled_descent = True
            self.ml_posture_maintained = True
            self.profile_frames = 0
            self.twist_frames = 0
            self.sag_frames = 0
            self.pike_frames = 0
            self.support_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(rowing_elbow_angle - 90) * 0.4))) if self.state != "PLANKING" else 100
        if not self.no_hip_rotation: form_score -= 35
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 20
        if not self.support_arm_locked: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Renegade Row",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "row_elbow_angle": round(rowing_elbow_angle, 1),
            "body_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }