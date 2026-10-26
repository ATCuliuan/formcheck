import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingOverheadExtensionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0
        
        self.full_stretch_reached = False  # Bottom depth (elbow <= 75°)
        self.lockout_reached = False       # Overhead lockout (elbow >= 155°)
        self.min_elbow = 180.0
        
        # Posture Tripwires (FRONT VIEW)
        self.arms_overhead = True          # Arms must stay elevated overhead (shoulder angle >= 140°)
        self.elbows_not_flared = True      # Catches chicken-wing wide elbow flare
        self.is_symmetrical = True        # Catches uneven arm extension
        self.torso_upright = True          # Catches hyperextending back/swaying
        self.controlled_descent = True     # Dropping weight behind neck
        self.ml_posture_maintained = True

        self.overhead_frames = 0
        self.flare_frames = 0
        self.symmetry_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.setup_frames = 0
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Face camera, hold weight with arms straight overhead"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_overhead_extension.tflite")
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

        vis_joints = [l_shoulder.visibility, r_shoulder.visibility, l_elbow.visibility,
                      r_elbow.visibility, l_wrist.visibility, r_wrist.visibility,
                      l_hip.visibility, r_hip.visibility]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(vis_joints) < 0.35:
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Overhead Extension", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Step back: Show full torso and overhead arms",
                "elbow_angle": 0.0, "depth_reached": False
            }

        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), 0.25)
        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0)) or 0.45

        # Bilateral Kinematics
        l_overhead_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_overhead_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_overhead_elevation = (l_overhead_angle + r_overhead_angle) / 2.0

        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow = min(l_elbow_angle, r_elbow_angle)
        avg_elbow = (l_elbow_angle + r_elbow_angle) / 2.0

        # Elbow flare width: Distance between elbows vs shoulder width
        elbow_span_x = abs(l_elbow.x - r_elbow.x)
        is_flared = elbow_span_x > (shoulder_width * 1.55)

        # Baseline: Both arms straight overhead (elevation >= 145° and elbows >= 150°)
        is_standing_overhead = (avg_overhead_elevation >= 145) and (min_elbow >= 150)
        if is_standing_overhead:
            self.setup_frames += 1
        else:
            self.setup_frames = 0

        if self.setup_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Overhead Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Extend arms straight up overhead to begin",
                "elbow_angle": round(avg_elbow, 1), "depth_reached": False
            }

        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # Kinematic Tripwires
        if self.state in ["LOWERING", "BOTTOM_STRETCH", "PRESSING"]:
            # Upper arm dropped out of overhead vertical line
            if avg_overhead_elevation < 135:
                self.overhead_frames += 1
                if self.overhead_frames >= 4: self.arms_overhead = False
            else:
                self.overhead_frames = max(0, self.overhead_frames - 1)

            # Excessive elbow flare
            if is_flared:
                self.flare_frames += 1
                if self.flare_frames >= 4: self.elbows_not_flared = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # Asymmetric arm press
            if abs(l_elbow_angle - r_elbow_angle) > 22.0:
                self.symmetry_frames += 1
                if self.symmetry_frames >= 4: self.is_symmetrical = False
            else:
                self.symmetry_frames = max(0, self.symmetry_frames - 1)

            self.min_elbow = min(self.min_elbow, avg_elbow)

        # State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "OVERHEAD_TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower weight behind head by bending elbows!"

        elif self.state in ["OVERHEAD_TOP", "PRESSING"] and avg_elbow < 140:
            self.state = "LOWERING"
            self.min_elbow = avg_elbow
            if self.post_rep_timer == 0:
                self.current_feedback = "Deep stretch behind the neck..."

        # Full Stretch ROM: Elbow flex reaches <= 75°
        elif self.state == "LOWERING" and avg_elbow <= 75:
            self.full_stretch_reached = True
            self.state = "BOTTOM_STRETCH"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full triceps stretch! Press straight back up."

        # Incomplete Rep reversal
        elif self.state == "LOWERING" and (avg_elbow - self.min_elbow) > 8:
            self.state = "PRESSING"
            self.current_feedback = "Perform a full rep! Lower deeply behind neck before pressing."

        elif self.state == "BOTTOM_STRETCH" and avg_elbow > 85:
            self.state = "PRESSING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lock out overhead..."

        # Eccentric drop detector
        if self.state in ["LOWERING", "PRESSING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - avg_elbow
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2: self.controlled_descent = False
            self.prev_elbow_angle = avg_elbow
        else:
            self.prev_elbow_angle = None

        # Rep Completion: Return overhead to lockout (>= 155°)
        if self.state in ["PRESSING", "BOTTOM_STRETCH", "LOWERING"] and min_elbow >= 155:
            is_bad_rep = False
            if not self.arms_overhead:
                self.current_feedback = "No rep: Keep upper arms pointing straight up overhead!"
                is_bad_rep = True
            elif not self.elbows_not_flared:
                self.current_feedback = "No rep: Keep elbows tucked in! Do not flare outward."
                is_bad_rep = True
            elif not self.is_symmetrical:
                self.current_feedback = "No rep: Press both arms evenly together."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the weight down behind your neck."
                is_bad_rep = True
            elif not self.full_stretch_reached:
                self.current_feedback = "No rep: Incomplete range! Lower weight all the way down."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect overhead triceps extension!"

            self.post_rep_timer = 40
            self.state = "OVERHEAD_TOP"
            self.full_stretch_reached = False
            self.min_elbow = 180.0
            self.arms_overhead = True
            self.elbows_not_flared = True
            self.is_symmetrical = True
            self.controlled_descent = True
            self.overhead_frames = 0
            self.flare_frames = 0
            self.symmetry_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        form_score = max(0, min(100, int(100 - abs(avg_elbow - 75) * 0.5)))
        if not self.arms_overhead: form_score -= 30
        if not self.elbows_not_flared: form_score -= 30
        if not self.is_symmetrical: form_score -= 20
        if not self.full_stretch_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Overhead Extension", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(avg_elbow, 1), "depth_reached": self.full_stretch_reached
        }