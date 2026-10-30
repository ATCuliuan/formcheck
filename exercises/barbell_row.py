import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BarbellRowTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera harness compatibility
        
        self.target_reached = False        # Peak concentric pull (elbow <= 85°)
        self.min_elbow_angle = 180.0       # Tracks pull depth
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Catches facing camera instead of profile
        self.torso_hinged = True           # Catches standing too upright (torso angle > 145°)
        self.neutral_spine = True          # Catches rounded lower/upper back
        self.no_momentum_heave = True      # Catches using hip swing/heave to pull bar
        self.controlled_eccentric = True   # Catches free-falling the bar down
        self.ml_posture_maintained = True

        # Baseline torso angle during static hinge setup
        self.baseline_torso_angle = None

        # Debounce counters
        self.profile_frames = 0
        self.torso_stand_frames = 0
        self.spine_frames = 0
        self.heave_frames = 0
        self.drop_frames = 0
        self.standing_frames = 0
        
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Stand sideways in a hip hinge, arms hanging straight"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_barbell_row.tflite")
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
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
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

        # In side view, evaluate dominant profile facing camera
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_knee.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Barbell Row", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": "Show full side profile",
                "elbow_angle": 0.0, "torso_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal kinematics
        torso_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        spine_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # Profile validation (horizontal shoulder span vs vertical torso height)
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        torso_height = abs(active_shoulder.y - active_hip.y) or 0.40
        is_side = (shoulder_span_x / torso_height) < 0.42

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Bent-over hinge position with arms extended)
        # ---------------------------------------------------------
        is_hinged_setup = (100 <= torso_angle <= 140) and (elbow_angle >= 150) and (knee_angle >= 135) and is_side
        if is_hinged_setup:
            self.standing_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = torso_angle
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * torso_angle
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Hinge at hips (~45°), let arms hang fully"
            return {
                "exercise": "Barbell Row", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "torso_angle": round(torso_angle, 1), "depth_reached": False
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
                if np.argmax(out, axis=1)[0] == 0 and self.state != "READY":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["PULLING", "TOP", "LOWERING"]:
            # A. Orientation Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Standing too upright cheat (> 145° torso angle)
            if torso_angle > 145:
                self.torso_stand_frames += 1
                if self.torso_stand_frames >= 4:
                    self.torso_hinged = False
            else:
                self.torso_stand_frames = max(0, self.torso_stand_frames - 1)

            # C. Rounded back / Spine breakdown (< 135°)
            if spine_angle < 135:
                self.spine_frames += 1
                if self.spine_frames >= 4:
                    self.neutral_spine = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

            # D. Hip momentum heave (Torso angle swings > 16° from baseline setup)
            if self.baseline_torso_angle is not None:
                torso_swing = abs(torso_angle - self.baseline_torso_angle)
                if torso_swing > 16.0:
                    self.heave_frames += 1
                    if self.heave_frames >= 3:
                        self.no_momentum_heave = False
                else:
                    self.heave_frames = max(0, self.heave_frames - 1)

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (NO HALF REPS - FULL ROM ONLY)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "READY"
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull barbell toward lower ribcage!"

        elif self.state in ["READY", "LOWERING"] and elbow_angle < 140:
            self.state = "PULLING"
            self.min_elbow_angle = elbow_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive elbows up and back..."

        # Full ROM: Elbows reach <= 85°
        elif self.state == "PULLING" and elbow_angle <= 85:
            self.target_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Squeeze back! Lower bar under control."

        # Early downward reversal before reaching full pull depth
        elif self.state == "PULLING" and (elbow_angle - self.min_elbow_angle) > 8:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Pull bar completely to lower chest."

        elif self.state == "TOP" and elbow_angle > 95:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent"

        # Eccentric drop detector
        if self.state in ["LOWERING", "PULLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_eccentric = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Re-extending arms (elbow angle >= 150°)
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "PULLING"] and elbow_angle >= 150:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways to the camera!"
                is_bad_rep = True
            elif not self.torso_hinged:
                self.current_feedback = "No rep: Torso stood too upright! Stay bent over ~45°."
                is_bad_rep = True
            elif not self.neutral_spine:
                self.current_feedback = "No rep: Keep back flat! Do not round your spine."
                is_bad_rep = True
            elif not self.no_momentum_heave:
                self.current_feedback = "No rep: No momentum! Do not heave body up to pull bar."
                is_bad_rep = True
            elif not self.controlled_eccentric:
                self.current_feedback = "No rep: Control the negative! Lower the bar slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor mechanics/body sway."
                is_bad_rep = True
            elif not self.target_reached:
                self.current_feedback = "No rep: Perform a full rep! Pull bar all the way up."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Clean barbell row! Good rep."

            self.post_rep_timer = 40
            self.state = "READY"
            self.target_reached = False
            self.min_elbow_angle = 180.0
            self.is_side_view = True
            self.torso_hinged = True
            self.neutral_spine = True
            self.no_momentum_heave = True
            self.controlled_eccentric = True
            self.ml_posture_maintained = True
            self.prev_elbow_angle = None
            self.profile_frames = 0
            self.torso_stand_frames = 0
            self.spine_frames = 0
            self.heave_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((100 - abs(elbow_angle - 85) * 0.5))))
        if not self.torso_hinged: form_score -= 30
        if not self.neutral_spine: form_score -= 35
        if not self.no_momentum_heave: form_score -= 25
        if not self.controlled_eccentric: form_score -= 20
        if not self.target_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Barbell Row", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "torso_angle": round(torso_angle, 1), "depth_reached": self.target_reached
        }