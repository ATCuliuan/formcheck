import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (LEG RAISE - SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_HIP_MIN_ANGLE = 162.0          # Both hips fully extended standing tall
SETUP_KNEE_MIN_ANGLE = 155.0         # Both knees locked
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.40 # Side profile ratio
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames standing upright
UNLOCK_RESET_FRAMES = 15             # Frames held standing tall to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds (Tuned for strict back & leg swing detection)
MAX_BACKWARD_SWING_RATIO = 0.025     # Max backward drift behind support leg / torso height
MAX_TORSO_BACKWARD_LEAN_ANGLE = 15.0 # Max degrees spine can lean backward past vertical
MIN_LIFT_KNEE_ANGLE = 150.0          # Bending lifting leg (hamstring cheat)
MIN_SUPPORT_KNEE_ANGLE = 152.0       # Buckling standing leg
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Free-falling leg cutoff (deg/frame)
EARLY_REVERSAL_DELTA = 8.0           # Early downward drop before hip level

# 3. Raise Height & Range of Motion
FULL_RAISE_MAX_HIP_ANGLE = 110.0     # Foot reached hip/waist level (hip <= 110°)
HALF_RAISE_MAX_HIP_ANGLE = 135.0     # Shallow range flagged strictly as bad rep
LIFTING_START_HIP_ANGLE = 155.0      # Raising leg initiates LIFTING
LOWERING_START_HIP_ANGLE = 118.0     # Lowering leg initiates LOWERING
REP_COMPLETE_HIP_ANGLE = 165.0       # Return to bottom evaluates completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_SWING = 2                   # Fast 2-frame trigger for pendulum kicks
DEBOUNCE_TORSO = 2                   # Fast 2-frame trigger for backward spine sway
DEBOUNCE_LIFT_KNEE = 4
DEBOUNCE_SUPPORT_KNEE = 4
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingActiveStraightLegRaiseTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility

        self.rep_started = False
        self.target_height_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.leg_swung_backward = False
        self.lifted_knee_straight = True
        self.support_knee_straight = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.facing_camera_frames = 0
        self.backward_swing_frames = 0
        self.lift_knee_frames = 0
        self.support_knee_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_torso_angle = None
        self.prev_hip_angle = None
        self.current_feedback = "Stand sideways to the camera to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_active_straight_leg_raise.tflite")
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

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.target_height_reached = False
        self.was_half_rep = False
        self.min_hip_angle = 180.0
        self.reversal_frames = 0
        self.prev_hip_angle = None

        self.is_side_view = True
        self.leg_swung_backward = False
        self.lifted_knee_straight = True
        self.support_knee_straight = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.facing_camera_frames = 0
        self.backward_swing_frames = 0
        self.lift_knee_frames = 0
        self.support_knee_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_toe = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        r_toe = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Leg Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Show full side profile",
                "hip_angle": 0.0, "knee_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        hip_span_x = abs(l_hip.x - r_hip.x)
        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0)) or 0.45

        shoulder_aspect = shoulder_span_x / torso_height
        is_turned_sideways = shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO and hip_span_x < 0.15

        facing_right = (l_toe.x > l_ankle.x) if l_vis > r_vis else (r_toe.x > r_ankle.x)

        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Active lifting leg isolation
        if l_hip_angle < r_hip_angle - 8:
            active_hip_angle = l_hip_angle
            active_knee_angle = l_knee_angle
            support_knee_angle = r_knee_angle
            active_ankle, support_ankle = l_ankle, r_ankle
            active_knee_pt, support_knee_pt = l_knee, r_knee
            active_shoulder, active_hip = l_shoulder, l_hip
        elif r_hip_angle < l_hip_angle - 8:
            active_hip_angle = r_hip_angle
            active_knee_angle = r_knee_angle
            support_knee_angle = l_knee_angle
            active_ankle, support_ankle = r_ankle, l_ankle
            active_knee_pt, support_knee_pt = r_knee, l_knee
            active_shoulder, active_hip = r_shoulder, r_hip
        else:
            active_hip_angle = min(l_hip_angle, r_hip_angle)
            active_knee_angle = l_knee_angle if l_hip_angle < r_hip_angle else r_knee_angle
            support_knee_angle = r_knee_angle if l_hip_angle < r_hip_angle else l_knee_angle
            active_ankle = l_ankle if l_hip_angle < r_hip_angle else r_ankle
            support_ankle = r_ankle if l_hip_angle < r_hip_angle else l_ankle
            active_knee_pt = l_knee if l_hip_angle < r_hip_angle else r_knee
            support_knee_pt = r_knee if l_hip_angle < r_hip_angle else l_knee
            active_shoulder = l_shoulder if l_hip_angle < r_hip_angle else r_shoulder
            active_hip = l_hip if l_hip_angle < r_hip_angle else r_hip

        # Torso inclination relative to vertical
        dx_spine = active_shoulder.x - active_hip.x
        dy_spine = max(abs(active_shoulder.y - active_hip.y), 0.1)
        spine_tilt_deg = np.degrees(np.arctan2(dx_spine, dy_spine))

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_standing_tall = (
            l_hip_angle > SETUP_HIP_MIN_ANGLE and r_hip_angle > SETUP_HIP_MIN_ANGLE
            and l_knee_angle > SETUP_KNEE_MIN_ANGLE and r_knee_angle > SETUP_KNEE_MIN_ANGLE
            and is_turned_sideways
        )

        if is_standing_tall:
            self.standing_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = spine_tilt_deg
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * spine_tilt_deg

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_turned_sideways else "Stand tall sideways, legs locked to begin"
            return {
                "exercise": "Standing Leg Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(min(l_hip_angle, r_hip_angle), 1),
                "knee_angle": round(min(l_knee_angle, r_knee_angle), 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER (SOFTMAX + DEBOUNCE)
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_started:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["STANDING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LIFTING", "PEAK"]:
            if shoulder_aspect > 0.48 or hip_span_x > 0.20:
                self.facing_camera_frames += 1
                if self.facing_camera_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.facing_camera_frames = max(0, self.facing_camera_frames - 1)

            # Strict Directional Backward Swing Tripwire:
            # Measures if active ankle or knee drifts behind the support leg's vertical line
            if facing_right:
                ankle_backward_drift = support_ankle.x - active_ankle.x
                knee_backward_drift = support_knee_pt.x - active_knee_pt.x
            else:
                ankle_backward_drift = active_ankle.x - support_ankle.x
                knee_backward_drift = active_knee_pt.x - support_knee_pt.x

            is_swinging_back = (
                (ankle_backward_drift > MAX_BACKWARD_SWING_RATIO * torso_height)
                or (knee_backward_drift > MAX_BACKWARD_SWING_RATIO * torso_height)
            )

            if is_swinging_back:
                self.backward_swing_frames += 1
                if self.backward_swing_frames >= DEBOUNCE_SWING:
                    self.leg_swung_backward = True
            else:
                self.backward_swing_frames = max(0, self.backward_swing_frames - 1)

            # Strict Backward Torso Lean Check:
            # Detects shoulders pitching backward behind hips relative to facing direction
            if self.baseline_torso_angle is not None:
                # If facing right, backward lean shifts shoulder x leftward (negative delta)
                # If facing left, backward lean shifts shoulder x rightward (positive delta)
                spine_delta = spine_tilt_deg - self.baseline_torso_angle
                backward_lean_degrees = -spine_delta if facing_right else spine_delta

                if backward_lean_degrees > MAX_TORSO_BACKWARD_LEAN_ANGLE:
                    self.torso_frames += 1
                    if self.torso_frames >= DEBOUNCE_TORSO:
                        self.torso_upright = False
                else:
                    self.torso_frames = max(0, self.torso_frames - 1)

            if active_knee_angle < MIN_LIFT_KNEE_ANGLE:
                self.lift_knee_frames += 1
                if self.lift_knee_frames >= DEBOUNCE_LIFT_KNEE:
                    self.lifted_knee_straight = False
            else:
                self.lift_knee_frames = max(0, self.lift_knee_frames - 1)

            if support_knee_angle < MIN_SUPPORT_KNEE_ANGLE:
                self.support_knee_frames += 1
                if self.support_knee_frames >= DEBOUNCE_SUPPORT_KNEE:
                    self.support_knee_straight = False
            else:
                self.support_knee_frames = max(0, self.support_knee_frames - 1)

            self.min_hip_angle = min(self.min_hip_angle, active_hip_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_tall:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Lift leg straight out in front without swinging."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Standing Leg Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "hip_angle": round(active_hip_angle, 1), "knee_angle": round(active_knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise leg straight out in front!"

        elif self.state in ["STANDING", "LOWERING"] and active_hip_angle <= LIFTING_START_HIP_ANGLE:
            self.state = "LIFTING"
            self.rep_started = True
            self.min_hip_angle = active_hip_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep both legs completely locked, spine upright..."

        # Full ROM: Foot reaches hip level (<= 110°)
        elif self.state == "LIFTING" and active_hip_angle <= FULL_RAISE_MAX_HIP_ANGLE:
            self.target_height_reached = True
            self.state = "PEAK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Great height! Lower slowly under control."

        # Early downward reversal (Flagged as Bad Rep)
        elif self.state == "LIFTING" and not self.target_height_reached and (active_hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                self.was_half_rep = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete raise! Lowering leg early..."

        elif self.state == "PEAK" and active_hip_angle > LOWERING_START_HIP_ANGLE:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent"

        if self.state == "LIFTING" and not ((active_hip_angle - self.min_hip_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop detector
        if self.rep_started and self.state in ["LOWERING", "LIFTING"]:
            if self.prev_hip_angle is not None:
                drop_speed = active_hip_angle - self.prev_hip_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_hip_angle = active_hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and active_hip_angle > REP_COMPLETE_HIP_ANGLE and self.state in ["LOWERING", "PEAK", "LIFTING"]:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways! Do not face camera."
                is_bad_rep = True
            elif self.leg_swung_backward:
                self.current_feedback = "Bad rep: Do not swing leg backward for momentum! Lift straight from standing."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Do not lean backward to heave leg up! Keep spine upright."
                is_bad_rep = True
            elif not self.lifted_knee_straight:
                self.current_feedback = "Bad rep: Do not bend lifting knee! Lock it straight."
                is_bad_rep = True
            elif not self.support_knee_straight:
                self.current_feedback = "Bad rep: Standing leg must stay locked out."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Lower leg slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected collapsed posture."
                is_bad_rep = True
            elif self.was_half_rep or not self.target_height_reached:
                if self.min_hip_angle <= HALF_RAISE_MAX_HIP_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Lift foot all the way to hip level."
                else:
                    self.current_feedback = "Bad rep: Leg raise was too shallow! Lift higher."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect straight leg raise!"
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(active_hip_angle - 90) * 0.4))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 35
        if self.leg_swung_backward: form_score -= 30
        if not self.torso_upright: form_score -= 30
        if not self.lifted_knee_straight: form_score -= 30
        if not self.support_knee_straight: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if (self.was_half_rep or not self.target_height_reached) and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Leg Raise", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "hip_angle": round(active_hip_angle, 1),
            "knee_angle": round(active_knee_angle, 1),
            "depth_reached": self.target_height_reached
        }