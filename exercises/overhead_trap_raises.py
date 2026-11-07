
import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (OVERHEAD TRAP RAISE - SAGITTAL VIEW)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_MAX_RAISE_ANGLE = 55.0         # Arms hanging at thighs
SETUP_MIN_ELBOW_ANGLE = 135.0        # Arms must be straight/extended to start
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.40 # Strict profile ratio (rejects front view)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames held with arms down to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 1.8       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds (TIGHTENED FOR STRICT BACKWARD/FORWARD LEAN)
MIN_ARM_STRAIGHT_ANGLE = 130.0       # Catching hand bend / elbow curl cheat (elbow < 130°)
MAX_TORSO_BACKWARD_LEAN_ANGLE = 7.0  # Tightened from 15.0° to 7.0° (catches subtle spine hyperextension)
MAX_TORSO_FORWARD_LEAN_ANGLE = 10.0  # Tightened from 15.0° to 10.0° (catches forward torso bowing)
MAX_KNEE_BOUNCE_DELTA = 14.0         # Knee deflection degrees indicating momentum cheat
MAX_ANKLE_SPREAD_RATIO = 0.45        # Max foot spread / torso length (catches jumping jack legs)
MAX_LATERAL_FLARE_SKEW = 0.38        # Flaring arms out to sides instead of raising vertically
ECCENTRIC_DROP_SPEED_MAX = 30.0      # Angular drop speed cutoff (deg/frame)

# 3. Range of Motion & Lockout Gates
FULL_OVERHEAD_MIN_ANGLE = 118.0      # Side-profile anatomical lockout angle
ASCENDING_START_ANGLE = 50.0         # Raising past this initiates ASCENDING
DESCENDING_TRIGGER_DELTA = 12.0      # Requires sustained downward drop from peak to shift to DESCENDING
REP_COMPLETE_ANGLE = 58.0            # Returning to thigh level completes the rep

# 4. Debounce Frame Thresholds
DEBOUNCE_REVERSAL = 4                # Requires 4 consecutive downward frames to prevent mid-rep jitter triggers
DEBOUNCE_CURL = 3                    # Fast trigger for bent elbows / arm curling
DEBOUNCE_PROFILE = 3                 # Fast trigger for facing camera
DEBOUNCE_TORSO_BACKWARD = 2          # Fast 2-frame tripwire for backward lean (trips almost instantly)
DEBOUNCE_TORSO_FORWARD = 2           # Fast 2-frame tripwire for forward lean
DEBOUNCE_JACK = 2                    # Fast trigger for jumping jack feet/arms
DEBOUNCE_KNEE = 3
DEBOUNCE_DROP = 2
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class OverheadTrapRaiseTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility

        self.rep_started = False
        self.overhead_reached = False      # Must be True to count as a good rep
        self.max_raise_angle = 0.0

        # Instructor Tripwires (SAGITTAL / SIDE VIEW)
        self.is_side_view = True           # Enforces side profile view
        self.arms_straight = True          # Catches hand bend / bicep curling
        self.no_torso_backward_lean = True # Catches leaning backward
        self.no_torso_forward_lean = True  # Catches leaning forward / bowing
        self.arms_vertical_plane = True    # Rejects lateral flare
        self.no_jumping_feet = True        # Rejects jumping feet out wide
        self.controlled_descent = True
        self.knee_bounce_warning = False
        self.ml_posture_maintained = True

        self.baseline_knee_angle = None
        self.baseline_torso_angle = None

        # Debounce counters
        self.reversal_frames = 0
        self.curl_frames = 0
        self.profile_frames = 0
        self.torso_back_frames = 0
        self.torso_fwd_frames = 0
        self.jack_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_raise_angle = None
        self.current_feedback = "Turn sideways to camera: show full side profile"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_overhead_trap_raises.tflite")
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
        self.overhead_reached = False
        self.max_raise_angle = 0.0
        self.reversal_frames = 0
        self.prev_raise_angle = None

        self.is_side_view = True
        self.arms_straight = True
        self.no_torso_backward_lean = True
        self.no_torso_forward_lean = True
        self.arms_vertical_plane = True
        self.no_jumping_feet = True
        self.controlled_descent = True
        self.knee_bounce_warning = False
        self.ml_posture_maintained = True

        self.curl_frames = 0
        self.profile_frames = 0
        self.torso_back_frames = 0
        self.torso_fwd_frames = 0
        self.jack_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

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
        l_toe = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        r_toe = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of working arm and torso",
                "elbow_angle": 0.0, "raise_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Joint angles
        raise_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        hip_span_x = abs(l_hip.x - r_hip.x)
        shoulder_aspect = shoulder_span_x / torso_len

        # Side profile verification
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_span_x < 0.16)
        facing_right = (l_toe.x > l_ankle.x) if l_vis > r_vis else (r_toe.x > r_ankle.x)

        # Directional spine tilt
        dx_spine = active_shoulder.x - active_hip.x
        dy_spine = max(abs(active_shoulder.y - active_hip.y), 0.1)
        spine_tilt_deg = np.degrees(np.arctan2(dx_spine, dy_spine))

        # Horizontal shoulder-over-hip shift normalized by torso length
        torso_drift_ratio = (active_shoulder.x - active_hip.x) / torso_len
        torso_backward_drift = -torso_drift_ratio if facing_right else torso_drift_ratio
        torso_forward_drift = torso_drift_ratio if facing_right else -torso_drift_ratio

        # Jumping Jack detection
        ankle_spread = abs(l_ankle.x - r_ankle.x) / torso_len
        is_jumping_legs = ankle_spread > MAX_ANKLE_SPREAD_RATIO

        wrist_span_x = abs(l_wrist.x - r_wrist.x) / torso_len
        is_flaring_lateral = wrist_span_x > MAX_LATERAL_FLARE_SKEW

        # Full Lockout Milestone: Either angle reaches 118° OR wrist ascends past eye/ear height
        is_overhead_achieved = (raise_angle >= FULL_OVERHEAD_MIN_ANGLE) or (active_wrist.y < (active_ear.y + 0.02))

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_resting_arms = (
            (raise_angle < SETUP_MAX_RAISE_ANGLE)
            and (elbow_angle > SETUP_MIN_ELBOW_ANGLE)
            and is_side
            and not is_jumping_legs
        )

        if is_resting_arms:
            self.ready_frames += 1
            if self.baseline_knee_angle is None:
                self.baseline_knee_angle = knee_angle
            else:
                self.baseline_knee_angle = 0.90 * self.baseline_knee_angle + 0.10 * knee_angle

            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = spine_tilt_deg
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * spine_tilt_deg

            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (must be side profile)"
            elif elbow_angle <= SETUP_MIN_ELBOW_ANGLE:
                feedback_msg = "Keep arms straight! Do not bend your elbows"
            elif is_jumping_legs:
                feedback_msg = "Keep feet together under hips"
            else:
                feedback_msg = "Let arms hang down sideways to start"

            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN BUFFER & INFERENCE (SOFTMAX + DEBOUNCE)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["RESTING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            # A. Hand / Elbow Bending & Curling Cheat
            if elbow_angle < MIN_ARM_STRAIGHT_ANGLE:
                self.curl_frames += 1
                if self.curl_frames >= DEBOUNCE_CURL:
                    self.arms_straight = False
            else:
                self.curl_frames = max(0, self.curl_frames - 1)

            # B. Side Profile Tripwire
            if shoulder_aspect > 0.46 or hip_span_x > 0.18:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # C. Dual-Vector Strict Torso Lean Tripwires (Angular + Coordinate Drift)
            if self.baseline_torso_angle is not None:
                spine_delta = spine_tilt_deg - self.baseline_torso_angle
                lean_direction_degrees = spine_delta if facing_right else -spine_delta

                # 1. Backward Lean Check (Angular >= 7.0° OR shoulder drifts backward by > 9% torso length)
                is_leaning_backward = (
                    (lean_direction_degrees < -MAX_TORSO_BACKWARD_LEAN_ANGLE)
                    or (torso_backward_drift > 0.09)
                )

                if is_leaning_backward:
                    self.torso_back_frames += 1
                    if self.torso_back_frames >= DEBOUNCE_TORSO_BACKWARD:
                        self.no_torso_backward_lean = False
                else:
                    self.torso_back_frames = max(0, self.torso_back_frames - 1)

                # 2. Forward Lean Check (Angular >= 10.0° OR shoulder drifts forward by > 12% torso length)
                is_leaning_forward = (
                    (lean_direction_degrees > MAX_TORSO_FORWARD_LEAN_ANGLE)
                    or (torso_forward_drift > 0.12)
                )

                if is_leaning_forward:
                    self.torso_fwd_frames += 1
                    if self.torso_fwd_frames >= DEBOUNCE_TORSO_FORWARD:
                        self.no_torso_forward_lean = False
                else:
                    self.torso_fwd_frames = max(0, self.torso_fwd_frames - 1)

            # D. Jumping Jack Legs / Arm Flare Tripwire
            if is_jumping_legs or (is_flaring_lateral and raise_angle > 65):
                self.jack_frames += 1
                if self.jack_frames >= DEBOUNCE_JACK:
                    if is_jumping_legs:
                        self.no_jumping_feet = False
                    if is_flaring_lateral:
                        self.arms_vertical_plane = False
            else:
                self.jack_frames = max(0, self.jack_frames - 1)

            # E. Knee Bounce Tripwire
            if self.baseline_knee_angle is not None and self.state == "ASCENDING":
                knee_deflection = abs(knee_angle - self.baseline_knee_angle)
                if knee_deflection > MAX_KNEE_BOUNCE_DELTA:
                    self.knee_frames += 1
                    if self.knee_frames >= DEBOUNCE_KNEE:
                        self.knee_bounce_warning = True
                else:
                    self.knee_frames = max(0, self.knee_frames - 1)

            self.max_raise_angle = max(self.max_raise_angle, raise_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (DEBOUNCED REVERSAL & LOCKOUT GATED)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_arms:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep torso upright and arms straight."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise arms up with straight elbows!"

        elif self.state in ["RESTING", "DESCENDING"] and raise_angle > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            self.rep_started = True
            self.max_raise_angle = raise_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise arms up, keep elbows locked..."

        # Milestone: Reached full overhead lockout
        if self.rep_started and is_overhead_achieved:
            self.overhead_reached = True
            if self.state == "ASCENDING":
                self.state = "TOP"
                if self.post_rep_timer == 0:
                    self.current_feedback = "Full overhead extension! Control the descent."

        # Debounced downward reversal detector
        if self.state in ["ASCENDING", "TOP"]:
            if (self.max_raise_angle - raise_angle) > DESCENDING_TRIGGER_DELTA:
                self.reversal_frames += 1
                if self.reversal_frames >= DEBOUNCE_REVERSAL:
                    self.state = "DESCENDING"
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Control the descent back down..."
            else:
                self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop speed monitor
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_raise_angle is not None:
                drop_speed = self.prev_raise_angle - raise_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_raise_angle = raise_angle
        else:
            self.prev_raise_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.rep_started and self.state == "DESCENDING" and raise_angle <= REP_COMPLETE_ANGLE:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to camera! Do not face front."
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "Bad rep: Do not bend or curl your arms! Keep elbows locked straight."
                is_bad_rep = True
            elif not self.no_torso_backward_lean:
                self.current_feedback = "Bad rep: You leaned backward! Keep your torso locked strictly upright."
                is_bad_rep = True
            elif not self.no_torso_forward_lean:
                self.current_feedback = "Bad rep: Do not bow or lean forward! Keep torso upright."
                is_bad_rep = True
            elif not self.no_jumping_feet:
                self.current_feedback = "Bad rep: Do not jump feet apart! Keep feet planted under hips."
                is_bad_rep = True
            elif not self.arms_vertical_plane:
                self.current_feedback = "Bad rep: Strict vertical raise only! Do not flare arms laterally."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Lower weights slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor posture/mechanics."
                is_bad_rep = True
            elif not self.overhead_reached:
                self.current_feedback = "Bad rep: Half rep! Raise arms all the way overhead past your head."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Keep knees steady, avoid leg drive."
                else:
                    self.current_feedback = "Excellent trap raise! Clean straight-arm execution."
                self.state = "RESTING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = 100 if self.state == "RESTING" else max(0, min(100, int((raise_angle / 118) * 100)))
        if not self.arms_straight: form_score -= 40
        if not self.is_side_view: form_score -= 40
        if not self.no_torso_backward_lean: form_score -= 35
        if not self.no_torso_forward_lean: form_score -= 30
        if not self.no_jumping_feet: form_score -= 40
        if not self.arms_vertical_plane: form_score -= 40
        if not self.controlled_descent: form_score -= 20
        if self.knee_bounce_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 30
        if not self.overhead_reached and self.state == "DESCENDING": form_score -= 35
        form_score = max(0, form_score)

        return {
            "exercise": "Overhead Trap Raise", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
        }
