import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (LYING LEG RAISE - NEAR-CAMERA LEG ONLY)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_LEG_RAISE_ANGLE_MAX = 35.0     # Both legs extended down close to floor during setup
SETUP_KNEE_STRAIGHT_MIN = 148.0      # Legs extended straight
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.38 # Enforces clean side profile view lying down
SETUP_MIN_READY_FRAMES = 6
UNLOCK_RESET_FRAMES = 12
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum landmark visibility sum

# 2. Form & Isolation Thresholds
MIN_KNEE_BEND_ANGLE = 140.0          # Bent knee momentum cheat threshold
MAX_LUMBAR_ARCH_GAP = 0.09           # Max vertical pelvis/spine lift off floor indicating arching core
MAX_OTHER_LEG_RAISE_ANGLE = 35.0     # Far-side leg must stay down on the floor (cannot raise!)
MAX_OVER_RAISE_ANGLE = 100.0         # Raising past 100° (pulling legs past 90° toward head) flags a fault

# 3. Raise Progression & Jitter-Free Reversal
ASCENDING_START_ANGLE = 34.0         # Near-camera leg raising off floor initiates ASCENDING
MIN_VALID_REP_PEAK_ANGLE = 42.0      # Below 42° is treated as floor shift/jitter (ignored, not a half rep)
FULL_TOP_MIN_ANGLE = 75.0            # Reaching near-vertical (~75-90°) confirms full range
DESCENDING_TRIGGER_DELTA = 14.0      # Must drop a full 14° below peak to trigger descent (prevents mid-raise jitter)
REP_COMPLETE_ANGLE = 30.0            # Returning leg to hover/bottom position completes rep
ANGLE_SMOOTHING_ALPHA = 0.65         # EMA weight for current frame angle to kill MediaPipe jitter

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_BENT_KNEE = 4
DEBOUNCE_ARCH = 4
DEBOUNCE_OTHER_LEG = 4               # Frames before flagging the back/far leg lifting up
DEBOUNCE_OVER_RAISE = 3              # Frames before flagging leg going past 90-100 degrees
DEBOUNCE_HALF_REP = 5                # Consecutive frames dropping required before confirming descent
DEBOUNCE_ML_FAULT = 5                # Consecutive frames required to trip ML fault

ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 # Softmax probability threshold
SEQUENCE_LENGTH = 30                 # ST-GCN temporal frame buffer size
POST_REP_FEEDBACK_FRAMES = 40        # Feedback display duration post-rep


class LyingLegRaiseTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.top_reached = False
        self.max_raise_angle = 0.0
        self.smoothed_raise_angle = None
        self.locked_side = None          # Locks 'RIGHT' or 'LEFT' (whichever side faces camera first)

        # Posture Tripwires (SIDE VIEW - SINGLE NEAR LEG)
        self.is_side_view = True
        self.legs_straight = True
        self.core_engaged = True
        self.other_leg_stationary = True # Ensures the non-camera leg stays down
        self.stayed_at_90 = True         # Ensures leg does not go past 90-100 degrees
        self.ml_posture_maintained = True

        self.baseline_hip_y = None

        # Debounce counters
        self.profile_frames = 0
        self.knee_frames = 0
        self.arch_frames = 0
        self.other_leg_frames = 0
        self.over_raise_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Lie flat on your back sideways to camera with legs straight"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_lying_leg_raise.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[LYING LEG RAISE ERROR] __init__ model_path={model_path} | error={exc}",
                    flush=True,
                )
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        """Maps MediaPipe 33 landmarks to the NTU-25 skeleton layout."""
        ntu = np.zeros((25, 3), dtype=np.float32)

        def xyz(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        ntu[3] = xyz(0)    # Nose
        ntu[4] = xyz(11)   # Left Shoulder
        ntu[5] = xyz(13)   # Left Elbow
        ntu[6] = xyz(15)   # Left Wrist
        ntu[7] = xyz(19)   # Left Hand/Index
        ntu[8] = xyz(12)   # Right Shoulder
        ntu[9] = xyz(14)   # Right Elbow
        ntu[10] = xyz(16)  # Right Wrist
        ntu[11] = xyz(20)  # Right Hand/Index
        ntu[12] = xyz(23)  # Left Hip
        ntu[13] = xyz(25)  # Left Knee
        ntu[14] = xyz(27)  # Left Ankle
        ntu[15] = xyz(31)  # Left Foot
        ntu[16] = xyz(24)  # Right Hip
        ntu[17] = xyz(26)  # Right Knee
        ntu[18] = xyz(28)  # Right Ankle
        ntu[19] = xyz(32)  # Right Foot

        ntu[0] = (ntu[12] + ntu[16]) / 2.0  # Base Spine (Mid-Hip)
        ntu[2] = (ntu[4] + ntu[8]) / 2.0    # Spine Shoulder / Neck (Mid-Shoulder)
        ntu[1] = (ntu[0] + ntu[2]) / 2.0    # Mid Spine
        ntu[20] = ntu[2]                    # Chest

        ntu[21] = xyz(17)  # Left Pinky
        ntu[22] = xyz(21)  # Left Thumb
        ntu[23] = xyz(18)  # Right Pinky
        ntu[24] = xyz(22)  # Right Thumb
        return ntu

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.top_reached = False
        self.max_raise_angle = 0.0

        self.is_side_view = True
        self.legs_straight = True
        self.core_engaged = True
        self.other_leg_stationary = True
        self.stayed_at_90 = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.knee_frames = 0
        self.arch_frames = 0
        self.other_leg_frames = 0
        self.over_raise_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Lying Leg Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Show full side profile lying down",
                "raise_angle": 0.0, "knee_angle": 0.0
            }

        # Lock onto the camera-facing side during setup so mid-rep leg raises never swap sides
        if self.locked_side is None:
            r_score = r_vis - (r_hip.z + r_knee.z)
            l_score = l_vis - (l_hip.z + l_knee.z)
            use_right = r_score >= l_score
        else:
            use_right = (self.locked_side == "RIGHT")

        if use_right:
            active_shoulder, active_hip, active_knee, active_ankle = r_shoulder, r_hip, r_knee, r_ankle
            other_shoulder, other_hip, other_knee, other_ankle = l_shoulder, l_hip, l_knee, l_ankle
        else:
            active_shoulder, active_hip, active_knee, active_ankle = l_shoulder, l_hip, l_knee, l_ankle
            other_shoulder, other_hip, other_knee, other_ankle = r_shoulder, r_hip, r_knee, r_ankle

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Raw & smoothed angle of the near-camera leg relative to the torso line
        raw_raise_angle = 180.0 - calculate_angle(active_shoulder, active_hip, active_knee)
        if self.smoothed_raise_angle is None:
            self.smoothed_raise_angle = raw_raise_angle
        else:
            self.smoothed_raise_angle = (
                ANGLE_SMOOTHING_ALPHA * raw_raise_angle
                + (1.0 - ANGLE_SMOOTHING_ALPHA) * self.smoothed_raise_angle
            )
        raise_angle = self.smoothed_raise_angle
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # Far-side (non-camera) leg elevation angle
        other_leg_visible = (other_hip.visibility > 0.35 and other_knee.visibility > 0.35)
        other_raise_angle = (
            180.0 - calculate_angle(other_shoulder, other_hip, other_knee)
            if other_leg_visible else 0.0
        )

        # Lumbar arch check (tracking active hip vertical lift off floor)
        current_hip_y = active_hip.y
        if self.baseline_hip_y is None:
            self.baseline_hip_y = current_hip_y

        hip_lift_offset = self.baseline_hip_y - current_hip_y
        is_lumbar_arched = hip_lift_offset > MAX_LUMBAR_ARCH_GAP

        # Both the near leg and far leg must be down at setup
        both_legs_down_setup = (
            (raise_angle <= SETUP_LEG_RAISE_ANGLE_MAX)
            and (other_raise_angle <= SETUP_LEG_RAISE_ANGLE_MAX)
        )
        is_setup_ready = (
            is_side
            and both_legs_down_setup
            and (knee_angle >= SETUP_KNEE_STRAIGHT_MIN)
            and not is_lumbar_arched
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD (LOCKS CAMERA-FACING LEG)
        # ---------------------------------------------------------
        if is_setup_ready:
            self.ready_frames += 1
            self.baseline_hip_y = 0.90 * self.baseline_hip_y + 0.10 * current_hip_y
            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                if not self.is_ready:
                    self.locked_side = "RIGHT" if (r_vis >= l_vis) else "LEFT"
                    print(f"[LYING LEG RAISE SETUP] Locked onto {self.locked_side} leg (near camera).", flush=True)
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (side view required)"
            elif knee_angle < SETUP_KNEE_STRAIGHT_MIN:
                feedback_msg = "Keep knees locked straight"
            elif is_lumbar_arched:
                feedback_msg = "Press lower back firmly into the floor (no arch)"
            else:
                feedback_msg = "Lower both legs flat to the floor to start"

            return {
                "exercise": "Lying Leg Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "raise_angle": round(raise_angle, 1), "knee_angle": round(knee_angle, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER & INFERENCE
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_started:
            try:
                buf_tvc = np.array(self.frame_buffer, dtype=np.float32)
                expected_shape = self.input_details[0]['shape']
                if len(expected_shape) == 5:
                    seq = np.transpose(buf_tvc, (2, 0, 1))[None, :, :, :, None]
                elif len(expected_shape) == 4 and tuple(expected_shape[1:]) == (buf_tvc.shape[2], buf_tvc.shape[0], buf_tvc.shape[1]):
                    seq = np.transpose(buf_tvc, (2, 0, 1))[None, :, :, :]
                else:
                    seq = np.expand_dims(buf_tvc, axis=0)

                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                bad_posture_prob = float(probs[0])

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["RESTING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(f"[LYING LEG RAISE FAULT] AI ST-GCN flagged poor core posture: prob={bad_posture_prob:.2f}", flush=True)
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[LYING LEG RAISE ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        # Catch if user tries to raise the far-side (non-camera) leg while resting or mid-rep
        if other_leg_visible and other_raise_angle > MAX_OTHER_LEG_RAISE_ANGLE:
            if other_knee.y < (active_hip.y - 0.08):
                self.other_leg_frames += 1
                if self.other_leg_frames >= DEBOUNCE_OTHER_LEG:
                    if self.other_leg_stationary:
                        print(
                            f"[LYING LEG RAISE FAULT] Far-side leg raised ({other_raise_angle:.1f}°)! "
                            f"Only the {self.locked_side} (camera-facing) leg may raise.",
                            flush=True,
                        )
                    self.other_leg_stationary = False
                    if not self.rep_started:
                        self.rep_started = True
                        self.state = "DESCENDING"
        else:
            self.other_leg_frames = max(0, self.other_leg_frames - 1)

        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[LYING LEG RAISE FAULT] Lost side profile: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # Over-Raise Check: Flag if leg goes past 90-100 degrees
            if raise_angle > MAX_OVER_RAISE_ANGLE:
                self.over_raise_frames += 1
                if self.over_raise_frames >= DEBOUNCE_OVER_RAISE:
                    if self.stayed_at_90:
                        print(
                            f"[LYING LEG RAISE FAULT] Over-raised past 90-100°: raise_angle={raise_angle:.1f}° "
                            f"(max={MAX_OVER_RAISE_ANGLE}°)",
                            flush=True,
                        )
                    self.stayed_at_90 = False
                    self.current_feedback = "Too high! Stop at 90 degrees—do not pull leg past vertical."
            else:
                self.over_raise_frames = max(0, self.over_raise_frames - 1)

            if knee_angle < MIN_KNEE_BEND_ANGLE:
                self.knee_frames += 1
                if self.knee_frames >= DEBOUNCE_BENT_KNEE:
                    if self.legs_straight:
                        print(f"[LYING LEG RAISE FAULT] Knee bent during raise: knee_angle={knee_angle:.1f}°", flush=True)
                    self.legs_straight = False
            else:
                self.knee_frames = max(0, self.knee_frames - 1)

            if is_lumbar_arched:
                self.arch_frames += 1
                if self.arch_frames >= DEBOUNCE_ARCH:
                    if self.core_engaged:
                        print(f"[LYING LEG RAISE FAULT] Lower back arched off floor: lift_offset={hip_lift_offset:.3f}", flush=True)
                    self.core_engaged = False
            else:
                self.arch_frames = max(0, self.arch_frames - 1)

            self.max_raise_angle = max(self.max_raise_angle, raise_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (DEBOUNCED REVERSAL - NO MID-RAISE HALF-REP JITTER)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_setup_ready:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = f"Reset complete! Raise only your {self.locked_side.lower()} (camera-side) leg to 90°."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Lying Leg Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "raise_angle": round(raise_angle, 1), "knee_angle": round(knee_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            if self.post_rep_timer == 0:
                self.current_feedback = f"Raise only your {self.locked_side.lower()} (camera-facing) leg to 90°!"

        elif self.state == "RESTING" and raise_angle > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            self.rep_started = True
            self.max_raise_angle = raise_angle
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise camera-side leg upward to 90°..."

        elif self.state == "ASCENDING" and raise_angle >= FULL_TOP_MIN_ANGLE:
            self.top_reached = True
            self.state = "TOP"
            self.half_rep_frames = 0
            if self.post_rep_timer == 0 and self.stayed_at_90:
                self.current_feedback = "90° reached! Stop here and slowly lower leg back down."

        # Debounced descent from ASCENDING: requires 5 consecutive frames dropping >= 14° below peak
        elif self.state == "ASCENDING" and not self.top_reached:
            if (self.max_raise_angle - raise_angle) > DESCENDING_TRIGGER_DELTA:
                self.half_rep_frames += 1
                if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                    self.state = "DESCENDING"
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Lowering leg back down..."
            else:
                self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Normal descent from TOP after reaching 90°
        elif self.state == "TOP" and (self.max_raise_angle - raise_angle) > 8.0:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0 and self.stayed_at_90:
                self.current_feedback = "Control the descent back toward the floor..."

        # Resume ASCENDING if user paused mid-way and continued lifting toward 90°
        elif self.state == "DESCENDING" and not self.top_reached and raise_angle > (self.max_raise_angle - 4.0) and raise_angle > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep raising leg toward 90°..."

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (ONLY AT BOTTOM RETURN)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "TOP", "ASCENDING"] and raise_angle <= REP_COMPLETE_ANGLE and other_raise_angle <= REP_COMPLETE_ANGLE:
            # Ignore minor floor jitter (< 42°)
            if self.max_raise_angle < MIN_VALID_REP_PEAK_ANGLE and self.other_leg_stationary:
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Lying Leg Raise", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "raise_angle": round(raise_angle, 1), "knee_angle": round(knee_angle, 1)
                }

            is_bad_rep = False
            is_half_rep = False

            if not self.other_leg_stationary:
                self.current_feedback = f"Bad rep: Only raise the {self.locked_side.lower()} (camera-facing) leg! Keep back leg down."
                is_bad_rep = True
            elif not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to camera! Lie flat to reset."
                is_bad_rep = True
            elif not self.stayed_at_90:
                self.current_feedback = "Bad rep: Leg went past 90-100°! Stop at 90 degrees. Lie flat to reset."
                is_bad_rep = True
            elif not self.legs_straight:
                self.current_feedback = "Bad rep: Keep raising knee locked straight! Do not bend leg."
                is_bad_rep = True
            elif not self.core_engaged:
                self.current_feedback = "Bad rep: Lower back arched off floor! Keep core pressed flat."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected core/posture breakdown."
                is_bad_rep = True
            elif not self.top_reached:
                self.current_feedback = "Bad rep: Half rep! Raise leg all the way to 90 degrees before lowering."
                is_half_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep or is_half_rep:
                self.bad_reps += 1
                print(
                    f"[LYING LEG RAISE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_angle={self.max_raise_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean 90° leg raise! Solid abdominal control."
                print(
                    f"[LYING LEG RAISE REP #{rep_num} -> GOOD REP] "
                    f"max_angle={self.max_raise_angle:.1f}°",
                    flush=True,
                )
                self.state = "RESTING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = 100 if self.state == "RESTING" else max(0, min(100, int((raise_angle / 80) * 100)))
        if not self.other_leg_stationary: form_score -= 45
        if not self.is_side_view: form_score -= 35
        if not self.stayed_at_90: form_score -= 35
        if not self.legs_straight: form_score -= 40
        if not self.core_engaged: form_score -= 35
        if not self.top_reached and self.state == "DESCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Lying Leg Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "raise_angle": round(raise_angle, 1),
            "knee_angle": round(knee_angle, 1)
        }