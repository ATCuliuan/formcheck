import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (DEADLIFT - SAGITTAL VIEW, FLOOR START)
# =====================================================================

# 1. Setup & Starting Guard (Bottom / Floor Position)
SETUP_BOTTOM_MAX_HINGE_ANGLE = 125.0 # Hips & knees hinged down holding the bar at the bottom
SETUP_WRIST_KNEE_MARGIN = 0.03       # Wrists at or below knee level at setup
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.42 # Side profile ratio
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames in bottom setup to confirm READY
UNLOCK_RESET_FRAMES = 15             # Frames held in bottom setup to clear LOCKED_FAULT (~0.5s)
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds
MIN_SPINE_ANGLE = 130.0              # Lumbar/thoracic spinal rounding cutoff
MIN_KNEE_SQUAT_ANGLE = 85.0          # Dropping hips way below knees (squat cheat)
MAX_HINGE_KNEE_LOCKOUT_ANGLE = 162.0 # Stiff-leg fault: knees locked straight (>= 162°) while torso is hinged
HINGE_PHASE_MAX_HIP_ANGLE = 135.0    # Hip angle below which torso is considered hinged forward
MAX_LOCKOUT_TORSO_SLANT_DEG = 12.0   # Max torso slant from vertical (0° = straight vertical) at top lockout
MAX_BAR_DRIFT_RATIO = 0.65           # Wrist forward drift relative to ankle line / torso length
MAX_HEEL_LIFT_DELTA = 0.045          # Ankle lift off baseline indicating heel rise
ECCENTRIC_DROP_SPEED_MAX = 28.0      # Free-fall descent cutoff (deg/frame)
EARLY_REVERSAL_DELTA = 7.0           # Early downward drop before reaching standing lockout

# 3. Deadlift Lockout & Range of Motion (Bottom -> Top -> Bottom)
ASCENDING_START_HINGE_ANGLE = 128.0  # Rising above this from bottom initiates ASCENDING (the pull)
HALF_LOCKOUT_MIN_HINGE_ANGLE = 138.0 # Pulling past 138° but short of 152° awards half rep
TOP_LOCKOUT_HINGE_ANGLE = 152.0      # Standing tall lockout angle required for full rep credit
TOP_KNEE_MIN_ANGLE = 158.0           # Knees extended at top lockout
TOP_HIP_MIN_ANGLE = 155.0            # Hips extended at top lockout
DESCENDING_START_HINGE_ANGLE = 145.0 # Hinging below this from TOP transitions to DESCENDING
REP_COMPLETE_HINGE_ANGLE = 125.0     # Returning bar to bottom position evaluates rep completion

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_BACK = 4
DEBOUNCE_SQUAT = 4
DEBOUNCE_KNEE_LOCK = 3               # Frames before flagging locked knees during hinged pull/descent
DEBOUNCE_SLANT = 3                   # Frames before flagging slanted/non-vertical torso at lockout
DEBOUNCE_DRIFT = 4
DEBOUNCE_HEEL = 3
DEBOUNCE_DROP = 2
DEBOUNCE_BOTTOM_COMPLETE = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class DeadliftTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.lockout_reached = False
        self.half_lockout_reached = False
        self.max_hinge_angle = 0.0

        # Biomechanical Tripwires (SAGITTAL / SIDE VIEW)
        self.is_side_view = True
        self.back_neutral = True
        self.not_squatted = True
        self.not_stiff_legged = True
        self.vertical_lockout = True
        self.bar_path_close = True
        self.heels_flat = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.back_frames = 0
        self.squat_frames = 0
        self.knee_lock_frames = 0
        self.slant_frames = 0
        self.drift_frames = 0
        self.heel_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.setup_frames = 0
        self.bottom_complete_frames = 0
        self.unlock_frames = 0
        self.ml_fault_frames = 0
        self.post_rep_timer = 0

        self.prev_hinge_angle = None
        self.baseline_ankle_y = None
        self.is_ready = False

        self.current_feedback = "Stand sideways, hinge down with soft knees and grip the bar"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_deadlift.tflite")
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

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.lockout_reached = False
        self.half_lockout_reached = False
        self.max_hinge_angle = 0.0
        self.bottom_complete_frames = 0
        self.reversal_frames = 0
        self.prev_hinge_angle = None

        self.is_side_view = True
        self.back_neutral = True
        self.not_squatted = True
        self.not_stiff_legged = True
        self.vertical_lockout = True
        self.bar_path_close = True
        self.heels_flat = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.back_frames = 0
        self.squat_frames = 0
        self.knee_lock_frames = 0
        self.slant_frames = 0
        self.drift_frames = 0
        self.heel_frames = 0
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
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of working body",
                "hip_angle": 0.0, "knee_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        spine_angle = calculate_angle(active_ear, active_shoulder, active_hip)
        hinge_angle = (hip_angle + knee_angle) / 2.0

        # Torso Vertical Tilt (0° = perfectly vertical upright, >12° = slanted forward or leaning/hyperextending back)
        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = max(abs(active_hip.y - active_shoulder.y), 0.05)
        torso_vertical_tilt_deg = np.degrees(np.arctan2(dx_torso, dy_torso))
        is_torso_vertical = torso_vertical_tilt_deg <= MAX_LOCKOUT_TORSO_SLANT_DEG

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # Bottom Starting Position Check (Hinged down holding the bar, knees bent between 85° and 162°)
        is_at_bottom = (
            (hinge_angle <= SETUP_BOTTOM_MAX_HINGE_ANGLE)
            or (active_wrist.y >= (active_knee.y - SETUP_WRIST_KNEE_MARGIN) and hinge_angle < 140.0)
        )
        is_knee_locked_in_hinge = knee_angle >= MAX_HINGE_KNEE_LOCKOUT_ANGLE
        is_bottom_setup = (
            is_side
            and is_at_bottom
            and (knee_angle >= MIN_KNEE_SQUAT_ANGLE)
            and (not is_knee_locked_in_hinge)
        )

        # Top Standing Lockout Check (Must be fully extended AND vertical with no slant)
        is_joints_extended = (
            (hinge_angle >= TOP_LOCKOUT_HINGE_ANGLE)
            or ((knee_angle >= TOP_KNEE_MIN_ANGLE) and (hip_angle >= TOP_HIP_MIN_ANGLE))
        )
        is_top_locked = is_joints_extended and is_torso_vertical

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Hold Bottom Deadlift Position to Start)
        # ---------------------------------------------------------
        if is_bottom_setup:
            self.setup_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y

            if self.setup_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.setup_frames = max(0, self.setup_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (side view)"
            elif knee_angle < MIN_KNEE_SQUAT_ANGLE:
                feedback_msg = "Raise hips slightly (do not deep-squat the setup)"
            elif is_at_bottom and is_knee_locked_in_hinge:
                feedback_msg = "Bend your knees slightly (do not lock knees in setup)"
            else:
                feedback_msg = "Hinge down with soft knees and hold the bar at the bottom to begin"
            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1),
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["BOTTOM_SETUP", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[DEADLIFT FAULT] AI ST-GCN flagged bad posture: prob={bad_posture_prob:.2f} "
                                f"(threshold={ML_FAULT_CONFIDENCE_THRESHOLD}), frames={self.ml_fault_frames}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES (ACTIVE REP: ASCENDING -> TOP -> DESCENDING)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            # 0. Side View Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[DEADLIFT FAULT] Lost side view: shoulder_span_ratio={shoulder_span_ratio:.2f} "
                            f"(max={SETUP_MAX_SHOULDER_SPAN_RATIO}), frames={self.profile_frames}",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # A. Back Rounding Check
            if spine_angle < MIN_SPINE_ANGLE:
                self.back_frames += 1
                if self.back_frames >= DEBOUNCE_BACK:
                    if self.back_neutral:
                        print(
                            f"[DEADLIFT FAULT] Back rounded: spine_angle={spine_angle:.1f}° "
                            f"(min={MIN_SPINE_ANGLE}°), frames={self.back_frames}",
                            flush=True,
                        )
                    self.back_neutral = False
            else:
                self.back_frames = max(0, self.back_frames - 1)

            # B. Squat Cheat Check
            if knee_angle < MIN_KNEE_SQUAT_ANGLE:
                self.squat_frames += 1
                if self.squat_frames >= DEBOUNCE_SQUAT:
                    if self.not_squatted:
                        print(
                            f"[DEADLIFT FAULT] Squat cheat detected: knee_angle={knee_angle:.1f}° "
                            f"(min={MIN_KNEE_SQUAT_ANGLE}°), frames={self.squat_frames}",
                            flush=True,
                        )
                    self.not_squatted = False
            else:
                self.squat_frames = max(0, self.squat_frames - 1)

            # C. Stiff-Leg / Early Knee Lockout Fault
            # If torso is still hinged forward (hip_angle < 135°) but knees are locked straight (>= 162°), flag bad rep!
            if hip_angle < HINGE_PHASE_MAX_HIP_ANGLE and knee_angle >= MAX_HINGE_KNEE_LOCKOUT_ANGLE:
                self.knee_lock_frames += 1
                if self.knee_lock_frames >= DEBOUNCE_KNEE_LOCK:
                    if self.not_stiff_legged:
                        print(
                            f"[DEADLIFT FAULT] Knees locked out during hinge (stiff-leg pull): "
                            f"knee_angle={knee_angle:.1f}° (max={MAX_HINGE_KNEE_LOCKOUT_ANGLE}°) while hip_angle={hip_angle:.1f}°",
                            flush=True,
                        )
                    self.not_stiff_legged = False
            else:
                self.knee_lock_frames = max(0, self.knee_lock_frames - 1)

            # D. Non-Vertical / Slanted Lockout Fault
            # When standing near the top (hips/knees extended), torso must be vertical (<= 12° slant, no leaning back or forward slant)
            if (hip_angle >= 150.0 and knee_angle >= 155.0) and not is_torso_vertical:
                self.slant_frames += 1
                if self.slant_frames >= DEBOUNCE_SLANT:
                    if self.vertical_lockout:
                        print(
                            f"[DEADLIFT FAULT] Slanted lockout (not vertical): "
                            f"torso_vertical_tilt={torso_vertical_tilt_deg:.1f}° (max={MAX_LOCKOUT_TORSO_SLANT_DEG}°), "
                            f"hip_angle={hip_angle:.1f}°, knee_angle={knee_angle:.1f}°",
                            flush=True,
                        )
                    self.vertical_lockout = False
            else:
                self.slant_frames = max(0, self.slant_frames - 1)

            # E. Bar Drift Check
            wrist_drift = abs(active_wrist.x - active_ankle.x) / torso_len
            if wrist_drift > MAX_BAR_DRIFT_RATIO:
                self.drift_frames += 1
                if self.drift_frames >= DEBOUNCE_DRIFT:
                    if self.bar_path_close:
                        print(
                            f"[DEADLIFT FAULT] Bar drifted forward: wrist_drift={wrist_drift:.2f} "
                            f"(max={MAX_BAR_DRIFT_RATIO}), frames={self.drift_frames}",
                            flush=True,
                        )
                    self.bar_path_close = False
            else:
                self.drift_frames = max(0, self.drift_frames - 1)

            # F. Heel Lift Check
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > MAX_HEEL_LIFT_DELTA:
                self.heel_frames += 1
                if self.heel_frames >= DEBOUNCE_HEEL:
                    if self.heels_flat:
                        heel_lift = self.baseline_ankle_y - active_ankle.y
                        print(
                            f"[DEADLIFT FAULT] Heels lifted off floor: heel_lift={heel_lift:.3f} "
                            f"(max={MAX_HEEL_LIFT_DELTA}), frames={self.heel_frames}",
                            flush=True,
                        )
                    self.heels_flat = False
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

            self.max_hinge_angle = max(self.max_hinge_angle, hinge_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (BOTTOM_SETUP -> ASCENDING -> TOP -> DESCENDING)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_bottom_setup:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "BOTTOM_SETUP"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Drive through heels and stand tall."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "BOTTOM_SETUP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through heels and pull the bar up!"

        # Pull Initiation from the Floor
        elif self.state == "BOTTOM_SETUP" and hinge_angle > ASCENDING_START_HINGE_ANGLE:
            self.state = "ASCENDING"
            self.rep_started = True
            self.max_hinge_angle = hinge_angle
            self.bottom_complete_frames = 0
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive hips forward, keep bar close to shins..."

        # Full Standing Vertical Lockout Reached at Top
        elif self.state == "ASCENDING" and is_top_locked:
            self.lockout_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Vertical lockout reached! Hinge hips back to lower the bar."

        # Early Downward Reversal before Lockout (Half Rep / Incomplete Lockout)
        elif self.state == "ASCENDING" and not self.lockout_reached and (self.max_hinge_angle - hinge_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_hinge_angle >= HALF_LOCKOUT_MIN_HINGE_ANGLE:
                    self.half_lockout_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Incomplete lockout! Lowering before standing vertically..."
                print(
                    f"[DEADLIFT FAULT] Early downward reversal before vertical lockout: "
                    f"max_hinge_angle={self.max_hinge_angle:.1f}°, torso_tilt={torso_vertical_tilt_deg:.1f}°",
                    flush=True,
                )

        # Normal Descent from Top Lockout
        elif self.state == "TOP" and hinge_angle < DESCENDING_START_HINGE_ANGLE:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the bar back down past your knees..."

        if self.state == "ASCENDING" and not ((self.max_hinge_angle - hinge_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop detector (monitors lowering speed)
        if self.rep_started and self.state in ["DESCENDING", "TOP"]:
            if self.prev_hinge_angle is not None:
                drop_speed = self.prev_hinge_angle - hinge_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[DEADLIFT FAULT] Uncontrolled descent speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_hinge_angle = hinge_angle
        else:
            self.prev_hinge_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (WHEN BAR RETURNS TO BOTTOM)
        # ---------------------------------------------------------
        if self.rep_started and is_at_bottom and hinge_angle <= REP_COMPLETE_HINGE_ANGLE and self.state in ["DESCENDING", "TOP", "ASCENDING"]:
            self.bottom_complete_frames += 1
            if self.bottom_complete_frames >= DEBOUNCE_BOTTOM_COMPLETE:
                is_bad_rep = False
                is_half_rep = False

                if not self.is_side_view:
                    self.current_feedback = "Bad rep: Stand sideways to camera! Hold bottom setup to reset."
                    is_bad_rep = True
                elif not self.not_stiff_legged:
                    self.current_feedback = "Bad rep: Knees locked out early! Bend knees and drive legs & hips together."
                    is_bad_rep = True
                elif not self.vertical_lockout:
                    self.current_feedback = "Bad rep: Slanted lockout! Stand straight vertical at the top (do not lean back or slant)."
                    is_bad_rep = True
                elif not self.back_neutral:
                    self.current_feedback = "Bad rep: Back rounded! Keep spine flat and chest proud."
                    is_bad_rep = True
                elif not self.not_squatted:
                    self.current_feedback = "Bad rep: You squatted! Push hips back instead of dropping them."
                    is_bad_rep = True
                elif not self.bar_path_close:
                    self.current_feedback = "Bad rep: Bar drifted away! Keep weights close to shins."
                    is_bad_rep = True
                elif not self.heels_flat:
                    self.current_feedback = "Bad rep: Keep heels planted flat on floor."
                    is_bad_rep = True
                elif not self.controlled_descent:
                    self.current_feedback = "Bad rep: Control the negative! Lower weights with control."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "Bad rep: AI flagged spinal curvature or kinetic breakdown."
                    is_bad_rep = True
                elif not self.lockout_reached:
                    if self.max_hinge_angle >= HALF_LOCKOUT_MIN_HINGE_ANGLE or self.half_lockout_reached:
                        self.current_feedback = "Half rep: Incomplete lockout! Stand all the way vertical at the top."
                        is_half_rep = True
                    else:
                        self.current_feedback = "Bad rep: Pull was too shallow! Drive hips to full vertical lockout."
                        is_bad_rep = True

                rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
                if is_bad_rep:
                    self.bad_reps += 1
                    print(
                        f"[DEADLIFT REP #{rep_num} -> BAD REP] "
                        f"Reason: {self.current_feedback} | max_hinge_angle={self.max_hinge_angle:.1f}°",
                        flush=True,
                    )
                    self.state = "LOCKED_FAULT"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                elif is_half_rep:
                    self.half_reps += 1
                    print(
                        f"[DEADLIFT REP #{rep_num} -> HALF REP] "
                        f"Reason: {self.current_feedback} | max_hinge_angle={self.max_hinge_angle:.1f}°",
                        flush=True,
                    )
                    self.state = "BOTTOM_SETUP"
                    self._reset_rep_tracking()
                else:
                    self.good_reps += 1
                    self.current_feedback = "Great deadlift! Clean vertical lockout."
                    print(
                        f"[DEADLIFT REP #{rep_num} -> GOOD REP] "
                        f"max_hinge_angle={self.max_hinge_angle:.1f}°",
                        flush=True,
                    )
                    self.state = "BOTTOM_SETUP"
                    self._reset_rep_tracking()

                self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((hinge_angle - 105) / 47 * 100))) if self.state != "BOTTOM_SETUP" else 100
        if not self.is_side_view: form_score -= 35
        if not self.not_stiff_legged: form_score -= 30
        if not self.vertical_lockout: form_score -= 30
        if not self.back_neutral: form_score -= 30
        if not self.not_squatted: form_score -= 25
        if not self.bar_path_close: form_score -= 20
        if not self.heels_flat: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.lockout_reached and self.state == "DESCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Deadlift", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1),
            "depth_reached": self.lockout_reached
        }