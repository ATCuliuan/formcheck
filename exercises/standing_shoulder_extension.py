import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SHOULDER EXTENSION - STRICT SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1-3: Stand Tall Sideways, Arms Locked Behind Hips)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.28 # Strict side profile (16:9 webcam calibrated)
SETUP_MAX_HIP_SPAN_RATIO = 0.22      # Hips aligned sideways to camera
SETUP_MAX_EXTENSION_ANGLE = 24.0     # Step 1 & 6: Allows holding bar/weights resting behind hips/glutes
SETUP_MIN_ELBOW_LOCKOUT = 148.0      # Step 3: Arms extended straight down, elbows frozen
SETUP_MAX_TORSO_TILT_DEG = 15.0      # Step 1 & 2: Stand tall with rib cage locked down (0° = vertical)
SETUP_MIN_READY_FRAMES = 8           # Consecutive frames to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held standing tall to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum upper-body keypoint visibility

# 2. Kinematic & Form Thresholds (Step 2, 3, 4, 6: Frozen Elbows, No Forward Lean, Posterior Arc)
MIN_ELBOW_LOCKOUT_ANGLE = 145.0      # Step 3: Squeeze triceps to keep elbows straight (no kickback bend)
MAX_TORSO_FORWARD_LEAN_DEG = 15.0    # Step 4: Move entirely from shoulder without leaning torso forward
MIN_NECK_NEUTRAL_ANGLE = 128.0       # Step 2: Keep neck neutral (no forward head thrust)
SHRUG_EAR_SHOULDER_FACTOR = 0.68     # Catches shrugging/rolling shoulders up toward ears at peak extension
MIN_POSTERIOR_ELBOW_DRIFT = 0.03     # Step 4: Elbow must move genuinely BEHIND shoulder line (positive drift)
MIN_POSTERIOR_WRIST_DRIFT = 0.05     # Step 4: Wrist must move genuinely BEHIND hip line (rejects front raise!)
ECCENTRIC_DROP_SPEED_MAX = 22.0      # Step 6: Lower under control (deg/frame cutoff)
EARLY_REVERSAL_DELTA = 5.0           # Early forward return before full ROM

# 3. Calibrated Extension Depth & Range of Motion (Step 4, 5, 6)
EXTENDING_START_ANGLE = 24.0         # Peeling arms backward past 24° initiates EXTENDING
MIN_VALID_REP_PEAK_ANGLE = 26.0      # Ignore minor resting posture shifts (< 26°)
HALF_ROM_MIN_EXTENSION = 27.0        # Pushing past 27° but short of 33° flags Half Rep (Bad Rep)
FULL_ROM_MIN_EXTENSION = 33.0        # Step 5: True anatomical posterior peak (~33°-45°+) without torso lean
RETURNING_START_ANGLE = 29.0         # Step 6: Slowly returning arms forward toward thighs
REP_COMPLETE_ANGLE = 23.0            # Step 6: Arms returned to back of thighs/hips completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 3
DEBOUNCE_DIRECTION = 3               # Frames before flagging lifting arm forward instead of backward
DEBOUNCE_ARM = 4
DEBOUNCE_TORSO = 4
DEBOUNCE_NECK = 4
DEBOUNCE_SHRUG = 4                   # Frames before flagging shoulder shrug/roll
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 3
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingShoulderExtensionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.target_extension_reached = False
        self.half_extension_reached = False
        self.max_extension = 0.0

        # Posture Tripwires (STRICT SIDE / SAGITTAL VIEW)
        self.is_side_view = True
        self.posterior_direction = True    # Step 4: Peel arms BACKWARD behind hips (rejects front raise)
        self.arms_straight = True          # Step 3: Elbows stay frozen and locked out
        self.torso_upright = True          # Step 4: Do not lean torso forward
        self.shoulders_depressed = True    # Step 2 & 4: Do not shrug or roll shoulders up toward ears
        self.neck_neutral = True           # Step 2: Keep spine and neck neutral
        self.controlled_return = True      # Step 6: Return slowly under control
        self.ml_posture_maintained = True

        self.baseline_torso_tilt = None
        self.baseline_ear_shoulder_ratio = None

        # Debounce counters
        self.profile_frames = 0
        self.direction_frames = 0
        self.arm_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.neck_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_extension = None
        self.current_feedback = "Stand sideways to camera: hold bar/weights behind hips with straight arms"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_extension.tflite")
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
        self.target_extension_reached = False
        self.half_extension_reached = False
        self.max_extension = 0.0
        self.reversal_frames = 0
        self.prev_extension = None

        self.is_side_view = True
        self.posterior_direction = True
        self.arms_straight = True
        self.torso_upright = True
        self.shoulders_depressed = True
        self.neck_neutral = True
        self.controlled_return = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.direction_frames = 0
        self.arm_frames = 0
        self.torso_frames = 0
        self.shrug_frames = 0
        self.neck_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Extension", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Full side profile not visible",
                "extension_angle": 0.0, "elbow_angle": 0.0, "depth_reached": False
            }

        use_right = (r_vis >= l_vis)
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_elbow = r_elbow if use_right else l_elbow
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip

        # 1. Sagittal Angles
        extension_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 2. Strict Side Profile Verification (16:9 Webcam Calibrated)
        torso_len = max(np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y), 0.35)
        shoulder_aspect = abs(l_shoulder.x - r_shoulder.x) / torso_len
        hip_aspect = abs(l_hip.x - r_hip.x) / torso_len
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        # 3. True Vertical Spine Inclination & Anti-Shrug Ratio
        dx_spine = active_shoulder.x - active_hip.x
        dy_spine = max(abs(active_shoulder.y - active_hip.y), 0.10)
        spine_tilt_signed_deg = np.degrees(np.arctan2(dx_spine, dy_spine))
        abs_torso_tilt_deg = abs(spine_tilt_signed_deg)
        is_upright_standing = (active_shoulder.y < active_hip.y) and (abs_torso_tilt_deg <= SETUP_MAX_TORSO_TILT_DEG)

        ear_shoulder_ratio = abs(active_ear.y - active_shoulder.y) / torso_len

        # 4. Step 4: Facing Direction & True Posterior Behind-the-Back Verification
        facing_right = nose.x > active_ear.x
        if facing_right:
            # Facing right (+X): Behind the back is toward the left (-X)
            posterior_wrist_drift = (active_hip.x - active_wrist.x) / torso_len
            posterior_elbow_drift = (active_shoulder.x - active_elbow.x) / torso_len
        else:
            # Facing left (-X): Behind the back is toward the right (+X)
            posterior_wrist_drift = (active_wrist.x - active_hip.x) / torso_len
            posterior_elbow_drift = (active_elbow.x - active_shoulder.x) / torso_len

        # Must genuinely move BEHIND the shoulder and hip (positive posterior drift!)
        is_peeling_backward = (
            (posterior_wrist_drift >= MIN_POSTERIOR_WRIST_DRIFT)
            and (posterior_elbow_drift >= MIN_POSTERIOR_ELBOW_DRIFT)
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Stand Tall Sideways, Arms Locked Down Behind Hips)
        # ---------------------------------------------------------
        is_standing_tall = (
            is_side
            and is_upright_standing
            and (extension_angle < SETUP_MAX_EXTENSION_ANGLE)
            and (elbow_angle >= SETUP_MIN_ELBOW_LOCKOUT)
        )

        if is_standing_tall:
            self.standing_frames += 1
            if self.baseline_torso_tilt is None:
                self.baseline_torso_tilt = spine_tilt_signed_deg
                self.baseline_ear_shoulder_ratio = ear_shoulder_ratio
            elif not self.rep_started and extension_angle < 18.0:
                self.baseline_torso_tilt = 0.90 * self.baseline_torso_tilt + 0.10 * spine_tilt_signed_deg
                self.baseline_ear_shoulder_ratio = 0.90 * self.baseline_ear_shoulder_ratio + 0.10 * ear_shoulder_ratio

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            elif not is_upright_standing:
                feedback_msg = "Stand tall: brace your core and glutes"
            elif elbow_angle < SETUP_MIN_ELBOW_LOCKOUT:
                feedback_msg = "Lock your elbows straight down behind your hips"
            else:
                feedback_msg = "Rest straight arms against the back of your thighs to begin"
            return {
                "exercise": "Shoulder Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "extension_angle": round(extension_angle, 1),
                "elbow_angle": round(elbow_angle, 1),
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
                        if self.ml_posture_maintained:
                            print(
                                f"[SHOULDER EXT FAULT] AI ST-GCN flagged posture fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES (ACTIVE ACROSS EXTENDING, PEAK, AND RETURNING)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["EXTENDING", "PEAK", "RETURNING"]:
            self.max_extension = max(self.max_extension, extension_angle)

            # A. Strict Side Profile Tripwire
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[SHOULDER EXT FAULT] Lost side view: ratio={shoulder_aspect:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Step 4: True Posterior Extension Check (Rejects lifting arms forward in front of thighs!)
            if extension_angle >= 26.0 and not is_peeling_backward:
                self.direction_frames += 1
                if self.direction_frames >= DEBOUNCE_DIRECTION:
                    if self.posterior_direction:
                        print(
                            f"[SHOULDER EXT FAULT] Arm moved forward instead of behind hips! "
                            f"wrist_drift={posterior_wrist_drift:.2f}, elbow_drift={posterior_elbow_drift:.2f}",
                            flush=True,
                        )
                    self.posterior_direction = False
            else:
                self.direction_frames = max(0, self.direction_frames - 1)

            # C. Step 3: Frozen Straight Elbows Check (Catches tricep kickback cheat)
            if elbow_angle < MIN_ELBOW_LOCKOUT_ANGLE:
                self.arm_frames += 1
                if self.arm_frames >= DEBOUNCE_ARM:
                    if self.arms_straight:
                        print(
                            f"[SHOULDER EXT FAULT] Elbow bent (kickback cheat): elbow={elbow_angle:.1f}° "
                            f"(min={MIN_ELBOW_LOCKOUT_ANGLE}°)",
                            flush=True,
                        )
                    self.arms_straight = False
            else:
                self.arm_frames = max(0, self.arm_frames - 1)

            # D. Step 4: Upright Torso Check (Catches leaning forward as arms peel back)
            torso_lean_delta = (
                abs(spine_tilt_signed_deg - self.baseline_torso_tilt)
                if self.baseline_torso_tilt is not None else abs_torso_tilt_deg
            )
            if torso_lean_delta > MAX_TORSO_FORWARD_LEAN_DEG or abs_torso_tilt_deg > (SETUP_MAX_TORSO_TILT_DEG + 8.0):
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_upright:
                        print(
                            f"[SHOULDER EXT FAULT] Torso leaned forward: delta={torso_lean_delta:.1f}° "
                            f"(max={MAX_TORSO_FORWARD_LEAN_DEG}°)",
                            flush=True,
                        )
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # E. Step 2 & 4: Shoulder Shrug / Anterior Roll Guard
            if (
                self.baseline_ear_shoulder_ratio is not None
                and ear_shoulder_ratio < (self.baseline_ear_shoulder_ratio * SHRUG_EAR_SHOULDER_FACTOR)
            ):
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    if self.shoulders_depressed:
                        print("[SHOULDER EXT FAULT] Shoulders shrugged/rolled upward toward ears!", flush=True)
                    self.shoulders_depressed = False
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # F. Step 2: Neutral Neck Check (Catches forward chin-poke cheat)
            if neck_angle < MIN_NECK_NEUTRAL_ANGLE:
                self.neck_frames += 1
                if self.neck_frames >= DEBOUNCE_NECK:
                    if self.neck_neutral:
                        print(f"[SHOULDER EXT FAULT] Forward head thrust: neck_angle={neck_angle:.1f}°", flush=True)
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FIXED STANDING -> EXTENDING -> PEAK -> RETURNING FLOW)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_tall:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Lock elbows and peel arms backward behind your hips."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Shoulder Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "extension_angle": round(extension_angle, 1),
                "elbow_angle": round(elbow_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep elbows frozen and peel arms backward behind your hips!"

        # Step 4: Initiate Posterior Peel (ONLY from STANDING so RETURNING never ping-pongs!)
        elif self.state == "STANDING" and extension_angle >= EXTENDING_START_ANGLE:
            self.state = "EXTENDING"
            self.rep_started = True
            self.max_extension = extension_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Peel arms backward and upward without leaning torso forward..."

        # Step 5: Hold the Peak (Extension >= 33° behind the back)
        elif self.state == "EXTENDING" and extension_angle >= FULL_ROM_MIN_EXTENSION and self.posterior_direction:
            self.target_extension_reached = True
            self.state = "PEAK"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Peak contraction! Squeeze upper back & triceps, then lower slowly."

        # Premature Reversal (Half Rep Branch -> Flagged as Bad Rep)
        elif self.state == "EXTENDING" and not self.target_extension_reached and (self.max_extension - extension_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_extension >= HALF_ROM_MIN_EXTENSION:
                    self.half_extension_reached = True
                self.state = "RETURNING"
                self.current_feedback = "Incomplete extension! Returning forward early..."
                print(
                    f"[SHOULDER EXT FAULT] Early reversal before peak extension: "
                    f"max_ext={self.max_extension:.1f}° (required>={FULL_ROM_MIN_EXTENSION}°)",
                    flush=True,
                )

        # Step 6: Lower Under Control from PEAK ONLY
        elif self.state == "PEAK" and extension_angle < RETURNING_START_ANGLE:
            self.state = "RETURNING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Slowly return arms to the back of your thighs..."

        if self.state == "EXTENDING" and not ((self.max_extension - extension_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 6: Eccentric drop detector
        if self.rep_started and self.state in ["RETURNING", "EXTENDING"]:
            if self.prev_extension is not None:
                drop_speed = self.prev_extension - extension_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_return:
                            print(
                                f"[SHOULDER EXT FAULT] Arms swung forward too fast: speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_return = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_extension = extension_angle
        else:
            self.prev_extension = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["RETURNING", "PEAK", "EXTENDING"] and extension_angle <= REP_COMPLETE_ANGLE:
            # Ignore minor resting twitches (< 26°) ONLY if posterior direction and side view were maintained
            if self.max_extension < MIN_VALID_REP_PEAK_ANGLE and self.is_side_view and self.posterior_direction:
                self._reset_rep_tracking()
                self.state = "STANDING"
                return {
                    "exercise": "Shoulder Extension", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "extension_angle": round(extension_angle, 1),
                    "elbow_angle": round(elbow_angle, 1),
                    "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Return arms to sides to reset."
                is_bad_rep = True
            elif not self.posterior_direction:
                self.current_feedback = "Bad rep: Peel your arms BACKWARD behind your hips (do not raise in front)!"
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "Bad rep: Lock your elbows straight! Do not bend into a tricep kickback."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Keep your torso upright! Do not lean forward as you push back."
                is_bad_rep = True
            elif not self.shoulders_depressed:
                self.current_feedback = "Bad rep: Keep your shoulders down and back! Do not shrug toward your ears."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "Bad rep: Keep your neck neutral! Do not poke your chin forward."
                is_bad_rep = True
            elif not self.controlled_return:
                self.current_feedback = "Bad rep: Control the return! Do not let your arms swing forward."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor posture or shoulder mechanics."
                is_bad_rep = True
            elif not self.target_extension_reached:
                if self.max_extension >= HALF_ROM_MIN_EXTENSION or self.half_extension_reached:
                    self.current_feedback = "Bad rep: Half rep! Peel your straight arms higher behind your back."
                else:
                    self.current_feedback = "Bad rep: Extension was too shallow! Push arms fully back."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SHOULDER EXT REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_ext={self.max_extension:.1f}°, elbow={elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Behind-the-Back Shoulder Extension! Great rear-delt and tricep lockout."
                print(
                    f"[SHOULDER EXT REP #{rep_num} -> GOOD REP] "
                    f"max_ext={self.max_extension:.1f}°, elbow={elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((extension_angle / 35) * 100))) if self.state != "STANDING" else 100
        if not self.is_side_view: form_score -= 35
        if not self.posterior_direction: form_score -= 40
        if not self.arms_straight: form_score -= 30
        if not self.torso_upright: form_score -= 35
        if not self.shoulders_depressed: form_score -= 20
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_return: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.target_extension_reached and self.state == "RETURNING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Extension",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "extension_angle": round(extension_angle, 1),
            "elbow_angle": round(elbow_angle, 1),
            "depth_reached": self.target_extension_reached
        }