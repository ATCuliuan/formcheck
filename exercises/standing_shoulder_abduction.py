import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SHOULDER ABDUCTION - FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 3: Front View, Arms at Sides, Soft Elbows)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.23 # Enforces front-facing view
SETUP_MAX_ABDUCTION_ANGLE = 34.0     # Step 1: Arms resting down at sides (loosened from 28° for natural stance)
SETUP_MIN_ELBOW_ANGLE = 125.0        # Step 3: Arms straight with comfortable soft bend
SETUP_MIN_READY_FRAMES = 6           # Frames holding resting stance
UNLOCK_RESET_FRAMES = 10             # Frames held with arms down to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35      # Minimum keypoint visibility

# 2. Kinematic & Form Thresholds (Both Arms Together, Soft Elbows Allowed, No Shrug, Lateral Plane)
MIN_ELBOW_STRAIGHT_ANGLE = 120.0     # Step 3: Allows natural soft bend & 2D scapular plane foreshortening
MAX_ASYMMETRY_DELTA = 26.0           # Max left vs right abduction difference (catches single-arm raise!)
SINGLE_ARM_TRIGGER_DEG = 42.0        # If one arm lifts >= 42° while the other stays < 32°, flags single-arm fault
MAX_TORSO_SWAY_RATIO = 0.15          # Step 1: Core engaged, no lateral torso sway
SHRUG_NECK_RATIO_FACTOR = 0.62       # Step 4: Pull shoulder blades down (catches upper-trap shrugging)
MIN_LATERAL_WRIST_SPAN_FACTOR = 1.35 # Step 4: Wrists must spread wider than shoulders (rejects front raise)
ECCENTRIC_DROP_SPEED_MAX = 26.0      # Step 6: Lower over a 2-3s count fighting gravity (deg/frame)
EARLY_REVERSAL_DELTA = 10.0          # Early downward drop before reaching shoulder height

# 3. Abduction Target Height & Range of Motion (Step 5: 90° Shoulder Height OR 180° Overhead Arc)
LIFTING_START_ABDUCTION = 32.0       # Raising either arm past 32° initiates LIFTING
MIN_VALID_REP_PEAK_ANGLE = 40.0      # Ignore minor resting arm twitches (< 40°)
HALF_ROM_MIN_ABDUCTION = 50.0        # Raising past 50° but short of 75° flags Half Rep (Bad Rep)
FULL_ROM_MIN_ABDUCTION = 75.0        # Step 5a: Standard ~90° shoulder-height target (>= 75° on both arms)
OVERHEAD_ROM_MIN_ABDUCTION = 145.0   # Step 5b: Full 180° overhead arc above crown of head
LOWERING_TRIGGER_DELTA = 12.0        # Dropping >= 12° from true peak initiates LOWERING
REP_COMPLETE_ABDUCTION = 35.0        # Step 6: Returning both arms down to thighs completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_STRAIGHT = 5
DEBOUNCE_ASYM = 4
DEBOUNCE_SWAY = 5
DEBOUNCE_SHRUG = 5
DEBOUNCE_PLANE = 5
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 6

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.78

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingShoulderAbductionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.target_height_reached = False
        self.overhead_arc_reached = False
        self.half_raise_reached = False
        self.max_abduction = 0.0           # Highest bilateral (min_abduction) peak
        self.max_single_arm_abduction = 0.0# Highest single-arm (max_abduction) peak

        # Posture Tripwires (STRICT FRONT / CORONAL VIEW)
        self.is_front_view = True
        self.both_arms_raised = True       # Enforces BOTH arms raising together!
        self.arms_straight = True          # Step 3: Keep arms straight with soft bend
        self.lateral_plane = True          # Step 4: Raise arms out to the sides (not front raise)
        self.shoulders_depressed = True    # Step 4: Pull shoulder blades down and back (no shrugging)
        self.torso_upright = True          # Step 1: Core engaged, tall spine
        self.controlled_descent = True     # Step 6: Lower slowly over 2-3 seconds
        self.ml_posture_maintained = True

        self.baseline_neck_ratio = None

        # Debounce counters
        self.front_frames = 0
        self.straight_frames = 0
        self.plane_frames = 0
        self.shrug_frames = 0
        self.asym_frames = 0
        self.sway_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_abduction = None
        self.current_feedback = "Face camera: stand tall with both arms at your sides"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_abduction.tflite")
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
        self.overhead_arc_reached = False
        self.half_raise_reached = False
        self.max_abduction = 0.0
        self.max_single_arm_abduction = 0.0
        self.reversal_frames = 0
        self.prev_abduction = None

        self.is_front_view = True
        self.both_arms_raised = True
        self.arms_straight = True
        self.lateral_plane = True
        self.shoulders_depressed = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.straight_frames = 0
        self.plane_frames = 0
        self.shrug_frames = 0
        self.asym_frames = 0
        self.sway_frames = 0
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

        vis_joints = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(vis_joints) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Abduction", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: Full upper body and both arms must be visible",
                "abduction_angle": 0.0, "elbow_angle": 0.0, "depth_reached": False
            }

        # 1. Bilateral Abduction & Elbow Angles
        l_abduction_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_abduction_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        min_abduction = min(l_abduction_angle, r_abduction_angle)
        max_abduction = max(l_abduction_angle, r_abduction_angle)
        arm_diff = abs(l_abduction_angle - r_abduction_angle)

        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow = min(l_elbow_angle, r_elbow_angle)

        # 2. Strict Front View & Scapular Shrug Metrics
        torso_height = max(abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0)), 0.30)
        raw_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_width / torso_height
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO

        shoulder_width = max(raw_shoulder_width, 0.22)
        wrist_span_x = abs(l_wrist.x - r_wrist.x)

        ear_shoulder_dist = (abs(l_ear.y - l_shoulder.y) + abs(r_ear.y - r_shoulder.y)) / 2.0
        neck_ratio = ear_shoulder_dist / torso_height

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Stand Tall Facing Camera, Arms at Sides)
        # ---------------------------------------------------------
        is_resting_at_sides = (
            is_facing_front
            and (max_abduction < SETUP_MAX_ABDUCTION_ANGLE)
            and (min_elbow >= SETUP_MIN_ELBOW_ANGLE)
        )

        if is_resting_at_sides:
            self.standing_frames += 1
            if self.baseline_neck_ratio is None:
                self.baseline_neck_ratio = neck_ratio
            elif not self.rep_started and max_abduction < 22.0:
                self.baseline_neck_ratio = 0.92 * self.baseline_neck_ratio + 0.08 * neck_ratio

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            elif min_elbow < SETUP_MIN_ELBOW_ANGLE:
                feedback_msg = "Straighten both arms down at your sides"
            else:
                feedback_msg = "Rest both arms down at your sides to begin"
            return {
                "exercise": "Shoulder Abduction", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "abduction_angle": round(min_abduction, 1),
                "elbow_angle": round(min_elbow, 1),
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
                                f"[SHOULDER ABDUCTION FAULT] AI ST-GCN flagged posture fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES (ACTIVE ACROSS LIFTING, PEAK, AND LOWERING)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LIFTING", "PEAK", "LOWERING"]:
            self.max_abduction = max(self.max_abduction, min_abduction)
            self.max_single_arm_abduction = max(self.max_single_arm_abduction, max_abduction)

            if min_abduction >= OVERHEAD_ROM_MIN_ABDUCTION and self.both_arms_raised:
                self.overhead_arc_reached = True

            # A. Strict Front View Check
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[SHOULDER ABDUCTION FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # B. Both Arms Must Abduct Together (Catches 1-arm raises or large left/right asymmetry!)
            is_single_arm_or_uneven = (
                (max_abduction >= SINGLE_ARM_TRIGGER_DEG and min_abduction < LIFTING_START_ABDUCTION)
                or (arm_diff > MAX_ASYMMETRY_DELTA)
            )
            if is_single_arm_or_uneven:
                self.asym_frames += 1
                if self.asym_frames >= DEBOUNCE_ASYM:
                    if self.both_arms_raised:
                        print(
                            f"[SHOULDER ABDUCTION FAULT] Single-arm or uneven raise: L={l_abduction_angle:.1f}°, "
                            f"R={r_abduction_angle:.1f}°",
                            flush=True,
                        )
                    self.both_arms_raised = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Raise BOTH arms evenly together!"
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            # C. Step 3: Keep Arms Straight (Comfortable Soft Bend Allowed >= 120°)
            if min_elbow < MIN_ELBOW_STRAIGHT_ANGLE:
                self.straight_frames += 1
                if self.straight_frames >= DEBOUNCE_STRAIGHT:
                    if self.arms_straight:
                        print(
                            f"[SHOULDER ABDUCTION FAULT] Elbows bent too much: min_elbow={min_elbow:.1f}° "
                            f"(min={MIN_ELBOW_STRAIGHT_ANGLE}°)",
                            flush=True,
                        )
                    self.arms_straight = False
            else:
                self.straight_frames = max(0, self.straight_frames - 1)

            # D. Step 4: True Lateral Abduction Plane (Uses wrist span at 65°-105° so front raises are caught accurately!)
            if 65.0 <= min_abduction <= 105.0 and self.both_arms_raised:
                if wrist_span_x < (shoulder_width * MIN_LATERAL_WRIST_SPAN_FACTOR):
                    self.plane_frames += 1
                    if self.plane_frames >= DEBOUNCE_PLANE:
                        if self.lateral_plane:
                            print("[SHOULDER ABDUCTION FAULT] Arms raised forward instead of out to the sides!", flush=True)
                        self.lateral_plane = False
                else:
                    self.plane_frames = max(0, self.plane_frames - 1)
            else:
                self.plane_frames = max(0, self.plane_frames - 1)

            # E. Step 4: Pull Shoulder Blades Down and Back (Anti-Shrug Guard below 105°)
            if max_abduction <= 105.0 and self.baseline_neck_ratio and neck_ratio < (self.baseline_neck_ratio * SHRUG_NECK_RATIO_FACTOR):
                self.shrug_frames += 1
                if self.shrug_frames >= DEBOUNCE_SHRUG:
                    if self.shoulders_depressed:
                        print("[SHOULDER ABDUCTION FAULT] Shoulders shrugged upward toward ears!", flush=True)
                    self.shoulders_depressed = False
            else:
                self.shrug_frames = max(0, self.shrug_frames - 1)

            # F. Step 1: Core Engaged / No Lateral Torso Sway
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            sway_ratio = abs(mid_shoulder_x - mid_hip_x) / torso_height
            if sway_ratio > MAX_TORSO_SWAY_RATIO:
                self.sway_frames += 1
                if self.sway_frames >= DEBOUNCE_SWAY:
                    if self.torso_upright:
                        print(f"[SHOULDER ABDUCTION FAULT] Torso swayed laterally: ratio={sway_ratio:.2f}", flush=True)
                    self.torso_upright = False
            else:
                self.sway_frames = max(0, self.sway_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (SUPPORTS BOTH 90° SHOULDER HEIGHT & 180° OVERHEAD ARC)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_at_sides:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Raise BOTH arms out to the sides."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Shoulder Abduction", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "abduction_angle": round(min_abduction, 1),
                "elbow_angle": round(min_elbow, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Turn palms forward and raise both arms out to shoulder height (or overhead)!"

        # Step 4: Initiate Abduction Upward (Triggers on max_abduction >= 32° so 1-arm raises are caught!)
        elif self.state == "STANDING" and max_abduction >= LIFTING_START_ABDUCTION:
            self.state = "LIFTING"
            self.rep_started = True
            self.max_abduction = min_abduction
            self.max_single_arm_abduction = max_abduction
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull shoulder blades down and raise both arms out to sides..."

        # Step 5: Target Height Reached (BOTH arms >= 75° shoulder height)
        elif self.state == "LIFTING" and min_abduction >= FULL_ROM_MIN_ABDUCTION and self.both_arms_raised:
            self.target_height_reached = True
            self.state = "PEAK"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Target height reached! Hold at 90° or continue to 180° overhead, then lower slowly."

        # Premature Reversal (Half Rep OR Single-Arm Raise returning down before both arms hit 75°)
        elif self.state == "LIFTING" and not self.target_height_reached and (
            (self.max_single_arm_abduction - max_abduction) > EARLY_REVERSAL_DELTA
        ):
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_abduction >= HALF_ROM_MIN_ABDUCTION:
                    self.half_raise_reached = True
                self.state = "LOWERING"
                if not self.both_arms_raised:
                    self.current_feedback = "One arm stayed down! Raise both arms together."
                else:
                    self.current_feedback = "Incomplete range: Lowering arms before shoulder height..."
                print(
                    f"[SHOULDER ABDUCTION FAULT] Early reversal: min_peak={self.max_abduction:.1f}°, "
                    f"max_peak={self.max_single_arm_abduction:.1f}°",
                    flush=True,
                )

        # Step 6: Debounced Descent from Peak (Allows smooth continuous ascent from 90° to 180° without false LOWERING!)
        elif self.state == "PEAK":
            if (self.max_abduction - min_abduction) >= LOWERING_TRIGGER_DELTA:
                self.reversal_frames += 1
                if self.reversal_frames >= 3:
                    self.state = "LOWERING"
                    self.reversal_frames = 0
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Lower slowly over 2-3 seconds, fighting gravity..."
            else:
                self.reversal_frames = max(0, self.reversal_frames - 1)

        if self.state == "LIFTING" and not ((self.max_single_arm_abduction - max_abduction) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 6: Eccentric drop detector
        if self.rep_started and self.state in ["LOWERING", "LIFTING"]:
            if self.prev_abduction is not None:
                drop_speed = self.prev_abduction - max_abduction
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[SHOULDER ABDUCTION FAULT] Uncontrolled arm drop: speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_abduction = max_abduction
        else:
            self.prev_abduction = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS & 1-ARM RAISES FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "PEAK", "LIFTING"] and max_abduction <= REP_COMPLETE_ABDUCTION:
            # Ignore minor resting twitches at thighs (< 40° on both arms)
            if self.max_single_arm_abduction < MIN_VALID_REP_PEAK_ANGLE and self.is_front_view and self.arms_straight:
                self._reset_rep_tracking()
                self.state = "STANDING"
                return {
                    "exercise": "Shoulder Abduction", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "abduction_angle": round(min_abduction, 1),
                    "elbow_angle": round(min_elbow, 1),
                    "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Rest arms at sides to reset."
                is_bad_rep = True
            elif not self.both_arms_raised:
                self.current_feedback = "Bad rep: Raise and lower BOTH arms evenly together!"
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "Bad rep: Keep your arms straight (only a soft bend in elbows)!"
                is_bad_rep = True
            elif not self.lateral_plane:
                self.current_feedback = "Bad rep: Abduct your arms straight out to the sides (not in front)!"
                is_bad_rep = True
            elif not self.shoulders_depressed:
                self.current_feedback = "Bad rep: Pull your shoulder blades down and back! Do not shrug."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Brace your core and keep your torso still! Do not sway."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower your arms slowly over 2-3 seconds! Fight gravity."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor posture or shoulder mechanics."
                is_bad_rep = True
            elif not self.target_height_reached:
                if self.max_abduction >= HALF_ROM_MIN_ABDUCTION or self.half_raise_reached:
                    self.current_feedback = "Bad rep: Half rep! Raise both arms at least to shoulder height (90°)."
                else:
                    self.current_feedback = "Bad rep: Raise was too shallow! Lift both arms to shoulder height."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SHOULDER ABDUCTION REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_peak={self.max_abduction:.1f}°, "
                    f"max_peak={self.max_single_arm_abduction:.1f}°, min_elbow={min_elbow:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.overhead_arc_reached:
                    self.current_feedback = "Perfect 180° Full-ROM Shoulder Abduction! Great control."
                else:
                    self.current_feedback = "Perfect 90° Shoulder Abduction! Clean scapular control."
                print(
                    f"[SHOULDER ABDUCTION REP #{rep_num} -> GOOD REP] "
                    f"max_abd={self.max_abduction:.1f}°, min_elbow={min_elbow:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((min_abduction / 75) * 100))) if self.state != "STANDING" else 100
        if not self.is_front_view: form_score -= 35
        if not self.both_arms_raised: form_score -= 40
        if not self.arms_straight: form_score -= 25
        if not self.lateral_plane: form_score -= 30
        if not self.shoulders_depressed: form_score -= 20
        if not self.torso_upright: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.target_height_reached and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Abduction",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "abduction_angle": round(min_abduction, 1),
            "elbow_angle": round(min_elbow, 1),
            "depth_reached": self.target_height_reached
        }