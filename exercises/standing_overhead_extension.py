import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (STANDING OVERHEAD EXTENSION - FRONT VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Stand Tall Facing Camera, Arms Straight Overhead)
SETUP_MIN_SHOULDER_SPAN_RATIO = 0.24 # Enforces strict front-facing view
SETUP_MIN_OVERHEAD_ELEVATION = 128.0 # Step 2: Upper arms pointing overhead (calibrated for 2D front view)
SETUP_MIN_ELBOW_LOCKOUT = 142.0      # Step 2: Arms extended straight overhead at start
SETUP_MAX_WRIST_GRIP_RATIO = 0.90    # Step 1: Hands cupping dumbbell together (or close grip)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames overhead to confirm READY
UNLOCK_RESET_FRAMES = 10             # Frames held in overhead lockout to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 0.35      # Minimum core joint visibility (shoulders, elbows, hips)

# 2. Kinematic & Form Thresholds (Step 3 & 4: Core Braced, Upper Arms Vertical, Elbows Tucked)
MIN_UPPER_ARM_ELEVATION = 115.0      # Step 4: Allows tight elbow tuck by ears in 2D front view
MAX_ELBOW_DROP_BELOW_SHOULDER = -0.02# Elbows must stay above shoulders (l_elbow.y < l_shoulder.y)
MAX_ELBOW_FLARE_RATIO = 1.52         # Step 4: Elbow span vs shoulder width (catches chicken-wing flare)
MAX_WRIST_SPREAD_RATIO = 1.05        # Rejects wide dumbbell shoulder press
MAX_ARM_ASYMMETRY_DEG = 32.0         # Checked only when wrists are unoccluded above head
MAX_TORSO_SWAY_RATIO = 0.14          # Step 3: Brace core & glutes (prevents lateral sway)
ECCENTRIC_DROP_SPEED_MAX = 26.0      # Step 4: Lower weight under control behind head (deg/frame)
EARLY_REVERSAL_DELTA = 9.0           # Early upward press before reaching full triceps stretch

# 3. Triceps Stretch & Lockout Range of Motion (Step 4, 5, 6)
LOWERING_START_ELBOW_ANGLE = 136.0   # Bending elbows behind head initiates LOWERING
MIN_VALID_REP_FLEXION_ANGLE = 126.0  # Ignore minor overhead holding twitches (> 126°)
HALF_STRETCH_MAX_ELBOW_ANGLE = 112.0 # Stopping between 88° and 112° flags Half Rep (Bad Rep)
FULL_STRETCH_MAX_ELBOW_ANGLE = 88.0  # Step 4: Deep stretch behind head (<= 88° or wrists drop behind neck)
PRESSING_START_ELBOW_ANGLE = 96.0    # Step 5: Driving weight back up from BOTTOM_STRETCH
REP_COMPLETE_ELBOW_ANGLE = 144.0     # Step 6: Full overhead triceps lockout (uses avg_elbow consistently!)

# 4. Debounce Frame Thresholds
DEBOUNCE_FRONT = 4
DEBOUNCE_OVERHEAD = 5
DEBOUNCE_FLARE = 5
DEBOUNCE_SYMMETRY = 5
DEBOUNCE_TORSO = 5
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.78

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class StandingOverheadExtensionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.full_stretch_reached = False  # Step 4: Deep stretch behind head
        self.was_half_rep = False
        self.min_elbow = 180.0

        # Posture Tripwires (STRICT FRONT VIEW)
        self.is_front_view = True
        self.arms_overhead = True          # Step 4: Upper arms stay vertical close to ears
        self.elbows_not_flared = True      # Step 4: Elbows tucked in (no outward flare)
        self.grip_maintained = True        # Step 1: Diamond/close grip maintained
        self.is_symmetrical = True         # Even drive with both arms
        self.torso_upright = True          # Step 3: Core & glutes braced, no torso sway
        self.controlled_descent = True     # Step 4: Slowly lower behind head under control
        self.ml_posture_maintained = True

        self.baseline_mid_hip_x = None

        # Debounce counters
        self.front_frames = 0
        self.overhead_frames = 0
        self.flare_frames = 0
        self.grip_frames = 0
        self.symmetry_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.setup_frames = 0
        self.unlock_frames = 0
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Face camera: cup dumbbell overhead with arms straight"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = SEQUENCE_LENGTH
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

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.full_stretch_reached = False
        self.was_half_rep = False
        self.min_elbow = 180.0
        self.reversal_frames = 0
        self.prev_elbow_angle = None

        self.is_front_view = True
        self.arms_overhead = True
        self.elbows_not_flared = True
        self.grip_maintained = True
        self.is_symmetrical = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.front_frames = 0
        self.overhead_frames = 0
        self.flare_frames = 0
        self.grip_frames = 0
        self.symmetry_frames = 0
        self.torso_frames = 0
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

        # Exclude wrists from the hard visibility gate so hands passing behind the head never reset the tracker!
        core_vis_joints = [
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_hip.visibility, r_hip.visibility
        ]
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if min(core_vis_joints) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Overhead Extension", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0,
                "feedback": "Step back: Show full torso and overhead arms",
                "elbow_angle": 0.0, "depth_reached": False
            }

        # 1. Front View & Body Scale Geometry
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0
        mid_ear_y = (l_ear.y + r_ear.y) / 2.0

        torso_height = max(abs(mid_shoulder_y - mid_hip_y), 0.30)
        raw_shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        shoulder_span_ratio = raw_shoulder_width / torso_height
        is_facing_front = shoulder_span_ratio >= SETUP_MIN_SHOULDER_SPAN_RATIO

        shoulder_width = max(raw_shoulder_width, 0.22)

        # 2. Occlusion-Safe Bilateral Overhead & Elbow Kinematics
        l_overhead_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_overhead_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_overhead_elevation = (l_overhead_angle + r_overhead_angle) / 2.0
        elbows_above_shoulders = (
            (l_elbow.y < l_shoulder.y + MAX_ELBOW_DROP_BELOW_SHOULDER)
            and (r_elbow.y < r_shoulder.y + MAX_ELBOW_DROP_BELOW_SHOULDER)
        )

        l_wrist_vis = l_wrist.visibility >= 0.35
        r_wrist_vis = r_wrist.visibility >= 0.35
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        if l_wrist_vis and r_wrist_vis:
            avg_elbow = (l_elbow_angle + r_elbow_angle) / 2.0
            min_elbow = min(l_elbow_angle, r_elbow_angle)
        elif l_wrist_vis:
            avg_elbow = l_elbow_angle
            min_elbow = l_elbow_angle
        elif r_wrist_vis:
            avg_elbow = r_elbow_angle
            min_elbow = r_elbow_angle
        else:
            avg_elbow = (l_elbow_angle + r_elbow_angle) / 2.0
            min_elbow = min(l_elbow_angle, r_elbow_angle)

        # In 2D front view, when hands drop behind the head/neck (wrist.y >= ear/neck level while elbows stay high),
        # confirm deep triceps stretch even if 2D foreshortening distorts the raw angle
        wrists_behind_neck = (
            elbows_above_shoulders
            and ((l_wrist.y >= mid_ear_y and r_wrist.y >= mid_ear_y) and avg_elbow <= 98.0)
        )

        # 3. Step 1 & 4: Diamond/Close Grip & Elbow Tuck Checks
        elbow_span_x = abs(l_elbow.x - r_elbow.x)
        is_flared = elbow_span_x > (shoulder_width * MAX_ELBOW_FLARE_RATIO)

        wrist_span_x = abs(l_wrist.x - r_wrist.x)
        is_close_grip = wrist_span_x <= (shoulder_width * SETUP_MAX_WRIST_GRIP_RATIO)

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1 & 2: Front View, Arms Locked Overhead)
        # ---------------------------------------------------------
        is_standing_overhead = (
            is_facing_front
            and elbows_above_shoulders
            and (avg_overhead_elevation >= SETUP_MIN_OVERHEAD_ELEVATION)
            and (avg_elbow >= SETUP_MIN_ELBOW_LOCKOUT)
            and (l_wrist.y < l_elbow.y)
            and (r_wrist.y < r_elbow.y)
            and is_close_grip
        )

        if is_standing_overhead:
            self.setup_frames += 1
            if self.baseline_mid_hip_x is None:
                self.baseline_mid_hip_x = mid_hip_x
            else:
                if not self.rep_started:
                    self.baseline_mid_hip_x = 0.90 * self.baseline_mid_hip_x + 0.10 * mid_hip_x

            if self.setup_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.setup_frames = max(0, self.setup_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_facing_front:
                feedback_msg = "Face the camera directly (front view required)"
            elif not is_close_grip:
                feedback_msg = "Cup dumbbell with both hands (or use a close grip)"
            else:
                feedback_msg = "Press weight completely overhead with arms straight to begin"
            return {
                "exercise": "Standing Overhead Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(avg_elbow, 1), "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER & INFERENCE
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["OVERHEAD_TOP", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[OVERHEAD EXT FAULT] AI ST-GCN flagged posture/arch fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "BOTTOM_STRETCH", "PRESSING"]:
            self.min_elbow = min(self.min_elbow, avg_elbow)

            # A. Front View Check
            if not is_facing_front:
                self.front_frames += 1
                if self.front_frames >= DEBOUNCE_FRONT:
                    if self.is_front_view:
                        print(f"[OVERHEAD EXT FAULT] Lost front view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_front_view = False
            else:
                self.front_frames = max(0, self.front_frames - 1)

            # B. Step 4: Upper Arms Still & Pointing Overhead (Elbows must stay above shoulders!)
            if (avg_overhead_elevation < MIN_UPPER_ARM_ELEVATION) or (not elbows_above_shoulders):
                self.overhead_frames += 1
                if self.overhead_frames >= DEBOUNCE_OVERHEAD:
                    if self.arms_overhead:
                        print(
                            f"[OVERHEAD EXT FAULT] Upper arms dropped down: elevation={avg_overhead_elevation:.1f}° "
                            f"(min={MIN_UPPER_ARM_ELEVATION}°)",
                            flush=True,
                        )
                    self.arms_overhead = False
            else:
                self.overhead_frames = max(0, self.overhead_frames - 1)

            # C. Step 4: Elbows Tucked Close to Ears (No Chicken-Wing Flare)
            if is_flared:
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_FLARE:
                    if self.elbows_not_flared:
                        print(
                            f"[OVERHEAD EXT FAULT] Elbows flared outward: span={elbow_span_x:.2f}, "
                            f"shoulder_w={shoulder_width:.2f}",
                            flush=True,
                        )
                    self.elbows_not_flared = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # D. Step 1: Close/Diamond Grip Maintained (Only checked when wrists are visible)
            if (l_wrist_vis and r_wrist_vis) and wrist_span_x > (shoulder_width * MAX_WRIST_SPREAD_RATIO):
                self.grip_frames += 1
                if self.grip_frames >= DEBOUNCE_FLARE:
                    if self.grip_maintained:
                        print("[OVERHEAD EXT FAULT] Hands separated too wide (shoulder press motion)!", flush=True)
                    self.grip_maintained = False
            else:
                self.grip_frames = max(0, self.grip_frames - 1)

            # E. Bilateral Symmetry Check (Only checked when both wrists are clearly visible and not occluded behind head)
            if (l_wrist_vis and r_wrist_vis) and (not wrists_behind_neck) and abs(l_elbow_angle - r_elbow_angle) > MAX_ARM_ASYMMETRY_DEG:
                self.symmetry_frames += 1
                if self.symmetry_frames >= DEBOUNCE_SYMMETRY:
                    if self.is_symmetrical:
                        print(
                            f"[OVERHEAD EXT FAULT] Asymmetric arm extension: L={l_elbow_angle:.1f}°, R={r_elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.is_symmetrical = False
            else:
                self.symmetry_frames = max(0, self.symmetry_frames - 1)

            # F. Step 3: Anchor Spine & Core (Removed buggy torso_arch_shrink scapula drop trap!)
            torso_lateral_sway = abs(mid_shoulder_x - mid_hip_x) / torso_height
            hip_shift = abs(mid_hip_x - self.baseline_mid_hip_x) / torso_height if self.baseline_mid_hip_x is not None else 0.0
            if torso_lateral_sway > MAX_TORSO_SWAY_RATIO or hip_shift > 0.16:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_upright:
                        print(
                            f"[OVERHEAD EXT FAULT] Torso or hips swayed: sway={torso_lateral_sway:.2f}, "
                            f"hip_shift={hip_shift:.2f}",
                            flush=True,
                        )
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE & DEBOUNCED REVERSAL)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_standing_overhead:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "OVERHEAD_TOP"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep upper arms tucked by ears and lower behind head."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Standing Overhead Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                "total_reps": total_reps, "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(avg_elbow, 1), "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "OVERHEAD_TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep upper arms by ears and lower weight behind your head!"

        # Step 4: Initiate Lowering Behind Head (ONLY from OVERHEAD_TOP to prevent ping-pong!)
        elif self.state == "OVERHEAD_TOP" and avg_elbow < LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            self.rep_started = True
            self.min_elbow = avg_elbow
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower slowly behind your head until forearms touch biceps..."

        # Step 4 Complete: Full Stretch Reached (avg_elbow <= 88° OR wrists dropped behind neck)
        elif self.state == "LOWERING" and (avg_elbow <= FULL_STRETCH_MAX_ELBOW_ANGLE or wrists_behind_neck):
            self.full_stretch_reached = True
            self.min_elbow = min(self.min_elbow, FULL_STRETCH_MAX_ELBOW_ANGLE)
            self.state = "BOTTOM_STRETCH"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Full triceps stretch! Drive weight straight back up."

        # Premature Reversal (Half Rep Branch: Pressing back up before full stretch)
        elif self.state == "LOWERING" and not self.full_stretch_reached and (avg_elbow - self.min_elbow) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.min_elbow <= HALF_STRETCH_MAX_ELBOW_ANGLE:
                    self.was_half_rep = True
                self.state = "PRESSING"
                self.current_feedback = "Incomplete stretch! Pressing up early..."
                print(
                    f"[OVERHEAD EXT FAULT] Early reversal before full stretch: min_elbow={self.min_elbow:.1f}° "
                    f"(required<={FULL_STRETCH_MAX_ELBOW_ANGLE}°)",
                    flush=True,
                )

        # Step 5: Drive Upward from BOTTOM_STRETCH
        elif self.state == "BOTTOM_STRETCH" and avg_elbow > PRESSING_START_ELBOW_ANGLE:
            self.state = "PRESSING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Contract triceps and lock out completely overhead!"

        if self.state == "LOWERING" and not ((avg_elbow - self.min_elbow) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 4: Eccentric drop speed detector
        if self.rep_started and self.state in ["LOWERING", "PRESSING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - avg_elbow
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[OVERHEAD EXT FAULT] Dropped weight too fast behind neck: speed={drop_speed:.1f} deg/frame",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = avg_elbow
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (Uses avg_elbow >= 144.0° Consistently!)
        # ---------------------------------------------------------
        if (
            self.rep_started
            and self.state in ["PRESSING", "BOTTOM_STRETCH", "LOWERING"]
            and avg_elbow >= REP_COMPLETE_ELBOW_ANGLE
            and (l_wrist.y < l_elbow.y and r_wrist.y < r_elbow.y)
        ):
            # Ignore minor overhead holding twitches (> 126°)
            if self.min_elbow > MIN_VALID_REP_FLEXION_ANGLE and not self.full_stretch_reached and self.is_front_view and self.arms_overhead:
                self._reset_rep_tracking()
                self.state = "OVERHEAD_TOP"
                return {
                    "exercise": "Standing Overhead Extension", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
                    "total_reps": total_reps, "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(avg_elbow, 1), "depth_reached": False
                }

            is_bad_rep = False

            if not self.is_front_view:
                self.current_feedback = "Bad rep: Face the camera directly! Lock out overhead to reset."
                is_bad_rep = True
            elif not self.arms_overhead:
                self.current_feedback = "Bad rep: Keep upper arms pointing straight up close to your ears!"
                is_bad_rep = True
            elif not self.elbows_not_flared:
                self.current_feedback = "Bad rep: Keep your elbows tucked in! Do not flare them outward."
                is_bad_rep = True
            elif not self.grip_maintained:
                self.current_feedback = "Bad rep: Keep hands together cupping the dumbbell! Do not press wide."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Brace your core and glutes! Do not arch or sway your back."
                is_bad_rep = True
            elif not self.is_symmetrical:
                self.current_feedback = "Bad rep: Bend and extend both arms evenly together!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower the weight slowly under control behind your head!"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged lower-back arch or posture breakdown."
                is_bad_rep = True
            elif self.was_half_rep or not self.full_stretch_reached:
                if self.min_elbow <= HALF_STRETCH_MAX_ELBOW_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Lower behind head until forearms touch biceps."
                else:
                    self.current_feedback = "Bad rep: Range was too shallow! Lower deeply behind your head."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[OVERHEAD EXT REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow={self.min_elbow:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect Overhead Triceps Extension! Full stretch and lockout."
                print(
                    f"[OVERHEAD EXT REP #{rep_num} -> GOOD REP] "
                    f"min_elbow={self.min_elbow:.1f}°",
                    flush=True,
                )
                self.state = "OVERHEAD_TOP"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = 100 if self.state == "OVERHEAD_TOP" else max(0, min(100, int(100 - abs(avg_elbow - 75) * 0.5)))
        if not self.is_front_view: form_score -= 35
        if not self.arms_overhead: form_score -= 30
        if not self.elbows_not_flared: form_score -= 30
        if not self.grip_maintained: form_score -= 25
        if not self.torso_upright: form_score -= 25
        if not self.is_symmetrical: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if not self.full_stretch_reached and self.state == "PRESSING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Overhead Extension", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, "half_reps": self.half_reps,
            "total_reps": total_reps, "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(avg_elbow, 1), "depth_reached": self.full_stretch_reached
        }