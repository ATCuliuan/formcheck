import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (ADJUST PARAMETERS HERE)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_ELBOW_MIN_ANGLE = 130.0       # Arm must hang extended down (> 130°)
SETUP_MIN_READY_FRAMES = 6          # Consecutive frames hanging arm to become READY
UNLOCK_RESET_FRAMES = 15            # Consecutive frames with arm hanging to clear LOCKED_FAULT (~0.5s)
BASELINE_EXPONENTIAL_ALPHA = 0.85   # Weight for baseline exponential moving average
BASELINE_NEW_SAMPLE_WEIGHT = 0.15   # Weight for new incoming baseline measurement

# 2. Kinematic & Form Thresholds
MAX_UPPER_ARM_ANGLE = 15.0          # Angle between torso and upper arm before flagging elbow flare/swing
MIN_VERTICAL_DROP_RATIO = 0.30      # Vertical drop of elbow below shoulder / upper arm length
SHRUG_EAR_DROP_RATIO = 0.85         # Ratio of baseline ear-to-shoulder height before flagging shrug
MAX_TORSO_SWAY = 0.75               # Horizontal shoulder-hip displacement / upper arm length
KNEE_BOUNCE_MAX_DEFLECTION = 14.0   # Knee angle change threshold for momentum/leg drive warning
ECCENTRIC_DROP_SPEED_MAX = 30.0     # Angular descent speed (deg/frame) for fast negative warning
MIN_VISIBILITY_THRESHOLD = 1.8      # Cumulative visibility score across key profile joints

# 3. Curl Depth & Range of Motion
FULL_CURL_PEAK_ANGLE = 75.0         # Maximum elbow angle required to credit a full contraction
HALF_CURL_THRESHOLD_ANGLE = 100.0   # If arm curls to <= 100° but misses 75°, it is flagged as half-rep failure
CURL_START_ELBOW_ANGLE = 125.0      # Moving below this angle transitions to CURLING
LOWERING_START_ELBOW_ANGLE = 80.0   # Moving above this angle transitions from TOP to LOWERING
EARLY_REVERSAL_DELTA = 5.0          # Upward bounce in degrees while curling to catch premature reversal
REP_COMPLETE_ELBOW_ANGLE = 130.0    # Arm extension angle required to evaluate rep completion

# 4. Debounce Frame Thresholds (Higher = Slower/More Deliberate detection)
DEBOUNCE_ELBOW_ELEVATION = 3        # Frames before flagging elbow forward swing/lift
DEBOUNCE_SHRUG = 3                  # Frames before flagging shoulder shrugging
DEBOUNCE_TORSO = 5                  # Frames before flagging torso lean-back / heave
DEBOUNCE_KNEE = 3                   # Frames before flagging knee momentum bounce
DEBOUNCE_DROP = 2                   # Frames before flagging fast/uncontrolled drop

# 5. Model & Buffer Settings
SEQUENCE_LENGTH = 30                # ST-GCN temporal frame buffer size

# =====================================================================


class DumbbellHammerCurlsTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.peak_contraction_reached = False
        self.min_elbow_angle = 180.0

        # Upper-Body Strict Tripwires (SIDE / SAGITTAL VIEW)
        self.elbow_stayed_down = True
        self.shoulder_not_shrugged = True
        self.torso_stable = True
        self.ml_posture_maintained = True

        # Advisory / Warning Flags
        self.fast_descent_warning = False
        self.knee_bounce_warning = False

        self.baseline_ear_shoulder_dist = None
        self.baseline_shoulder_hip_dist = None
        self.baseline_knee_angle = None

        # Debounce counters
        self.elevation_frames = 0
        self.shrug_frames = 0
        self.torso_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0

        self.prev_elbow_angle = None
        self.extended_frames = 0
        self.unlock_frames = 0
        self.is_ready = False
        self.current_feedback = "Show body sideways, let arm hang"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_hammer_curls.tflite")
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
        self.peak_contraction_reached = False
        self.min_elbow_angle = 180.0
        self.prev_elbow_angle = None

        self.elbow_stayed_down = True
        self.shoulder_not_shrugged = True
        self.torso_stable = True
        self.ml_posture_maintained = True
        self.knee_bounce_warning = False
        self.fast_descent_warning = False

        self.elevation_frames = 0
        self.shrug_frames = 0
        self.torso_frames = 0
        self.knee_frames = 0
        self.drop_frames = 0

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

        # Profile visibility check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Show your arm and torso (side view)",
                "elbow_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Joint angle calculations
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        upper_arm_len = np.sqrt((active_shoulder.x - active_elbow.x)**2 + (active_shoulder.y - active_elbow.y)**2)
        if upper_arm_len == 0:
            upper_arm_len = 0.25

        ear_shoulder_dist = abs(active_shoulder.y - active_ear.y)
        shoulder_hip_dist = abs(active_hip.y - active_shoulder.y)

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        arm_extended = elbow_angle > SETUP_ELBOW_MIN_ANGLE

        if arm_extended:
            self.extended_frames += 1
            if self.extended_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
                if self.baseline_ear_shoulder_dist is None:
                    self.baseline_ear_shoulder_dist = ear_shoulder_dist
                if self.baseline_shoulder_hip_dist is None:
                    self.baseline_shoulder_hip_dist = shoulder_hip_dist
                if self.baseline_knee_angle is None:
                    self.baseline_knee_angle = knee_angle
                else:
                    self.baseline_knee_angle = (
                        BASELINE_EXPONENTIAL_ALPHA * self.baseline_knee_angle
                        + BASELINE_NEW_SAMPLE_WEIGHT * knee_angle
                    )
        else:
            self.extended_frames = max(0, self.extended_frames - 1)

        # Allow execution to proceed once ready or if currently mid-rep
        if not self.is_ready and self.state not in ["EXTENDED", "CURLING", "TOP", "LOWERING", "LOCKED_FAULT"]:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Let arm hang down to start",
                "elbow_angle": round(elbow_angle, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["CURLING", "TOP", "LOWERING"]:
            # A. Shoulder-to-Elbow Elevation & Forward Upper Arm Swing
            vertical_drop_ratio = (active_elbow.y - active_shoulder.y) / upper_arm_len
            is_elevating = (upper_arm_angle > MAX_UPPER_ARM_ANGLE) or (vertical_drop_ratio < MIN_VERTICAL_DROP_RATIO)

            if is_elevating:
                self.elevation_frames += 1
                if self.elevation_frames >= DEBOUNCE_ELBOW_ELEVATION:
                    self.elbow_stayed_down = False
            else:
                self.elevation_frames = max(0, self.elevation_frames - 1)

            # B. Shoulder Shrug Check
            if self.baseline_ear_shoulder_dist is not None:
                ear_compressed = ear_shoulder_dist < (self.baseline_ear_shoulder_dist * SHRUG_EAR_DROP_RATIO)
                if ear_compressed:
                    self.shrug_frames += 1
                    if self.shrug_frames >= DEBOUNCE_SHRUG:
                        self.shoulder_not_shrugged = False
                else:
                    self.shrug_frames = max(0, self.shrug_frames - 1)

            # C. Torso Lean-Back / Spine Heave
            torso_sway = abs(active_shoulder.x - active_hip.x) / upper_arm_len
            if torso_sway > MAX_TORSO_SWAY:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    self.torso_stable = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            # D. Knee Bounce / Leg-Drive Momentum (Advisory)
            if self.baseline_knee_angle is not None:
                knee_deflection = abs(knee_angle - self.baseline_knee_angle)
                if knee_deflection > KNEE_BOUNCE_MAX_DEFLECTION:
                    self.knee_frames += 1
                    if self.knee_frames >= DEBOUNCE_KNEE:
                        self.knee_bounce_warning = True
                else:
                    self.knee_frames = max(0, self.knee_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH LOCKED_FAULT FREEZE)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if arm_extended:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "EXTENDED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Curl the weight up."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1)
            }

        # Setup to Initial Extended Transition
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Curl the weight up!"

        # Curl initiation
        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < CURL_START_ELBOW_ANGLE:
            self.state = "CURLING"
            self.rep_started = True
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.current_feedback = "Curl up to your chest..."

        # Full peak contraction reached (<= 75°)
        elif self.state == "CURLING" and elbow_angle <= FULL_CURL_PEAK_ANGLE:
            self.state = "TOP"
            self.peak_contraction_reached = True
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.current_feedback = "Good squeeze! Lower the weight."

        # Premature reversal detection while curling: arm stopped before 75° and reversed downward
        elif self.state == "CURLING" and not self.peak_contraction_reached and (elbow_angle - self.min_elbow_angle) > EARLY_REVERSAL_DELTA:
            self.state = "LOWERING"
            self.current_feedback = "Incomplete curl! Lowering arm early..."

        # Normal descent transition from TOP
        elif self.state == "TOP" and elbow_angle > LOWERING_START_ELBOW_ANGLE:
            self.state = "LOWERING"
            self.current_feedback = "Lower arm down..."

        # Track minimum angle while in curling phase
        if self.state in ["CURLING", "TOP"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # Eccentric Free-Fall Monitor (Advisory)
        if self.state in ["LOWERING", "CURLING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = elbow_angle - self.prev_elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        self.fast_descent_warning = True
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "TOP", "CURLING"] and self.rep_started and elbow_angle >= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False

            # Strict Form Tripwires (Form faults first)
            if not self.elbow_stayed_down:
                self.current_feedback = "Bad rep: Keep elbows pinned down! Return arm down to reset."
                is_bad_rep = True
            elif not self.shoulder_not_shrugged:
                self.current_feedback = "Bad rep: Relax your traps! Return arm down to reset."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "Bad rep: Do not heave or lean back! Return arm down to reset."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected arm mechanics breakdown."
                is_bad_rep = True

            # Incomplete Rep & Half Rep Check (Flagged as Bad Rep)
            elif not self.peak_contraction_reached:
                if self.min_elbow_angle <= HALF_CURL_THRESHOLD_ANGLE:
                    self.current_feedback = "Bad rep: Half rep! Curled to ~100°, squeeze to under 75°. Return arm down to reset."
                else:
                    self.current_feedback = "Bad rep: Movement was too shallow! Return arm down to reset."
                is_bad_rep = True

            # Rep Evaluation
            if is_bad_rep:
                self.bad_reps += 1
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                if self.knee_bounce_warning:
                    self.current_feedback = "Good rep! Reminder: Keep knees steady, avoid leg drive."
                elif self.fast_descent_warning:
                    self.current_feedback = "Good rep! Reminder: Lower slower for muscle control."
                else:
                    self.current_feedback = "Great hammer curl! Clean rep."
                self.state = "EXTENDED"
                self._reset_rep_tracking()

            # Baseline updates on rep completion
            if self.baseline_ear_shoulder_dist is not None:
                self.baseline_ear_shoulder_dist = (
                    BASELINE_EXPONENTIAL_ALPHA * self.baseline_ear_shoulder_dist
                    + BASELINE_NEW_SAMPLE_WEIGHT * ear_shoulder_dist
                )
            if self.baseline_shoulder_hip_dist is not None:
                self.baseline_shoulder_hip_dist = (
                    BASELINE_EXPONENTIAL_ALPHA * self.baseline_shoulder_hip_dist
                    + BASELINE_NEW_SAMPLE_WEIGHT * shoulder_hip_dist
                )

        # Dynamic Scoring
        form_score = max(0, min(100, int(((150 - elbow_angle) / 80) * 100)))
        if not self.elbow_stayed_down:
            form_score -= 30
        if not self.shoulder_not_shrugged:
            form_score -= 20
        if not self.torso_stable:
            form_score -= 20
        if self.knee_bounce_warning:
            form_score -= 10
        if self.fast_descent_warning:
            form_score -= 8
        if not self.peak_contraction_reached and self.rep_started:
            form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Hammer Curls",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1)
        }