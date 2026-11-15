import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (PLANK TRICEPS EXTENSION - SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1 & 2: Forearm Plank, Elbows Under Shoulders, Rigid Line)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.35 # Enforces strict side profile view
SETUP_MAX_ELBOW_ANGLE = 118.0        # Forearms on floor, elbows bent ~90° at start
SETUP_MIN_ELBOW_ANGLE = 60.0         # Prevents collapsing chest flat onto hands
SETUP_MAX_UPPER_ARM_TILT_DEG = 32.0  # Elbows stacked directly under shoulders (Step 1)
SETUP_MAX_FOREARM_SLOPE = 0.10       # Forearms resting flat on floor (elbow.y ~ wrist.y)
SETUP_BODY_MIN_ANGLE = 152.0         # Rigid straight line from head/shoulders to heels (Step 2)
SETUP_KNEE_MIN_ANGLE = 148.0         # Legs extended straight on toes (knees off floor)
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames in forearm plank to confirm READY
UNLOCK_RESET_FRAMES = 12             # Frames held in forearm plank to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.4       # Minimum side-profile landmark visibility sum[cite: 9]

# 2. Kinematic & Form Thresholds (Step 2, 4, 5: Straight Body, Elbows Tucked, Controlled Lower)
MIN_KNEE_STRAIGHT_ANGLE = 142.0      # Catches knees bending / touching floor
MIN_CORE_SAG_BODY_ANGLE = 145.0      # Hips sagging below the shoulder-to-ankle line
MIN_CORE_PIKE_BODY_ANGLE = 142.0     # Hips piking upward into an A-frame
MIN_NECK_NEUTRAL_ANGLE = 100.0       # Keep neck neutral in line with spine
MAX_ELBOW_FLARE_SPAN_RATIO = 0.34    # Step 4: Keep elbows tucked (prevents flaring elbows out wide)
ECCENTRIC_DROP_SPEED_MAX = 24.0      # Step 5: Lower with control (max deg/frame drop speed)

# 3. Press & Return Range of Motion (Step 3 -> Step 5)
PRESSING_START_ELBOW_ANGLE = 115.0   # Extending elbows off the floor initiates PRESSING_UP
MIN_VALID_REP_PEAK_ANGLE = 122.0     # Ignore tiny forearm shifts (< 122°)
HALF_REP_MIN_ELBOW_ANGLE = 125.0     # Stopping between 125° and 148° flags a Half Rep (Bad Rep)
FULL_LOCKOUT_MIN_ELBOW_ANGLE = 148.0 # Step 3: Arms straight at top lockout
LOWERING_TRIGGER_DELTA = 10.0        # Bending elbows from peak initiates LOWERING
REP_COMPLETE_ELBOW_ANGLE = 112.0     # Step 5: Forearms returning to floor completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 3                 # Frames before flagging lost side view
DEBOUNCE_KNEE_GROUND = 4             # Frames before flagging knees dropping to floor
DEBOUNCE_SAG = 4                     # Frames before flagging hip sag
DEBOUNCE_PIKE = 4                    # Frames before flagging hip pike
DEBOUNCE_NECK = 5                    # Frames before flagging neck craning/dropping
DEBOUNCE_FLARE = 4                   # Frames before flagging elbow flare
DEBOUNCE_DROP = 2                    # Frames before flagging uncontrolled drop to forearms[cite: 9]
DEBOUNCE_HALF_REP = 4                # Frames dropping before confirming premature lowering
DEBOUNCE_ML_FAULT = 5                # Consecutive frames for ST-GCN fault[cite: 9]

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 # Softmax confidence threshold[cite: 9]

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal window[cite: 9]
POST_REP_FEEDBACK_FRAMES = 40        # Post-rep feedback hold frames[cite: 9]

# =====================================================================


class PlankTricepsExtensionTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept at 0 for API compatibility (half reps count as bad reps)[cite: 9]

        self.rep_started = False
        self.lockout_reached = False
        self.half_press_reached = False
        self.max_elbow_angle = 0.0

        # Posture Tripwires (SIDE / SAGITTAL PROFILE)
        self.is_side_view = True           # Enforces strict side profile
        self.knees_off_floor = True        # Step 2: Stay on toes, knees off floor[cite: 9]
        self.core_maintained = True        # Step 2 & 5: No hip sagging[cite: 9]
        self.no_pike = True                # Step 2 & 5: No hip piking[cite: 9]
        self.neck_neutral = True           # Step 2: Head to heels straight line[cite: 9]
        self.elbows_tucked = True          # Step 4: Keep elbows tucked close to sides
        self.controlled_descent = True     # Step 5: Lower forearms with control[cite: 9]
        self.ml_posture_maintained = True  # ST-GCN posture check[cite: 9]

        # Debounce counters
        self.profile_frames = 0
        self.knee_ground_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.neck_frames = 0
        self.flare_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.plank_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_elbow_angle = None
        self.current_feedback = "Set up in a forearm plank sideways to camera (elbows under shoulders)"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_plank_triceps_extension.tflite")
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
        self.half_press_reached = False
        self.max_elbow_angle = 0.0
        self.prev_elbow_angle = None
        self.reversal_frames = 0

        self.is_side_view = True
        self.knees_off_floor = True
        self.core_maintained = True
        self.no_pike = True
        self.neck_neutral = True
        self.elbows_tucked = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.knee_ground_frames = 0
        self.sag_frames = 0
        self.pike_frames = 0
        self.neck_frames = 0
        self.flare_frames = 0
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

        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Plank Triceps Extension", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile on the floor",
                "elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal Joint Angles[cite: 9]
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 1. Strict Side Profile & Horizontal Plank Orientation Check
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_ratio = abs(l_shoulder.x - r_shoulder.x) / torso_len
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        dy_body = abs(active_ankle.y - active_shoulder.y)
        dx_body = abs(active_ankle.x - active_shoulder.x)
        is_horizontal = dx_body > (dy_body * 0.90)

        # 2. Step 1: Forearm Plank Geometry (Elbows under shoulders, forearms flat on floor)
        dx_upper_arm = abs(active_shoulder.x - active_elbow.x)
        dy_upper_arm = max(abs(active_elbow.y - active_shoulder.y), 0.05)
        upper_arm_tilt_deg = np.degrees(np.arctan2(dx_upper_arm, dy_upper_arm))
        elbows_under_shoulders = upper_arm_tilt_deg <= SETUP_MAX_UPPER_ARM_TILT_DEG

        forearm_flat_on_floor = abs(active_elbow.y - active_wrist.y) <= SETUP_MAX_FOREARM_SLOPE
        is_forearm_bent = (SETUP_MIN_ELBOW_ANGLE <= elbow_angle <= SETUP_MAX_ELBOW_ANGLE)

        # 3. Step 2: Knees Off Floor & True Shoulder-to-Ankle Line Deviation (Sag vs Pike)
        is_knee_on_floor = (knee_angle < MIN_KNEE_STRAIGHT_ANGLE) or (active_knee.y > max(active_hip.y, active_ankle.y) + 0.03)
        expected_hip_y = (active_shoulder.y + active_ankle.y) / 2.0
        hip_vertical_deviation = active_hip.y - expected_hip_y  # Positive = sagging toward floor, Negative = piking up

        # 4. Step 4: Elbows Tucked Check (Prevents wide outward elbow flare)
        elbow_span_ratio = abs(l_elbow.x - r_elbow.x) / torso_len
        is_elbow_flaring = elbow_span_ratio > MAX_ELBOW_FLARE_SPAN_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1 & 2: Forearm Plank Position)
        # ---------------------------------------------------------
        is_forearm_plank = (
            is_side
            and is_horizontal
            and is_forearm_bent
            and elbows_under_shoulders
            and forearm_flat_on_floor
            and (body_angle >= SETUP_BODY_MIN_ANGLE)
            and (knee_angle >= SETUP_KNEE_MIN_ANGLE)
            and not is_knee_on_floor
        )

        if is_forearm_plank:
            self.plank_frames += 1
            if self.plank_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.plank_frames = max(0, self.plank_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            elif not is_horizontal:
                feedback_msg = "Get down on the floor in a horizontal forearm plank"
            elif is_knee_on_floor or knee_angle < SETUP_KNEE_MIN_ANGLE:
                feedback_msg = "Extend legs straight on your toes (lift knees off floor)"
            elif body_angle < SETUP_BODY_MIN_ANGLE:
                feedback_msg = "Brace core and glutes: keep body in a straight line"
            elif not is_forearm_bent or not forearm_flat_on_floor:
                feedback_msg = "Start in a forearm plank: rest forearms and hands flat on floor"
            elif not elbows_under_shoulders:
                feedback_msg = "Place your elbows directly under your shoulders"
            else:
                feedback_msg = "Hold steady forearm plank to begin"

            return {
                "exercise": "Plank Triceps Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["FOREARM_PLANK", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[PLANK TRICEPS FAULT] AI ST-GCN flagged plank breakdown: prob={bad_posture_prob:.2f}",
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
        if self.rep_started and self.state in ["PRESSING_UP", "LOCKOUT_TOP", "LOWERING"]:
            # A. Strict Side View Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[PLANK TRICEPS FAULT] Lost side view: ratio={shoulder_span_ratio:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Knees Touching Floor (Step 2)
            if is_knee_on_floor:
                self.knee_ground_frames += 1
                if self.knee_ground_frames >= DEBOUNCE_KNEE_GROUND:
                    if self.knees_off_floor:
                        print(f"[PLANK TRICEPS FAULT] Knees dropped to floor: knee_angle={knee_angle:.1f}°", flush=True)
                    self.knees_off_floor = False
            else:
                self.knee_ground_frames = max(0, self.knee_ground_frames - 1)

            # C. Core Sagging (Step 2 & 5: Hips sagging below shoulder-ankle line)
            if (hip_vertical_deviation > 0.03) and (body_angle < MIN_CORE_SAG_BODY_ANGLE):
                self.sag_frames += 1
                if self.sag_frames >= DEBOUNCE_SAG:
                    if self.core_maintained:
                        print(f"[PLANK TRICEPS FAULT] Hips sagged: body_angle={body_angle:.1f}°", flush=True)
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # D. Hip Piking (Step 2 & 5: Butt piking upward above shoulder-ankle line)
            if (hip_vertical_deviation < -0.04) and (body_angle < MIN_CORE_PIKE_BODY_ANGLE):
                self.pike_frames += 1
                if self.pike_frames >= DEBOUNCE_PIKE:
                    if self.no_pike:
                        print(f"[PLANK TRICEPS FAULT] Hips piked upward: body_angle={body_angle:.1f}°", flush=True)
                    self.no_pike = False
            else:
                self.pike_frames = max(0, self.pike_frames - 1)

            # E. Step 4: Keep Elbows Tucked (No outward flare)
            if is_elbow_flaring:
                self.flare_frames += 1
                if self.flare_frames >= DEBOUNCE_FLARE:
                    if self.elbows_tucked:
                        print(f"[PLANK TRICEPS FAULT] Elbows flared outward: ratio={elbow_span_ratio:.2f}", flush=True)
                    self.elbows_tucked = False
            else:
                self.flare_frames = max(0, self.flare_frames - 1)

            # F. Neutral Neck Check
            if neck_angle < MIN_NECK_NEUTRAL_ANGLE:
                self.neck_frames += 1
                if self.neck_frames >= DEBOUNCE_NECK:
                    if self.neck_neutral:
                        print(f"[PLANK TRICEPS FAULT] Head/neck dropped: neck_angle={neck_angle:.1f}°", flush=True)
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            self.max_elbow_angle = max(self.max_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (FOREARM PLANK -> PRESS UP -> LOWER TO FOREARMS)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_forearm_plank:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "FOREARM_PLANK"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Drive through palms to extend elbows."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Plank Triceps Extension", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "FOREARM_PLANK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive through your palms and extend elbows until arms are straight!"

        # Step 3: Press Up from Forearm Plank
        elif self.state in ["FOREARM_PLANK", "LOWERING"] and elbow_angle > PRESSING_START_ELBOW_ANGLE:
            self.state = "PRESSING_UP"
            self.rep_started = True
            self.max_elbow_angle = elbow_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Press up! Keep elbows tucked and body straight..."

        # Step 3 Complete: Full Straight-Arm Lockout Reached (>= 148°)
        elif self.state == "PRESSING_UP" and elbow_angle >= FULL_LOCKOUT_MIN_ELBOW_ANGLE:
            self.lockout_reached = True
            self.state = "LOCKOUT_TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Full lockout! Slowly lower forearms back to the floor."

        # Premature Lowering (Half Rep Branch: Lowered back down before straightening arms)
        elif self.state == "PRESSING_UP" and not self.lockout_reached and (self.max_elbow_angle - elbow_angle) > LOWERING_TRIGGER_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_elbow_angle >= HALF_REP_MIN_ELBOW_ANGLE:
                    self.half_press_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Incomplete lockout! Lowering early..."
                print(
                    f"[PLANK TRICEPS FAULT] Early lowering before full arm extension: max_elbow={self.max_elbow_angle:.1f}° "
                    f"(required={FULL_LOCKOUT_MIN_ELBOW_ANGLE}°)",
                    flush=True,
                )

        # Step 5: Lower with Control from Top Lockout
        elif self.state == "LOCKOUT_TOP" and (self.max_elbow_angle - elbow_angle) > LOWERING_TRIGGER_DELTA:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower forearms slowly to floor without sagging hips..."

        if self.state == "PRESSING_UP" and not ((self.max_elbow_angle - elbow_angle) > LOWERING_TRIGGER_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Step 5: Eccentric drop speed detector (catches crashing forearms onto floor)[cite: 9]
        if self.rep_started and self.state in ["LOWERING", "PRESSING_UP"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[PLANK TRICEPS FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (WHEN FOREARMS RETURN TO FLOOR)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["LOWERING", "LOCKOUT_TOP", "PRESSING_UP"] and elbow_angle <= REP_COMPLETE_ELBOW_ANGLE:
            # Ignore minor setup shifts on the floor (< 122°)
            if self.max_elbow_angle < MIN_VALID_REP_PEAK_ANGLE and self.knees_off_floor and self.is_side_view:
                self._reset_rep_tracking()
                self.state = "FOREARM_PLANK"
                return {
                    "exercise": "Plank Triceps Extension", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                    "depth_reached": False
                }

            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to camera! Hold forearm plank to reset."
                is_bad_rep = True
            elif not self.knees_off_floor:
                self.current_feedback = "Bad rep: Knees touched floor! Stay on your toes. Hold forearm plank to reset."
                is_bad_rep = True
            elif not self.elbows_tucked:
                self.current_feedback = "Bad rep: Keep elbows tucked close to your sides! Do not flare outward."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "Bad rep: Hips sagged! Keep body in a rigid straight line."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "Bad rep: Hips piked up! Keep body in a straight line from head to heels."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "Bad rep: Keep neck neutral in line with your spine!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Lower forearms slowly with control! Do not drop to the floor."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected broken plank mechanics! Hold forearm plank to reset."
                is_bad_rep = True
            elif not self.lockout_reached:
                if self.max_elbow_angle >= HALF_REP_MIN_ELBOW_ANGLE or self.half_press_reached:
                    self.current_feedback = "Bad rep: Half rep! Extend elbows until arms are fully straight."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Press was too shallow! Extend arms fully to lockout."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep or is_half_rep:
                self.bad_reps += 1
                print(
                    f"[PLANK TRICEPS REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_elbow={self.max_elbow_angle:.1f}°, body={body_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Plank Triceps Extension! Great lockout and core control."
                print(
                    f"[PLANK TRICEPS REP #{rep_num} -> GOOD REP] "
                    f"max_elbow={self.max_elbow_angle:.1f}°, body={body_angle:.1f}°",
                    flush=True,
                )
                self.state = "FOREARM_PLANK"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring[cite: 9]
        form_score = 100 if self.state == "FOREARM_PLANK" else max(0, min(100, int((elbow_angle / 150) * 100)))
        if not self.is_side_view: form_score -= 35
        if not self.knees_off_floor: form_score -= 40
        if not self.elbows_tucked: form_score -= 30
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 25
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.lockout_reached and self.state == "LOWERING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Plank Triceps Extension",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "body_angle": round(body_angle, 1),
            "depth_reached": self.lockout_reached
        }