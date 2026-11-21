import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (BURPEES + PUSH-UP - BALANCED TUNING)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_KNEE_MIN_ANGLE = 155.0            # Knees extended to confirm standing posture
SETUP_HIP_MIN_ANGLE = 150.0             # Hips extended to confirm standing posture
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.42    # Standard side profile ratio limit
SETUP_MIN_READY_FRAMES = 6              # Consecutive frames standing upright to confirm READY
UNLOCK_RESET_FRAMES = 12                # Frames standing tall to clear LOCKED_FAULT (~0.4s)
LANDING_COOLDOWN_FRAMES = 12            # Blocks jump-landing knee bend from starting a fake rep
MIN_VISIBILITY_THRESHOLD = 2.2          # Minimum keypoint visibility for dominant profile

# 2. Balanced Burpee & Push-Up Phase Thresholds
DROP_KNEE_MAX_ANGLE = 120.0             # Bending knees below 120° initiates DROPPING
DROP_HIP_MAX_ANGLE = 105.0              # Hinging hips below 105° initiates DROPPING
HANDS_NEAR_FLOOR_HIP_Y_DIFF = 0.10      # Wrists dropping near/below hip level confirms floor descent
PLANK_VERTICAL_DROP_MAX = 0.35          # Max shoulder-to-ankle vertical drop for horizontal plank
PLANK_HORIZONTAL_SPAN_MIN = 0.30        # Min shoulder-to-ankle horizontal stretch for plank
PLANK_BODY_MIN_ANGLE = 132.0            # Balanced plank extension angle (catches real planks, blocks squat-thrusts)
HALF_PLANK_BODY_MIN_ANGLE = 115.0       # Half-plank kickback threshold (flagged as Bad Rep)

# Push-Up Depth (Balanced for fast burpee push-ups)
PUSHUP_FULL_DEPTH_ELBOW_ANGLE = 102.0   # Elbow bend <= 102° required for valid burpee push-up depth
PUSHUP_HALF_DEPTH_ELBOW_ANGLE = 122.0   # Bending between 102° and 122° flags a shallow half push-up (Bad Rep)

# 3. Form Tripwires (Side View + Hip Sag + Push-Up Elbow Flare + Jump Finish)
MAX_HIP_SAG_BODY_ANGLE = 130.0          # Hips sinking below shoulder/ankle line (< 130°) flags core sag
VERTICAL_CHECK_ELBOW_ANGLE = 120=.0      # Check shoulder-to-elbow flare when lowered into push-up (<= 120°)
MIN_ELBOW_VERTICAL_DEVIATION_DEG = 35.0 # Below 35° = shoulder & elbow vertically aligned / flared (BAD REP)
MIN_JUMP_HEIGHT_DELTA = 0.025           # Ankle lift off floor for jump credit
ASCENT_RECOVERY_VERTICAL_DROP = 0.40    # Torso lifting from floor transitions to ASCENDING
REP_COMPLETE_KNEE_ANGLE = 155.0         # Standing knee extension to finish rep
REP_COMPLETE_HIP_ANGLE = 150.0          # Standing hip extension to finish rep

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4                    # Frames before flagging lost side profile
DEBOUNCE_SAG = 3                        # Frames before flagging sagging hips in plank
DEBOUNCE_ELBOW_VERTICAL = 4             # Fast 2-frame latch when push-up elbows flare under shoulders
DEBOUNCE_JUMP = 2                       # Frames required to confirm jump or overhead reach

# 5. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class BurpeesTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.plank_reached = False
        self.half_plank_reached = False
        self.pushup_depth_reached = False
        self.half_pushup_reached = False
        self.min_elbow_angle = 180.0
        self.max_plank_body_angle = 0.0
        self.jump_finished = False

        # Strict Tripwires
        self.is_side_view = True
        self.core_maintained = True
        self.elbow_not_vertical_aligned = True

        # Debounce & Cooldown counters
        self.profile_frames = 0
        self.sag_frames = 0
        self.elbow_vertical_frames = 0
        self.jump_frames = 0
        self.standing_frames = 0
        self.unlock_frames = 0
        self.cooldown_timer = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_ankle_y = None
        self.current_feedback = "Stand sideways to the camera to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_burpees.tflite")
        if not os.path.exists(model_path):
            print(f"[BURPEE ERROR] __init__ model file not found: {model_path}", flush=True)
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[BURPEE ERROR] __init__ model_path={model_path} | error={exc}",
                    flush=True,
                )
                self.interpreter = None

    def _extract_body_keypoints(self, landmarks):
        ntu = np.zeros((25, 3), dtype=np.float32)

        def xyz(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        ntu[3] = xyz(0)
        ntu[4] = xyz(11)
        ntu[5] = xyz(13)
        ntu[6] = xyz(15)
        ntu[7] = xyz(19)
        ntu[8] = xyz(12)
        ntu[9] = xyz(14)
        ntu[10] = xyz(16)
        ntu[11] = xyz(20)
        ntu[12] = xyz(23)
        ntu[13] = xyz(25)
        ntu[14] = xyz(27)
        ntu[15] = xyz(31)
        ntu[16] = xyz(24)
        ntu[17] = xyz(26)
        ntu[18] = xyz(28)
        ntu[19] = xyz(32)

        ntu[0] = (ntu[12] + ntu[16]) / 2.0
        ntu[2] = (ntu[4] + ntu[8]) / 2.0
        ntu[1] = (ntu[0] + ntu[2]) / 2.0
        ntu[20] = ntu[2]

        ntu[21] = xyz(17)
        ntu[22] = xyz(21)
        ntu[23] = xyz(18)
        ntu[24] = xyz(22)
        return ntu

    def _reset_rep_tracking(self, current_ankle_y=None):
        self.rep_started = False
        self.plank_reached = False
        self.half_plank_reached = False
        self.pushup_depth_reached = False
        self.half_pushup_reached = False
        self.min_elbow_angle = 180.0
        self.max_plank_body_angle = 0.0
        self.jump_finished = False

        self.is_side_view = True
        self.core_maintained = True
        self.elbow_not_vertical_aligned = True

        self.profile_frames = 0
        self.sag_frames = 0
        self.elbow_vertical_frames = 0
        self.jump_frames = 0

        if current_ankle_y is not None:
            self.baseline_ankle_y = current_ankle_y

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Side view dominant side detection
        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility, l_elbow.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, r_shoulder.visibility, r_elbow.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        if max(l_vis, r_vis) < MIN_VISIBILITY_THRESHOLD:
            self.frame_buffer.clear()
            return {
                "exercise": "Burpees", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: full side profile not visible",
                "elbow_angle": 0.0, "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        # Side-view Biomechanical Angles
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # Profile validation using 2D torso length
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        torso_len = max(np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y), 0.25)
        shoulder_span_ratio = shoulder_span_x / torso_len
        is_turned_sideways = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        is_standing_upright = (
            knee_angle > SETUP_KNEE_MIN_ANGLE
            and hip_angle > SETUP_HIP_MIN_ANGLE
            and is_turned_sideways
        )

        if is_standing_upright:
            self.standing_frames += 1
            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
                if self.baseline_ankle_y is None:
                    self.baseline_ankle_y = active_ankle.y
                else:
                    self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera (side profile)" if not is_turned_sideways else "Stand tall sideways to start"
            return {
                "exercise": "Burpees", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER (NO INFERENCE INVOKED)
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # ---------------------------------------------------------
        # 3. KINEMATIC GEOMETRY & STATE MACHINE
        # ---------------------------------------------------------
        vertical_drop = abs(active_shoulder.y - active_ankle.y)
        horizontal_span = abs(active_shoulder.x - active_ankle.x)
        is_horizontal = (vertical_drop < PLANK_VERTICAL_DROP_MAX) and (horizontal_span > PLANK_HORIZONTAL_SPAN_MIN)
        hands_reaching_floor = active_wrist.y > (active_hip.y - HANDS_NEAR_FLOOR_HIP_Y_DIFF)

        # Fault lockout state handling
        if self.state == "LOCKED_FAULT":
            if is_standing_upright:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "STANDING"
                    self.unlock_frames = 0
                    self.cooldown_timer = LANDING_COOLDOWN_FRAMES
                    self._reset_rep_tracking(active_ankle.y)
                    self.current_feedback = "Reset complete! Drop down into a burpee."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Burpees", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1),
                "depth_reached": False
            }

        # Initialize to STANDING once ready
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.baseline_ankle_y = active_ankle.y
            if self.post_rep_timer == 0:
                self.current_feedback = "Drop down into a plank and push-up!"

        # 1. Drop: Requires cooldown to finish AND hands moving toward floor (blocks landing bounce)
        elif (
            self.state == "STANDING"
            and self.cooldown_timer == 0
            and hands_reaching_floor
            and (knee_angle < DROP_KNEE_MAX_ANGLE or hip_angle < DROP_HIP_MAX_ANGLE)
        ):
            self.state = "DROPPING"
            self.rep_started = True
            if self.post_rep_timer == 0:
                self.current_feedback = "Kick legs back into a plank and push-up!"

        # 2. Plank & Push-Up Phase: Enters PLANK when body stretches horizontally OR elbows bend near floor
        elif self.state == "DROPPING" and (is_horizontal or (hands_reaching_floor and elbow_angle <= PUSHUP_HALF_DEPTH_ELBOW_ANGLE)):
            self.state = "PLANK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Lower chest to 90°, then jump feet in!"

        # ---------------------------------------------------------
        # 4. CONTINUOUS BOTTOM TRACKING & FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["DROPPING", "PLANK"] and (is_horizontal or hands_reaching_floor):
            if is_horizontal:
                self.max_plank_body_angle = max(self.max_plank_body_angle, body_angle)
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

            if self.max_plank_body_angle >= PLANK_BODY_MIN_ANGLE:
                self.plank_reached = True
            elif self.max_plank_body_angle >= HALF_PLANK_BODY_MIN_ANGLE:
                self.half_plank_reached = True

            if self.min_elbow_angle <= PUSHUP_FULL_DEPTH_ELBOW_ANGLE:
                self.pushup_depth_reached = True
                self.plank_reached = True
            elif self.min_elbow_angle <= PUSHUP_HALF_DEPTH_ELBOW_ANGLE:
                self.half_pushup_reached = True

            # A. Side View Check
            if not is_turned_sideways:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[BURPEE FAULT] Lost side view: ratio={shoulder_span_ratio:.2f} "
                            f"(max={SETUP_MAX_SHOULDER_SPAN_RATIO})",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Hip Sag Check (Hips sinking below both shoulders and ankles)
            if is_horizontal and active_hip.y > active_shoulder.y and active_hip.y > active_ankle.y and body_angle < MAX_HIP_SAG_BODY_ANGLE:
                self.sag_frames += 1
                if self.sag_frames >= DEBOUNCE_SAG:
                    if self.core_maintained:
                        print(
                            f"[BURPEE FAULT] Core sagging in plank/push-up: body_angle={body_angle:.1f}° "
                            f"(min={MAX_HIP_SAG_BODY_ANGLE}°)",
                            flush=True,
                        )
                    self.core_maintained = False
            else:
                self.sag_frames = max(0, self.sag_frames - 1)

            # C. Push-Up Elbow Flare / Shoulder-to-Elbow Vertical Alignment Check
            if is_horizontal and elbow_angle <= VERTICAL_CHECK_ELBOW_ANGLE:
                dx_elbow = abs(active_shoulder.x - active_elbow.x)
                dy_elbow = max(abs(active_elbow.y - active_shoulder.y), 0.02)
                elbow_vertical_tilt = np.degrees(np.arctan2(dx_elbow, dy_elbow))

                if elbow_vertical_tilt <= MIN_ELBOW_VERTICAL_DEVIATION_DEG:
                    self.elbow_vertical_frames += 1
                    if self.elbow_vertical_frames >= DEBOUNCE_ELBOW_VERTICAL or elbow_angle <= PUSHUP_FULL_DEPTH_ELBOW_ANGLE:
                        if self.elbow_not_vertical_aligned:
                            print(
                                f"[BURPEE FAULT] Push-up elbows flared/aligned with shoulders: "
                                f"tilt={elbow_vertical_tilt:.1f}° (must be > {MIN_ELBOW_VERTICAL_DEVIATION_DEG}°), "
                                f"elbow_angle={elbow_angle:.1f}°",
                                flush=True,
                            )
                        self.elbow_not_vertical_aligned = False
                else:
                    self.elbow_vertical_frames = max(0, self.elbow_vertical_frames - 1)
            else:
                self.elbow_vertical_frames = max(0, self.elbow_vertical_frames - 1)

        # Transition: Rising from floor (Torso shifts toward vertical)
        if self.state == "PLANK" and vertical_drop > ASCENT_RECOVERY_VERTICAL_DROP and self.rep_started:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Jump up and reach overhead!"
        elif self.state == "DROPPING" and is_standing_upright and not is_horizontal and not self.plank_reached:
            # Cancel false knee-bend starts cleanly without flagging a bad rep
            self.state = "STANDING"
            self._reset_rep_tracking(active_ankle.y)

        # Jump & Overhead Extension Verification
        if self.state in ["ASCENDING", "STANDING"] and self.rep_started:
            jumped = (self.baseline_ankle_y is not None) and ((self.baseline_ankle_y - active_ankle.y) > MIN_JUMP_HEIGHT_DELTA)
            arms_up = active_wrist.y < active_shoulder.y
            if jumped or arms_up:
                self.jump_frames += 1
                if self.jump_frames >= DEBOUNCE_JUMP:
                    self.jump_finished = True

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.state == "ASCENDING" and self.rep_started and knee_angle > REP_COMPLETE_KNEE_ANGLE and hip_angle > REP_COMPLETE_HIP_ANGLE:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Stand tall to reset."
                is_bad_rep = True
            elif not self.core_maintained:
                self.current_feedback = "Bad rep: Hips sagged in the plank! Squeeze core and stand tall to reset."
                is_bad_rep = True
            elif not self.elbow_not_vertical_aligned:
                self.current_feedback = "Bad rep: Push-up elbows flared/aligned with shoulders! Tuck elbows back and stand tall to reset."
                is_bad_rep = True
            elif not self.plank_reached:
                if self.half_plank_reached:
                    self.current_feedback = "Bad rep: Half rep! Incomplete kickback—kick legs all the way into a plank. Stand tall to reset."
                else:
                    self.current_feedback = "Bad rep: Incomplete kickback! Plank was too shallow. Stand tall to reset."
                is_bad_rep = True
            elif not self.pushup_depth_reached:
                if self.half_pushup_reached:
                    self.current_feedback = "Bad rep: Half rep! Shallow push-up—lower chest to 90° at the bottom. Stand tall to reset."
                else:
                    self.current_feedback = "Bad rep: Push-up skipped! Lower chest at the bottom of the plank. Stand tall to reset."
                is_bad_rep = True
            elif not self.jump_finished:
                self.current_feedback = "Bad rep: Finish with an explosive jump or overhead reach! Stand tall to reset."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[BURPEE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_elbow_angle={self.min_elbow_angle:.1f}°, "
                    f"max_body_angle={self.max_plank_body_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking(active_ankle.y)
            else:
                self.good_reps += 1
                self.current_feedback = "Great explosive burpee!"
                print(
                    f"[BURPEE REP #{rep_num} -> GOOD REP] "
                    f"min_elbow_angle={self.min_elbow_angle:.1f}°, max_body_angle={self.max_plank_body_angle:.1f}°",
                    flush=True,
                )
                self.state = "STANDING"
                self.cooldown_timer = LANDING_COOLDOWN_FRAMES
                self._reset_rep_tracking(active_ankle.y)

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = 100
        if not self.is_side_view: form_score -= 35
        if not self.core_maintained: form_score -= 30
        if not self.elbow_not_vertical_aligned: form_score -= 30
        if not self.plank_reached and self.rep_started: form_score -= 25
        if not self.pushup_depth_reached and self.state == "ASCENDING": form_score -= 25
        if not self.jump_finished and self.state == "ASCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Burpees", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1),
            "depth_reached": self.pushup_depth_reached
        }