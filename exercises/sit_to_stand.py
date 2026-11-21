import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SIT TO STAND - STRICT VERTICAL SHIN)
# =====================================================================

# 1. Setup & Starting Guard (Seated Side View, Tall Vertical Spine, Vertical Shins)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.32 # Side profile tolerance
SETUP_MAX_HIP_SPAN_RATIO = 0.25      # Hips aligned sideways to camera
SETUP_SEATED_KNEE_MIN = 70.0         # Seated knee bend min
SETUP_SEATED_KNEE_MAX = 112.0        # Neutral ~90° chair sit
SETUP_SEATED_HIP_MIN = 70.0
SETUP_SEATED_HIP_MAX = 115.0
SETUP_MAX_TORSO_LEAN_DEG = 22.0      # Sit tall with vertical upper body
SETUP_MAX_SHIN_TILT_DEG = 24.0       # Strict vertical knee-to-ankle alignment at setup (16:9 corrected)
SETUP_MIN_READY_FRAMES = 8
UNLOCK_RESET_FRAMES = 10
MIN_VISIBILITY_THRESHOLD = 2.0

# 2. Kinematic & Form Thresholds (Strict Vertical Knee-to-Ankle + Vertical Body)
ASPECT_RATIO_16_9 = 1.77             # Corrects MediaPipe 16:9 X-compression for true vertical shin angle
MAX_SHIN_TILT_DEG = 28.0             # Strict max knee-to-ankle tilt from vertical (above 15° = BAD REP!)
MAX_KNEE_ANKLE_X_RATIO = 0.42       # Max horizontal |knee.x - ankle.x| / shin_len ratio
MAX_TORSO_FORWARD_LEAN = 35.0        # Keep body mostly vertical with only a soft hip lean
STAND_MAX_TORSO_TILT_DEG = 18.0      # Body must be vertical at standing top
MIN_SPINE_NEUTRAL_ANGLE = 120.0      # Do not round your back (ear-shoulder-hip angle)
MIN_WRIST_ABOVE_KNEE_MARGIN = 0.06   # Arms crossed at chest or out in front (NOT pushing on knees!)
MAX_HEEL_LIFT_DELTA = 0.05           # Feet flat on floor, drive through heels
ECCENTRIC_SEAT_DROP_MAX = 22.0       # Control the descent (no plopping onto chair)

# 3. Extension & Soft-Bend Standing Lockout
STANDING_UP_START_ANGLE = 108.0      # Lowered from 115° so liftoff is tracked immediately!
MIN_VALID_REP_PEAK_ANGLE = 120.0     # Ignore minor seated weight shifts (< 120°)
HALF_STAND_MIN_KNEE_ANGLE = 126.0    # Rising past 126° without reaching soft stand flags Half Rep
FULL_STAND_MIN_KNEE_ANGLE = 150.0    # Soft knee bend allowed at standing lockout (~150°+)
FULL_STAND_MIN_HIP_ANGLE = 148.0     # Soft hip lockout with vertical torso (~148°+)
SITTING_DOWN_START_ANGLE = 138.0     # Descending from standing lockout
EARLY_REVERSAL_DELTA = 10.0          # Sitting back down early before reaching standing position
REP_COMPLETE_KNEE_ANGLE = 105.0      # Fully seated back in chair completes rep

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_LEAN = 4
DEBOUNCE_SHIN = 4                    # Fast 2-frame trigger if knee-to-ankle is not vertical!
DEBOUNCE_SPINE = 4                   # Frames before flagging rounded back
DEBOUNCE_HANDS_KNEE = 4              # Frames before flagging pushing hands on knees/thighs
DEBOUNCE_HEEL = 4                    # Frames before flagging heels lifting off floor
DEBOUNCE_DROP = 2
DEBOUNCE_HALF_REP = 4
DEBOUNCE_ML_FAULT = 5

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40

# =====================================================================


class SitToStandTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.stand_reached = False
        self.was_half_rep = False
        self.max_knee_angle = 0.0
        self.max_hip_angle = 0.0
        self.max_shin_tilt = 0.0           # Tracks worst knee-to-ankle tilt during the entire rep

        # Posture Tripwires (STRICT SIDE VIEW)
        self.is_side_view = True
        self.torso_upright = True          # Body stays vertical / soft lean only
        self.shins_vertical = True         # Knee-to-ankle MUST stay vertical!
        self.spine_neutral = True          # Spine long and tall, do not round back
        self.hands_free = True             # Arms at chest or out in front (no pushing on knees)
        self.heels_planted = True          # Push firmly through heels
        self.controlled_descent = True     # Slowly lower without plopping
        self.ml_posture_maintained = True

        self.baseline_ankle_y = None

        # Debounce counters
        self.profile_frames = 0
        self.lean_frames = 0
        self.shin_frames = 0
        self.spine_frames = 0
        self.hand_cheat_frames = 0
        self.heel_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_knee_angle = None
        self.current_feedback = "Sit tall sideways to camera: keep knee-to-ankle vertical"
        self.mp_pose = mp.solutions.pose
        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_sit_to_stand.tflite")
        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as exc:
                print(
                    f"[SIT TO STAND ERROR] __init__ model_path={model_path} | error={exc}",
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

    def _reset_rep_tracking(self):
        self.rep_started = False
        self.stand_reached = False
        self.was_half_rep = False
        self.max_knee_angle = 0.0
        self.max_hip_angle = 0.0
        self.max_shin_tilt = 0.0
        self.prev_knee_angle = None

        self.is_side_view = True
        self.torso_upright = True
        self.shins_vertical = True
        self.spine_neutral = True
        self.hands_free = True
        self.heels_planted = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.lean_frames = 0
        self.shin_frames = 0
        self.spine_frames = 0
        self.hand_cheat_frames = 0
        self.heel_frames = 0
        self.drop_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
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
                "exercise": "Sit to Stand", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile seated in chair",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        use_right = (r_vis >= l_vis)
        active_ear = r_ear if use_right else l_ear
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee
        active_ankle = r_ankle if use_right else l_ankle

        # 1. Sagittal Joint Angles
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        spine_neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 2. Strict Side Profile Verification
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_aspect = abs(l_shoulder.x - r_shoulder.x) / torso_len
        hip_aspect = abs(l_hip.x - r_hip.x) / torso_len
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        # 3. Torso Vertical Inclination (0° = perfectly vertical spine)
        dx_torso = abs(active_shoulder.x - active_hip.x) * ASPECT_RATIO_16_9
        dy_torso = max(abs(active_shoulder.y - active_hip.y), 0.10)
        torso_lean_angle = np.degrees(np.arctan2(dx_torso, dy_torso))

        # 4. Strict 16:9-Corrected Knee-to-Ankle Vertical Angle
        # Scales X by 1.77 so webcam aspect ratio never hides forward/backward knee tilt!
        dx_shin_scaled = abs(active_knee.x - active_ankle.x) * ASPECT_RATIO_16_9
        dy_shin = max(abs(active_ankle.y - active_knee.y), 0.08)
        shin_len = np.hypot(dx_shin_scaled, dy_shin) or 0.25
        shin_tilt_angle = float(np.degrees(np.arctan2(dx_shin_scaled, dy_shin)))
        knee_ankle_x_ratio = dx_shin_scaled / shin_len

        is_shin_vertical_setup = (shin_tilt_angle <= SETUP_MAX_SHIN_TILT_DEG)
        is_shin_tilted_fault = (shin_tilt_angle > MAX_SHIN_TILT_DEG) or (knee_ankle_x_ratio > MAX_KNEE_ANKLE_X_RATIO)

        # 5. Hands Crossed Over Chest or Extended Forward (No pushing off knees/chair)
        dist_wrist_to_knee = np.hypot(active_wrist.x - active_knee.x, active_wrist.y - active_knee.y) / torso_len
        hands_pushing_on_knees_or_seat = (
            (active_wrist.y > (active_hip.y - MIN_WRIST_ABOVE_KNEE_MARGIN))
            or (dist_wrist_to_knee < 0.28)
        )

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Side View, Seated Tall, Vertical Shins, Hands Ready)
        # ---------------------------------------------------------
        is_seated = (
            is_side
            and (SETUP_SEATED_KNEE_MIN <= knee_angle <= SETUP_SEATED_KNEE_MAX)
            and (SETUP_SEATED_HIP_MIN <= hip_angle <= SETUP_SEATED_HIP_MAX)
            and (torso_lean_angle <= SETUP_MAX_TORSO_LEAN_DEG)
            and is_shin_vertical_setup
            and not hands_pushing_on_knees_or_seat
        )

        if is_seated:
            self.ready_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y

            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (side view required)"
            elif not is_shin_vertical_setup:
                feedback_msg = f"Align knee directly over ankle (shin tilt {shin_tilt_angle:.0f}° > {SETUP_MAX_SHIN_TILT_DEG:.0f}°)"
            elif not (SETUP_SEATED_KNEE_MIN <= knee_angle <= SETUP_SEATED_KNEE_MAX):
                feedback_msg = "Sit on chair with knees bent (~90°) and feet flat"
            elif torso_lean_angle > SETUP_MAX_TORSO_LEAN_DEG:
                feedback_msg = "Keep your body vertical and sit tall"
            elif hands_pushing_on_knees_or_seat:
                feedback_msg = "Cross arms over chest or extend them out in front of you"
            else:
                feedback_msg = "Sit tall with vertical knee-to-ankle alignment to begin"

            return {
                "exercise": "Sit to Stand", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["SEATED", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[SIT TO STAND FAULT] AI ST-GCN flagged posture fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception as exc:
                expected_shape = self.input_details[0]['shape'] if hasattr(self, "input_details") else None
                print(
                    f"[SIT TO STAND ERROR] process_frame: expected_shape={expected_shape} | error={exc}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # 3. KINEMATIC TRIPWIRES (MONITORED DURING LIFTOFF AND ACTIVE REP!)
        # ---------------------------------------------------------
        # Monitor shin verticality even at the very start of liftoff from SEATED (knee_angle > 95°)
        if self.state == "SEATED" and knee_angle > 95.0:
            if is_shin_tilted_fault:
                self.shin_frames += 1
                self.max_shin_tilt = max(self.max_shin_tilt, shin_tilt_angle)
                if self.shin_frames >= DEBOUNCE_SHIN:
                    self.shins_vertical = False
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

        if self.rep_started and self.state in ["STANDING_UP", "STANDING", "SITTING_DOWN"]:
            self.max_knee_angle = max(self.max_knee_angle, knee_angle)
            self.max_hip_angle = max(self.max_hip_angle, hip_angle)
            self.max_shin_tilt = max(self.max_shin_tilt, shin_tilt_angle)

            # A. Strict Side View Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[SIT TO STAND FAULT] Lost side view: ratio={shoulder_aspect:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Strict Knee-to-Ankle Vertical Check (Checked FIRST so non-vertical shins ALWAYS flag!)
            if is_shin_tilted_fault:
                self.shin_frames += 1
                if self.shin_frames >= DEBOUNCE_SHIN:
                    if self.shins_vertical:
                        print(
                            f"[SIT TO STAND FAULT] Knee-to-ankle NOT vertical! "
                            f"shin_tilt={shin_tilt_angle:.1f}° (max={MAX_SHIN_TILT_DEG}°), "
                            f"x_ratio={knee_ankle_x_ratio:.2f}",
                            flush=True,
                        )
                    self.shins_vertical = False
                    if self.post_rep_timer == 0:
                        self.current_feedback = "Keep knee-to-ankle vertical! Do not let knees shift forward."
            else:
                self.shin_frames = max(0, self.shin_frames - 1)

            # C. Vertical Body Check (Allows only a soft hip lean, no excessive forward torso fold)
            if torso_lean_angle > MAX_TORSO_FORWARD_LEAN:
                self.lean_frames += 1
                if self.lean_frames >= DEBOUNCE_LEAN:
                    if self.torso_upright:
                        print(
                            f"[SIT TO STAND FAULT] Body leaned too far forward: lean={torso_lean_angle:.1f}° "
                            f"(max={MAX_TORSO_FORWARD_LEAN}°)",
                            flush=True,
                        )
                    self.torso_upright = False
            else:
                self.lean_frames = max(0, self.lean_frames - 1)

            # D. Do Not Round Your Back
            if spine_neck_angle < MIN_SPINE_NEUTRAL_ANGLE:
                self.spine_frames += 1
                if self.spine_frames >= DEBOUNCE_SPINE:
                    if self.spine_neutral:
                        print(
                            f"[SIT TO STAND FAULT] Rounded back/neck: spine_angle={spine_neck_angle:.1f}°",
                            flush=True,
                        )
                    self.spine_neutral = False
            else:
                self.spine_frames = max(0, self.spine_frames - 1)

            # E. Hands Pushing on Knees/Thighs Cheat
            if hands_pushing_on_knees_or_seat and knee_angle < 142.0:
                self.hand_cheat_frames += 1
                if self.hand_cheat_frames >= DEBOUNCE_HANDS_KNEE:
                    if self.hands_free:
                        print("[SIT TO STAND FAULT] Hands dropped to push off knees/chair!", flush=True)
                    self.hands_free = False
            else:
                self.hand_cheat_frames = max(0, self.hand_cheat_frames - 1)

            # F. Drive Through Heels (Keep Feet Flat)
            if self.baseline_ankle_y is not None and (self.baseline_ankle_y - active_ankle.y) > MAX_HEEL_LIFT_DELTA:
                self.heel_frames += 1
                if self.heel_frames >= DEBOUNCE_HEEL:
                    if self.heels_planted:
                        print("[SIT TO STAND FAULT] Heels lifted off the floor!", flush=True)
                    self.heels_planted = False
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (SEATED -> STANDING_UP -> STANDING -> SITTING_DOWN)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_seated:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "SEATED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Keep knee-to-ankle vertical and stand up."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Sit to Stand", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "SEATED"
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep knee-to-ankle vertical and drive through heels to stand!"

        # Initiate Drive Up ONLY from SEATED
        elif self.state == "SEATED" and knee_angle > STANDING_UP_START_ANGLE:
            self.state = "STANDING_UP"
            self.rep_started = True
            self.max_knee_angle = knee_angle
            self.max_hip_angle = hip_angle
            self.max_shin_tilt = max(self.max_shin_tilt, shin_tilt_angle)
            self.half_rep_frames = 0
            if self.post_rep_timer == 0 and self.shins_vertical:
                self.current_feedback = "Stand up tall (keep knee-to-ankle vertical)..."

        # Full Standing Reached (Allows a natural soft bend in the knees >= 150° with vertical body!)
        elif (
            self.state == "STANDING_UP"
            and knee_angle >= FULL_STAND_MIN_KNEE_ANGLE
            and hip_angle >= FULL_STAND_MIN_HIP_ANGLE
            and torso_lean_angle <= STAND_MAX_TORSO_TILT_DEG
        ):
            self.stand_reached = True
            self.state = "STANDING"
            self.half_rep_frames = 0
            if self.post_rep_timer == 0 and self.shins_vertical:
                self.current_feedback = "Tall vertical stand reached! Sit back down keeping shins vertical."

        # Premature Reversal (Half Rep Branch: Sitting back down before reaching standing posture)
        elif self.state == "STANDING_UP" and not self.stand_reached and (self.max_knee_angle - knee_angle) > EARLY_REVERSAL_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                if self.max_knee_angle >= HALF_STAND_MIN_KNEE_ANGLE:
                    self.was_half_rep = True
                self.state = "SITTING_DOWN"
                self.current_feedback = "Incomplete stand! Sitting back down early..."
                print(
                    f"[SIT TO STAND FAULT] Early reversal before vertical stand: "
                    f"max_knee={self.max_knee_angle:.1f}°, max_hip={self.max_hip_angle:.1f}°",
                    flush=True,
                )

        # Lower with Control from STANDING Lockout ONLY
        elif self.state == "STANDING" and knee_angle < SITTING_DOWN_START_ANGLE:
            self.state = "SITTING_DOWN"
            if self.post_rep_timer == 0 and self.shins_vertical:
                self.current_feedback = "Sit hips back slowly, keeping knee-to-ankle vertical..."

        if self.state == "STANDING_UP" and not ((self.max_knee_angle - knee_angle) > EARLY_REVERSAL_DELTA):
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # Eccentric Seat Drop Detector (Catches plopping onto the chair)
        if self.rep_started and self.state in ["SITTING_DOWN", "STANDING_UP"]:
            if self.prev_knee_angle is not None:
                drop_speed = self.prev_knee_angle - knee_angle
                if drop_speed > ECCENTRIC_SEAT_DROP_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[SIT TO STAND FAULT] Plopped onto chair: drop_speed={drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_SEAT_DROP_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_knee_angle = knee_angle
        else:
            self.prev_knee_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (VERTICAL SHIN CHECK PRIORITIZED!)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["SITTING_DOWN", "STANDING", "STANDING_UP"] and knee_angle <= REP_COMPLETE_KNEE_ANGLE:
            # Ignore minor seated adjustments (< 120°) ONLY if shins stayed vertical
            if (
                self.max_knee_angle < MIN_VALID_REP_PEAK_ANGLE
                and self.shins_vertical
                and self.max_shin_tilt <= MAX_SHIN_TILT_DEG
                and self.is_side_view
                and self.hands_free
            ):
                self._reset_rep_tracking()
                self.state = "SEATED"
                return {
                    "exercise": "Sit to Stand", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
                }

            is_bad_rep = False

            # Hard guarantee: if shins_vertical tripped OR peak shin tilt exceeded MAX_SHIN_TILT_DEG, flag Bad Rep!
            if not self.shins_vertical or self.max_shin_tilt > MAX_SHIN_TILT_DEG:
                self.current_feedback = "Bad rep: Knee-to-ankle was not vertical! Keep knees stacked over ankles."
                is_bad_rep = True
            elif not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to camera! Sit tall to reset."
                is_bad_rep = True
            elif not self.hands_free:
                self.current_feedback = "Bad rep: Keep arms crossed at chest or out in front! Do not push off knees."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Keep your body vertical! Do not lean excessively forward."
                is_bad_rep = True
            elif not self.spine_neutral:
                self.current_feedback = "Bad rep: Keep your spine long and straight! Do not round your back."
                is_bad_rep = True
            elif not self.heels_planted:
                self.current_feedback = "Bad rep: Keep feet flat and drive firmly through your heels!"
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Do not plop onto the chair! Lower slowly under control."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected posture or balance breakdown."
                is_bad_rep = True
            elif not self.stand_reached:
                if self.max_knee_angle >= HALF_STAND_MIN_KNEE_ANGLE or self.was_half_rep:
                    self.current_feedback = "Bad rep: Half rep! Stand all the way up into a vertical posture."
                else:
                    self.current_feedback = "Bad rep: Incomplete stand! Stand up tall before sitting back down."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SIT TO STAND REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_shin_tilt={self.max_shin_tilt:.1f}°, "
                    f"max_knee={self.max_knee_angle:.1f}°, max_hip={self.max_hip_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Sit-to-Stand! Great vertical knee-to-ankle alignment and posture."
                print(
                    f"[SIT TO STAND REP #{rep_num} -> GOOD REP] "
                    f"max_shin_tilt={self.max_shin_tilt:.1f}°, max_knee={self.max_knee_angle:.1f}°, "
                    f"max_hip={self.max_hip_angle:.1f}°",
                    flush=True,
                )
                self.state = "SEATED"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = 100 if self.state == "SEATED" else max(0, min(100, int((knee_angle / 150.0) * 100)))
        if not self.shins_vertical or self.max_shin_tilt > MAX_SHIN_TILT_DEG: form_score -= 40
        if not self.is_side_view: form_score -= 35
        if not self.hands_free: form_score -= 35
        if not self.torso_upright: form_score -= 30
        if not self.spine_neutral: form_score -= 25
        if not self.heels_planted: form_score -= 20
        if not self.controlled_descent: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 30
        if not self.stand_reached and self.state == "SITTING_DOWN": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Sit to Stand",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1)
        }