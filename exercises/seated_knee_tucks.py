import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (SEATED KNEE TUCKS - FLOOR EXECUTION)
# =====================================================================

# 1. Starting Position (Step 1.1 - 1.7: Sit on Floor, Legs Extended, Hands Behind Hips, Feet Slightly Lifted)
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.34 # Side profile view tolerance
SETUP_MAX_HIP_SPAN_RATIO = 0.26      # Hips aligned sideways to camera
SETUP_MIN_TORSO_LEAN_DEG = 14.0      # Step 1.5: Lean upper body slightly backward
SETUP_MAX_TORSO_LEAN_DEG = 68.0      # Step 1.5: Keep chest lifted (do not lie flat on floor)
SETUP_EXTENDED_MIN_KNEE_ANGLE = 132.0# Step 1.2 & 3.5: Legs extended nearly straight in front of you
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames in starting V-hover to confirm READY
UNLOCK_RESET_FRAMES = 8              # Frames held in starting position to clear LOCKED_FAULT
MIN_VISIBILITY_THRESHOLD = 2.0       # Minimum side-profile landmark visibility sum

# 2. Foot Hover & Butt-to-Ankle Angle Window (Step 1.7, 2.2, 3.2: Slightly Off Floor, <= 30° Ceiling)
MIN_BUTT_TO_ANKLE_HOVER_DEG = 5   # Step 1.7 & 3.2: Feet elevated off floor (below -6° = heels touching floor!)
MAX_BUTT_TO_ANKLE_ANGLE_DEG = 25.0   # Butt-to-ankle angle must stay low (~30° max upward angle)
MAX_LEG_SPLIT_RATIO = 0.16           # Step 1.7: Keep both legs together (no single-leg or bicycle kick)

# 3. Pulling Knees In & Extending Thresholds (Step 2 & Step 3)
TUCKING_START_KNEE_ANGLE = 122.0     # Step 2.1: Bending knees from extended position initiates TUCKING
TIGHT_TUCK_MAX_KNEE_ANGLE = 90.0     # Step 2.1 & 2.6: Knees bent in close to chest (<= 90°)
TIGHT_TUCK_MAX_TORSO_ANGLE = 95.0    # Step 2.2 & 2.3: Thighs pulled to stomach & torso moves slightly forward
HALF_TUCK_MAX_KNEE_ANGLE = 114.0     # Partial knee bend (91° to 114°) flags Half Rep (Bad Rep)
EXTENDING_START_KNEE_ANGLE = 98.0    # Step 3.1: Extending legs forward from TUCKED position
REP_COMPLETE_KNEE_ANGLE = 132.0      # Step 3.5: Stop extending when legs are nearly straight (>= 132°)
EARLY_REVERSAL_DELTA = 10.0          # Extending early before knees reach chest
MIN_VALID_REP_ROM_DEG = 20.0         # Ignore minor straight-leg balancing shifts

# 4. Movement Control & Posture Thresholds (Step 1.3, 2.5, 3.5)
MAX_LYING_FLAT_LEAN_DEG = 74.0       # Step 3.5: Lower back losing controlled position / collapsing flat
MAX_KNEE_ANGULAR_SPEED_DEG = 28.0    # Step 2.5: Keep movements controlled and avoid swinging/kicking

# 5. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 4
DEBOUNCE_HANDS = 4                   # Frames before flagging hands lifted off floor / not supporting behind hips
DEBOUNCE_POSTURE = 4                 # Frames before flagging collapsed back
DEBOUNCE_FEET_FLOOR = 4              # Frames before flagging feet touching the floor
DEBOUNCE_HIGH_LEGS = 8               # Frames before flagging butt-to-ankle angle > 30°
DEBOUNCE_LEGS_TOGETHER = 4           # Frames before flagging legs separating / single-leg tuck
DEBOUNCE_SPEED = 3                   # Frames before flagging uncontrolled swinging/kicking
DEBOUNCE_HALF_REP = 4                # Frames before confirming early extension reversal
DEBOUNCE_ML_FAULT = 5

ML_FAULT_CONFIDENCE_THRESHOLD = 0.75
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class SeatedKneeTucksTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.tuck_depth_reached = False
        self.was_half_rep = False
        self.min_knee_angle = 180.0
        self.max_knee_angle = 0.0
        self.min_torso_angle = 180.0
        self.max_butt_ankle_deg = 0.0
        self.prev_knee_angle = None

        # Posture Tripwires (MATCHED TO EXECUTION GUIDE)
        self.is_side_view = True
        self.hands_planted_behind = True   # Step 1.3 & 1.4: Hands flat on floor slightly behind hips
        self.seated_lean_maintained = True # Step 1.5 & 3.5: Chest lifted, controlled lower back (not lying flat)
        self.feet_hovered = True           # Step 1.7 & 3.2: Feet elevated without touching the floor
        self.butt_to_ankle_under_30 = True # Step 1.7: Feet lifted slightly (butt-to-ankle <= 30°)
        self.legs_together = True          # Step 1.7: Keep both legs together
        self.controlled_movement = True    # Step 2.5: Controlled movement, avoid swinging
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.hand_frames = 0
        self.posture_frames = 0
        self.feet_floor_frames = 0
        self.high_leg_frames = 0
        self.leg_split_frames = 0
        self.speed_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Sit on floor sideways: hands behind hips, legs extended & slightly lifted"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_seated_knee_tucks.tflite")
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
        self.tuck_depth_reached = False
        self.was_half_rep = False
        self.min_knee_angle = 180.0
        self.max_knee_angle = 0.0
        self.min_torso_angle = 180.0
        self.max_butt_ankle_deg = 0.0
        self.prev_knee_angle = None

        self.is_side_view = True
        self.hands_planted_behind = True
        self.seated_lean_maintained = True
        self.feet_hovered = True
        self.butt_to_ankle_under_30 = True
        self.legs_together = True
        self.controlled_movement = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.hand_frames = 0
        self.posture_frames = 0
        self.feet_floor_frames = 0
        self.high_leg_frames = 0
        self.leg_split_frames = 0
        self.speed_frames = 0
        self.half_rep_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
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
                "exercise": "Seated Knee Tucks", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side seated profile on floor",
                "knee_angle": 0.0, "torso_angle": 0.0
            }

        use_right = (r_vis >= l_vis)
        active_shoulder = r_shoulder if use_right else l_shoulder
        active_wrist = r_wrist if use_right else l_wrist
        active_hip = r_hip if use_right else l_hip
        active_knee = r_knee if use_right else l_knee
        active_ankle = r_ankle if use_right else l_ankle

        # 1. Sagittal Angles (Knee Extension & Torso-to-Thigh V-Angle)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        torso_angle = calculate_angle(active_shoulder, active_hip, active_knee)

        # 2. Side Profile & Upper-Body Lean Geometry
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.38
        shoulder_aspect = abs(l_shoulder.x - r_shoulder.x) / torso_len
        hip_aspect = abs(l_hip.x - r_hip.x) / torso_len
        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        dx_torso = abs(active_shoulder.x - active_hip.x)
        dy_torso = max(abs(active_hip.y - active_shoulder.y), 0.05)
        torso_lean_deg = np.degrees(np.arctan2(dx_torso, dy_torso))  # 0° = vertical, 90° = lying flat

        # 3. Step 1.3: Hands on Floor Slightly Behind Hips (Opposite the direction of knees/feet)
        leg_dir_sign = 1.0 if (active_ankle.x >= active_hip.x) else -1.0
        # Wrist should be near floor level (Y near hip Y) and not reaching forward past the knees
        wrist_on_floor = active_wrist.y >= (active_hip.y - 0.16)
        wrist_not_grabbing_legs = ((active_wrist.x - active_hip.x) * leg_dir_sign) < (0.25 * torso_len)
        hands_supporting_behind = wrist_on_floor and wrist_not_grabbing_legs

        # 4. Step 1.7 & 3.2: Butt-to-Ankle Angle (Feet Lifted Slightly Off Floor, Not Passing 30°)
        dx_butt_ankle = max(abs(active_ankle.x - active_hip.x), 0.08)
        # Floor level is approximated by the seated hip/butt Y (active_hip.y)
        dy_butt_ankle_up = active_hip.y - active_ankle.y  # Positive = ankle higher than butt/floor
        butt_to_ankle_deg = float(np.degrees(np.arctan2(dy_butt_ankle_up, dx_butt_ankle)))

        feet_off_floor = butt_to_ankle_deg >= MIN_BUTT_TO_ANKLE_HOVER_DEG
        legs_under_30_deg = butt_to_ankle_deg <= MAX_BUTT_TO_ANKLE_ANGLE_DEG

        # 5. Step 1.7: Keep Both Legs Together (When both ankles/knees are visible)
        both_legs_vis = (l_ankle.visibility > 0.45 and r_ankle.visibility > 0.45)
        leg_split_ratio = (abs(l_ankle.y - r_ankle.y) / torso_len) if both_legs_vis else 0.0
        are_legs_together = leg_split_ratio <= MAX_LEG_SPLIT_RATIO

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1.1 - 1.7: Sit with Both Legs Extended & Slightly Lifted)
        # ---------------------------------------------------------
        is_seated_posture = (
            (active_hip.y > active_shoulder.y)
            and (SETUP_MIN_TORSO_LEAN_DEG <= torso_lean_deg <= SETUP_MAX_TORSO_LEAN_DEG)
        )
        # Requires legs extended in front (knee_angle >= 132°) so straightening from bent knees never triggers a false rep!
        is_legs_extended_start = knee_angle >= SETUP_EXTENDED_MIN_KNEE_ANGLE

        is_setup_ready = (
            is_side
            and is_seated_posture
            and hands_supporting_behind
            and is_legs_extended_start
            and feet_off_floor
            and legs_under_30_deg
        )

        if is_setup_ready:
            self.ready_frames += 1
            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif torso_lean_deg > SETUP_MAX_TORSO_LEAN_DEG or active_hip.y <= active_shoulder.y:
                feedback_msg = "Sit up with chest lifted (do not lie flat on your back)"
            elif torso_lean_deg < SETUP_MIN_TORSO_LEAN_DEG:
                feedback_msg = "Lean your upper body slightly backward"
            elif not hands_supporting_behind:
                feedback_msg = "Place both hands flat on the floor slightly behind your hips"
            elif not is_legs_extended_start:
                feedback_msg = "Extend both legs straight out in front of you to start"
            elif not feet_off_floor:
                feedback_msg = "Lift both feet slightly off the floor"
            elif not legs_under_30_deg:
                feedback_msg = "Lower your legs slightly (keep butt-to-ankle angle under 30°)"
            else:
                feedback_msg = "Hold extended starting position to begin"

            return {
                "exercise": "Seated Knee Tucks", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "knee_angle": round(knee_angle, 1), "torso_angle": round(torso_angle, 1)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["EXTENDED", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[SEATED KNEE TUCKS FAULT] AI ST-GCN flagged posture fault: prob={bad_posture_prob:.2f}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES (STEPS 1, 2, 3)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["TUCKING", "TUCKED", "EXTENDING"]:
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)
            self.max_knee_angle = max(self.max_knee_angle, knee_angle)
            self.min_torso_angle = min(self.min_torso_angle, torso_angle)
            self.max_butt_ankle_deg = max(self.max_butt_ankle_deg, butt_to_ankle_deg)

            # A. Side View Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(f"[SEATED KNEE TUCKS FAULT] Lost side view: ratio={shoulder_aspect:.2f}", flush=True)
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Step 1.3: Hands Remain on Floor Behind Hips (No grabbing knees/legs!)
            if not hands_supporting_behind:
                self.hand_frames += 1
                if self.hand_frames >= DEBOUNCE_HANDS:
                    if self.hands_planted_behind:
                        print("[SEATED KNEE TUCKS FAULT] Hands left floor behind hips / grabbed legs!", flush=True)
                    self.hands_planted_behind = False
            else:
                self.hand_frames = max(0, self.hand_frames - 1)

            # C. Step 1.5 & 3.5: Controlled Upper-Body Lean (Do not collapse flat onto back)
            if torso_lean_deg > MAX_LYING_FLAT_LEAN_DEG:
                self.posture_frames += 1
                if self.posture_frames >= DEBOUNCE_POSTURE:
                    if self.seated_lean_maintained:
                        print(f"[SEATED KNEE TUCKS FAULT] Collapsed flat onto back: lean={torso_lean_deg:.1f}°", flush=True)
                    self.seated_lean_maintained = False
            else:
                self.posture_frames = max(0, self.posture_frames - 1)

            # D. Step 2.2 & 3.2: Keep Both Feet Elevated Without Touching the Floor
            if not feet_off_floor:
                self.feet_floor_frames += 1
                if self.feet_floor_frames >= DEBOUNCE_FEET_FLOOR:
                    if self.feet_hovered:
                        print(f"[SEATED KNEE TUCKS FAULT] Feet touched the floor: angle={butt_to_ankle_deg:.1f}°", flush=True)
                    self.feet_hovered = False
            else:
                self.feet_floor_frames = max(0, self.feet_floor_frames - 1)

            # E. Butt-to-Ankle Angle Ceiling (Must NOT pass ~30° upward from butt to ankle)
            if not legs_under_30_deg:
                self.high_leg_frames += 1
                if self.high_leg_frames >= DEBOUNCE_HIGH_LEGS:
                    if self.butt_to_ankle_under_30:
                        print(
                            f"[SEATED KNEE TUCKS FAULT] Butt-to-ankle angle passed 30°! "
                            f"angle={butt_to_ankle_deg:.1f}° (max={MAX_BUTT_TO_ANKLE_ANGLE_DEG}°)",
                            flush=True,
                        )
                    self.butt_to_ankle_under_30 = False
            else:
                self.high_leg_frames = max(0, self.high_leg_frames - 1)

            # F. Step 1.7: Keep Legs Together
            if not are_legs_together:
                self.leg_split_frames += 1
                if self.leg_split_frames >= DEBOUNCE_LEGS_TOGETHER:
                    if self.legs_together:
                        print(f"[SEATED KNEE TUCKS FAULT] Legs separated: split_ratio={leg_split_ratio:.2f}", flush=True)
                    self.legs_together = False
            else:
                self.leg_split_frames = max(0, self.leg_split_frames - 1)

            # G. Step 2.5 & 3.1: Controlled Movement Speed (Both Tucking & Extending)
            if self.prev_knee_angle is not None:
                angular_speed = abs(knee_angle - self.prev_knee_angle)
                if angular_speed > MAX_KNEE_ANGULAR_SPEED_DEG:
                    self.speed_frames += 1
                    if self.speed_frames >= DEBOUNCE_SPEED:
                        if self.controlled_movement:
                            print(f"[SEATED KNEE TUCKS FAULT] Uncontrolled fast leg swing: speed={angular_speed:.1f}", flush=True)
                        self.controlled_movement = False
                else:
                    self.speed_frames = max(0, self.speed_frames - 1)
            self.prev_knee_angle = knee_angle
        else:
            self.prev_knee_angle = None

        # ---------------------------------------------------------
        # 4. STATE MACHINE (EXTENDED START -> TUCKING -> TUCKED -> EXTENDING -> EXTENDED)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_setup_ready:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "EXTENDED"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Slowly bend both knees toward your chest."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Seated Knee Tucks", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "knee_angle": round(knee_angle, 1), "torso_angle": round(torso_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            if self.post_rep_timer == 0:
                self.current_feedback = "Starting position locked! Slowly bend both knees toward your chest."

        # Step 2: Pulling Your Knees In (Initiates ONLY from EXTENDED so EXTENDING never ping-pongs!)
        elif self.state == "EXTENDED" and knee_angle < TUCKING_START_KNEE_ANGLE:
            self.state = "TUCKING"
            self.rep_started = True
            self.min_knee_angle = knee_angle
            self.max_knee_angle = max(knee_angle, SETUP_EXTENDED_MIN_KNEE_ANGLE)
            self.min_torso_angle = torso_angle
            self.max_butt_ankle_deg = butt_to_ankle_deg
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull thighs closer to your stomach and bring upper body slightly forward..."

        # Step 2.4 & 2.6: Reached Tucked Position (Knees <= 90° and thighs close to chest <= 95°)
        elif self.state == "TUCKING" and (
            knee_angle <= TIGHT_TUCK_MAX_KNEE_ANGLE
            and torso_angle <= TIGHT_TUCK_MAX_TORSO_ANGLE
        ):
            self.tuck_depth_reached = True
            self.state = "TUCKED"
            self.half_rep_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Tucked position reached! Slowly extend both legs forward until nearly straight."

        # Premature Extension (Half Rep Branch: Extending legs before reaching full tuck)
        elif self.state == "TUCKING" and not self.tuck_depth_reached and (knee_angle - self.min_knee_angle) > EARLY_REVERSAL_DELTA:
            self.half_rep_frames += 1
            if self.half_rep_frames >= DEBOUNCE_HALF_REP:
                if self.min_knee_angle <= HALF_TUCK_MAX_KNEE_ANGLE:
                    self.was_half_rep = True
                self.state = "EXTENDING"
                self.current_feedback = "Incomplete knee tuck! Extending legs early..."
                print(
                    f"[SEATED KNEE TUCKS FAULT] Early reversal before full tuck: min_knee={self.min_knee_angle:.1f}°, "
                    f"min_torso={self.min_torso_angle:.1f}°",
                    flush=True,
                )

        # Step 3: Extending Your Legs Forward from TUCKED
        elif self.state == "TUCKED" and knee_angle > EXTENDING_START_KNEE_ANGLE:
            self.state = "EXTENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Slowly extend legs until nearly straight (keep feet elevated under 30°)..."

        if self.state == "TUCKING" and not ((knee_angle - self.min_knee_angle) > EARLY_REVERSAL_DELTA):
            self.half_rep_frames = max(0, self.half_rep_frames - 1)

        # ---------------------------------------------------------
        # 5. REP COMPLETION (WHEN LEGS EXTEND NEARLY STRAIGHT >= 132°)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["EXTENDING", "TUCKED", "TUCKING"] and knee_angle >= REP_COMPLETE_KNEE_ANGLE:
            # Ignore minor balancing shifts if knees barely bent (< 20° ROM)
            if (self.max_knee_angle - self.min_knee_angle) < MIN_VALID_REP_ROM_DEG and self.is_side_view and self.butt_to_ankle_under_30:
                self._reset_rep_tracking()
                self.state = "EXTENDED"
                return {
                    "exercise": "Seated Knee Tucks", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "knee_angle": round(knee_angle, 1), "torso_angle": round(torso_angle, 1)
                }

            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stay sideways to camera! Extend legs to reset."
                is_bad_rep = True
            elif not self.hands_planted_behind:
                self.current_feedback = "Bad rep: Keep both hands flat on the floor behind your hips for support!"
                is_bad_rep = True
            elif not self.butt_to_ankle_under_30:
                self.current_feedback = "Bad rep: Butt-to-ankle angle passed 30°! Keep feet lifted only slightly off floor."
                is_bad_rep = True
            elif not self.feet_hovered:
                self.current_feedback = "Bad rep: Keep both feet elevated! Do not allow them to touch the floor."
                is_bad_rep = True
            elif not self.legs_together:
                self.current_feedback = "Bad rep: Keep both legs together throughout the movement!"
                is_bad_rep = True
            elif not self.seated_lean_maintained:
                self.current_feedback = "Bad rep: Keep your chest lifted and lower back controlled! Do not lie flat."
                is_bad_rep = True
            elif not self.controlled_movement:
                self.current_feedback = "Bad rep: Keep movements slow and controlled! Avoid swinging or kicking."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI detected excessive body swinging or balance loss."
                is_bad_rep = True
            elif not self.tuck_depth_reached:
                if self.min_knee_angle <= TIGHT_TUCK_MAX_KNEE_ANGLE and self.min_torso_angle > TIGHT_TUCK_MAX_TORSO_ANGLE:
                    self.current_feedback = "Bad rep: Bring your upper body slightly forward to meet your knees!"
                elif self.min_knee_angle <= HALF_TUCK_MAX_KNEE_ANGLE or self.was_half_rep:
                    self.current_feedback = "Bad rep: Half rep! Pull your knees closer to your chest before extending."
                else:
                    self.current_feedback = "Bad rep: Tuck was too shallow! Bring knees all the way to your chest."
                is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[SEATED KNEE TUCKS REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | min_knee={self.min_knee_angle:.1f}°, "
                    f"ext_knee={knee_angle:.1f}°, max_butt_ankle_deg={self.max_butt_ankle_deg:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean Seated Knee Tuck! Great chest tuck and controlled straight-leg extension."
                print(
                    f"[SEATED KNEE TUCKS REP #{rep_num} -> GOOD REP] "
                    f"min_knee={self.min_knee_angle:.1f}°, ext_knee={knee_angle:.1f}°, "
                    f"max_butt_ankle_deg={self.max_butt_ankle_deg:.1f}°",
                    flush=True,
                )
                self.state = "EXTENDED"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        form_score = 100 if self.state == "EXTENDED" else max(0, min(100, int((180 - knee_angle) / 1.1)))
        if not self.is_side_view: form_score -= 35
        if not self.hands_planted_behind: form_score -= 25
        if not self.butt_to_ankle_under_30: form_score -= 35
        if not self.feet_hovered: form_score -= 25
        if not self.legs_together: form_score -= 20
        if not self.seated_lean_maintained: form_score -= 30
        if not self.controlled_movement: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.tuck_depth_reached and self.state == "EXTENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Seated Knee Tucks",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1),
            "torso_angle": round(torso_angle, 1)
        }