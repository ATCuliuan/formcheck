import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (OVERHEAD TRAP RAISE - STRICT SIDE VIEW)
# =====================================================================

# 1. Setup & Starting Guard (Step 1-3: Side View, Plate at Hips, Tall Core)
SETUP_MAX_RAISE_ANGLE = 55.0         # Plate resting near hips/thighs[cite: 8]
SETUP_MIN_ELBOW_ANGLE = 125.0        # Arms straight or with a very soft bend at elbows
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.10 # Strict side profile ratio (rejects front view)
SETUP_MAX_HIP_SPAN_RATIO = 0.16      # Hips aligned sideways to camera
SETUP_MIN_READY_FRAMES = 6           # Consecutive frames to confirm READY[cite: 8]
UNLOCK_RESET_FRAMES = 15             # Frames held with plate at hips to clear LOCKED_FAULT (~0.5s)[cite: 8]
MIN_VISIBILITY_THRESHOLD = 1.8       # Minimum upper-body keypoint visibility[cite: 8]

# 2. Kinematic & Form Thresholds (Step 4-6: Soft Elbows, Braced Core, Controlled Negative)
MIN_ARM_STRAIGHT_ANGLE = 120.0       # Catches elbow curling/bending cheat while allowing soft elbow bend
MAX_TORSO_LEAN_DELTA = 5.0          # Max degrees spine can lean/heave backward or forward from baseline[cite: 8]
ECCENTRIC_DROP_SPEED_MAX = 28.0      # Angular drop speed cutoff (deg/frame) to enforce controlled negative

# 3. Overhead Arc Progression & Half-Rep Thresholds (Step 4 & 5: Full Overhead Squeeze)
ASCENDING_START_ANGLE = 52.0         # Lifting plate away from hips initiates ASCENDING
MIN_VALID_REP_PEAK_ANGLE = 65.0      # Below 65° is ignored as minor hip/setup adjustment
HALF_RAISE_MIN_ANGLE = 75.0          # Stopping between 75° (chest/eye level) and 142° flags Half Rep (Bad Rep)
FULL_OVERHEAD_MIN_ANGLE = 142.0      # Reaching overhead (>= 142°) confirms full lower-trap lockout
DESCENDING_TRIGGER_DELTA = 12.0      # Drop from max peak to transition into DESCENDING
REP_COMPLETE_ANGLE = 55.0            # Returning plate to hip level completes the rep

# 4. Debounce Frame Thresholds
DEBOUNCE_CURL = 4                    # Frames before flagging bent elbows / arm curling
DEBOUNCE_PROFILE = 2                # Fast trigger for losing side view[cite: 8]
DEBOUNCE_TORSO = 2                  # Frames before flagging backward spinal heave
DEBOUNCE_DROP = 2                    # Frames before flagging uncontrolled drop[cite: 8]
DEBOUNCE_HALF_REP = 4                # Consecutive frames dropping before confirming early reversal
DEBOUNCE_ML_FAULT = 5                # Consecutive frames for ST-GCN fault[cite: 8]

# 5. Machine Learning Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.75 # Softmax confidence threshold[cite: 8]

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30                 # ST-GCN temporal window[cite: 8]
POST_REP_FEEDBACK_FRAMES = 40        # Post-rep feedback hold frames[cite: 8]

# =====================================================================


class OverheadTrapRaiseTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.rep_started = False
        self.overhead_reached = False
        self.half_raise_reached = False
        self.max_raise_angle = 0.0

        # Instructor Tripwires (STRICT SIDE VIEW - UPPER BODY ONLY)
        self.is_side_view = True           # Enforces side profile view[cite: 8]
        self.arms_straight = True          # Catches bicep curling / excessive elbow bend[cite: 8]
        self.torso_upright = True          # Catches leaning back to heave the plate[cite: 8]
        self.controlled_descent = True     # Catches dropping the plate on the negative[cite: 8]
        self.ml_posture_maintained = True  # ST-GCN posture check (catches shrugging/rounding)[cite: 8]

        self.baseline_torso_angle = None

        # Debounce counters
        self.curl_frames = 0
        self.profile_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0
        self.ready_frames = 0
        self.unlock_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.prev_raise_angle = None
        self.current_feedback = "Turn sideways to camera: hold plate at hip level"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_overhead_trap_raises.tflite")
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
        self.overhead_reached = False
        self.half_raise_reached = False
        self.max_raise_angle = 0.0
        self.prev_raise_angle = None
        self.reversal_frames = 0

        self.is_side_view = True
        self.arms_straight = True
        self.torso_upright = True
        self.controlled_descent = True
        self.ml_posture_maintained = True

        self.curl_frames = 0
        self.profile_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        self.ml_fault_frames = 0

        self.frame_buffer.clear()

    def process_frame(self, landmarks):
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
                "exercise": "Overhead Trap Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of upper body and hips",
                "elbow_angle": 0.0, "raise_angle": 0.0
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        # Sagittal Joint Angles[cite: 8]
        raise_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        # Strict Side Profile Verification (No foot/toe dependencies)
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        hip_span_x = abs(l_hip.x - r_hip.x)
        shoulder_aspect = shoulder_span_x / torso_len
        hip_aspect = hip_span_x / torso_len

        is_side = (shoulder_aspect < SETUP_MAX_SHOULDER_SPAN_RATIO) and (hip_aspect < SETUP_MAX_HIP_SPAN_RATIO)

        # Sagittal Spine Tilt (0° = vertical upright)[cite: 8]
        dx_spine = active_shoulder.x - active_hip.x
        dy_spine = max(abs(active_shoulder.y - active_hip.y), 0.10)
        spine_tilt_deg = np.degrees(np.arctan2(dx_spine, dy_spine))

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Step 1-3: Side View, Plate at Hips, Arms Extended)
        # ---------------------------------------------------------
        is_resting_arms = (
            is_side
            and (raise_angle < SETUP_MAX_RAISE_ANGLE)
            and (elbow_angle > SETUP_MIN_ELBOW_ANGLE)
        )

        if is_resting_arms:
            self.ready_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = spine_tilt_deg
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * spine_tilt_deg

            if self.ready_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.ready_frames = max(0, self.ready_frames - 1)

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera (strict side view required)"
            elif elbow_angle <= SETUP_MIN_ELBOW_ANGLE:
                feedback_msg = "Extend your arms (keep only a soft bend in elbows)"
            else:
                feedback_msg = "Hold weight at hip level to start"

            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
            }

        # ---------------------------------------------------------
        # 2. ST-GCN BUFFER & INFERENCE (SOFTMAX + DEBOUNCE)
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

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["RESTING", "WAITING", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[TRAP RAISE FAULT] AI ST-GCN flagged posture/shrug fault: prob={bad_posture_prob:.2f}",
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
        if self.rep_started and self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            # A. Elbow Bending / Bicep Curling Cheat[cite: 8]
            if elbow_angle < MIN_ARM_STRAIGHT_ANGLE:
                self.curl_frames += 1
                if self.curl_frames >= DEBOUNCE_CURL:
                    if self.arms_straight:
                        print(
                            f"[TRAP RAISE FAULT] Excessive elbow bend / curl cheat: elbow_angle={elbow_angle:.1f}° "
                            f"(min={MIN_ARM_STRAIGHT_ANGLE}°)",
                            flush=True,
                        )
                    self.arms_straight = False
            else:
                self.curl_frames = max(0, self.curl_frames - 1)

            # B. Strict Side Profile Tripwire[cite: 8]
            # B. Strict Side Profile Tripwire (Uses config constants directly)
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[TRAP RAISE FAULT] Lost side profile (Front View Detected): "
                            f"shoulder_aspect={shoulder_aspect:.2f} (max={SETUP_MAX_SHOULDER_SPAN_RATIO}), "
                            f"hip_aspect={hip_aspect:.2f} (max={SETUP_MAX_HIP_SPAN_RATIO})",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

                
            # C. Torso Lean / Heaving Cheat (Step 2: Core Braced, Tall Spine)
            if self.baseline_torso_angle is not None:
                spine_delta = abs(spine_tilt_deg - self.baseline_torso_angle)
                if spine_delta > MAX_TORSO_LEAN_DELTA:
                    self.torso_frames += 1
                    if self.torso_frames >= DEBOUNCE_TORSO:
                        if self.torso_upright:
                            print(
                                f"[TRAP RAISE FAULT] Torso leaned/heaved: spine_delta={spine_delta:.1f}° "
                                f"(max={MAX_TORSO_LEAN_DELTA}°)",
                                flush=True,
                            )
                        self.torso_upright = False
                else:
                    self.torso_frames = max(0, self.torso_frames - 1)

            self.max_raise_angle = max(self.max_raise_angle, raise_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (WITH FULL OVERHEAD CHECK & HALF-REP DETECTION)
        # ---------------------------------------------------------
        if self.state == "LOCKED_FAULT":
            if is_resting_arms:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "RESTING"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Engage scapula and raise overhead."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Pull shoulders down/back and raise plate overhead!"

        # Step 4: Initiate Smooth Overhead Arc
        elif self.state in ["RESTING", "DESCENDING"] and raise_angle > ASCENDING_START_ANGLE:
            self.state = "ASCENDING"
            self.rep_started = True
            self.max_raise_angle = raise_angle
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Lift plate in a smooth arc all the way overhead..."

        # Step 5: Full Overhead Lockout & Trap Squeeze (>= 142°)
        elif self.state == "ASCENDING" and raise_angle >= FULL_OVERHEAD_MIN_ANGLE:
            self.overhead_reached = True
            self.state = "TOP"
            self.reversal_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Squeeze lower/middle traps at the top! Lower slowly."

        # Premature Reversal (Half Rep Branch: Stopped before reaching overhead)
        elif self.state == "ASCENDING" and not self.overhead_reached and (self.max_raise_angle - raise_angle) > DESCENDING_TRIGGER_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_raise_angle >= HALF_RAISE_MIN_ANGLE:
                    self.half_raise_reached = True
                self.state = "DESCENDING"
                self.current_feedback = "Incomplete overhead height! Lowering early..."
                print(
                    f"[TRAP RAISE FAULT] Early reversal before overhead lockout: max_raise={self.max_raise_angle:.1f}° "
                    f"(required={FULL_OVERHEAD_MIN_ANGLE}°)",
                    flush=True,
                )

        # Step 6: Lower Under Control from TOP
        elif self.state == "TOP" and (self.max_raise_angle - raise_angle) > DESCENDING_TRIGGER_DELTA:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Fight the weight on the way down to hip level..."

        if self.state == "ASCENDING" and not ((self.max_raise_angle - raise_angle) > DESCENDING_TRIGGER_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        # Eccentric drop speed monitor (Step 6: Lower under control)[cite: 8]
        if self.rep_started and self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_raise_angle is not None:
                drop_speed = self.prev_raise_angle - raise_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_descent:
                            print(
                                f"[TRAP RAISE FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_descent = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_raise_angle = raise_angle
        else:
            self.prev_raise_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION & EVALUATION (HALF REPS FLAGGED AS BAD REPS)
        # ---------------------------------------------------------
        if self.rep_started and self.state in ["DESCENDING", "TOP", "ASCENDING"] and raise_angle <= REP_COMPLETE_ANGLE:
            # Ignore tiny setup twitches near the hips (< 65°)
            if self.max_raise_angle < MIN_VALID_REP_PEAK_ANGLE and self.is_side_view and self.arms_straight:
                self._reset_rep_tracking()
                self.state = "RESTING"
                return {
                    "exercise": "Overhead Trap Raise", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": self.half_reps, "total_reps": total_reps,
                    "form_score": 100, "feedback": self.current_feedback,
                    "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
                }

            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to camera! Do not face front."
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "Bad rep: Keep arms straight (or soft bend only)! Do not curl weight."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "Bad rep: Keep core braced and spine tall! Do not lean back."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "Bad rep: Control the negative! Lower the plate slowly to hips."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged shoulder shrugging or posture breakdown."
                is_bad_rep = True
            elif not self.overhead_reached:
                if self.max_raise_angle >= HALF_RAISE_MIN_ANGLE or self.half_raise_reached:
                    self.current_feedback = "Bad rep: Half rep! Lift the plate all the way overhead."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Raise was too shallow! Lift plate all the way overhead."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep or is_half_rep:
                self.bad_reps += 1
                print(
                    f"[TRAP RAISE REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_raise={self.max_raise_angle:.1f}°, elbow={elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Excellent Overhead Trap Raise! Great overhead arc and trap control."
                print(
                    f"[TRAP RAISE REP #{rep_num} -> GOOD REP] "
                    f"max_raise={self.max_raise_angle:.1f}°, elbow={elbow_angle:.1f}°",
                    flush=True,
                )
                self.state = "RESTING"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring[cite: 8]
        form_score = 100 if self.state == "RESTING" else max(0, min(100, int((raise_angle / 145) * 100)))
        if not self.arms_straight: form_score -= 40
        if not self.is_side_view: form_score -= 40
        if not self.torso_upright: form_score -= 30
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.overhead_reached and self.state == "DESCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Overhead Trap Raise", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps,
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1), "raise_angle": round(raise_angle, 1)
        }