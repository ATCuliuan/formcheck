import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

# =====================================================================
# TRACKER TUNING CONFIGURATION (DUMBBELL KICKBACK - CLEANED & STREAMLINED)
# =====================================================================

# 1. Setup & Starting Guard
SETUP_MAX_SHOULDER_SPAN_RATIO = 0.48 
SETUP_TORSO_MAX_ANGLE = 60.0         # torso angle is measured from HORIZONTAL (0-90), so 165 never failed
MAX_SHOULDER_ELBOW_X_DRIFT = 0.15    # (no longer used: replaced by the upper-arm-pinned check)
MIN_FOREARM_HORIZONTAL_ANGLE = 55.0  
MAX_FOREARM_HORIZONTAL_ANGLE = 125.0
SETUP_ELBOW_START_MAX_ANGLE = 120.0  
SETUP_UPPER_ARM_PINNED_ANGLE = 55.0  
SETUP_MIN_READY_FRAMES = 4           
UNLOCK_RESET_FRAMES = 10             
SETUP_LOST_RESET_FRAMES = 12         # frames out of hinge (between reps) before setup is required again
MIN_VISIBILITY_THRESHOLD = 1.8       

# 2. Kinematic & Form Thresholds
MAX_TORSO_UPRIGHT_ANGLE = 70.0       # from horizontal; standing up mid-rep (e.g. a standing curl) trips this
LOCKOUT_MIN_FOREARM_ALIGNMENT = 0.75 # cos(angle) between forearm and torso axis required at lockout
MAX_UPPER_ARM_DROP_ANGLE = 55.0      
MAX_ELBOW_HIKE_ANGLE = 55.0          
MAX_TORSO_HEAVE_DELTA = 22.0         
ECCENTRIC_DROP_SPEED_MAX = 30.0      

# 3. Extension Depth & Range of Motion
FULL_LOCKOUT_MIN_ANGLE = 148.0       
HALF_EXT_THRESHOLD_ANGLE = 120.0     
EXTENDING_START_ELBOW_ANGLE = 110.0  
RETURNING_START_ELBOW_ANGLE = 138.0  
EARLY_REVERSAL_DELTA = 10.0          
REP_COMPLETE_ELBOW_ANGLE = 110.0     

# 4. Debounce Frame Thresholds
DEBOUNCE_PROFILE = 6
DEBOUNCE_TORSO = 6
DEBOUNCE_ELBOW_DROP = 6
DEBOUNCE_OVEREXTENSION = 6
DEBOUNCE_CURL = 3
DEBOUNCE_HEAVE = 5
DEBOUNCE_DROP = 3
DEBOUNCE_HALF_REP = 4

# 5. ST-GCN Machine Learning Model Sensitivity
ML_FAULT_CONFIDENCE_THRESHOLD = 0.82
DEBOUNCE_ML_FAULT = 6

# 6. Model & Buffer Settings
SEQUENCE_LENGTH = 30
POST_REP_FEEDBACK_FRAMES = 40


class DumbbellKickbackTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0

        self.target_reached = False
        self.half_ext_reached = False
        self.max_extension = 0.0
        self.rep_in_progress = False

        # Posture Tripwires
        self.is_side_view = True
        self.torso_hinged = True
        self.elbow_pinned = True
        self.arm_not_overextended = True
        self.wrist_behind_elbow = True
        self.no_torso_heave = True
        self.controlled_eccentric = True
        self.ml_posture_maintained = True

        self.baseline_torso_angle = None

        # Debounce counters
        self.profile_frames = 0
        self.torso_frames = 0
        self.elbow_drop_frames = 0
        self.overextension_frames = 0
        self.curl_detected_frames = 0
        self.heave_frames = 0
        self.drop_frames = 0
        self.reversal_frames = 0
        self.ml_fault_frames = 0

        self.standing_frames = 0
        self.unlock_frames = 0
        self.setup_lost_frames = 0
        self.prev_elbow_angle = None
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Hinge forward, align shoulder over elbow, let forearm hang down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = SEQUENCE_LENGTH
        self.frame_buffer = deque(maxlen=self.sequence_length)

        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_kickback.tflite")
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
        self.rep_in_progress = False
        self.target_reached = False
        self.half_ext_reached = False
        self.max_extension = 0.0
        self.prev_elbow_angle = None
        self.reversal_frames = 0

        self.is_side_view = True
        self.torso_hinged = True
        self.elbow_pinned = True
        self.arm_not_overextended = True
        self.wrist_behind_elbow = True
        self.no_torso_heave = True
        self.controlled_eccentric = True
        self.ml_posture_maintained = True

        self.profile_frames = 0
        self.torso_frames = 0
        self.elbow_drop_frames = 0
        self.overextension_frames = 0
        self.curl_detected_frames = 0
        self.heave_frames = 0
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
                "exercise": "Dumbbell Kickback",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Show full side profile",
                "elbow_angle": 0.0,
                "torso_angle": 0.0,
                "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip

        # Clean vector-based torso angle relative to horizontal
        torso_dx = active_shoulder.x - active_hip.x
        torso_dy = active_shoulder.y - active_hip.y
        torso_angle = np.degrees(np.arctan2(abs(torso_dy), abs(torso_dx)))

        upper_arm_angle = calculate_angle(active_hip, active_shoulder, active_elbow)
        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        torso_height = abs(active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_ratio = shoulder_span_x / torso_height
        is_side = shoulder_span_ratio < SETUP_MAX_SHOULDER_SPAN_RATIO

        is_torso_hinged = torso_angle <= SETUP_TORSO_MAX_ANGLE
        # Upper arm must run along the torso (elbow by the rib cage), not hang vertically
        # under the shoulder. Combined with the hinge check, this separates a kickback
        # setup from a standing curl setup.
        is_shoulder_elbow_aligned = upper_arm_angle <= SETUP_UPPER_ARM_PINNED_ANGLE

        forearm_dx = active_wrist.x - active_elbow.x
        forearm_dy = active_wrist.y - active_elbow.y
        forearm_angle_horiz = np.degrees(np.arctan2(abs(forearm_dy), abs(forearm_dx)))
        is_forearm_vertical = (MIN_FOREARM_HORIZONTAL_ANGLE <= forearm_angle_horiz <= MAX_FOREARM_HORIZONTAL_ANGLE) or (active_wrist.y > active_elbow.y)

        is_in_hinge = (
            is_side
            and is_torso_hinged
            and is_shoulder_elbow_aligned
            and is_forearm_vertical
            and (elbow_angle <= SETUP_ELBOW_START_MAX_ANGLE)
        )

        # -----------------
        # 1. SETUP GUARD
        # -----------------
        if is_in_hinge:
            self.standing_frames += 1
            if self.baseline_torso_angle is None:
                self.baseline_torso_angle = torso_angle
            else:
                self.baseline_torso_angle = 0.90 * self.baseline_torso_angle + 0.10 * torso_angle

            if self.standing_frames > SETUP_MIN_READY_FRAMES:
                self.is_ready = True
        else:
            self.standing_frames = max(0, self.standing_frames - 1)

        # Previously is_ready stayed True forever once set, so standing up or switching
        # to another movement kept counting reps. Drop readiness if the hinge setup is
        # lost while no rep is in progress.
        if self.is_ready and not self.rep_in_progress and self.state in ["READY", "WAITING"]:
            if is_in_hinge:
                self.setup_lost_frames = 0
            else:
                self.setup_lost_frames += 1
                if self.setup_lost_frames >= SETUP_LOST_RESET_FRAMES:
                    self.is_ready = False
                    self.setup_lost_frames = 0
                    self.standing_frames = 0
                    self.state = "WAITING"
                    self.baseline_torso_angle = None

        if not self.is_ready and self.state != "LOCKED_FAULT":
            self.state = "WAITING"
            self.frame_buffer.clear()
            if not is_side:
                feedback_msg = "Turn sideways to camera"
            elif not is_torso_hinged:
                feedback_msg = "Hinge forward so your torso is horizontal or sloped"
            elif not is_shoulder_elbow_aligned:
                feedback_msg = "Pin your upper arm along your torso, elbow by your ribs"
            elif not is_forearm_vertical:
                feedback_msg = "Let your forearm hang straight down vertically"
            else:
                feedback_msg = "Hold position to start..."

            return {
                "exercise": "Dumbbell Kickback",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "elbow_angle": round(elbow_angle, 1),
                "torso_angle": round(torso_angle, 1),
                "depth_reached": False
            }

        torso_vec_x = active_hip.x - active_shoulder.x
        torso_vec_y = active_hip.y - active_shoulder.y
        torso_mag = np.hypot(torso_vec_x, torso_vec_y) or 1e-4
        unit_torso_x = torso_vec_x / torso_mag
        unit_torso_y = torso_vec_y / torso_mag

        forearm_vec_x = active_wrist.x - active_elbow.x
        forearm_vec_y = active_wrist.y - active_elbow.y
        posterior_projection = (forearm_vec_x * unit_torso_x) + (forearm_vec_y * unit_torso_y)
        forearm_len = np.hypot(forearm_vec_x, forearm_vec_y) or 1e-4
        # 1.0 = forearm parallel to torso pointing back toward the hips (true kickback lockout)
        forearm_alignment = posterior_projection / forearm_len

        dist_wrist_to_shoulder = np.hypot(active_wrist.x - active_shoulder.x, active_wrist.y - active_shoulder.y)
        dist_elbow_to_shoulder = np.hypot(active_elbow.x - active_shoulder.x, active_elbow.y - active_shoulder.y)

        # -----------------
        # 2. ST-GCN BUFFER
        # -----------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length and self.rep_in_progress:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                bad_posture_prob = probs[0]

                if bad_posture_prob >= ML_FAULT_CONFIDENCE_THRESHOLD and self.state not in ["WAITING", "READY", "LOCKED_FAULT"]:
                    self.ml_fault_frames += 1
                    if self.ml_fault_frames >= DEBOUNCE_ML_FAULT:
                        if self.ml_posture_maintained:
                            print(
                                f"[KICKBACK FAULT] AI ST-GCN flagged bad posture: prob={bad_posture_prob:.2f} "
                                f"(threshold={ML_FAULT_CONFIDENCE_THRESHOLD}), state={self.state}",
                                flush=True,
                            )
                        self.ml_posture_maintained = False
                else:
                    self.ml_fault_frames = max(0, self.ml_fault_frames - 1)
            except Exception:
                pass

        # -----------------
        # 3. TRIPWIRES
        # -----------------
        if self.rep_in_progress and self.state in ["EXTENDING", "LOCKED", "RETURNING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= DEBOUNCE_PROFILE:
                    if self.is_side_view:
                        print(
                            f"[KICKBACK FAULT] Lost side view: shoulder_span_ratio={shoulder_span_ratio:.2f} "
                            f"(max={SETUP_MAX_SHOULDER_SPAN_RATIO})",
                            flush=True,
                        )
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            if torso_angle > MAX_TORSO_UPRIGHT_ANGLE:
                self.torso_frames += 1
                if self.torso_frames >= DEBOUNCE_TORSO:
                    if self.torso_hinged:
                        print(
                            f"[KICKBACK FAULT] Torso upright cheat: torso_angle={torso_angle:.1f}° "
                            f"(max={MAX_TORSO_UPRIGHT_ANGLE}°)",
                            flush=True,
                        )
                    self.torso_hinged = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            if upper_arm_angle > MAX_UPPER_ARM_DROP_ANGLE and active_elbow.y > active_shoulder.y:
                self.elbow_drop_frames += 1
                if self.elbow_drop_frames >= DEBOUNCE_ELBOW_DROP:
                    if self.elbow_pinned:
                        print(
                            f"[KICKBACK FAULT] Dropped elbow: upper_arm_angle={upper_arm_angle:.1f}° "
                            f"(max={MAX_UPPER_ARM_DROP_ANGLE}°)",
                            flush=True,
                        )
                    self.elbow_pinned = False
            else:
                self.elbow_drop_frames = max(0, self.elbow_drop_frames - 1)

            is_elbow_overextended = active_elbow.y < (min(active_shoulder.y, active_hip.y) - 0.05)
            if is_elbow_overextended or (upper_arm_angle > MAX_ELBOW_HIKE_ANGLE and active_elbow.y < active_shoulder.y):
                self.overextension_frames += 1
                if self.overextension_frames >= DEBOUNCE_OVEREXTENSION:
                    if self.arm_not_overextended:
                        print(
                            f"[KICKBACK FAULT] Arm overextended/hiked: upper_arm_angle={upper_arm_angle:.1f}° "
                            f"(max={MAX_ELBOW_HIKE_ANGLE}°)",
                            flush=True,
                        )
                    self.arm_not_overextended = False
            else:
                self.overextension_frames = max(0, self.overextension_frames - 1)

            is_curling = (
                (dist_wrist_to_shoulder < (dist_elbow_to_shoulder * 0.95))
                or (posterior_projection < 0.0)
                or (active_wrist.y < active_elbow.y and elbow_angle < 120)
            )
            if is_curling:
                self.curl_detected_frames += 1
                if self.curl_detected_frames >= DEBOUNCE_CURL:
                    if self.wrist_behind_elbow:
                        print(
                            f"[KICKBACK FAULT] Bicep curl / front swing cheat: posterior_projection={posterior_projection:.3f}, "
                            f"elbow_angle={elbow_angle:.1f}°",
                            flush=True,
                        )
                    self.wrist_behind_elbow = False
            else:
                self.curl_detected_frames = max(0, self.curl_detected_frames - 1)

            if self.baseline_torso_angle is not None:
                if abs(torso_angle - self.baseline_torso_angle) > MAX_TORSO_HEAVE_DELTA:
                    self.heave_frames += 1
                    if self.heave_frames >= DEBOUNCE_HEAVE:
                        if self.no_torso_heave:
                            print(
                                f"[KICKBACK FAULT] Torso heave / momentum: angle_delta={abs(torso_angle - self.baseline_torso_angle):.1f}° "
                                f"(max={MAX_TORSO_HEAVE_DELTA}°)",
                                flush=True,
                            )
                        self.no_torso_heave = False
                else:
                    self.heave_frames = max(0, self.heave_frames - 1)

            self.max_extension = max(self.max_extension, elbow_angle)

        # -----------------
        # 4. STATE MACHINE
        # -----------------
        if self.state == "LOCKED_FAULT":
            if is_in_hinge:
                self.unlock_frames += 1
                if self.unlock_frames >= UNLOCK_RESET_FRAMES:
                    self.state = "READY"
                    self.unlock_frames = 0
                    self._reset_rep_tracking()
                    self.current_feedback = "Reset complete! Pin elbow to ribs and extend back."
            else:
                self.unlock_frames = 0

            return {
                "exercise": "Dumbbell Kickback",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": self.current_feedback,
                "elbow_angle": round(elbow_angle, 1),
                "torso_angle": round(torso_angle, 1),
                "depth_reached": False
            }

        if self.state == "WAITING" and self.is_ready:
            self.state = "READY"
            self.rep_in_progress = False
            if self.post_rep_timer == 0:
                self.current_feedback = "Extend your forearm straight back!"

        elif self.state in ["READY", "RETURNING"] and elbow_angle > EXTENDING_START_ELBOW_ANGLE:
            if posterior_projection > 0.01 and dist_wrist_to_shoulder > dist_elbow_to_shoulder:
                self.state = "EXTENDING"
                self.rep_in_progress = True
                self.max_extension = elbow_angle
                self.reversal_frames = 0
                if self.post_rep_timer == 0:
                    self.current_feedback = "Lock arm straight back..."

        elif self.state == "EXTENDING" and elbow_angle >= FULL_LOCKOUT_MIN_ANGLE:
            if (
                posterior_projection > 0.05
                and forearm_alignment >= LOCKOUT_MIN_FOREARM_ALIGNMENT
                and dist_wrist_to_shoulder > dist_elbow_to_shoulder
                and is_torso_hinged
            ):
                self.target_reached = True
                self.state = "LOCKED"
                if self.post_rep_timer == 0:
                    self.current_feedback = "Full triceps contraction! Lower under control."
            elif not (posterior_projection > 0.05 and dist_wrist_to_shoulder > dist_elbow_to_shoulder):
                # Arm is straight but the forearm is not behind the elbow: curl / front swing
                self.wrist_behind_elbow = False
            # Otherwise (e.g. forearm only slightly off-axis): stay in EXTENDING and keep
            # checking; lockout simply isn't credited until the forearm lines up.

        elif self.state == "EXTENDING" and not self.target_reached and (self.max_extension - elbow_angle) > EARLY_REVERSAL_DELTA:
            self.reversal_frames += 1
            if self.reversal_frames >= DEBOUNCE_HALF_REP:
                if self.max_extension >= HALF_EXT_THRESHOLD_ANGLE:
                    self.half_ext_reached = True
                self.state = "RETURNING"
                self.current_feedback = "Incomplete extension! Returning forearm early..."
                print(
                    f"[KICKBACK FAULT] Early reversal: max_extension={self.max_extension:.1f}° "
                    f"(required >= {FULL_LOCKOUT_MIN_ANGLE}°)",
                    flush=True,
                )

        elif self.state == "LOCKED" and elbow_angle < RETURNING_START_ELBOW_ANGLE:
            self.state = "RETURNING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the weight forward"

        if self.state == "EXTENDING" and not ((self.max_extension - elbow_angle) > EARLY_REVERSAL_DELTA):
            self.reversal_frames = max(0, self.reversal_frames - 1)

        if self.rep_in_progress and self.state in ["RETURNING", "EXTENDING"]:
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > ECCENTRIC_DROP_SPEED_MAX:
                    self.drop_frames += 1
                    if self.drop_frames >= DEBOUNCE_DROP:
                        if self.controlled_eccentric:
                            print(
                                f"[KICKBACK FAULT] Uncontrolled drop speed: {drop_speed:.1f} deg/frame "
                                f"(max={ECCENTRIC_DROP_SPEED_MAX})",
                                flush=True,
                            )
                        self.controlled_eccentric = False
                else:
                    self.drop_frames = max(0, self.drop_frames - 1)
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # -----------------
        # 5. REP COMPLETION
        # -----------------
        if self.rep_in_progress and self.state in ["RETURNING", "LOCKED", "EXTENDING"] and elbow_angle <= REP_COMPLETE_ELBOW_ANGLE:
            is_bad_rep = False
            is_half_rep = False

            if not self.is_side_view:
                self.current_feedback = "Bad rep: Stand sideways to the camera! Reset in hinge."
                is_bad_rep = True
            elif not self.wrist_behind_elbow:
                self.current_feedback = "Bad rep: Do not curl! Kick your arm straight back behind you."
                is_bad_rep = True
            elif not self.torso_hinged:
                self.current_feedback = "Bad rep: Stay hinged! Do not stand up to swing."
                is_bad_rep = True
            elif not self.elbow_pinned:
                self.current_feedback = "Bad rep: Keep upper arm pinned high! Do not swing elbow."
                is_bad_rep = True
            elif not self.arm_not_overextended:
                self.current_feedback = "Bad rep: Do not hike elbow past your back! Keep arm along rib cage."
                is_bad_rep = True
            elif not self.no_torso_heave:
                self.current_feedback = "Bad rep: Do not heave your chest up to swing the weight."
                is_bad_rep = True
            elif not self.controlled_eccentric:
                self.current_feedback = "Bad rep: Control the negative! Do not drop forearm."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "Bad rep: AI flagged poor mechanics/body sway."
                is_bad_rep = True
            elif not self.target_reached:
                if self.max_extension >= HALF_EXT_THRESHOLD_ANGLE or self.half_ext_reached:
                    self.current_feedback = "Bad rep: Half rep! Full extension not reached. Extend arm back fully to reset."
                    is_half_rep = True
                else:
                    self.current_feedback = "Bad rep: Movement was too shallow! Lock arm out to reset."
                    is_bad_rep = True

            rep_num = self.good_reps + self.bad_reps + self.half_reps + 1
            if is_bad_rep:
                self.bad_reps += 1
                print(
                    f"[KICKBACK REP #{rep_num} -> BAD REP] "
                    f"Reason: {self.current_feedback} | max_extension={self.max_extension:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            elif is_half_rep:
                self.bad_reps += 1
                print(
                    f"[KICKBACK REP #{rep_num} -> BAD REP (HALF REP)] "
                    f"Reason: {self.current_feedback} | max_extension={self.max_extension:.1f}°",
                    flush=True,
                )
                self.state = "LOCKED_FAULT"
                self.unlock_frames = 0
                self._reset_rep_tracking()
            else:
                self.good_reps += 1
                self.current_feedback = "Clean kickback! Good rep."
                print(
                    f"[KICKBACK REP #{rep_num} -> GOOD REP] "
                    f"max_extension={self.max_extension:.1f}°",
                    flush=True,
                )
                self.state = "READY"
                self._reset_rep_tracking()

            self.post_rep_timer = POST_REP_FEEDBACK_FRAMES

        # Dynamic Scoring
        form_score = max(0, min(100, int((elbow_angle / 148) * 100)))
        if not self.is_side_view: form_score -= 35
        if not self.torso_hinged: form_score -= 30
        if not self.elbow_pinned: form_score -= 35
        if not self.arm_not_overextended: form_score -= 25
        if not self.wrist_behind_elbow: form_score -= 30
        if not self.no_torso_heave: form_score -= 25
        if not self.controlled_eccentric: form_score -= 20
        if not self.target_reached and self.rep_in_progress: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Kickback",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "torso_angle": round(torso_angle, 1),
            "depth_reached": self.target_reached
        }