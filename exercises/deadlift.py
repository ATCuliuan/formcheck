import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DeadliftTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility

        self.depth_reached = False         # Full depth: strict hip hinge <= 118° AND wrists passing below kneecap
        self.was_half_rep = False          # Latches an early reversal to enforce a bad rep
        self.min_hinge_angle = 180.0       # Tracks peak hinge depth
        
        # Biomechanical Tripwires (SAGITTAL / SIDE VIEW)
        self.is_side_view = True           # Profile camera orientation check
        self.back_neutral = True           # Catches excessive lumbar/thoracic rounding
        self.not_squatted = True           # Catches dropping hips into a squat instead of hinging
        self.bar_path_close = True         # Catches barbell drifting away from shins
        self.heels_flat = True             # Catches lifting heels off floor
        self.controlled_descent = True     # Catches free-falling or uncontrolled drop
        self.ml_posture_maintained = True

        # Debounce counters
        self.profile_frames = 0
        self.back_frames = 0
        self.squat_frames = 0
        self.drift_frames = 0
        self.heel_frames = 0
        self.drop_frames = 0
        self.standing_frames = 0
        self.lockout_frames = 0
        self.post_rep_timer = 0
        
        self.prev_hinge_angle = None
        self.baseline_ankle_y = None
        self.is_ready = False

        self.current_feedback = "Stand sideways to the camera holding weights"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model

        self.interpreter = None

        model_path = os.path.join("models", "tflite", "st_gcn_deadlift.tflite")

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

    def process_frame(self, landmarks):
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Determine dominant side closest to camera
        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile of working body",
                "hip_angle": 0.0, "knee_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        # Sagittal Kinematics
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        spine_angle = calculate_angle(active_ear, active_shoulder, active_hip)
        hinge_angle = (hip_angle + knee_angle) / 2.0

        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < 0.42

        # Stable Lockout Check at Standing
        is_standing = (knee_angle > 158.0) and (hip_angle > 155.0) and is_side

        # ---------------------------------------------------------
        # 1. SETUP GUARD
        # ---------------------------------------------------------
        if is_standing:
            self.standing_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Stand tall holding weights to begin"
            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1),
                "depth_reached": False
            }

        # ---------------------------------------------------------
        # 2. ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)

                self.interpreter.invoke()

                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            # A. Orientation Check
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # B. Spinal Flexion / Rounding
            if spine_angle < 130.0:
                self.back_frames += 1
                if self.back_frames >= 4:
                    self.back_neutral = False
            else:
                self.back_frames = max(0, self.back_frames - 1)

            # C. Squatting the deadlift (Knees forward, hips drop straight down)
            if knee_angle < 85.0:
                self.squat_frames += 1
                if self.squat_frames >= 4:
                    self.not_squatted = False
            else:
                self.squat_frames = max(0, self.squat_frames - 1)

            # D. Bar Path Drift (Wrists drifting forward away from shin/ankle line)
            wrist_drift = abs(active_wrist.x - active_ankle.x) / torso_len
            if wrist_drift > 0.65:
                self.drift_frames += 1
                if self.drift_frames >= 4:
                    self.bar_path_close = False
            else:
                self.drift_frames = max(0, self.drift_frames - 1)

            # E. Heel Lift (Transferring weight onto toes)
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.045:
                self.heel_frames += 1
                if self.heel_frames >= 3:
                    self.heels_flat = False
            else:
                self.heel_frames = max(0, self.heel_frames - 1)

            self.min_hinge_angle = min(self.min_hinge_angle, hinge_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (STRICT NO HALF REPS - INVALIDATES ON EARLY REVERSAL)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Hinge hips back, lower past knees"

        # Start descent when hinge angle dips below 145°
        elif self.state in ["STANDING", "ASCENDING"] and hinge_angle < 145.0:
            self.state = "DESCENDING"
            self.min_hinge_angle = hinge_angle
            self.lockout_frames = 0
            if self.post_rep_timer == 0:
                self.current_feedback = "Push hips back, keep bar close..."

        # Full Depth Hinge: Must achieve genuine hip hinge (<= 118°) AND wrists clearly below knees
        elif self.state == "DESCENDING" and (hinge_angle <= 118.0 or active_wrist.y > (active_knee.y + 0.02)):
            self.depth_reached = True
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Depth reached! Drive through heels and stand."

        # Half rep trigger: lifter starts coming back up before depth is hit
        elif self.state == "DESCENDING" and (hinge_angle - self.min_hinge_angle) > 7.0 and not self.depth_reached:
            self.was_half_rep = True
            self.state = "ASCENDING"
            self.current_feedback = "No rep: You reversed early! Lower weights past your knees."

        # Ascending out of bottom
        elif self.state == "BOTTOM" and hinge_angle > 122.0:
            self.state = "ASCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Drive hips forward to lockout!"

        # Eccentric drop detector
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_hinge_angle is not None:
                drop_speed = self.prev_hinge_angle - hinge_angle
                if drop_speed > 28.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_hinge_angle = hinge_angle
        else:
            self.prev_hinge_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Re-extending into standing lockout
        # ---------------------------------------------------------
        if is_standing and hinge_angle > 152.0 and self.state in ["ASCENDING", "BOTTOM", "DESCENDING"]:
            self.lockout_frames += 1
            if self.lockout_frames >= 3:
                is_bad_rep = False

                if not self.is_side_view:
                    self.current_feedback = "No rep: Stand sideways to the camera!"
                    is_bad_rep = True
                elif not self.back_neutral:
                    self.current_feedback = "No rep: Back rounded! Keep spine flat and chest proud."
                    is_bad_rep = True
                elif not self.not_squatted:
                    self.current_feedback = "No rep: You squatted! Push hips back instead of dropping them."
                    is_bad_rep = True
                elif not self.bar_path_close:
                    self.current_feedback = "No rep: Bar drifted away! Keep weights close to shins."
                    is_bad_rep = True
                elif not self.heels_flat:
                    self.current_feedback = "No rep: Keep heels planted flat on floor."
                    is_bad_rep = True
                elif not self.controlled_descent:
                    self.current_feedback = "No rep: Control the negative! Lower weights with control."
                    is_bad_rep = True
                elif not self.ml_posture_maintained:
                    self.current_feedback = "No rep: AI flagged spinal curvature or kinetic breakdown."
                    is_bad_rep = True
                elif self.was_half_rep or not self.depth_reached:
                    self.current_feedback = "No rep: Incomplete range! Lower weights fully past knees."
                    is_bad_rep = True

                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Great deadlift! Clean rep."

                self.post_rep_timer = 40
                self.state = "STANDING"
                self.depth_reached = False
                self.was_half_rep = False
                self.min_hinge_angle = 180.0
                self.is_side_view = True
                self.back_neutral = True
                self.not_squatted = True
                self.bar_path_close = True
                self.heels_flat = True
                self.controlled_descent = True
                self.ml_posture_maintained = True
                
                self.lockout_frames = 0
                self.profile_frames = 0
                self.back_frames = 0
                self.squat_frames = 0
                self.drift_frames = 0
                self.heel_frames = 0
                self.drop_frames = 0
                self.prev_hinge_angle = None
                self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((160 - hinge_angle) / 45 * 100))) if self.state != "STANDING" else 100
        if not self.back_neutral: form_score -= 30
        if not self.not_squatted: form_score -= 25
        if not self.bar_path_close: form_score -= 20
        if not self.heels_flat: form_score -= 20
        if not self.controlled_descent: form_score -= 20
        if (self.was_half_rep or not self.depth_reached) and self.state == "ASCENDING": form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Deadlift",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "depth_reached": self.depth_reached
        }