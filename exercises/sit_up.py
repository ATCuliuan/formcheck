import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class SitUpTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera harness compatibility

        self.depth_reached = False         # Full torso compression (hip <= 65°)
        self.min_hip_angle = 180.0         # Tracks peak compression

        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Profile camera alignment check
        self.neck_neutral = True           # Catches neck yanking / violent chin-tuck
        self.feet_planted = True           # Catches lifting/swinging feet for momentum
        self.controlled_descent = True     # Catches free-falling backwards
        self.ml_posture_maintained = True  # ST-GCN AI flag

        # Debounce counters
        self.profile_frames = 0
        self.neck_frames = 0
        self.feet_frames = 0
        self.drop_frames = 0
        self.lying_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.baseline_ankle_y = None
        self.prev_hip_angle = None
        self.current_feedback = "Lie flat on floor with knees bent sideways to camera"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load TFLite model

        self.interpreter = None

        model_path = os.path.join("models", "tflite", "st_gcn_sit_up.tflite")

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

        # Calculate side-dominant visibility
        l_vis = sum([l_ear.visibility, l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_ear.visibility, r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.5: 
            self.depth_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Sit-Up",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Full side profile not visible",
                "hip_angle": 0.0, "neck_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        # Sagittal kinematics
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # Orientation & floor check
        torso_len = np.hypot(active_shoulder.x - active_hip.x, active_shoulder.y - active_hip.y) or 0.40
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        is_side = (shoulder_span_x / torso_len) < 0.42

        # ---------------------------------------------------------
        # 1. SETUP GUARD (Lying flat, knees bent)
        # ---------------------------------------------------------
        is_lying = (hip_angle > 105.0) and (knee_angle < 135.0) and is_side
        if is_lying:
            self.lying_frames += 1
            if self.baseline_ankle_y is None:
                self.baseline_ankle_y = active_ankle.y
            else:
                self.baseline_ankle_y = 0.90 * self.baseline_ankle_y + 0.10 * active_ankle.y
        else:
            self.lying_frames = 0

        if self.lying_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.depth_reached = False
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera" if not is_side else "Lie flat on floor with knees bent to begin"
            return {
                "exercise": "Sit-Up",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "hip_angle": round(hip_angle, 1), "neck_angle": round(neck_angle, 1),
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
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "BOTTOM":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # 3. KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["ASCENDING", "TOP", "DESCENDING"]:
            if not is_side:
                self.profile_frames += 1
                if self.profile_frames >= 4:
                    self.is_side_view = False
            else:
                self.profile_frames = max(0, self.profile_frames - 1)

            # A. Neck Yanking / Violent chin tuck (< 130°)
            if neck_angle < 130.0:
                self.neck_frames += 1
                if self.neck_frames >= 3:
                    self.neck_neutral = False
            else:
                self.neck_frames = max(0, self.neck_frames - 1)

            # B. Feet Lifting / Momentum kick
            if self.baseline_ankle_y and (self.baseline_ankle_y - active_ankle.y) > 0.055:
                self.feet_frames += 1
                if self.feet_frames >= 3:
                    self.feet_planted = False
            else:
                self.feet_frames = max(0, self.feet_frames - 1)

            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

        # ---------------------------------------------------------
        # 4. STATE MACHINE (STRICT NO HALF-REPS)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "BOTTOM"
            if self.post_rep_timer == 0:
                self.current_feedback = "Curl torso up, compress toward knees!"

        elif self.state in ["BOTTOM", "DESCENDING"] and hip_angle < 95.0:
            self.state = "ASCENDING"
            self.min_hip_angle = hip_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Squeeze abs, sit all the way up..."

        # Full ROM: Torso reaches thighs (hip angle <= 65°)
        elif self.state == "ASCENDING" and hip_angle <= 65.0:
            self.depth_reached = True
            self.state = "TOP"
            if self.post_rep_timer == 0:
                self.current_feedback = "Full compression! Lower back down with control."

        # Incomplete Rep reversal: User reverses back to floor early
        elif self.state == "ASCENDING" and (hip_angle - self.min_hip_angle) > 8.0:
            self.state = "DESCENDING"
            self.current_feedback = "Perform a full rep! Sit all the way up to your knees."

        elif self.state == "TOP" and hip_angle > 72.0:
            self.state = "DESCENDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent back to floor..."

        # Eccentric drop detector (slamming back to floor)
        if self.state in ["DESCENDING", "ASCENDING"]:
            if self.prev_hip_angle is not None:
                drop_speed = hip_angle - self.prev_hip_angle
                if drop_speed > 26.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_hip_angle = hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # 5. REP COMPLETION: Return to Floor (hip angle >= 105°)
        # ---------------------------------------------------------
        if self.state in ["DESCENDING", "TOP", "ASCENDING"] and hip_angle >= 105.0:
            is_bad_rep = False

            if not self.is_side_view:
                self.current_feedback = "No rep: Lie sideways to camera!"
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Stop yanking your neck! Lead with chest and abs."
                is_bad_rep = True
            elif not self.feet_planted:
                self.current_feedback = "No rep: Keep feet flat! Do not kick legs for momentum."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Do not drop heavily to floor."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged body twist or broken kinetic chain."
                is_bad_rep = True
            elif not self.depth_reached:
                self.current_feedback = "No rep: Incomplete range! Sit all the way up to knees."
                is_bad_rep = True

            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Great sit-up! Clean rep."

            self.post_rep_timer = 40
            self.state = "BOTTOM"
            self.depth_reached = False
            self.min_hip_angle = 180.0
            self.is_side_view = True
            self.neck_neutral = True
            self.feet_planted = True
            self.controlled_descent = True
            self.ml_posture_maintained = True
            self.profile_frames = 0
            self.neck_frames = 0
            self.feet_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int((115 - hip_angle) / 50 * 100))) if self.state != "BOTTOM" else 100
        if not self.neck_neutral: form_score -= 35
        if not self.feet_planted: form_score -= 35
        if not self.controlled_descent: form_score -= 20
        if not self.depth_reached and self.state == "DESCENDING": form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Sit-Up",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "hip_angle": round(hip_angle, 1),
            "neck_angle": round(neck_angle, 1),
            "depth_reached": self.depth_reached
        }