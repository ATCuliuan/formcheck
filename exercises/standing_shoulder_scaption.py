import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingShoulderScaptionTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility
        
        self.target_height_reached = False # Full ROM: >= 130 deg
        self.max_elevation = 0.0           # Tracks peak elevation reached
        
        # Core Exercise Constraints
        self.wrists_vertical = True       # Enforces vertical wrist (thumbs-up / neutral grip)
        self.arms_y_shape = True          # Enforces diagonal Y-shape path (rejects lateral jumping jacks)
        self.hands_too_close = False      # Enforces arms stay outside the inner body midline
        
        # Posture Tripwires
        self.arms_bent = False            # Catches excessive elbow curling (< 135 deg)
        self.torso_leaned = False         # Catches backward torso lean (< 165 deg)
        self.is_asymmetrical = False      # Catches see-saw uneven elevation
        self.ml_posture_maintained = True # ST-GCN AI flag

        # Debounce counters
        self.wrist_frames = 0
        self.y_shape_frames = 0
        self.inner_body_frames = 0
        self.bent_frames = 0
        self.lean_frames = 0
        self.asym_frames = 0
        self.standing_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, arms at sides with wrists vertical (thumbs up)"
        self.mp_pose = mp.solutions.pose

        # Temporal buffer for ST-GCN
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_standing_shoulder_scaption.tflite")

        
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
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_index = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value]
        r_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        total_vis = sum([
            l_shoulder.visibility, r_shoulder.visibility,
            l_elbow.visibility, r_elbow.visibility,
            l_wrist.visibility, r_wrist.visibility,
            l_hip.visibility, r_hip.visibility
        ])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if total_vis < 5.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Scaption",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Torso and arms not fully visible",
                "elevation_angle": 0.0, "depth_reached": False
            }

        # Torso and shoulder geometry
        shoulder_width = max(abs(l_shoulder.x - r_shoulder.x), 0.25)
        body_center_x = (l_shoulder.x + r_shoulder.x) / 2.0

        # Elevation Angles (Arm raising relative to torso)
        l_elevation_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_elevation_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        avg_elevation = (l_elevation_angle + r_elevation_angle) / 2.0

        # Elbow Angles to catch excessive curling / bent arms
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow_angle = min(l_elbow_angle, r_elbow_angle)

        # Torso Angle (Shoulder -> Hip -> Knee) to catch backward lean
        l_torso_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_torso_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_torso_angle = (l_torso_angle + r_torso_angle) / 2.0

        # ---------------------------------------------------------
        # 1. VERTICAL WRIST CHECK (Hand Vector: Wrist -> Index Knuckle)
        # ---------------------------------------------------------
        l_hand_dx = abs(l_index.x - l_wrist.x)
        l_hand_dy = abs(l_index.y - l_wrist.y)
        r_hand_dx = abs(r_index.x - r_wrist.x)
        r_hand_dy = abs(r_index.y - r_wrist.y)

        # In vertical thumbs-up alignment, hand dy dominates dx.
        l_wrist_is_flat = l_hand_dx > (l_hand_dy * 1.15)
        r_wrist_is_flat = r_hand_dx > (r_hand_dy * 1.15)
        wrist_fails_vertical = l_wrist_is_flat or r_wrist_is_flat

        # ---------------------------------------------------------
        # 2. DIAGONAL Y-SHAPE GEOMETRY (Shoulder to Wrist Vector)
        # ---------------------------------------------------------
        l_arm_dx = abs(l_wrist.x - l_shoulder.x)
        l_arm_dy = max(abs(l_wrist.y - l_shoulder.y), 0.05)
        r_arm_dx = abs(r_wrist.x - r_shoulder.x)
        r_arm_dy = max(abs(r_wrist.y - r_shoulder.y), 0.05)

        l_slope = l_arm_dx / l_arm_dy
        r_slope = r_arm_dx / r_arm_dy
        arm_fails_y_shape = (l_slope > 1.28) or (r_slope > 1.28)

        # ---------------------------------------------------------
        # 3. INNER BODY LINE CHECK (WRIST TOO CLOSE TO MIDLINE)
        # ---------------------------------------------------------
        # Left wrist crosses toward body midline: l_wrist.x goes rightward toward center
        # Right wrist crosses toward body midline: r_wrist.x goes leftward toward center
        inner_margin = 0.15 * shoulder_width
        l_wrist_inner = l_wrist.x > (body_center_x - inner_margin)
        r_wrist_inner = r_wrist.x < (body_center_x + inner_margin)
        wrist_crosses_inner_body = l_wrist_inner or r_wrist_inner

        # ---------------------------------------------------------
        # SETUP GUARD
        # ---------------------------------------------------------
        if avg_elevation < 30 and avg_torso_angle > 165:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Shoulder Scaption",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand tall, arms at sides, wrists vertical (thumbs up)",
                "elevation_angle": round(avg_elevation, 1), "depth_reached": False
            }

        # ---------------------------------------------------------
        # ST-GCN TEMPORAL BUFFER
        # ---------------------------------------------------------
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)

                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)

                self.interpreter.invoke()

                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                predicted_class = np.argmax(output_data, axis=1)[0]
                if predicted_class == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # KINEMATIC FORM TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["LIFTING", "PEAK"]:
            # Rule 1: Vertical Wrist Check (Active between 50° and 125°)
            if 50 <= avg_elevation <= 125:
                if wrist_fails_vertical:
                    self.wrist_frames += 1
                    if self.wrist_frames >= 3:
                        self.wrists_vertical = False
                else:
                    self.wrist_frames = max(0, self.wrist_frames - 1)

            # Rule 2: Diagonal Y-Shape Check (Active between 60° and 120°)
            if 60 <= avg_elevation <= 120:
                if arm_fails_y_shape:
                    self.y_shape_frames += 1
                    if self.y_shape_frames >= 3:
                        self.arms_y_shape = False
                else:
                    self.y_shape_frames = max(0, self.y_shape_frames - 1)

            # Rule 3: Inner Body Line Check (Active between 45° and 125°)
            if 45 <= avg_elevation <= 125:
                if wrist_crosses_inner_body:
                    self.inner_body_frames += 1
                    if self.inner_body_frames >= 3:
                        self.hands_too_close = True
                else:
                    self.inner_body_frames = max(0, self.inner_body_frames - 1)

            # Elbow Bend Check (Allows soft anatomical bend >= 135°)
            if min_elbow_angle < 135:
                self.bent_frames += 1
                if self.bent_frames >= 6:
                    self.arms_bent = True
            else:
                self.bent_frames = max(0, self.bent_frames - 1)

            # Torso Lean Check
            if avg_torso_angle < 165:
                self.lean_frames += 1
                if self.lean_frames >= 4:
                    self.torso_leaned = True
            else:
                self.lean_frames = max(0, self.lean_frames - 1)

            # Asymmetrical Elevation Check
            if abs(l_elevation_angle - r_elevation_angle) > 20:
                self.asym_frames += 1
                if self.asym_frames >= 4:
                    self.is_asymmetrical = True
            else:
                self.asym_frames = max(0, self.asym_frames - 1)

            self.max_elevation = max(self.max_elevation, avg_elevation)

        # ---------------------------------------------------------
        # STATE MACHINE (NO HALF REPS - FULL Y-ELEVATION ONLY)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise arms diagonally into a Y-shape!"

        elif self.state in ["STANDING", "LOWERING"] and avg_elevation >= 40:
            self.state = "LIFTING"
            self.max_elevation = avg_elevation
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise arms in a diagonal Y, thumbs up..."

        # Full ROM: Both arms reach >= 130 deg
        elif self.state == "LIFTING" and l_elevation_angle >= 130 and r_elevation_angle >= 130:
            self.target_height_reached = True
            self.state = "PEAK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Peak Y height reached! Lower slowly."

        # Early downward reversal before reaching peak overhead Y
        elif self.state == "LIFTING" and (self.max_elevation - avg_elevation) > 8:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Raise arms all the way into overhead Y."

        elif self.state == "PEAK" and avg_elevation < 120:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent..."

        # ---------------------------------------------------------
        # REP COMPLETION: Return down (avg elevation < 40 deg)
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "PEAK", "LIFTING"] and avg_elevation < 40:
            is_bad_rep = False

            # Form Cheats First
            if self.hands_too_close:
                self.current_feedback = "No rep: Arms too close to the body! Raise outward in a diagonal Y-shape."
                is_bad_rep = True
            elif not self.wrists_vertical:
                self.current_feedback = "No rep: Keep wrists vertical (thumbs pointing up)! Do not flatten hands."
                is_bad_rep = True
            elif not self.arms_y_shape:
                self.current_feedback = "No rep: Raise arms in a diagonal Y-shape, not straight out to the sides!"
                is_bad_rep = True
            elif self.torso_leaned:
                self.current_feedback = "No rep: Do not lean backward to heave arms up."
                is_bad_rep = True
            elif self.arms_bent:
                self.current_feedback = "No rep: Arms bent too much! Maintain soft extension, do not curl."
                is_bad_rep = True
            elif self.is_asymmetrical:
                self.current_feedback = "No rep: Both arms must lift equally into the Y."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected shoulder shrugging/form breakdown."
                is_bad_rep = True
            elif not self.target_height_reached:
                self.current_feedback = "No rep: Perform a full rep! Raise arms all the way into overhead Y."
                is_bad_rep = True

            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect diagonal Y-raise scaption!"

            self.post_rep_timer = 40

            # Reset flags for next rep
            self.state = "STANDING"
            self.target_height_reached = False
            self.max_elevation = 0.0
            self.wrists_vertical = True
            self.arms_y_shape = True
            self.hands_too_close = False
            self.arms_bent = False
            self.torso_leaned = False
            self.is_asymmetrical = False
            self.ml_posture_maintained = True
            self.wrist_frames = 0
            self.y_shape_frames = 0
            self.inner_body_frames = 0
            self.bent_frames = 0
            self.lean_frames = 0
            self.asym_frames = 0
            self.frame_buffer.clear()

        # Dynamic Form Score
        form_score = max(0, min(100, int((avg_elevation / 130) * 100)))
        if self.hands_too_close: form_score -= 30
        if not self.wrists_vertical: form_score -= 30
        if not self.arms_y_shape: form_score -= 35
        if self.torso_leaned: form_score -= 35
        if self.arms_bent: form_score -= 25
        if self.is_asymmetrical: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.target_height_reached: form_score -= 25
        form_score = max(0, form_score)

        return {
            "exercise": "Shoulder Scaption",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elevation_angle": round(avg_elevation, 1),
            "depth_reached": self.target_height_reached
        }