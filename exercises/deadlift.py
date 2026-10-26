import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp

def calculate_damped_3d_angle(a, b, c):
    """Calculates angle with damped Z to prevent perspective warping."""
    # Scale Z by 0.5 to keep dominant planar movement stable while preserving depth
    ba = np.array([a.x - b.x, a.y - b.y, (a.z - b.z) * 0.5])
    bc = np.array([c.x - b.x, c.y - b.y, (c.z - b.z) * 0.5])
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

class DeadliftTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 

        self.hinge_reached = False         # Full depth hinge (hip <= 120 deg)
        self.half_hinge_reached = False    # Half depth hinge (hip <= 138 deg)
        self.min_hip_angle = 180.0         
        self.reversal_frames = 0           

        # Posture Tripwires
        self.squatted_the_movement = False 
        self.neck_neutral = True           
        self.ml_posture_maintained = True  
        self.fast_descent_warning = False  

        self.squat_frames = 0
        self.neck_frames = 0
        self.drop_frames = 0

        self.prev_hip_angle = None
        self.standing_frames = 0
        self.is_ready = False
        self.current_feedback = "Stand at an angle to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load LiteRT Model
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

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility])
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 1.6:
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step into camera view",
                "hip_angle": 0.0, "knee_angle": 0.0
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        hip_angle = calculate_damped_3d_angle(active_shoulder, active_hip, active_knee)
        neck_angle = calculate_damped_3d_angle(active_ear, active_shoulder, active_hip)

        # Plate-occlusion tolerant ankle estimation
        if active_ankle.visibility > 0.35:
            knee_angle = calculate_damped_3d_angle(active_hip, active_knee, active_ankle)
        else:
            class SyntheticAnkle:
                def __init__(self, k):
                    self.x, self.y, self.z = k.x, k.y + 0.35, k.z
            knee_angle = calculate_damped_3d_angle(active_hip, active_knee, SyntheticAnkle(active_knee))

        # 1. Starting Guard (Lowered to > 138° for easy session start)
        if hip_angle > 138:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 6:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Deadlift", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Stand tall at an angle to begin",
                "hip_angle": round(hip_angle, 1), "knee_angle": round(knee_angle, 1)
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(seq, axis=0))
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # 3. Tripwires (Evaluated during hinge)
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            if knee_angle < 85:
                self.squat_frames += 1
                if self.squat_frames >= 4:
                    self.squatted_the_movement = True
            else:
                self.squat_frames = 0

            if neck_angle < 108:
                self.neck_frames += 1
                if self.neck_frames >= 5:
                    self.neck_neutral = False
            else:
                self.neck_frames = 0

            self.min_hip_angle = min(self.min_hip_angle, hip_angle)

        # 4. State Machine & Flexible Transitions
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.current_feedback = "Hinge hips back..."

        # Start Descent
        elif self.state in ["STANDING", "ASCENDING"] and hip_angle <= 138:
            self.state = "DESCENDING"
            self.min_hip_angle = hip_angle
            self.reversal_frames = 0
            self.current_feedback = "Push hips back, keep back flat..."

        # Full Depth Hinge (hip <= 120 deg)
        elif self.state == "DESCENDING" and hip_angle <= 120:
            self.hinge_reached = True
            self.half_hinge_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Drive through heels, stand up!"

        # Half Rep Reversal (Hinged to at least 138°, then reversed by > 8°)
        elif self.state == "DESCENDING" and hip_angle <= 138 and (hip_angle - self.min_hip_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_hinge_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Standing up early..."

        # Moving Upwards
        elif self.state == "BOTTOM" and hip_angle > 122:
            self.state = "ASCENDING"
            self.current_feedback = "Drive hips forward to lockout!"

        # Fast Descent Monitor (Advisory only)
        if self.state == "DESCENDING":
            if self.prev_hip_angle is not None:
                drop_speed = self.prev_hip_angle - hip_angle
                if drop_speed > 32.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.fast_descent_warning = True
            self.prev_hip_angle = hip_angle
        else:
            self.prev_hip_angle = None

        # 5. Rep Completion: Accessible from BOTH "ASCENDING" and "BOTTOM"
        # Triggers cleanly once hip opens back past 138°
        if self.state in ["ASCENDING", "BOTTOM"] and hip_angle >= 138:
            is_bad_rep = False
            
            # Posture Violations
            if self.squatted_the_movement:
                self.current_feedback = "No rep: You squatted! Push hips back instead of down."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Keep neck neutral with your spine."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged excessive back rounding."
                is_bad_rep = True
                
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            elif self.hinge_reached:
                self.good_reps += 1
                if self.fast_descent_warning:
                    self.current_feedback = "Good rep! Reminder: Lower with more control next time."
                else:
                    self.current_feedback = "Great deadlift!"
            elif self.half_hinge_reached:
                self.half_reps += 1
                if self.fast_descent_warning:
                    self.current_feedback = "Half Rep! Reminder: Control descent & hinge slightly deeper."
                else:
                    self.current_feedback = "Counted as Half Rep: Hinge slightly deeper next time!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Hinge was too shallow."

            # Reset flags
            self.state = "STANDING"
            self.hinge_reached = False
            self.half_hinge_reached = False
            self.min_hip_angle = 180.0
            self.reversal_frames = 0
            self.squat_frames = 0
            self.neck_frames = 0
            self.drop_frames = 0
            self.squatted_the_movement = False
            self.neck_neutral = True
            self.fast_descent_warning = False
            self.prev_hip_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(((150 - hip_angle) / 45) * 100)))
        if self.squatted_the_movement: form_score -= 25
        if not self.neck_neutral: form_score -= 15
        if self.fast_descent_warning: form_score -= 10
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_hinge_reached and not self.hinge_reached: form_score -= 10
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
            "knee_angle": round(knee_angle, 1)
        }