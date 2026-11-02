import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class PushupTracker:
    def __init__(self):
        self.state = "STANDING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Partial ROM pushup counter
        
        self.depth_reached = False         # Full ROM: Elbow <= 95 deg
        self.half_depth_reached = False    # Half ROM: Elbow <= 120 deg
        self.min_elbow_angle = 180.0       # Tracks peak depth
        self.reversal_frames = 0           # Debounce counter
        
        # Posture Tripwires (SIDE / SAGITTAL PROFILE)
        self.core_maintained = True        # Catches hip sag / banana back
        self.no_pike = True                # Catches butt shooting into the air
        self.neck_neutral = True           # Catches chin-poking to fake depth
        self.controlled_descent = True     # Catches free-falling to floor
        self.ml_posture_maintained = True  
        
        self.prev_elbow_angle = None
        self.plank_frames = 0
        self.is_ready = False

        self.current_feedback = "Get down on the floor in a side-profile plank"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model

        
        self.interpreter = None

        
        model_path = os.path.join("models", "tflite", "st_gcn_pushup.tflite")

        
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
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Dynamic profile visibility selection
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_hip.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_hip.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Push-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Show full side profile on floor",
                "elbow_angle": 0.0, "body_angle": 0.0, "depth_reached": False
            }

        active_ear = r_ear if r_vis > l_vis else l_ear
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_ankle = r_ankle if r_vis > l_vis else l_ankle

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)
        neck_angle = calculate_angle(active_ear, active_shoulder, active_hip)

        # 1. Orientation Guard (Must be horizontal on floor)
        dy = abs(active_ankle.y - active_shoulder.y)
        dx = abs(active_ankle.x - active_shoulder.x)
        is_vertical = dy > (dx * 1.1)

        if is_vertical:
            self.is_ready = False
            self.plank_frames = 0
            self.state = "STANDING"
            self.frame_buffer.clear()
            return {
                "exercise": "Push-Up", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Get down on the floor into a plank",
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # 2. Setup Guard: Stable Plank Hold
        if elbow_angle > 150 and body_angle > 155:
            self.plank_frames += 1
        else:
            self.plank_frames = 0

        if self.plank_frames > 12 and not self.is_ready:
            self.is_ready = True
            self.state = "PLANK"

        if not self.is_ready:
            return {
                "exercise": "Push-Up", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Hold straight plank to begin",
                "elbow_angle": round(elbow_angle, 1), "body_angle": round(body_angle, 1),
                "depth_reached": False
            }

        # 3. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)

                
                self.interpreter.invoke()

                
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                if np.argmax(output_data, axis=1)[0] == 0:
                    self.ml_posture_maintained = False
            except Exception: pass

        # ---------------------------------------------------------
        # 4. SIDE-VIEW FORM TRIPWIRES
        # ---------------------------------------------------------
        
        # A. Hip Sag: Hips collapse toward floor (larger Y on screen)
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            if active_hip.y > active_shoulder.y and body_angle < 150:
                self.core_maintained = False

        # B. Pike Cheating: Hips peak upward into a mountain shape
        if self.state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
            if active_hip.y < active_shoulder.y and body_angle < 140:
                self.no_pike = False

        # C. Neck Craning: Dropping chin/head to fake chest depth
        if neck_angle < 130 and self.state in ["DESCENDING", "BOTTOM"]:
            self.neck_neutral = False

        # Track deepest elbow angle
        if self.state in ["DESCENDING", "BOTTOM"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # ---------------------------------------------------------
        # 5. STATE MACHINE (With Half-Rep Logic)
        # ---------------------------------------------------------
        if elbow_angle <= 140 and self.state in ["PLANK", "ASCENDING"]:
            self.state = "DESCENDING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Lower chest toward the floor..."

        # Full Depth (chest to floor level, elbow <= 95 deg)
        elif self.state == "DESCENDING" and elbow_angle <= 95:
            self.depth_reached = True
            self.half_depth_reached = True
            self.state = "BOTTOM"
            self.current_feedback = "Depth good! Drive the floor away."

        # Half Rep Reversal: Reached at least 120°, then pushed back up (> 8° above min point)
        elif self.state == "DESCENDING" and elbow_angle <= 120 and (elbow_angle - self.min_elbow_angle) > 8:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_depth_reached = True
                self.state = "ASCENDING"
                self.current_feedback = "Half rep: Pushing up early..."

        elif self.state == "BOTTOM" and elbow_angle > 105:
            self.state = "ASCENDING"
            self.current_feedback = "Push back to full lockout!"

        # D. Eccentric Drop Detector: Track lowering speed
        if self.state == "DESCENDING":
            if self.prev_elbow_angle is not None:
                drop_speed = self.prev_elbow_angle - elbow_angle
                if drop_speed > 20.0:  # Dropped > 20° in a single frame (~33ms)
                    self.controlled_descent = False
            self.prev_elbow_angle = elbow_angle
        else:
            self.prev_elbow_angle = None

        # Rep Evaluation: Re-locked elbows at top of plank
        if elbow_angle > 150 and self.state == "ASCENDING":
            is_bad_rep = False
            
            # Form Cheats First
            if not self.core_maintained:
                self.current_feedback = "No rep: Hips sagged! Squeeze abs and glutes."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "No rep: Butt too high! Flatten into a real plank."
                is_bad_rep = True
            elif not self.neck_neutral:
                self.current_feedback = "No rep: Keep neck neutral! Lower chest, not chin."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Lower under control, do not drop to the floor."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected broken plank mechanics."
                is_bad_rep = True
                
            # Rep Quality / ROM Checks
            if is_bad_rep:
                self.bad_reps += 1
            elif self.depth_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect push-up!"
            elif self.half_depth_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Lower chest deeper (elbows to 90°)!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Push-up was too shallow."

            # Reset flags for next rep
            self.state = "PLANK"
            self.depth_reached = False
            self.half_depth_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            self.core_maintained = True
            self.no_pike = True
            self.neck_neutral = True
            self.controlled_descent = True
            self.prev_elbow_angle = None
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(100 - abs(elbow_angle - 90) * 0.4)))
        if not self.core_maintained: form_score -= 35
        if not self.no_pike: form_score -= 25
        if not self.neck_neutral: form_score -= 15
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_depth_reached and not self.depth_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Push-Up",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "elbow_angle": round(elbow_angle, 1),
            "body_angle": round(body_angle, 1),
            "depth_reached": self.depth_reached
        }