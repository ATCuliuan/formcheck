import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class DumbbellHammerCurlsTracker:
    def __init__(self):
        self.state = "WAITING"
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # <-- NEW: Counter for partial ROM reps
        
        self.peak_contraction_reached = False  # Full ROM (< 70 deg)
        self.half_curl_reached = False         # <-- NEW: Half ROM (<= 100 deg)
        self.min_elbow_angle = 180.0          # <-- NEW: Deepest angle recorded in rep
        self.reversal_frames = 0               # <-- NEW: Debounce counter for jitter
        self.arm_path_straight = True
        self.legs_locked = True

        # Instructor Tripwires
        self.elbows_pinned = True         # Catches the "Shoulder Swing"
        self.torso_stable = True          # Catches the "Body Rock" momentum cheat
        self.ml_posture_maintained = True # AI catches the half-rep
        self.is_ready = False
        self.extended_frames = 0

        self.current_feedback = "Stand tall, arms fully extended"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # Load LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_dumbbell_hammer_curls.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Hammer Curl model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Hammer Curl weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]



        # Use the side of the body that is most visible to the camera
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])
        
        # Count all completed executions (good, bad, half) in total
        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Stand sideways or angle camera to see your arm",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_elbow = r_elbow if r_vis > l_vis else l_elbow
        active_wrist = r_wrist if r_vis > l_vis else l_wrist
        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        elbow_angle = calculate_angle(active_shoulder, active_elbow, active_wrist)

        # 1. Starting Position Guard (Arms must be hanging straight down)
        if elbow_angle > 150:
            self.extended_frames += 1
        else:
            self.extended_frames = 0

        if self.extended_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Dumbbell Hammer Curls",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Lower weights fully to start",
                "knee_angle": round(elbow_angle, 1),
                "hip_angle": 0.0,
                "depth_reached": False
            }

        # 2. ST-GCN Temporal Buffer Update
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
                if predicted_class == 0 and self.state != "EXTENDED":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # 3. Form Tripwires

        wristdrift = abs ( active_elbow.x - active_wrist.x)
        if wristdrift > 0.10 and self.state in ["CURLING", "TOP"]:
            self.wrist_drift =  False 

        elbow_drift = abs(active_shoulder.x - active_elbow.x)
        if elbow_drift > 0.12 and self.state in ["CURLING", "TOP"]:
            self.elbows_pinned = False

        torso_sway = abs(active_shoulder.x - active_hip.x)
        if torso_sway > 0.10 and self.state in ["CURLING", "LOWERING"]:
            self.torso_stable = False

        # Keep track of lowest elbow angle reached during the upward curl
        if self.state in ["CURLING", "TOP"]:
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "EXTENDED"
            self.current_feedback = "Curl it up!"

        elif self.state in ["EXTENDED", "LOWERING"] and elbow_angle < 130:
            self.state = "CURLING"
            self.min_elbow_angle = elbow_angle
            self.reversal_frames = 0
            self.current_feedback = "Squeeze the bicep & brachialis..."

        elif self.state == "CURLING" and elbow_angle < 70:
            # Reached full contraction at the top
            self.state = "TOP"
            self.peak_contraction_reached = True
            self.half_curl_reached = True
            self.current_feedback = "Lower under control"

        # Half rep check: Arm curled past 100° and starts lowering (reverses by > 10° for 3 frames)
        elif self.state == "CURLING" and elbow_angle <= 100 and (elbow_angle - self.min_elbow_angle) > 10:
            self.reversal_frames += 1
            if self.reversal_frames >= 3:
                self.half_curl_reached = True
                self.state = "LOWERING"
                self.current_feedback = "Half rep: Lowering arm..."

        elif self.state == "TOP" and elbow_angle > 90:
            self.state = "LOWERING"

        elif self.state == "LOWERING" and elbow_angle >= 150:
            is_bad_rep = False

            # Form Evaluation (Checked at the bottom of the rep)
            if not self.elbows_pinned:
                self.current_feedback = "No rep: You swung your elbow forward! Pin it to your ribs."
                is_bad_rep = True
            elif not self.torso_stable:
                self.current_feedback = "No rep: Stop rocking your body. Use your arms."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected improper mechanics."
                is_bad_rep = True
            elif not self.wrist_drift:
                self.current_feedback = "No rep: Keep your forearm vertical! Do not curl diagonally."
                is_bad_rep = True
            elif not self.legs_locked:
                self.current_feedback = "No rep: Do not use your legs! Strict press only."
                is_bad_rep = True
                
            # Rep Quality / ROM Evaluation
            if is_bad_rep:
                self.bad_reps += 1
            elif self.peak_contraction_reached:
                self.good_reps += 1
                self.current_feedback = "Perfect curl! Full range of motion."
            elif self.half_curl_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Curl higher toward your shoulder!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Curl was too shallow to count."

            # Reset tripwires and rep tracking for next rep
            self.state = "EXTENDED"
            self.peak_contraction_reached = False
            self.half_curl_reached = False
            self.min_elbow_angle = 180.0
            self.reversal_frames = 0
            self.elbows_pinned = True
            self.torso_stable = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring based on elbow angle
        form_score = max(0, min(100, int(( (160 - elbow_angle) / 90 ) * 100)))
        if not self.elbows_pinned: form_score -= 30
        if not self.torso_stable: form_score -= 30
        if not self.ml_posture_maintained: form_score -= 30
        if self.half_curl_reached and not self.peak_contraction_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Dumbbell Hammer Curls",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(elbow_angle, 1),
            "hip_angle": 0.0,
            "depth_reached": self.peak_contraction_reached
        }