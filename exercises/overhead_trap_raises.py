import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class OverheadTrapRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.overhead_reached = False
        
        # Instructor Tripwires
        self.arms_straight = True
        self.core_stable = True           # Catches the dangerous lumbar arch
        self.ml_posture_maintained = True 
        
        self.ready_frames = 0
        self.is_ready = False
        self.current_feedback = "Let arms hang straight down"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load Native LiteRT Model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_overhead_trap_raises.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Overhead Trap Raise model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Overhead Trap Raise weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        # Full 25-joint extraction for ST-GCN consistency
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

        # Dynamic Visibility Check
        l_vis = sum([l_shoulder.visibility, l_elbow.visibility, l_wrist.visibility, l_hip.visibility])
        r_vis = sum([r_shoulder.visibility, r_elbow.visibility, r_wrist.visibility, r_hip.visibility])

        total_reps = self.good_reps + self.bad_reps

        if max(l_vis, r_vis) < 3.0:
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back to show your torso and arms",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        # Calculate kinematics for BOTH arms
        l_shoulder_abduction = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_shoulder_abduction = calculate_angle(r_hip, r_shoulder, r_elbow)
        min_abduction = min(l_shoulder_abduction, r_shoulder_abduction)

        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        min_elbow = min(l_elbow_angle, r_elbow_angle)

        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_hip = r_hip if r_vis > l_vis else l_hip

        # 1. Starting Guard: Arms down, elbows locked
        if min_abduction < 45 and min_elbow > 150:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15: 
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Overhead Trap Raise", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Let arms hang completely straight to start",
                "knee_angle": round(min_elbow, 1), "hip_angle": round(min_abduction, 1)
            }

        # 2. ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.interpreter and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer, dtype=np.float32)
                tensor_input = np.expand_dims(seq, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "WAITING":
                    self.ml_posture_maintained = False
            except Exception: pass

        # 3. Kinematic Form Tripwires
        
        # The T-Rex Raise (Bent Elbows)
        if min_elbow < 150 and self.state in ["ASCENDING", "TOP"]:
            self.arms_straight = False

        # The Lumbar Arch (Leaning back to force the weight up)
        torso_sway = abs(active_shoulder.x - active_hip.x)
        if torso_sway > 0.15 and self.state in ["ASCENDING", "TOP"]:
            self.core_stable = False

        # 4. State Machine
        if self.state == "WAITING" and self.is_ready:
            self.state = "RESTING"
            self.current_feedback = "Raise arms overhead!"

        elif self.state in ["RESTING", "DESCENDING"] and min_abduction > 60:
            self.state = "ASCENDING"
            self.current_feedback = "Keep arms completely straight..."

        elif self.state == "ASCENDING" and min_abduction >= 150:
            self.overhead_reached = True
            self.state = "TOP"
            self.current_feedback = "Squeeze the traps! Lower slowly."

        elif self.state == "TOP" and min_abduction < 130:
            self.state = "DESCENDING"
            self.current_feedback = "Control the descent"

        elif self.state == "DESCENDING" and min_abduction <= 50:
            is_bad_rep = False
            
            if not self.overhead_reached:
                self.current_feedback = "No rep: Raise arms higher (in line with torso)."
                is_bad_rep = True
            elif not self.arms_straight:
                self.current_feedback = "No rep: You bent your elbows."
                is_bad_rep = True
            elif not self.core_stable:
                self.current_feedback = "No rep: Do not arch your lower back to lift the weight."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI flagged poor posture."
                is_bad_rep = True
                
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Excellent trap raise!"
                
            self.state = "RESTING"
            self.overhead_reached = False
            self.arms_straight = True
            self.core_stable = True
            self.ml_posture_maintained = True
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = max(0, min(100, int(( min_abduction / 160 ) * 100)))
        if not self.overhead_reached and self.state == "DESCENDING": form_score -= 30
        if not self.arms_straight: form_score -= 35
        if not self.core_stable: form_score -= 40
        form_score = max(0, form_score)

        return {
            "exercise": "Overhead Trap Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(min_elbow, 1),
            "hip_angle": round(min_abduction, 1)
        }