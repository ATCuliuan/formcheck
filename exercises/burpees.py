import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class BurpeesTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Counts shallow or partial burpees
        
        # Posture & Movement Tripwires (Side View)
        self.plank_reached = False
        self.half_plank_reached = False    # Catches incomplete kick-backs
        self.core_maintained = True       # Catches hip sag / banana back
        self.no_pike = True                # Catches butt shooting up into the air
        self.jump_finished = False         # Catches skipping the jump/reach at the top
        self.ml_posture_maintained = True
        
        self.is_ready = False
        self.standing_frames = 0
        self.baseline_ankle_y = None       # Tracks floor level for jump height

        self.current_feedback = "Stand sideways to the camera"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_burpees.tflite")

        if os.path.exists(model_path):
            try:
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"✅ ST-GCN TFLite Burpees model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading TFLite Burpees weights: {e}")
                self.interpreter = None
        else:
            print(f"⚠️ ST-GCN TFLite model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # In side view, one side dominates visibility
        l_vis = sum([l_hip.visibility, l_knee.visibility, l_ankle.visibility, l_shoulder.visibility])
        r_vis = sum([r_hip.visibility, r_knee.visibility, r_ankle.visibility, r_shoulder.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if max(l_vis, r_vis) < 2.5: 
            self.frame_buffer.clear()
            return {
                "exercise": "Burpees", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Step back: full profile not visible",
                "knee_angle": 0.0, "hip_angle": 0.0
            }

        active_hip = r_hip if r_vis > l_vis else l_hip
        active_knee = r_knee if r_vis > l_vis else l_knee
        active_ankle = r_ankle if r_vis > l_vis else l_ankle
        active_shoulder = r_shoulder if r_vis > l_vis else l_shoulder
        active_wrist = r_wrist if r_vis > l_vis else l_wrist

        # Side-view Biomechanical Angles
        knee_angle = calculate_angle(active_hip, active_knee, active_ankle)
        hip_angle = calculate_angle(active_shoulder, active_hip, active_knee)
        body_angle = calculate_angle(active_shoulder, active_hip, active_ankle)

        # 1. Starting Guard (Standing tall sideways)
        if knee_angle > 160 and hip_angle > 155:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 15:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            return {
                "exercise": "Burpees", "state": self.state,
                "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
                "half_reps": self.half_reps, "total_reps": total_reps,
                "form_score": 0, "feedback": "Stand tall sideways to start",
                "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
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
                
                if np.argmax(output_data, axis=1)[0] == 0 and self.state != "STANDING":
                    self.ml_posture_maintained = False
            except Exception:
                pass

        # ---------------------------------------------------------
        # SIDE-VIEW GEOMETRY & STATE MACHINE
        # ---------------------------------------------------------
        
        # In a side view:
        # - Vertical distance between shoulder & ankle collapses when on the floor
        # - Horizontal body span expands when kicking out into a full plank
        vertical_drop = abs(active_shoulder.y - active_ankle.y)
        horizontal_span = abs(active_shoulder.x - active_ankle.x)
        
        is_horizontal = vertical_drop < 0.28 and horizontal_span > 0.35

        # Initialize to STANDING once ready
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            self.baseline_ankle_y = active_ankle.y
            self.current_feedback = "Drop down into a plank!"

        # Drop: Bending down to plant hands
        elif self.state == "STANDING" and (knee_angle < 120 or hip_angle < 100):
            self.state = "DROPPING"
            self.current_feedback = "Kick your legs straight back!"

        # Full Plank reached (horizontal on the floor + flat body line)
        elif self.state == "DROPPING" and is_horizontal and body_angle >= 150:
            self.state = "PLANK"
            self.plank_reached = True
            self.half_plank_reached = True
            self.current_feedback = "Good plank! Jump feet back forward."

        # Half Plank branch: Body is down, but hips are heavily bent or knees not kicked back fully
        elif self.state == "DROPPING" and is_horizontal and body_angle < 150:
            self.half_plank_reached = True
            self.state = "PLANK"
            self.current_feedback = "Half rep: Straighten your legs into a flat plank!"

        # Side Tripwire: Hip Sag (Hips sit lower on screen than shoulders & ankles)
        if self.state == "PLANK" and (active_hip.y > active_shoulder.y and active_hip.y > active_ankle.y):
            if body_angle < 145:
                self.core_maintained = False

        # Side Tripwire: Pike (Hips peak higher on screen than shoulders & ankles)
        if self.state == "PLANK" and (active_hip.y < active_shoulder.y and active_hip.y < active_ankle.y):
            if body_angle < 135:
                self.no_pike = False

        # Transition: Rising from floor (Torso moves back to vertical)
        elif self.state == "PLANK" and vertical_drop > 0.45:
            self.state = "ASCENDING"
            self.current_feedback = "Jump up and reach overhead!"

        # Jump / Extension Verification:
        # Measures ankle floor lift (> 0.04 screen height) or hands raised above shoulders
        if self.state in ["ASCENDING", "STANDING"]:
            jumped = self.baseline_ankle_y is not None and (self.baseline_ankle_y - active_ankle.y) > 0.035
            arms_up = active_wrist.y < active_shoulder.y
            if jumped or arms_up:
                self.jump_finished = True

        # Rep Completion Evaluation
        elif self.state == "ASCENDING" and knee_angle > 160 and hip_angle > 155:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.core_maintained:
                self.current_feedback = "No rep: Hips sagged to the floor (protect your back)."
                is_bad_rep = True
            elif not self.no_pike:
                self.current_feedback = "No rep: Butt too high! Flatten into a real plank."
                is_bad_rep = True
            elif not self.jump_finished:
                self.current_feedback = "No rep: Finish with a jump or hands overhead!"
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: Lazy extension at the top."
                is_bad_rep = True
            
            # Rep Quality / ROM Evaluation
            if is_bad_rep:
                self.bad_reps += 1
            elif self.plank_reached:
                self.good_reps += 1
                self.current_feedback = "Great burpee!"
            elif self.half_plank_reached:
                self.half_reps += 1
                self.current_feedback = "Counted as Half Rep: Kick your feet all the way back next time!"
            else:
                self.bad_reps += 1
                self.current_feedback = "No rep: Incomplete kick-back."

            # Reset state for next rep
            self.state = "STANDING"
            self.plank_reached = False
            self.half_plank_reached = False
            self.core_maintained = True
            self.no_pike = True
            self.jump_finished = False
            self.ml_posture_maintained = True
            self.baseline_ankle_y = active_ankle.y
            self.frame_buffer.clear()

        # Dynamic Scoring
        form_score = 100
        if not self.core_maintained: form_score -= 30
        if not self.no_pike: form_score -= 20
        if not self.jump_finished: form_score -= 25
        if not self.ml_posture_maintained: form_score -= 25
        if self.half_plank_reached and not self.plank_reached: form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Burpees", "state": self.state,
            "good_reps": self.good_reps, "bad_reps": self.bad_reps, 
            "half_reps": self.half_reps, "total_reps": total_reps,
            "form_score": form_score, "feedback": self.current_feedback,
            "knee_angle": round(knee_angle, 1), "hip_angle": round(hip_angle, 1)
        }