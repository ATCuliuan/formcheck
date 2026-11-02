import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class HurdleStepTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                

        # Sagittal Stage Machine:
        # 0 = Standing tall in side profile
        # 1 = Stepping forward (knee drive up & over)
        # 2 = Touched down on far side (Waiting for return)
        # 3 = Stepping backward to origin
        self.cycle_stage = 0              
        self.active_side = None           # 'LEFT' or 'RIGHT' based on camera side profile
        self.hurdle_cleared = False
        
        # Sagittal Posture Tripwires
        self.support_leg_bent = False     # Standing knee buckle
        self.torso_leaned = False         # Anterior / Posterior trunk lean (swaying back or bowing forward)
        self.hands_on_head = True         # Wrists drop from head
        self.foot_slid = False            # Dragging foot forward without driving knee up
        self.ml_posture_maintained = True 

        # Baselines
        self.baseline_hip_x = None
        self.baseline_knee_y = None
        self.baseline_ankle_x = None
        
        # Debounce counters
        self.support_bend_frames = 0
        self.torso_lean_frames = 0
        self.hands_drop_frames = 0
        self.slide_frames = 0             
        self.ml_error_frames = 0
        self.touchdown_frames = 0
        self.standing_frames = 0
        self.is_ready = False

        self.current_feedback = "Side view: Place hands behind head, stand tall"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load TFLite model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_hurdle_step.tflite")
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
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        if max(l_vis, r_vis) < 2.5:
            self.frame_buffer.clear()
            return {
                "exercise": "Hurdle Step", "state": "WAITING",
                "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                "half_reps": 0, "total_reps": self.good_reps + self.bad_reps,
                "form_score": 0, "feedback": "Step back: Show full body side profile",
                "knee_angle": 0.0, "hip_angle": 0.0, "depth_reached": False
            }

        # Select primary visible profile side (near side facing camera)
        if r_vis >= l_vis:
            active_ear, active_shoulder = r_ear, r_shoulder
            active_hip, active_knee, active_ankle = r_hip, r_knee, r_ankle
            moving_wrist = r_wrist
            other_knee, other_ankle = l_knee, l_ankle
        else:
            active_ear, active_shoulder = l_ear, l_shoulder
            active_hip, active_knee, active_ankle = l_hip, l_knee, l_ankle
            moving_wrist = l_wrist
            other_knee, other_ankle = r_knee, r_ankle

        # Sagittal Knee Angles
        support_knee_angle = calculate_angle(active_hip, active_knee, active_ankle)

        # Sagittal Trunk Inclination (Spine angle relative to true vertical)
        dx_trunk = abs(active_shoulder.x - active_hip.x)
        dy_trunk = abs(active_shoulder.y - active_hip.y)
        trunk_lean_from_vertical = np.degrees(np.arctan2(dx_trunk, dy_trunk)) if dy_trunk > 0 else 0.0

        # Sagittal Hands on Head Check (wrist above clavicle / near head height)
        wrist_at_head = moving_wrist.y < (active_shoulder.y + 0.05)

        # ---------------------------------------------------------
        # 1. SAGITTAL SETUP GUARD
        # ---------------------------------------------------------
        if not self.is_ready:
            is_standing_tall = (support_knee_angle > 150.0) and (trunk_lean_from_vertical < 18.0)

            if is_standing_tall and wrist_at_head:
                self.standing_frames += 1
            else:
                self.standing_frames = 0

            if self.standing_frames > 6:
                self.is_ready = True
                self.state = "STANDING"
                self.baseline_hip_x = active_hip.x
                self.baseline_ankle_x = active_ankle.x
                self.current_feedback = "Ready! Step over hurdle and return."
            else:
                self.state = "WAITING"
                self.frame_buffer.clear()
                return {
                    "exercise": "Hurdle Step", "state": self.state,
                    "good_reps": self.good_reps, "bad_reps": self.bad_reps,
                    "half_reps": 0, "total_reps": self.good_reps + self.bad_reps,
                    "form_score": 0, "feedback": "Stand tall, hands behind head (Side Profile)",
                    "knee_angle": round(support_knee_angle, 1),
                    "hip_angle": round(trunk_lean_from_vertical, 1),
                    "depth_reached": False
                }

        # ---------------------------------------------------------
        # 2. SAGITTAL STATE MACHINE (TRACKING KNEE DRIVE & STRIDE)
        # ---------------------------------------------------------
        # Detect step by knee elevation difference or ankle anterior displacement
        knee_vertical_lift = active_knee.y - other_knee.y
        ankle_stride_disp = abs(active_ankle.x - other_ankle.x)

        # Stage 0: Standing -> First Step Forward
        if self.cycle_stage == 0:
            self.state = "STANDING"
            # Stepping begins when either knee drives up or foot steps out horizontally
            if abs(knee_vertical_lift) > 0.06 or ankle_stride_disp > 0.09:
                self.state = "STEPPING"
                self.cycle_stage = 1
                self.hurdle_cleared = False
                self.baseline_knee_y = min(active_knee.y, other_knee.y)
                self.current_feedback = "Step forward... Drive knee high over hurdle!"

        # Stage 2: Touched down on far side -> Return Step
        elif self.cycle_stage == 2:
            self.state = "HALFWAY"
            if abs(knee_vertical_lift) > 0.05 or ankle_stride_disp > 0.07:
                self.state = "STEPPING"
                self.cycle_stage = 3
                self.current_feedback = "Returning! Step back over hurdle to start."

        # ---------------------------------------------------------
        # 3. SAGITTAL POSTURE TRIPWIRES
        # ---------------------------------------------------------
        if self.cycle_stage in [1, 3]:
            # A. Hands Dropped From Head (10 frames debounce)
            if not wrist_at_head:
                self.hands_drop_frames += 1
                if self.hands_drop_frames >= 10:
                    self.hands_on_head = False
            else:
                self.hands_drop_frames = max(0, self.hands_drop_frames - 2)

            # B. Support Leg Excessive Buckle (< 130 deg, 8 frames debounce)
            # Only checked on the planted supporting leg
            standing_knee_angle = max(support_knee_angle, calculate_angle(l_hip, l_knee, l_ankle) if r_vis >= l_vis else calculate_angle(r_hip, r_knee, r_ankle))
            if standing_knee_angle < 130.0:
                self.support_bend_frames += 1
                if self.support_bend_frames >= 8:
                    self.support_leg_bent = True
            else:
                self.support_bend_frames = max(0, self.support_bend_frames - 1)

            # C. Anterior / Posterior Trunk Lean (> 22° forward bow or backward heave)
            if trunk_lean_from_vertical > 22.0:
                self.torso_lean_frames += 1
                if self.torso_lean_frames >= 8:
                    self.torso_leaned = True
            else:
                self.torso_lean_frames = max(0, self.torso_lean_frames - 1)

            # D. Foot Slide Cheat (Ankle moves forward along X while knee never lifts)
            if self.baseline_knee_y is not None:
                max_knee_rise = self.baseline_knee_y - min(active_knee.y, other_knee.y)
                if ankle_stride_disp > 0.12 and max_knee_rise < 0.03:
                    self.slide_frames += 1
                    if self.slide_frames >= 8:
                        self.foot_slid = True
                else:
                    self.slide_frames = max(0, self.slide_frames - 1)

            # E. Sagittal Hurdle Clearance:
            # Moving knee must elevate significantly toward hip level
            highest_knee_y = min(active_knee.y, other_knee.y)
            lowest_hip_y = max(active_hip.y, (l_hip.y + r_hip.y)/2)
            if highest_knee_y < (lowest_hip_y + 0.15):
                self.hurdle_cleared = True

        # ST-GCN Buffer
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        # ---------------------------------------------------------
        # 4. SAGITTAL TOUCHDOWN & REP RESOLUTION
        # ---------------------------------------------------------
        # Both feet planted when vertical difference is small and knees are extended
        both_feet_down = abs(active_ankle.y - other_ankle.y) < 0.06 and support_knee_angle > 140.0

        if both_feet_down:
            self.touchdown_frames += 1

            # Midway Touchdown (Stage 1 -> Stage 2)
            if self.cycle_stage == 1 and self.touchdown_frames >= 2:
                self.cycle_stage = 2
                self.touchdown_frames = 0
                self.state = "HALFWAY"
                self.slide_frames = 0
                self.current_feedback = "Touched down! Step back to start."

            # Final Return Touchdown (Stage 3 -> Completed Rep)
            elif self.cycle_stage == 3 and self.touchdown_frames >= 2:
                is_bad_rep = False

                if self.foot_slid:
                    self.current_feedback = "No rep: Do not slide foot! Drive knee upward."
                    is_bad_rep = True
                elif not self.hurdle_cleared:
                    self.current_feedback = "No rep: Lift knee higher over the hurdle!"
                    is_bad_rep = True
                elif not self.hands_on_head:
                    self.current_feedback = "No rep: Keep hands behind head."
                    is_bad_rep = True
                elif self.support_leg_bent:
                    self.current_feedback = "No rep: Keep support leg straight."
                    is_bad_rep = True
                elif self.torso_leaned:
                    self.current_feedback = "No rep: Do not lean torso forward or back."
                    is_bad_rep = True

                if is_bad_rep:
                    self.bad_reps += 1
                else:
                    self.good_reps += 1
                    self.current_feedback = "Clean repetition! Full hurdle step complete."

                # Reset to ready for next repetition
                self.cycle_stage = 0
                self.state = "STANDING"
                self.hurdle_cleared = False
                self.support_leg_bent = False
                self.torso_leaned = False
                self.hands_on_head = True
                self.foot_slid = False
                
                self.baseline_knee_y = None
                self.slide_frames = 0
                self.support_bend_frames = 0
                self.torso_lean_frames = 0
                self.hands_drop_frames = 0
                self.touchdown_frames = 0
                self.frame_buffer.clear()
        else:
            self.touchdown_frames = 0

        form_score = 100
        if self.support_leg_bent: form_score -= 25
        if self.torso_leaned: form_score -= 25
        if not self.hands_on_head: form_score -= 15
        if self.foot_slid: form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Hurdle Step",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": 0,
            "total_reps": self.good_reps + self.bad_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(support_knee_angle, 1),
            "hip_angle": round(trunk_lean_from_vertical, 1),
            "depth_reached": self.hurdle_cleared
        }