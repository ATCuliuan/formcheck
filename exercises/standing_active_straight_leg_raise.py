import os
from collections import deque
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from core.kinematics import calculate_angle

class StandingActiveStraightLegRaiseTracker:
    def __init__(self):
        self.state = "WAITING"   
        self.good_reps = 0
        self.bad_reps = 0
        self.half_reps = 0                 # Kept for camera test harness compatibility
        
        self.target_height_reached = False # Full ROM (hip <= 110 deg)
        self.min_hip_angle = 180.0         # Tracks peak height
        
        # Posture Tripwires (SIDE / SAGITTAL VIEW)
        self.is_side_view = True           # Catches facing camera instead of profile
        self.leg_swung_backward = False    # Catches swinging the leg back to gain momentum
        self.lifted_knee_straight = True   # Catches bending lifting leg (hamstring cheat)
        self.support_knee_straight = True  # Catches bending standing leg
        self.torso_upright = True          # Catches leaning backward to counter-balance
        self.controlled_descent = True     # Catches free-falling the leg
        self.ml_posture_maintained = True 

        # Debounce counters
        self.facing_camera_frames = 0
        self.backward_swing_frames = 0
        self.lift_knee_frames = 0
        self.support_knee_frames = 0
        self.torso_frames = 0
        self.drop_frames = 0
        
        self.prev_hip_angle = None
        self.standing_frames = 0
        self.post_rep_timer = 0
        self.is_ready = False

        self.current_feedback = "Stand sideways to the camera to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Load LiteRT model
        self.interpreter = None
        model_path = os.path.join("models", "tflite", "st_gcn_standing_active_straight_leg_raise.tflite")

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
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_toe = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        r_toe = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        l_vis = sum([l_shoulder.visibility, l_hip.visibility, l_knee.visibility, l_ankle.visibility])
        r_vis = sum([r_shoulder.visibility, r_hip.visibility, r_knee.visibility, r_ankle.visibility])

        total_reps = self.good_reps + self.bad_reps + self.half_reps

        if self.post_rep_timer > 0:
            self.post_rep_timer -= 1

        if max(l_vis, r_vis) < 2.0: 
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Leg Raise",
                "state": "WAITING",
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": "Step back: Show full side profile",
                "hip_angle": 0.0, "knee_angle": 0.0, "depth_reached": False
            }

        # Side profile validation (shoulders and hips should not appear wide across screen)
        shoulder_span_x = abs(l_shoulder.x - r_shoulder.x)
        hip_span_x = abs(l_hip.x - r_hip.x)
        torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2.0) - ((l_hip.y + r_hip.y) / 2.0))
        if torso_height == 0:
            torso_height = 0.45

        shoulder_aspect = shoulder_span_x / torso_height
        is_turned_sideways = shoulder_aspect < 0.40 and hip_span_x < 0.15

        # Determine which direction is forward along the X axis using toe orientation
        facing_right = (l_toe.x > l_ankle.x) if l_vis > r_vis else (r_toe.x > r_ankle.x)

        # Calculate joint angles for both limbs
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Baseline: standing tall with both legs extended and facing side profile
        is_standing_tall = (l_hip_angle > 162 and r_hip_angle > 162 and 
                            l_knee_angle > 155 and r_knee_angle > 155 and
                            is_turned_sideways)

        if is_standing_tall:
            self.standing_frames += 1
        else:
            self.standing_frames = 0

        if self.standing_frames > 8:
            self.is_ready = True

        if not self.is_ready:
            self.state = "WAITING"
            self.frame_buffer.clear()
            feedback_msg = "Turn sideways to camera (side profile)" if not is_turned_sideways else "Stand tall sideways, legs locked to begin"
            return {
                "exercise": "Standing Leg Raise",
                "state": self.state,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "half_reps": self.half_reps,
                "total_reps": total_reps,
                "form_score": 0,
                "feedback": feedback_msg,
                "hip_angle": round(min(l_hip_angle, r_hip_angle), 1),
                "knee_angle": round(min(l_knee_angle, r_knee_angle), 1),
                "depth_reached": False
            }

        # Determine active (lifting) leg vs support leg
        if l_hip_angle < r_hip_angle - 10:
            active_hip_angle = l_hip_angle
            active_knee_angle = l_knee_angle
            support_knee_angle = r_knee_angle
            active_ankle, support_ankle = l_ankle, r_ankle
            active_shoulder, active_hip = l_shoulder, l_hip
        elif r_hip_angle < l_hip_angle - 10:
            active_hip_angle = r_hip_angle
            active_knee_angle = r_knee_angle
            support_knee_angle = l_knee_angle
            active_ankle, support_ankle = r_ankle, l_ankle
            active_shoulder, active_hip = r_shoulder, r_hip
        else:
            active_hip_angle = min(l_hip_angle, r_hip_angle)
            active_knee_angle = l_knee_angle if l_hip_angle < r_hip_angle else r_knee_angle
            support_knee_angle = r_knee_angle if l_hip_angle < r_hip_angle else l_knee_angle
            active_ankle = l_ankle if l_hip_angle < r_hip_angle else r_ankle
            support_ankle = r_ankle if l_hip_angle < r_hip_angle else l_ankle
            active_shoulder = l_shoulder if l_hip_angle < r_hip_angle else r_shoulder
            active_hip = l_hip if l_hip_angle < r_hip_angle else r_hip

        # ST-GCN Buffer
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
        # SIDE-VIEW TRIPWIRES
        # ---------------------------------------------------------
        if self.state in ["LIFTING", "PEAK"]:
            # 0. Front Facing Tripwire: User turning chest/hips open toward camera
            if shoulder_aspect > 0.48 or hip_span_x > 0.20:
                self.facing_camera_frames += 1
                if self.facing_camera_frames >= 4:
                    self.is_side_view = False
            else:
                self.facing_camera_frames = max(0, self.facing_camera_frames - 1)

            # 1. Backward Leg Swing Tripwire: Swinging foot behind the standing leg
            # If facing right, backward is active_ankle.x < support_ankle.x
            # If facing left, backward is active_ankle.x > support_ankle.x
            if facing_right:
                backward_distance = support_ankle.x - active_ankle.x
            else:
                backward_distance = active_ankle.x - support_ankle.x

            if backward_distance > 0.08 * torso_height:
                self.backward_swing_frames += 1
                if self.backward_swing_frames >= 3:
                    self.leg_swung_backward = True
            else:
                self.backward_swing_frames = max(0, self.backward_swing_frames - 1)

            # 2. Hamstring Cheat: Lifting knee bends
            if active_knee_angle < 150:
                self.lift_knee_frames += 1
                if self.lift_knee_frames >= 4:
                    self.lifted_knee_straight = False
            else:
                self.lift_knee_frames = max(0, self.lift_knee_frames - 1)
                
            # 3. Support Knee Cheat: Standing leg buckles
            if support_knee_angle < 152:
                self.support_knee_frames += 1
                if self.support_knee_frames >= 4:
                    self.support_knee_straight = False
            else:
                self.support_knee_frames = max(0, self.support_knee_frames - 1)

            # 4. Torso Counter-balance
            torso_lean = abs(active_shoulder.x - active_hip.x)
            if torso_lean > 0.16:
                self.torso_frames += 1
                if self.torso_frames >= 4:
                    self.torso_upright = False
            else:
                self.torso_frames = max(0, self.torso_frames - 1)

            self.min_hip_angle = min(self.min_hip_angle, active_hip_angle)

        # ---------------------------------------------------------
        # STATE MACHINE (NO HALF REPS - FULL ROM ONLY)
        # ---------------------------------------------------------
        if self.state == "WAITING" and self.is_ready:
            self.state = "STANDING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Raise leg straight out in front!"

        elif self.state in ["STANDING", "LOWERING"] and active_hip_angle <= 155:
            self.state = "LIFTING"
            self.min_hip_angle = active_hip_angle
            if self.post_rep_timer == 0:
                self.current_feedback = "Keep both legs completely locked..."

        # Full ROM: Foot reaches waist/hip height (hip angle <= 110 deg)
        elif self.state == "LIFTING" and active_hip_angle <= 110:
            self.target_height_reached = True
            self.state = "PEAK"
            if self.post_rep_timer == 0:
                self.current_feedback = "Great height! Lower slowly under control."

        # Early downward reversal before reaching target height
        elif self.state == "LIFTING" and (active_hip_angle - self.min_hip_angle) > 8:
            self.state = "LOWERING"
            self.current_feedback = "Perform a full rep! Lift foot all the way to hip level."

        elif self.state == "PEAK" and active_hip_angle > 118:
            self.state = "LOWERING"
            if self.post_rep_timer == 0:
                self.current_feedback = "Control the descent"

        # Eccentric Free-Fall Detector
        if self.state in ["LOWERING", "LIFTING"]:
            if self.prev_hip_angle is not None:
                drop_speed = active_hip_angle - self.prev_hip_angle
                if drop_speed > 24.0:
                    self.drop_frames += 1
                    if self.drop_frames >= 2:
                        self.controlled_descent = False
            self.prev_hip_angle = active_hip_angle
        else:
            self.prev_hip_angle = None

        # ---------------------------------------------------------
        # REP COMPLETION (Return to bottom >= 165 deg)
        # ---------------------------------------------------------
        if self.state in ["LOWERING", "PEAK", "LIFTING"] and active_hip_angle > 165:
            is_bad_rep = False
            
            # Form Cheats First
            if not self.is_side_view:
                self.current_feedback = "No rep: Stand sideways! Do not face the camera."
                is_bad_rep = True
            elif self.leg_swung_backward:
                self.current_feedback = "No rep: Do not swing your leg backward for momentum! Lift straight from standing."
                is_bad_rep = True
            elif not self.lifted_knee_straight:
                self.current_feedback = "No rep: Do not bend your lifting knee! Lock it straight."
                is_bad_rep = True
            elif not self.support_knee_straight:
                self.current_feedback = "No rep: Standing leg must stay locked out."
                is_bad_rep = True
            elif not self.torso_upright:
                self.current_feedback = "No rep: Do not lean backward to heave the leg up."
                is_bad_rep = True
            elif not self.controlled_descent:
                self.current_feedback = "No rep: Control the negative! Lower the leg slowly."
                is_bad_rep = True
            elif not self.ml_posture_maintained:
                self.current_feedback = "No rep: AI detected collapsed posture."
                is_bad_rep = True
            elif not self.target_height_reached:
                self.current_feedback = "No rep: Perform a full rep! Lift your foot all the way to hip level."
                is_bad_rep = True
            
            # Rep Counting
            if is_bad_rep:
                self.bad_reps += 1
            else:
                self.good_reps += 1
                self.current_feedback = "Perfect straight leg raise!"
            
            self.post_rep_timer = 40

            # Reset flags for next rep
            self.state = "STANDING"
            self.target_height_reached = False
            self.min_hip_angle = 180.0
            self.is_side_view = True
            self.leg_swung_backward = False
            self.lifted_knee_straight = True
            self.support_knee_straight = True
            self.torso_upright = True
            self.controlled_descent = True
            self.prev_hip_angle = None
            self.ml_posture_maintained = True
            self.facing_camera_frames = 0
            self.backward_swing_frames = 0
            self.lift_knee_frames = 0
            self.support_knee_frames = 0
            self.torso_frames = 0
            self.drop_frames = 0
            self.frame_buffer.clear()

        # Dynamic Form Scoring
        form_score = max(0, min(100, int(100 - abs(active_hip_angle - 90) * 0.4)))
        if not self.is_side_view: form_score -= 35
        if self.leg_swung_backward: form_score -= 25
        if not self.lifted_knee_straight: form_score -= 30
        if not self.support_knee_straight: form_score -= 25
        if not self.torso_upright: form_score -= 25
        if not self.controlled_descent: form_score -= 20
        if not self.ml_posture_maintained: form_score -= 30
        if not self.target_height_reached: form_score -= 20
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Leg Raise",
            "state": self.state,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "half_reps": self.half_reps,
            "total_reps": total_reps,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "hip_angle": round(active_hip_angle, 1),
            "knee_angle": round(active_knee_angle, 1),
            "depth_reached": self.target_height_reached
        }