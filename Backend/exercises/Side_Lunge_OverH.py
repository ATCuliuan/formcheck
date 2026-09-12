import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class SideLungeOverheadReachTracker:
    """
    Coaching logic for a Side Lunge + Overhead Reach:
      1. STANDING  -> feet together/hip-width, arms at sides, both knees straight.
      2. LUNGING   -> step out laterally, one knee bends and the hips sit back
                      into it, both arms reach DOWN toward the bent knee/floor.
      3. BOTTOM    -> lunge depth reached on the working leg.
      4. RISING    -> drive back up to standing while BOTH arms sweep UP overhead.
      5. STANDING  -> rep complete once the leg is straight again AND the
                      overhead reach was actually achieved on the way up.

    This is a compound movement, so it fails in compound ways. Three
    instructor-level faults get tracked independently:
      - squatting on both legs instead of loading one hip (leg not isolated)
      - the working knee caving inward past the ankle (valgus collapse)
      - standing back up without ever finishing the overhead reach (half-rep)
    """

    LUNGE_DEPTH_ANGLE = 110.0     # working-knee angle at a good lunge depth
    STANDING_KNEE_ANGLE = 160.0   # knee angle counted as "leg straight"
    SQUAT_FAULT_ANGLE = 130.0     # if BOTH knees drop below this, it's a squat, not a lunge
    STRAIGHT_LEG_MIN = 150.0      # the non-working leg should stay roughly this straight
    ARM_OVERHEAD_ANGLE = 150.0    # hip-shoulder-wrist angle counted as "reached overhead"
    ARM_DOWN_ANGLE = 40.0         # arms relaxed at the sides
    VALGUS_TOLERANCE = 0.035      # normalized-coordinate tolerance for knee-over-ankle tracking
    ENGAGED_KNEE_ANGLE = 150.0    # below this we trust which leg is "the working leg"

    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.current_side = None  # "LEFT" or "RIGHT" working leg

        # Per-rep tripwires
        self.depth_reached = False
        self.leg_isolated = True
        self.knee_stable = True
        self.reach_completed = False
        self.ml_posture_maintained = True

        # Setup Guard
        self.ready_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall, feet hip-width, to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = ST_GCN_SIDE_LUNGE_REACH_MODEL

        if os.path.exists(model_path):
            try:
                self.model = STGCN(num_classes=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception:
                self.model = None

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Both arms reach together on this exercise, so we average them
        # rather than pick a "dominant" side like the push-up tracker does.
        l_arm_angle = calculate_angle(l_hip, l_shoulder, l_wrist)
        r_arm_angle = calculate_angle(r_hip, r_shoulder, r_wrist)
        arm_angle = (l_arm_angle + r_arm_angle) / 2.0

        working_knee_angle = min(l_knee_angle, r_knee_angle)
        support_knee_angle = max(l_knee_angle, r_knee_angle)

        # 1. Hard Reset: both knees dropping together means this is a squat, not a lunge.
        if working_knee_angle < self.SQUAT_FAULT_ANGLE and support_knee_angle < self.SQUAT_FAULT_ANGLE:
            self.is_ready = False
            self.ready_frames = 0
            self.state = "SQUATTING"
            self.depth_reached = False
            self.current_side = None
            self.frame_buffer.clear()
            return {
                "exercise": "Side Lunge + Overhead Reach",
                "state": self.state,
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Keep one leg straight — shift your weight side to side, don't squat",
                "knee_angle": round(working_knee_angle, 1),
                "arm_angle": round(arm_angle, 1),
                "side": self.current_side,
                "depth_reached": False,
            }

        # 2. Setup Guard: hold a neutral standing position before we start counting.
        is_neutral_stance = (
            working_knee_angle > self.STANDING_KNEE_ANGLE
            and support_knee_angle > self.STANDING_KNEE_ANGLE
            and arm_angle < self.ARM_DOWN_ANGLE
        )

        if is_neutral_stance:
            self.ready_frames += 1
        else:
            self.ready_frames = 0

        if self.ready_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "STANDING"

        if not self.is_ready:
            return {
                "exercise": "Side Lunge + Overhead Reach",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Feet hip-width, arms relaxed at your sides to start",
                "knee_angle": round(working_knee_angle, 1),
                "arm_angle": round(arm_angle, 1),
                "side": self.current_side,
                "depth_reached": False,
            }

        # 3. Identify the working (bent) leg once it's meaningfully engaged.
        if working_knee_angle < self.ENGAGED_KNEE_ANGLE:
            self.current_side = "RIGHT" if r_knee_angle < l_knee_angle else "LEFT"
            active_knee = r_knee if self.current_side == "RIGHT" else l_knee
            active_ankle = r_ankle if self.current_side == "RIGHT" else l_ankle

            # 4. Knee Tracking Tripwire: bent knee should stay over the ankle, not cave inward.
            if self.current_side == "RIGHT":
                valgus_offset = active_ankle.x - active_knee.x  # positive = caved left/medially
            else:
                valgus_offset = active_knee.x - active_ankle.x  # positive = caved right/medially
            if valgus_offset > self.VALGUS_TOLERANCE:
                self.knee_stable = False

        # 5. Leg Isolation Tripwire: the support leg should stay close to straight.
        if support_knee_angle < self.STRAIGHT_LEG_MIN:
            self.leg_isolated = False

        # 6. Reach Tripwire: mark the overhead reach the moment it's achieved this rep.
        if arm_angle > self.ARM_OVERHEAD_ANGLE:
            self.reach_completed = True

        # 7. Buffer & AI
        frame_coords = self._extract_body_keypoints(landmarks)
        self.frame_buffer.append(frame_coords)

        if self.model and len(self.frame_buffer) == self.sequence_length:
            try:
                seq = np.array(self.frame_buffer)
                tensor_input = torch.tensor(np.transpose(seq, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if torch.argmax(self.model(tensor_input), dim=1).item() == 0:
                        self.ml_posture_maintained = False
            except Exception:
                pass

        # 8. State Machine & Coaching
        if working_knee_angle > self.STANDING_KNEE_ANGLE:
            if self.state in ("LUNGING", "BOTTOM", "RISING"):
                if self.depth_reached:
                    if not self.leg_isolated:
                        self.current_feedback = "No rep: Both knees bent — keep the support leg straighter"
                    elif not self.knee_stable:
                        self.current_feedback = "No rep: Working knee caved inward — track it over your toes"
                    elif not self.reach_completed:
                        self.current_feedback = "No rep: Finish the movement — reach all the way overhead"
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected broken form"
                    else:
                        self.rep_count += 1
                        self.current_feedback = "Great side lunge and reach!"
                else:
                    self.current_feedback = "No rep: Didn't sit low enough into the lunge"

            self.state = "STANDING"
            self.depth_reached = False
            self.leg_isolated = True
            self.knee_stable = True
            self.reach_completed = False
            self.ml_posture_maintained = True
            self.current_side = None

        elif working_knee_angle <= self.LUNGE_DEPTH_ANGLE:
            self.depth_reached = True
            self.state = "BOTTOM"
            self.current_feedback = f"Good depth on the {self.current_side.lower() if self.current_side else ''} side — now drive up and reach!"

        else:
            if self.state in ("STANDING",):
                self.state = "LUNGING"
                self.current_feedback = "Sit back into that hip, reach down toward your knee"
            elif self.state in ("BOTTOM", "RISING"):
                self.state = "RISING"
                self.current_feedback = "Drive up through your heel, sweep your arms overhead!"

        # 9. Dynamic Form Scoring
        target_knee = 95.0
        form_score = max(0, min(100, int(100 - abs(working_knee_angle - target_knee) * 0.5)))
        if not self.leg_isolated:
            form_score -= 40
        if not self.knee_stable:
            form_score -= 35
        if not self.reach_completed and self.state == "STANDING":
            form_score -= 30
        form_score = max(0, form_score)

        return {
            "exercise": "Side Lunge + Overhead Reach",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "knee_angle": round(working_knee_angle, 1),
            "arm_angle": round(arm_angle, 1),
            "side": self.current_side,
            "depth_reached": self.depth_reached,
        }