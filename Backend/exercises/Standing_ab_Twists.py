import os
import numpy as np
from collections import deque
import torch
import mediapipe as mp
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN


class StandingAbTwistTracker:
    """
    Coaching logic for a Standing Ab Twist:
      1. CENTER    -> stand tall, feet planted, torso squared to the camera.
      2. TWISTING  -> rotate the torso to one side while hips stay square.
      3. PEAK      -> full rotation reached (this is the "crunch" of the twist).
      4. Unwind back through CENTER to complete the rep.

    The key coaching cue this exercise lives or dies on: the twist has to come
    from the torso/obliques, NOT from swinging the hips or the whole body.
    So unlike the push-up/jumping-jack trackers, this one isolates a
    "shoulder rotation" signal from a "hip rotation" signal and flags the rep
    if the hips did most of the work.
    """

    PEAK_ANGLE = 25.0      # degrees of isolated torso rotation to count as a full twist
    RETURN_ANGLE = 10.0    # degrees below which we consider the torso "back at center"
    HIP_STABILITY_LIMIT = 15.0  # hips rotating past this = swinging, not twisting
    UPRIGHT_MIN_ANGLE = 155.0   # shoulder-hip-ankle angle; lower = leaning/hunching

    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.peak_reached = False
        self.current_direction = None  # "LEFT" or "RIGHT", set when a peak is hit

        # Posture tripwires
        self.core_maintained = True        # hips stayed square (isolated twist)
        self.ml_posture_maintained = True
        self.twist_frames = 0              # frames spent moving from center toward peak

        # Setup Guards
        self.center_frames = 0
        self.is_ready = False

        self.current_feedback = "Stand tall and centered to begin"
        self.mp_pose = mp.solutions.pose

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ST-GCN Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        model_path = STANDING_AB_TWISTS_MODEL

        if os.path.exists(model_path):
            try:
                self.model = STGCN(num_classes=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                self.model.eval()
                print(f"✅ Standing Ab Twists model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading Standing Ab Twists model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found at {model_path}")

    def _extract_body_keypoints(self, landmarks):
        target_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        coords = []
        for idx in target_indices:
            coords.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        while len(coords) < 25:
            coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    @staticmethod
    def _rotation_theta(left_pt, right_pt):
        """
        Signed rotation of a left-right landmark pair about the vertical axis,
        using the depth (z) MediaPipe estimates relative to the hip center.
        ~0 degrees = squared to the camera. Sign flips with twist direction.
        """
        dx = right_pt.x - left_pt.x
        dz = right_pt.z - left_pt.z
        return np.degrees(np.arctan2(dz, dx + 1e-6))

    def process_frame(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # This is a symmetric, bilateral movement, so we work off midpoints/pairs
        # rather than picking a "dominant" side like the push-up tracker does.
        shoulder_theta = self._rotation_theta(l_shoulder, r_shoulder)
        hip_theta = self._rotation_theta(l_hip, r_hip)

        # Isolated torso twist = how much the shoulders rotated beyond the hips.
        twist_angle = shoulder_theta - hip_theta

        shoulder_mid = type("P", (), {
            "x": (l_shoulder.x + r_shoulder.x) / 2,
            "y": (l_shoulder.y + r_shoulder.y) / 2,
        })()
        hip_mid = type("P", (), {
            "x": (l_hip.x + r_hip.x) / 2,
            "y": (l_hip.y + r_hip.y) / 2,
        })()
        ankle_mid = type("P", (), {
            "x": (l_ankle.x + r_ankle.x) / 2,
            "y": (l_ankle.y + r_ankle.y) / 2,
        })()
        torso_angle = calculate_angle(shoulder_mid, hip_mid, ankle_mid)

        # 1. Orientation / Posture Guard: must be standing tall, not leaning or hunched.
        if torso_angle < self.UPRIGHT_MIN_ANGLE:
            self.is_ready = False
            self.center_frames = 0
            self.state = "LEANING"
            self.peak_reached = False
            self.frame_buffer.clear()
            return {
                "exercise": "Standing Ab Twist",
                "state": self.state,
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Stand up tall, stop leaning",
                "twist_angle": round(twist_angle, 1),
                "hip_angle": round(hip_theta, 1),
                "peak_reached": False,
            }

        # 2. Setup Guard: hold a squared, centered stance before we start counting.
        if abs(twist_angle) < self.RETURN_ANGLE:
            self.center_frames += 1
        else:
            self.center_frames = 0

        if self.center_frames > 15 and not self.is_ready:
            self.is_ready = True
            self.state = "CENTER"

        if not self.is_ready:
            return {
                "exercise": "Standing Ab Twist",
                "state": "WAITING",
                "rep_count": self.rep_count,
                "form_score": 0,
                "feedback": "Feet planted, square up to the camera to start",
                "twist_angle": round(twist_angle, 1),
                "hip_angle": round(hip_theta, 1),
                "peak_reached": False,
            }

        # 3. Hip Sway Tripwire: the twist should come from the torso, not the hips.
        if abs(hip_theta) > self.HIP_STABILITY_LIMIT:
            self.core_maintained = False

        if self.state == "TWISTING":
            self.twist_frames += 1

        # 4. Buffer & AI
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

        # 5. State Machine & Coaching
        if abs(twist_angle) > self.PEAK_ANGLE:
            if not self.peak_reached:
                self.current_direction = "RIGHT" if twist_angle > 0 else "LEFT"
            self.peak_reached = True
            self.state = "PEAK"
            self.current_feedback = "Full rotation — now unwind back to center"

        elif abs(twist_angle) < self.RETURN_ANGLE:
            if self.state in ("TWISTING", "PEAK"):
                if self.peak_reached:
                    if not self.core_maintained:
                        self.current_feedback = "No rep: Hips swiveled — isolate the twist to your core"
                    elif not self.ml_posture_maintained:
                        self.current_feedback = "No rep: AI detected broken form at the peak"
                    else:
                        self.rep_count += 1
                        if self.twist_frames < 4:
                            self.current_feedback = "Good rep! Slow down — control beats speed here."
                        else:
                            self.current_feedback = f"Clean twist to the {self.current_direction.lower()}!"
                else:
                    self.current_feedback = "No rep: Didn't rotate far enough — twist further"

            self.state = "CENTER"
            self.peak_reached = False
            self.core_maintained = True
            self.ml_posture_maintained = True
            self.twist_frames = 0

        else:
            # In the transition band between center and a full peak.
            if self.state in ("CENTER", "TWISTING"):
                self.state = "TWISTING"
                self.current_feedback = "Rotate through your torso, keep hips facing forward"
            # If we're unwinding from PEAK back toward center, hold state until
            # it drops below RETURN_ANGLE so tempo isn't double-counted.

        # 6. Dynamic Form Scoring
        target_twist = 40.0  # a strong, controlled rotation
        form_score = max(0, min(100, int(100 - abs(abs(twist_angle) - target_twist) * 1.2)))
        if not self.core_maintained:
            form_score -= 50
        if not self.peak_reached and self.state == "CENTER":
            form_score -= 40
        if 0 < self.twist_frames < 4:
            form_score -= 15
        form_score = max(0, form_score)

        return {
            "exercise": "Standing Ab Twist",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": form_score,
            "feedback": self.current_feedback,
            "twist_angle": round(twist_angle, 1),
            "hip_angle": round(hip_theta, 1),
            "peak_reached": self.peak_reached,
        }