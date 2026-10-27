import numpy as np

class ExerciseSubjectLock:
    def __init__(self, max_jump_distance=0.28, max_lost_frames=45):
        """
        max_jump_distance: Max normalized screen distance (0.0 to 1.0) hip can move per frame.
                           Increased to 0.28 to allow dropping into a plank.
        max_lost_frames: ~1.5s grace period (at 30 FPS).
        """
        self.max_jump_distance = max_jump_distance
        self.max_lost_frames = max_lost_frames
        
        self.locked_hip = None          # (x, y) anchor of active lifter
        self.locked_torso_len = None    # 2D Euclidean torso length
        self.lost_frames = 0
        self.prev_candidate_x = None

    @property
    def is_locked(self):
        return self.locked_hip is not None

    def validate_and_update(self, landmarks):
        """
        Validates whether the detected skeleton belongs to the locked user.
        Returns: (is_valid: bool, landmarks or None)
        """
        if landmarks is None or len(landmarks) == 0:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.reset()
            return False, None

        # Extract landmarks: Shoulders (11, 12), Hips (23, 24)
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        current_hip_x = (l_hip.x + r_hip.x) / 2.0
        current_hip_y = (l_hip.y + r_hip.y) / 2.0
        
        shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0

        # FIX 1: True 2D torso length (invariant to standing vs lying horizontal)
        current_torso_len = np.hypot(current_hip_x - shoulder_x, current_hip_y - shoulder_y)

        # 1. Initial Acquisition: Lock onto the first detected person
        if self.locked_hip is None:
            self.locked_hip = (current_hip_x, current_hip_y)
            self.locked_torso_len = current_torso_len
            self.lost_frames = 0
            self.prev_candidate_x = current_hip_x
            return True, landmarks

        # 2. Velocity Filter: Detect rapid horizontal walk-bys
        horizontal_speed = abs(current_hip_x - self.prev_candidate_x) if self.prev_candidate_x is not None else 0.0
        self.prev_candidate_x = current_hip_x
        is_walking_fast = horizontal_speed > 0.08

        # 3. Spatial & Scale Checks
        prev_x, prev_y = self.locked_hip
        jump_dist = np.hypot(current_hip_x - prev_x, current_hip_y - prev_y)
        
        # Scale ratio based on true anatomical torso length
        scale_ratio = current_torso_len / self.locked_torso_len if self.locked_torso_len > 0 else 1.0

        # FIX 2: Widen bounds so floor transitions aren't rejected as depth changes
        is_behind = scale_ratio < 0.55
        is_too_close = scale_ratio > 1.65
        is_displaced = jump_dist > self.max_jump_distance

        # Reject candidate if any background/passerby flag trips
        if is_displaced or is_behind or is_too_close or is_walking_fast:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.locked_hip = (current_hip_x, current_hip_y)
                self.locked_torso_len = current_torso_len
                self.lost_frames = 0
                return True, landmarks
            return False, None

        # Verified: Primary lifter confirmed
        self.lost_frames = 0
        self.locked_hip = (
            0.80 * prev_x + 0.20 * current_hip_x,
            0.80 * prev_y + 0.20 * current_hip_y
        )
        self.locked_torso_len = 0.85 * self.locked_torso_len + 0.15 * current_torso_len
        return True, landmarks

    def reset(self):
        self.locked_hip = None
        self.locked_torso_len = None
        self.lost_frames = 0
        self.prev_candidate_x = None