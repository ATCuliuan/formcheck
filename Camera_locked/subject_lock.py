import numpy as np

class ExerciseSubjectLock:
    def __init__(self, max_jump_distance=0.18, max_lost_frames=45):
        """
        max_jump_distance: Max normalized screen distance (0.0 to 1.0) hip can teleport.
        max_lost_frames: ~1.5s grace period (at 30 FPS) for occlusions or passersby.
        """
        self.max_jump_distance = max_jump_distance
        self.max_lost_frames = max_lost_frames
        
        self.locked_hip = None          # (x, y) anchor of active lifter
        self.locked_torso_h = None      # visual torso scale
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
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        current_torso_h = abs(current_hip_y - shoulder_y)

        # 1. Initial Acquisition: Lock onto the first detected person
        if self.locked_hip is None:
            self.locked_hip = (current_hip_x, current_hip_y)
            self.locked_torso_h = current_torso_h
            self.lost_frames = 0
            self.prev_candidate_x = current_hip_x
            return True, landmarks

        # 2. Velocity Filter: Detect rapid horizontal walk-bys
        horizontal_speed = abs(current_hip_x - self.prev_candidate_x) if self.prev_candidate_x is not None else 0.0
        self.prev_candidate_x = current_hip_x
        is_walking_fast = horizontal_speed > 0.06  # Fast transverse movement

        # 3. Spatial & Scale Checks
        prev_x, prev_y = self.locked_hip
        jump_dist = np.sqrt((current_hip_x - prev_x)**2 + (current_hip_y - prev_y)**2)
        scale_ratio = current_torso_h / self.locked_torso_h if self.locked_torso_h > 0 else 1.0

        is_behind = scale_ratio < 0.72       # Smaller visual torso = person behind
        is_too_close = scale_ratio > 1.35    # Abnormally large visual torso = foreground blocker
        is_displaced = jump_dist > self.max_jump_distance  # Teleported to someone on the side

        # Reject candidate if any background/passerby flag trips
        if is_displaced or is_behind or is_too_close or is_walking_fast:
            self.lost_frames += 1
            # If the original user left the frame for > 1.5 seconds, hand lock to new person
            if self.lost_frames > self.max_lost_frames:
                self.locked_hip = (current_hip_x, current_hip_y)
                self.locked_torso_h = current_torso_h
                self.lost_frames = 0
                return True, landmarks
            return False, None

        # Verified: Primary lifter confirmed
        self.lost_frames = 0
        self.locked_hip = (
            0.85 * prev_x + 0.15 * current_hip_x,
            0.85 * prev_y + 0.15 * current_hip_y
        )
        self.locked_torso_h = 0.85 * self.locked_torso_h + 0.15 * current_torso_h
        return True, landmarks

    def reset(self):
        self.locked_hip = None
        self.locked_torso_h = None
        self.lost_frames = 0
        self.prev_candidate_x = None