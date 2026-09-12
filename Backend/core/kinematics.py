import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the interior angle between three 2D/3D points:
    a: First joint (e.g., Hip)
    b: Middle joint / vertex (e.g., Knee)
    c: End joint (e.g., Ankle)
    Returns: angle in degrees (0 to 180)
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    # Vectors ba and bc
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return float(np.degrees(angle))

def normalize_landmarks(landmarks, mp_pose):
    """
    Recycled from your old code:
    Centers on midpoint of hips and scales by shoulder width.
    """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    ref_x = (left_hip.x + right_hip.x) / 2.0
    ref_y = (left_hip.y + right_hip.y) / 2.0

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    body_scale = np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2
    )
    body_scale = max(body_scale, 1e-6)

    normalized = []
    for lm in landmarks:
        norm_x = (lm.x - ref_x) / body_scale
        norm_y = (lm.y - ref_y) / body_scale
        normalized.extend([norm_x, norm_y])
    return normalized