import os, torch, numpy as np, mediapipe as mp
from collections import deque
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_LUNGE_TWIST_MODEL

class CompactLungeTwistTracker:
    def __init__(self):
        # Flattened state initializations
        self.state, self.rep_count, self.side = "STANDING", 0, None
        self.ready_frames, self.good_depth, self.correct_twist = 0, False, True
        self.buf = deque(maxlen=30)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Condensed Model Loader
        try:
            self.model = STGCN(2).to(self.dev)
            self.model.load_state_dict(torch.load(ST_GCN_LUNGE_TWIST_MODEL, map_location=self.dev))
            self.model.eval()
        except Exception: 
            self.model = None

    def process_frame(self, lmks):
        # Lambda helper to shorten landmark extraction
        get = lambda idx: lmks[idx.value]
        L = mp.solutions.pose.PoseLandmark

        ls, rs = get(L.LEFT_SHOULDER), get(L.RIGHT_SHOULDER)
        lh, rh = get(L.LEFT_HIP), get(L.RIGHT_HIP)
        lk, rk = get(L.LEFT_KNEE), get(L.RIGHT_KNEE)
        la, ra = get(L.LEFT_ANKLE), get(L.RIGHT_ANKLE)

        # 1-Liner: Extract keypoints, pad to 25 joints, and append to buffer
        idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.buf.append(np.array([[lmks[i].x, lmks[i].y, lmks[i].z] for i in idx] + [[0]*3]*8, dtype=np.float32)[:25])

        # Kinematics
        lk_ang, rk_ang = calculate_angle(lh, lk, la), calculate_angle(rh, rk, ra)
        is_standing = lk_ang > 155 and rk_ang > 155

        # Calibration Guard
        self.ready_frames = self.ready_frames + 1 if is_standing else 0
        if self.ready_frames < 10 and self.state == "WAITING":
            return {"state": "WAITING", "feedback": "Stand tall, feet together to start."}

        # Dynamic Tracking
        if not is_standing:
            # Front foot is lower on the screen (larger Y value)
            self.side = "LEFT" if la.y > ra.y else "RIGHT"
            
            # Forgiving depth check (130 degrees is a very shallow, beginner-safe lunge)
            if (lk_ang if self.side == "LEFT" else rk_ang) < 130: 
                self.good_depth = True

            # Twist detection using Z-Depth difference between shoulders
            twist_magnitude = abs(ls.z - rs.z)
            if twist_magnitude > 0.15 and self.state == "LUNGING":
                self.state = "TWISTED"
                # If Left leg is front, Left shoulder should move back (larger Z) and Right moves front (smaller Z)
                self.correct_twist = (ls.z > rs.z) if self.side == "LEFT" else (rs.z > ls.z)

        # Flattened State Machine & Coaching
        fb = "Step forward into a comfortable lunge."
        if is_standing:
            if self.state == "TWISTED":
                self.rep_count += 1
                fb = "Great rep!" if self.correct_twist else "Good! But remember to twist TOWARD your front leg."
            elif self.state == "LUNGING":
                fb = "Don't forget to twist before you stand back up!"
                
            self.state, self.good_depth, self.correct_twist = "STANDING", False, True
        else:
            if self.state == "STANDING" and self.good_depth:
                self.state = "LUNGING"
            
            if self.state == "LUNGING":
                fb = f"Good lunge. Now twist your torso to the {self.side.lower()}!"
            elif self.state == "TWISTED":
                fb = "Nice twist! Now push off the front foot and return to center."

        return {
            "exercise": "Beginner Lunge Twist",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": 100 if self.correct_twist else 75,
            "feedback": fb,
            "side": self.side
        }