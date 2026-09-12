import os, torch, numpy as np, mediapipe as mp
from collections import deque
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_WOODCHOP_MODEL

class CompactCenterWoodchopTracker:
    def __init__(self):
        # Flattened state initializations
        self.state, self.rep_count, self.arms_straight = "OVERHEAD", 0, True
        self.good_posture = True
        self.buf = deque(maxlen=30)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Condensed Model Loader
        try:
            self.model = STGCN(2).to(self.dev)
            self.model.load_state_dict(torch.load(ST_GCN_WOODCHOP_MODEL, map_location=self.dev))
            self.model.eval()
        except Exception: 
            self.model = None

    def process_frame(self, lmks):
        # Lambda helper to eliminate repetitive array indexing
        get = lambda idx: lmks[idx.value]
        L = mp.solutions.pose.PoseLandmark

        ls, rs = get(L.LEFT_SHOULDER), get(L.RIGHT_SHOULDER)
        le, re = get(L.LEFT_ELBOW), get(L.RIGHT_ELBOW)
        lw, rw = get(L.LEFT_WRIST), get(L.RIGHT_WRIST)
        lk, rk = get(L.LEFT_KNEE), get(L.RIGHT_KNEE)

        # 1-Liner: Extract keypoints, pad to 25 joints, and append to buffer
        idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.buf.append(np.array([[lmks[i].x, lmks[i].y, lmks[i].z] for i in idx] + [[0]*3]*8, dtype=np.float32)[:25])

        # Posture AI Check
        if self.model and len(self.buf) == 30:
            with torch.no_grad():
                tensor_in = torch.tensor(np.array(self.buf).transpose(2, 0, 1)).float().unsqueeze(0).to(self.dev)
                self.good_posture = torch.argmax(self.model(tensor_in), 1).item() != 0

        # Kinematics & Spatial Thresholds
        hands_y = (lw.y + rw.y) / 2
        shoulders_y = (ls.y + rs.y) / 2
        knees_y = (lk.y + rk.y) / 2
        avg_elbow = (calculate_angle(ls, le, lw) + calculate_angle(rs, re, rw)) / 2

        # In MediaPipe, smaller Y is higher up on the screen
        is_overhead = hands_y < (shoulders_y - 0.05)
        is_chopped = hands_y > (knees_y - 0.15)  # Forgiving depth: just get hands near the knees

        # Catch the "T-Rex Chop" (allowing a generous 120-degree bend for tight shoulders)
        if self.state != "OVERHEAD" and avg_elbow < 120:
            self.arms_straight = False

        # Condensed State Machine & Coaching Logic
        fb = "Clasp your hands and reach high above your head to start."
        
        if is_overhead:
            if self.state == "CHOPPED":
                self.rep_count += 1
                fb = "Great power!" if self.arms_straight else "Good! But try to keep your arms longer on the swing."
            elif self.state == "CHOPPING":
                fb = "Use your core and swing it down!"
            self.state, self.arms_straight = "OVERHEAD", True
        else:
            if self.state == "OVERHEAD":
                self.state = "CHOPPING"
            
            if self.state == "CHOPPING":
                fb = "Chop down between your legs! Soften your knees."
                if is_chopped: 
                    self.state = "CHOPPED"
            elif self.state == "CHOPPED":
                fb = "Drive through your heels and swing back up!"

        return {
            "exercise": "Beginner Center Woodchop",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": 100 if self.arms_straight else 75,
            "feedback": fb,
            "hands_low_enough": is_chopped
        }