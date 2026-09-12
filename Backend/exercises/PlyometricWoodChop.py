import os, torch, numpy as np, mediapipe as mp
from collections import deque
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_PLYO_CHOP_MODEL

class CompactPlyoWoodchopTracker:
    def __init__(self):
        # Flattened state initializations
        self.state, self.rep_count = "LOW", 0
        self.arms_straight, self.soft_landing, self.explosive_up = True, True, False
        self.buf = deque(maxlen=30)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Condensed Model Loader
        try:
            self.model = STGCN(2).to(self.dev)
            self.model.load_state_dict(torch.load(ST_GCN_PLYO_CHOP_MODEL, map_location=self.dev))
            self.model.eval()
        except Exception: 
            self.model = None

    def process_frame(self, lmks):
        # Lambda helper to eliminate repetitive array indexing
        get = lambda idx: lmks[idx.value]
        L = mp.solutions.pose.PoseLandmark

        ls, rs = get(L.LEFT_SHOULDER), get(L.RIGHT_SHOULDER)
        lh, rh = get(L.LEFT_HIP), get(L.RIGHT_HIP)
        lk, rk = get(L.LEFT_KNEE), get(L.RIGHT_KNEE)
        la, ra = get(L.LEFT_ANKLE), get(L.RIGHT_ANKLE)
        lft, rft = get(L.LEFT_FOOT_INDEX), get(L.RIGHT_FOOT_INDEX)
        le, re = get(L.LEFT_ELBOW), get(L.RIGHT_ELBOW)
        lw, rw = get(L.LEFT_WRIST), get(L.RIGHT_WRIST)

        # 1-Liner: Extract keypoints, pad to 25 joints, and append to buffer
        idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.buf.append(np.array([[lmks[i].x, lmks[i].y, lmks[i].z] for i in idx] + [[0]*3]*8, dtype=np.float32)[:25])

        # Kinematics & Spatial Thresholds
        avg_knee = (calculate_angle(lh, lk, la) + calculate_angle(rh, rk, ra)) / 2
        avg_ankle = (calculate_angle(lk, la, lft) + calculate_angle(rk, ra, rft)) / 2
        avg_elbow = (calculate_angle(ls, le, lw) + calculate_angle(rs, re, rw)) / 2
        
        hands_y = (lw.y + rw.y) / 2

        is_high = hands_y < ((ls.y + rs.y) / 2 - 0.05)
        is_low = hands_y > ((lh.y + rh.y) / 2)

        # Dynamic Fault Catching
        if self.state == "HIGH":
            # Beginner Plyo: Did their heels leave the ground (plantarflexion > 125)?
            if avg_ankle > 125: self.explosive_up = True
            if avg_elbow < 110: self.arms_straight = False
        if self.state == "CHOPPING" and avg_knee < 140:
            # Landing absorption: knees must bend below 140 degrees on the way down
            self.soft_landing = True

        # Condensed State Machine & Coaching Logic
        fb = "Start low, hands by your knees."
        
        if is_high and self.state in ("LOW", "CHOPPING"):
            self.state = "HIGH"
            fb = "Explode up! Get on your toes or add a small hop!"
            self.explosive_up, self.arms_straight, self.soft_landing = False, True, False

        elif not is_high and self.state == "HIGH":
            self.state = "CHOPPING"
            fb = "Chop hard and bend your knees to absorb the landing!"
            
        elif is_low and self.state == "CHOPPING":
            self.rep_count += 1
            if not self.explosive_up: fb = "Good, but next time push hard enough to lift your heels!"
            elif not self.soft_landing: fb = "Ouch! Don't land with stiff legs. Bend your knees to absorb the chop."
            elif not self.arms_straight: fb = "Nice jump, but keep your arms longer on the swing!"
            else: fb = "Perfect plyo chop! Great power and safe landing."
            self.state = "LOW"

        # Generous Form Scoring
        score = 100
        if not self.explosive_up: score -= 20
        if not self.soft_landing: score -= 30  # Higher penalty for safety
        if not self.arms_straight: score -= 15

        return {
            "exercise": "Beginner Plyo Woodchop",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": max(0, score),
            "feedback": fb if self.state != "LOW" else (fb if self.rep_count > 0 else "Start low, hands by your knees."),
            "explosive_jump": self.explosive_up
        }