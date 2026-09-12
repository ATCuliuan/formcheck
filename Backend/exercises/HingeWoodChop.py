import os, torch, numpy as np, mediapipe as mp
from collections import deque
from formcheck.Backend.core.kinematics import calculate_angle
from formcheck.Backend.st_gcn_model import STGCN
from formcheck.Backend.path_config import ST_GCN_HINGE_CHOP_MODEL

class CompactHingeWoodchopTracker:
    def __init__(self):
        # Flattened state initializations
        self.state, self.rep_count = "HIGH", 0
        self.arms_straight, self.avoided_squat = True, True
        self.buf = deque(maxlen=30)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Condensed Model Loader
        try:
            self.model = STGCN(2).to(self.dev)
            self.model.load_state_dict(torch.load(ST_GCN_HINGE_CHOP_MODEL, map_location=self.dev))
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
        le, re = get(L.LEFT_ELBOW), get(L.RIGHT_ELBOW)
        lw, rw = get(L.LEFT_WRIST), get(L.RIGHT_WRIST)

        # 1-Liner: Extract keypoints, pad to 25 joints, and append to buffer
        idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.buf.append(np.array([[lmks[i].x, lmks[i].y, lmks[i].z] for i in idx] + [[0]*3]*8, dtype=np.float32)[:25])

        # Kinematics (Averaging L/R sides for robustness)
        avg_knee = (calculate_angle(lh, lk, la) + calculate_angle(rh, rk, ra)) / 2
        avg_hip = (calculate_angle(ls, lh, lk) + calculate_angle(rs, rh, rk)) / 2
        avg_elbow = (calculate_angle(ls, le, lw) + calculate_angle(rs, re, rw)) / 2
        
        hands_y = (lw.y + rw.y) / 2
        shoulders_y = (ls.y + rs.y) / 2
        hips_y = (lh.y + rh.y) / 2

        # Movement Triggers
        is_high = hands_y < shoulders_y
        is_low = hands_y > hips_y
        is_hinged = avg_hip < 135  # Forgiving: just a modest hip hinge required

        # Catching beginner compensations dynamically
        if self.state != "HIGH":
            if avg_elbow < 110: self.arms_straight = False
            # If knees bend past 120, they turned the hinge into a squat
            if avg_knee < 120: self.avoided_squat = False 

        # Condensed State Machine & Coaching Logic
        fb = "Stand tall, arms high and slightly to one side."
        
        # Reset/Top Position requires hands high AND hips fully extended
        if is_high and avg_hip > 150:
            if self.state == "CHOPPED":
                self.rep_count += 1
                if not self.avoided_squat: fb = "Good chop, but keep your legs straighter! Push hips BACK, don't squat DOWN."
                elif not self.arms_straight: fb = "Nice! Try to keep those arms a bit longer next time."
                else: fb = "Perfect hinge chop! Great glute engagement."
            elif self.state == "CHOPPING":
                fb = "Finish the chop! Bring your hands down past your hips."
                
            self.state, self.arms_straight, self.avoided_squat = "HIGH", True, True
            
        else:
            if self.state == "HIGH":
                self.state = "CHOPPING"
            
            if self.state == "CHOPPING":
                fb = "Push your hips BACK and chop diagonally down!"
                if is_low and is_hinged: 
                    self.state = "CHOPPED"
            elif self.state == "CHOPPED":
                fb = "Squeeze your glutes and stand powerfully back up!"

        # Generous Form Scoring
        score = 100
        if not self.avoided_squat: score -= 20
        if not self.arms_straight: score -= 15

        return {
            "exercise": "Beginner Hinge Woodchop",
            "state": self.state,
            "rep_count": self.rep_count,
            "form_score": max(0, score),
            "feedback": fb,
            "is_hinged": is_hinged
        }