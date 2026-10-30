import os
import cv2
import json
import numpy as np
import mediapipe as mp

INPUT_DIR = "raw_videos"
BASE_OUT_DIR = "train"

# Standard, stable import
mp_pose = mp.solutions.pose

def extract_keypoints(landmarks):
    lm = landmarks.landmark
    def get_pt(idx):
        return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

    # Extract MediaPipe joints and strictly map to the 25-joint ST-GCN/NTU standard
    ntu_joints = np.zeros((25, 3))
    
    l_shoulder, r_shoulder = get_pt(11), get_pt(12)
    l_hip, r_hip = get_pt(23), get_pt(24)
    
    mid_hip = (l_hip + r_hip) / 2.0
    spine_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_spine = (mid_hip + spine_shoulder) / 2.0

    ntu_joints[0] = mid_hip                    
    ntu_joints[1] = mid_spine                  
    ntu_joints[2] = spine_shoulder             
    ntu_joints[3] = get_pt(0)                  
    ntu_joints[4] = l_shoulder                 
    ntu_joints[5] = get_pt(13)                 
    ntu_joints[6] = get_pt(15)                 
    ntu_joints[7] = get_pt(19)                  
    ntu_joints[8] = r_shoulder                 
    ntu_joints[9] = get_pt(14)                 
    ntu_joints[10] = get_pt(16)                
    ntu_joints[11] = get_pt(20)                 
    ntu_joints[12] = l_hip                     
    ntu_joints[13] = get_pt(25)                
    ntu_joints[14] = get_pt(27)                
    ntu_joints[15] = get_pt(31)                
    ntu_joints[16] = r_hip                     
    ntu_joints[17] = get_pt(26)                
    ntu_joints[18] = get_pt(28)                
    ntu_joints[19] = get_pt(32)                
    ntu_joints[20] = spine_shoulder            
    ntu_joints[21] = get_pt(17)                
    ntu_joints[22] = get_pt(21)                
    ntu_joints[23] = get_pt(18)                
    ntu_joints[24] = get_pt(22)                
    
    return ntu_joints.tolist()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                sequence_data.append(extract_keypoints(results.pose_landmarks))
    cap.release()
    return sequence_data

if __name__ == "__main__":
    # 1. Define both processing pipelines
    pipelines = [
        ("raw_videos", "train"),
        ("raw_bad_videos", "train_bad")
    ]

    # 2. Loop through both folders automatically
    for input_dir, output_dir in pipelines:
        print(f"\n--- Processing folder: {input_dir} ---")
        os.makedirs(input_dir, exist_ok=True)
        
        for file_name in os.listdir(input_dir):
            if file_name.endswith(('.mp4', '.avi')):
                video_path = os.path.join(input_dir, file_name)
                
                # 3. Clean the filename to accurately match your dictionary keys
                clean_name = file_name.replace("_bad", "").lower() 
                
                if "pullup" in clean_name: exercise_name = "pullup"
                elif "wall_pushup" in clean_name: exercise_name = "wall_pushup"
                elif "sit_up" in clean_name: exercise_name = "sit_up"
                elif "dumbbell_floor_press" in clean_name: exercise_name = "dumbbell_floor_press"
                elif "decline_pushup" in clean_name: exercise_name = "decline_pushup"
                elif "dumbbell_floor_flye" in clean_name: exercise_name = "dumbbell_floor_flye"
                elif "renegade_row" in clean_name: exercise_name = "renegade_row"
                elif "good_morning" in clean_name: exercise_name = "good_morning"
                elif "standing_overhead_extension" in clean_name: exercise_name = "standing_overhead_extension"
                elif "dumbbell_kickback" in clean_name: exercise_name = "dumbbell_kickback"
                elif "floor_skull_crusher" in clean_name: exercise_name = "floor_skull_crusher"
                elif "plank_triceps_extension" in clean_name: exercise_name = "plank_triceps_extension"
                elif "seated_knee_tucks" in clean_name: exercise_name = "seated_knee_tucks"
                elif "lying_leg_raise" in clean_name: exercise_name = "lying_leg_raise"
                elif "russian_twist" in clean_name: exercise_name = "russian_twist"
                elif "reverse_curl" in clean_name: exercise_name = "reverse_curl"
                elif "concentration_curl" in clean_name: exercise_name = "concentration_curl"
                elif "tricep_dips" in clean_name: exercise_name = "tricep_dips"
                else: continue  
                
                base_name = os.path.splitext(file_name)[0]
                
                subject_dir = os.path.join(output_dir, base_name)
                joints_dir = os.path.join(subject_dir, "joints3d_25")
                json_path = os.path.join(joints_dir, f"{exercise_name}.json")
                
                # Fast-forward check so you don't re-process old videos
                if os.path.exists(json_path):
                    print(f"⏭️ Skipping {file_name} (Already processed)")
                    continue
                    
                os.makedirs(joints_dir, exist_ok=True)
                
                print(f"Tracking skeleton in {file_name}...")
                sequence_data = process_video(video_path)
                
                if len(sequence_data) > 10:
                    with open(json_path, 'w') as f:
                        json.dump({"joints3d_25": sequence_data}, f)
                    
                    ann_path = os.path.join(subject_dir, "rep_ann.json")
                    with open(ann_path, 'w') as f:
                        json.dump({exercise_name: [0, len(sequence_data) - 1]}, f)
                    
                    print(f"✅ Saved {file_name} to {output_dir}: {len(sequence_data)} frames")