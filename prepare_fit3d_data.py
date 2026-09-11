import os
import json
import numpy as np

SEQUENCE_LENGTH = 30

def sample_frames(frames_data, target_length):
    current_length = len(frames_data)
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return np.array(frames_data)[indices]

def generate_bad_pushup_form(sequence):
    """
    Synthetically alters a valid push-up to simulate a core collapse (sagging hips).
    Drops the entire pelvic cluster: Pelvis(0), L_Hip(1), and R_Hip(4).
    """
    bad_sequence = np.copy(sequence)
    # Drop the entire core heavily (0.30) to create a distinct mathematical class
    for joint in [0, 1, 4]:
        bad_sequence[:, joint, 1] -= 0.30  
    return bad_sequence

def extract_and_augment_pushup():
    base_folder = "train"
    folder_path = os.path.join("..", base_folder) if os.path.exists(os.path.join("..", base_folder)) else base_folder
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Directory {folder_path} not found.")
        return

    dataset_x = []
    dataset_y = []

    for subject in sorted(os.listdir(folder_path)):
        subject_dir = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_dir):
            continue
        
        # TARGET PUSH-UP FILES
        joints_path = os.path.join(subject_dir, "joints3d_25", "pushup.json")
        ann_path = os.path.join(subject_dir, "rep_ann.json")
        if not os.path.exists(joints_path) or not os.path.exists(ann_path):
            continue

        with open(joints_path, 'r') as f:
            motion_data = json.load(f)["joints3d_25"]
        with open(ann_path, 'r') as f:
            # Extract from the pushup array
            rep_boundaries = json.load(f).get("pushup", [])

        for i in range(len(rep_boundaries) - 1):
            start_f = rep_boundaries[i]
            end_f = rep_boundaries[i + 1]
            sampled_good = sample_frames(motion_data[start_f:end_f], SEQUENCE_LENGTH)
            
            # 1. Real Good Rep (Class 1)
            tensor_good = np.transpose(sampled_good, (2, 0, 1))
            dataset_x.append(tensor_good)
            dataset_y.append(1)
            
            # 2. Synthetic Bad Rep (Class 0 - Sagging Hips)
            sampled_bad = generate_bad_pushup_form(sampled_good)
            tensor_bad = np.transpose(sampled_bad, (2, 0, 1))
            dataset_x.append(tensor_bad)
            dataset_y.append(0)

    os.makedirs("dataset", exist_ok=True)
    # Save as pushup-specific files to avoid overwriting the squat dataset
    np.save("dataset/train_pushup_X.npy", np.array(dataset_x, dtype=np.float32))
    np.save("dataset/train_pushup_y.npy", np.array(dataset_y, dtype=np.int64))
    
    good_count = sum(dataset_y)
    bad_count = len(dataset_y) - good_count
    print(f"✅ Generated Balanced Push-Up Dataset: {good_count} Good | {bad_count} Bad (Total: {len(dataset_y)})")

if __name__ == "__main__":
    extract_and_augment_pushup()