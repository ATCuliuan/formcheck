import os
import json
import numpy as np

SEQUENCE_LENGTH = 30

def sample_frames(frames_data, target_length):
    """Interpolates a movement sequence to exactly 30 frames."""
    current_length = len(frames_data)
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return np.array(frames_data)[indices]

def generate_bad_form(sequence):
    """
    Mathematically distorts a perfect squat to simulate injury-risk form.
    Simulates a 'Good Morning' squat by dropping the shoulders and neck forward.
    """
    bad_sequence = np.copy(sequence)
    
    # In a typical 25-joint model: Neck=8, Left Shoulder=11, Right Shoulder=14
    # We shift the Y and Z coordinates to simulate the chest falling forward
    for joint in [8, 11, 14]:
        bad_sequence[:, joint, 1] -= 0.15  # Drop Y axis (downward)
        bad_sequence[:, joint, 2] += 0.15  # Push Z axis (forward)
        
    return bad_sequence

def process_directory(base_folder, output_prefix):
    dataset_x = []
    dataset_y = []
    
    # Check parent directory if running inside nested formcheck folder
    folder_path = os.path.join("..", base_folder) if os.path.exists(os.path.join("..", base_folder)) else base_folder
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Directory {folder_path} not found. Skipping.")
        return

    # Loop through every subject folder (s03, s04, s05...)
    for subject in os.listdir(folder_path):
        subject_dir = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_dir): 
            continue
        
        joints_path = os.path.join(subject_dir, "joints3d_25", "squat.json")
        ann_path = os.path.join(subject_dir, "rep_ann.json")
        
        if not os.path.exists(joints_path) or not os.path.exists(ann_path):
            continue
            
        with open(joints_path, 'r') as f:
            motion_data = json.load(f)["joints3d_25"]
        with open(ann_path, 'r') as f:
            rep_boundaries = json.load(f).get("squat", [])

        # Slice the continuous video into individual reps
        for i in range(len(rep_boundaries) - 1):
            start_f = rep_boundaries[i]
            end_f = rep_boundaries[i + 1]
            
            # 1. The Good Rep (Label 1)
            sampled_good = sample_frames(motion_data[start_f:end_f], SEQUENCE_LENGTH)
            tensor_good = np.transpose(sampled_good, (2, 0, 1)) # (3, 30, 25)
            dataset_x.append(tensor_good)
            dataset_y.append(1)
            
            # 2. The Bad Rep (Label 0)
            sampled_bad = generate_bad_form(sampled_good)
            tensor_bad = np.transpose(sampled_bad, (2, 0, 1)) # (3, 30, 25)
            dataset_x.append(tensor_bad)
            dataset_y.append(0)

    if dataset_x:
        os.makedirs("dataset", exist_ok=True)
        np.save(f"dataset/{output_prefix}_X.npy", np.array(dataset_x, dtype=np.float32))
        np.save(f"dataset/{output_prefix}_y.npy", np.array(dataset_y, dtype=np.int64))
        print(f"✅ Extracted {len(dataset_x)} total sequences ({len(dataset_x)//2} Good, {len(dataset_x)//2} Bad) from {base_folder}")
    else:
        print(f"⚠️ No usable squat data found in {base_folder}")

if __name__ == "__main__":
    print("Extracting & Augmenting Training Data...")
    process_directory("train", "train")