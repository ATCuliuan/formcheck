import os
import glob
import numpy as np

SEQUENCE_LENGTH = 30

# Updated to use relative paths based on your new clean folder structure
GOOD_DIR = os.path.join("UI-PRMD", "Segmented Movements", "Kinect", "Positions")
BAD_DIR = os.path.join("UI-PRMD", "Incorrect Segmented Movements", "Kinect", "Positions")

# Mapping UI-PRMD (22 joints) to ST-GCN (25 joints)
UI_TO_STGCN = [
    0, 1, 3, 4,        # 0:Base Spine, 1:Mid Spine, 2:Neck, 3:Head
    7, 8, 9, 9,        # 4:L Shoulder, 5:L Elbow, 6:L Wrist, 7:L Hand
    11, 12, 13, 13,    # 8:R Shoulder, 9:R Elbow, 10:R Wrist, 11:R Hand
    14, 15, 16, 17,    # 12:L Hip, 13:L Knee, 14:L Ankle, 15:L Foot
    18, 19, 20, 21,    # 16:R Hip, 17:R Knee, 18:R Ankle, 19:R Foot
    2,                 # 20:Spine Shoulder / Chest
    9, 9, 13, 13       # 21-24: Thumb/Hand tips (duplicated from wrists)
]

def load_and_map_sequence(file_path):
    """Reads a UI-PRMD comma-separated positions file and maps to 25 joints."""
    data = np.loadtxt(file_path, delimiter=',')
    frames = data.shape[0]
    data_3d = data.reshape((frames, 22, 3))
    mapped_data = data_3d[:, UI_TO_STGCN, :]
    return mapped_data

def sample_frames(frames_data, target_length):
    current_length = len(frames_data)
    if current_length == 0:
        return np.zeros((target_length, 25, 3))
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return frames_data[indices]

def normalize_skeleton(sequence):
    """Scales the skeleton based on shoulder width to prevent camera-distance data leaks."""
    norm_seq = np.copy(sequence)
    for i in range(len(norm_seq)):
        frame = norm_seq[i]
        center = np.copy(frame[2]) # Spine Shoulder
        shoulder_dist = np.linalg.norm(frame[4] - frame[8]) 
        if shoulder_dist < 1e-5: shoulder_dist = 1.0 
            
        for j in range(25):
            if np.all(frame[j] == 0.0):
                norm_seq[i, j] = 0.0
            else:
                norm_seq[i, j] = (frame[j] - center) / shoulder_dist
    return norm_seq

def augment_sequence(sequence):
    """Injects slight 3D noise to simulate different body types and camera jitter."""
    noise = np.random.normal(0, 0.025, sequence.shape) 
    return sequence + noise

def process_folder(movement_code, folder_path, label, dataset_x, dataset_y):
    """Reads all segmented files in a folder, maps, normalizes, and augments."""
    search_pattern = os.path.join(folder_path, f"{movement_code}_*positions*.txt")
    file_list = glob.glob(search_pattern)
    
    for file_path in file_list:
        try:
            motion_data = load_and_map_sequence(file_path)
            sampled_seq = sample_frames(motion_data, SEQUENCE_LENGTH)
            
            # Apply Normalization before saving
            normalized_seq = normalize_skeleton(sampled_seq)
            
            # Base Rep
            dataset_x.append(normalized_seq)
            dataset_y.append(label)
            
            # Create 4 extra variations using noise to ensure a robust dataset size
            for _ in range(4):
                augmented_seq = augment_sequence(normalized_seq)
                dataset_x.append(augmented_seq)
                dataset_y.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def extract_and_augment(movement_code, exercise_name):
    if not os.path.exists(GOOD_DIR) or not os.path.exists(BAD_DIR):
        print(f"❌ Directories not found. Please verify paths.")
        return

    dataset_x = []
    dataset_y = []

    # Process Correct Movements (Label 1)
    process_folder(movement_code, GOOD_DIR, 1, dataset_x, dataset_y)
    
    # Process Incorrect Movements (Label 0)
    process_folder(movement_code, BAD_DIR, 0, dataset_x, dataset_y)

    if not dataset_y:
        print(f"⚠️ No files found for {exercise_name} ({movement_code}). Check folders.")
        return

    os.makedirs("dataset", exist_ok=True)
    out_x = f"dataset/train_{exercise_name}_X.npy"
    out_y = f"dataset/train_{exercise_name}_y.npy"
    
    np.save(out_x, np.array(dataset_x, dtype=np.float32))
    np.save(out_y, np.array(dataset_y, dtype=np.int64))
    
    good_count = sum(dataset_y)
    bad_count = len(dataset_y) - good_count
    print(f"✅ Generated Real-World UI-PRMD {exercise_name.replace('_', ' ').title()} Dataset: {good_count} Good | {bad_count} Bad (Total: {len(dataset_y)})")

if __name__ == "__main__":
    # Execution list for UI-PRMD
    extract_and_augment("m01", "squat")
    extract_and_augment("m02", "hurdle_step")
    extract_and_augment("m03", "inline_lunge")
    extract_and_augment("m04", "side_lunge")
    extract_and_augment("m05", "sit_to_stand")
    extract_and_augment("m06", "standing_active_straight_leg_raise")
    extract_and_augment("m07", "standing_shoulder_abduction")
    extract_and_augment("m08", "standing_shoulder_extension")
    extract_and_augment("m09", "standing_shoulder_internal_external_rotation")
    extract_and_augment("m10", "standing_shoulder_scaption")