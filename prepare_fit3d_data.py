import os
import json
import numpy as np

SEQUENCE_LENGTH = 30

def sample_frames(frames_data, target_length):
    current_length = len(frames_data)
    if current_length == 0:
        return np.zeros((target_length, 25, 3))
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return np.array(frames_data)[indices]

# ==========================================
# 1. AI PRE-PROCESSING (Masks & Normalization)
# ==========================================
def apply_joint_mask(sequence, exercise):
    masked_seq = np.copy(sequence)
    
    upper_body_exercises = [
        "dumbbell_overhead_shoulder_press", "dumbbell_hammer_curls", 
        "band_pull_apart", "dumbbell_biceps_curls", "overhead_trap_raises", 
        "side_lateral_raise"
    ]
    plank_core_exercises = [
        "pushup", "barbell_row", "standing_ab_twists"
    ]
    full_body_crop = [
         "deadlift", "burpees", "mule_kick", "dumbbell_reverse_lunge"
    ]
    
    if exercise in upper_body_exercises:
        # Mask everything from the hips down
        joints_to_mask = [0, 12, 13, 14, 15, 16, 17, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]: masked_seq[:, j, :] = 0.0
                
    elif exercise in plank_core_exercises:
        # Keep hips, mask knees and below
        joints_to_mask = [13, 14, 15, 17, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]: masked_seq[:, j, :] = 0.0
            
    elif exercise in full_body_crop:
        # Keep everything but the extreme feet points to prevent edge-frame jitter
        joints_to_mask = [15, 19] 
        for j in joints_to_mask:
            if j < masked_seq.shape[1]: masked_seq[:, j, :] = 0.0
                
    return masked_seq

def normalize_skeleton(sequence):
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

# ==========================================
# 2. SYNTHETIC BAD FORM FUNCTIONS (Noise Boosted to 0.025)
# ==========================================
def generate_augmented_good_form(sequence):
    noise = np.random.normal(0, 0.01, sequence.shape) 
    return sequence + noise

def generate_augmented_bad_pushup(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.40)
    for joint in [0, 1, 4]: bad_seq[:, joint, 1] -= drop_amount  
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_burpees(sequence):
    bad_seq = np.copy(sequence)
    hunch_amount = np.random.uniform(0.15, 0.35)
    for joint in [2, 3, 4, 8, 20]: bad_seq[:, joint, 1] -= hunch_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_mule_kick(sequence):
    bad_seq = np.copy(sequence)
    arch_amount = np.random.uniform(0.10, 0.25)
    for joint in [0, 20]: bad_seq[:, joint, 1] -= arch_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_dumbbell_reverse_lunge(sequence):
    bad_seq = np.copy(sequence)
    lean_amount = np.random.uniform(0.15, 0.30)
    for joint in [2, 3, 4, 8, 20]: bad_seq[:, joint, 1] -= lean_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_dumbbell_overhead_shoulder_press(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [14, 15, 16, 17]: bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_dumbbell_hammer_curls(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [16, 17]: bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_band_pull_apart(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [5, 9]: bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_barbell_row(sequence):
    bad_seq = np.copy(sequence)
    hunch_amount = np.random.uniform(0.15, 0.35)
    for joint in [2, 3, 4, 8, 20]: bad_seq[:, joint, 1] -= hunch_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_deadlift(sequence):
    bad_seq = np.copy(sequence)
    hunch_amount = np.random.uniform(0.15, 0.40)
    for joint in [2, 3, 4, 8, 20]: bad_seq[:, joint, 1] -= hunch_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_dumbbell_biceps_curls(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [16, 17]: bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_overhead_trap_raises(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [14, 15, 16, 17]: bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_side_lateral_raise(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [14, 15, 16, 17]: bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)


# ==========================================
# 3. EXTRACTION AND AUGMENTATION LOGIC
# ==========================================
def extract_and_augment(exercise):
    good_folder = "fit3d_train"
    bad_folder = "train_bad"

    dataset_x = []
    dataset_y = []

    # 1. LOAD REAL BAD FORM FIRST (The Priority)
    real_bad_count = 0
    if os.path.exists(bad_folder):
        for subject in sorted(os.listdir(bad_folder)):
            subject_dir = os.path.join(bad_folder, subject)
            if not os.path.isdir(subject_dir): continue
            
            joints_path = os.path.join(subject_dir, "joints3d_25", f"{exercise}.json")
            ann_path = os.path.join(subject_dir, "rep_ann.json")
            
            if os.path.exists(joints_path) and os.path.exists(ann_path):
                with open(joints_path, 'r') as f:
                    motion_data = json.load(f)["joints3d_25"]
                with open(ann_path, 'r') as f:
                    rep_boundaries = json.load(f).get(exercise, [])

                for i in range(len(rep_boundaries) - 1):
                    start_f = rep_boundaries[i]
                    end_f = rep_boundaries[i + 1]
                    total_frames = end_f - start_f
                    
                    chunks = []
                    if total_frames > 90:
                        stride = 45 
                        for j in range(start_f, end_f - 90 + 1, stride):
                            chunks.append(motion_data[j : j + 90])
                    else:
                        chunks.append(motion_data[start_f:end_f])
                        
                    for chunk in chunks:
                        sampled_bad = sample_frames(chunk, SEQUENCE_LENGTH)
                        
                        dataset_x.append(sampled_bad)
                        dataset_y.append(0) 
                        real_bad_count += 1
                        
                        for _ in range(2):
                            dataset_x.append(generate_augmented_good_form(sampled_bad)) 
                            dataset_y.append(0)
                            real_bad_count += 1

    # 2. LOAD GOOD FORM (Directly from Fit3D Train Folder)
    if not os.path.exists(good_folder):
        print(f"[{exercise.upper()}] Error: '{good_folder}' directory not found.")
        return

    good_reps_list = [] 
    
    for subject in sorted(os.listdir(good_folder)):
        subject_dir = os.path.join(good_folder, subject)
        if not os.path.isdir(subject_dir): continue
        
        joints_path = os.path.join(subject_dir, "joints3d_25", f"{exercise}.json")
        ann_path = os.path.join(subject_dir, "rep_ann.json")
        
        if not os.path.exists(joints_path) or not os.path.exists(ann_path):
            continue

        with open(joints_path, 'r') as f:
            motion_data = json.load(f)["joints3d_25"]
        with open(ann_path, 'r') as f:
            rep_boundaries = json.load(f).get(exercise, [])

        if not rep_boundaries: continue

        for i in range(len(rep_boundaries) - 1):
            start_f = rep_boundaries[i]
            end_f = rep_boundaries[i + 1]
            total_frames = end_f - start_f
            
            chunks = []
            if total_frames > 90:
                stride = 45 
                for j in range(start_f, end_f - 90 + 1, stride):
                    chunks.append(motion_data[j : j + 90])
            else:
                chunks.append(motion_data[start_f:end_f])
                
            for chunk in chunks:
                sampled_good = sample_frames(chunk, SEQUENCE_LENGTH)
                
                dataset_x.append(sampled_good)
                dataset_y.append(1)
                good_reps_list.append(sampled_good)
                
                for _ in range(2):
                    dataset_x.append(generate_augmented_good_form(sampled_good))
                    dataset_y.append(1)
                    good_reps_list.append(sampled_good)

    total_good_count = len(good_reps_list)
    if total_good_count == 0:
        print(f"[{exercise.upper()}] Warning: No Good Form reps found in {good_folder}.")
        return
        
    # 3. THE SAFETY NET (Fill deficit with synthetic data)
    deficit = total_good_count - real_bad_count
    
    if deficit > 0:
        print(f"[{exercise.upper()}] Safety Net Triggered: Generating {deficit} synthetic bad reps...")
        for i in range(deficit):
            base_good_rep = good_reps_list[i % len(good_reps_list)] 
            
            if exercise == "pushup": dataset_x.append(generate_augmented_bad_pushup(base_good_rep))
            elif exercise == "burpees": dataset_x.append(generate_augmented_bad_burpees(base_good_rep))
            elif exercise == "mule_kick": dataset_x.append(generate_augmented_bad_mule_kick(base_good_rep))
            elif exercise == "dumbbell_reverse_lunge": dataset_x.append(generate_augmented_bad_dumbbell_reverse_lunge(base_good_rep))
            elif exercise == "dumbbell_overhead_shoulder_press": dataset_x.append(generate_augmented_bad_dumbbell_overhead_shoulder_press(base_good_rep))
            elif exercise == "dumbbell_hammer_curls": dataset_x.append(generate_augmented_bad_dumbbell_hammer_curls(base_good_rep))
            elif exercise == "band_pull_apart": dataset_x.append(generate_augmented_bad_band_pull_apart(base_good_rep))
            elif exercise == "barbell_row": dataset_x.append(generate_augmented_bad_barbell_row(base_good_rep))
            elif exercise == "deadlift": dataset_x.append(generate_augmented_bad_deadlift(base_good_rep))
            elif exercise == "dumbbell_biceps_curls": dataset_x.append(generate_augmented_bad_dumbbell_biceps_curls(base_good_rep))
            elif exercise == "overhead_trap_raises": dataset_x.append(generate_augmented_bad_overhead_trap_raises(base_good_rep))
            elif exercise == "side_lateral_raise": dataset_x.append(generate_augmented_bad_side_lateral_raise(base_good_rep))
            else: dataset_x.append(generate_augmented_good_form(base_good_rep)) 
            
            dataset_y.append(0)

    elif deficit < 0:
        good_deficit = abs(deficit)
        print(f"[{exercise.upper()}] Upsampling Good Form to maintain perfect 50/50 balance...")
        for i in range(good_deficit):
            base_good_rep = good_reps_list[i % len(good_reps_list)] 
            dataset_x.append(generate_augmented_good_form(base_good_rep))
            dataset_y.append(1)
            total_good_count += 1
        
    # 4. APPLY MASK, NORMALIZE, AND SAVE ARRAYS
    os.makedirs("dataset", exist_ok=True)
    out_x = f"dataset/train_{exercise}_X.npy"
    out_y = f"dataset/train_{exercise}_y.npy"
    
    processed_dataset_x = []
    for seq in dataset_x:
        masked_seq = apply_joint_mask(seq, exercise)
        normalized_seq = normalize_skeleton(masked_seq)
        processed_dataset_x.append(normalized_seq)
    
    np.save(out_x, np.array(processed_dataset_x, dtype=np.float32))
    np.save(out_y, np.array(dataset_y, dtype=np.int64))
    
    final_bad_count = len(dataset_y) - total_good_count
    print(f"✅ Generated Fit3D Native {exercise.title()} Dataset: {total_good_count} Good | {final_bad_count} Bad (Total: {len(dataset_y)})\n")


if __name__ == "__main__":
    extract_and_augment("pushup")
    extract_and_augment("burpees")#
    extract_and_augment("mule_kick")
    extract_and_augment("dumbbell_reverse_lunge")#
    extract_and_augment("dumbbell_overhead_shoulder_press")
    extract_and_augment("dumbbell_hammer_curls")
    extract_and_augment("band_pull_apart")#
    extract_and_augment("barbell_row")#
    extract_and_augment("deadlift")#
    extract_and_augment("dumbbell_biceps_curls")
    extract_and_augment("overhead_trap_raises")#
    extract_and_augment("side_lateral_raise")#