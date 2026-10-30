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

def apply_joint_mask(sequence, exercise):
    # NTU Leg Indices: Hips (12, 16), Knees (13, 17), Ankles (14, 18), Feet (15, 19), Mid-Hip (0)
    masked_seq = np.copy(sequence)
    
    upper_body_exercises = [
        "standing_overhead_extension", "dumbbell_kickback", "floor_skull_crusher",
        "reverse_curl", "concentration_curl", "dumbbell_floor_press",
        "dumbbell_floor_flye", "pullup", "tricep_dips"
    ]
    
    plank_core_exercises = [
        "renegade_row", "wall_pushup", "decline_pushup", "plank_triceps_extension","good_morning"
    ]
    
    if exercise in upper_body_exercises:
        # Mask everything from the hips down
        joints_to_mask = [0, 12, 13, 14, 15, 16, 17, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]:
                masked_seq[:, j, :] = 0.0
                
    elif exercise in plank_core_exercises:
        # Keep the hips for twist/sag detection, mask knees and below
        joints_to_mask = [13, 14, 15, 17, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]:
                masked_seq[:, j, :] = 0.0

    elif exercise == "seated_knee_tucks":
        # Keep knees and hips for the tuck motion, but mask the feet/ankles which often crop out
        joints_to_mask = [14, 15, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]:
                masked_seq[:, j, :] = 0.0

    return masked_seq

def generate_augmented_good_form(sequence):
    noise = np.random.normal(0, 0.01, sequence.shape) 
    return sequence + noise

def normalize_skeleton(sequence):
    norm_seq = np.copy(sequence)
    for i in range(len(norm_seq)):
        frame = norm_seq[i]
        
        # 1. Find the center point: Spine Shoulder (NTU Index 2)
        center = np.copy(frame[2])
        
        # 2. Find the scale: Distance between Left Shoulder (4) and Right Shoulder (8)
        shoulder_dist = np.linalg.norm(frame[4] - frame[8])
        if shoulder_dist < 1e-5:
            shoulder_dist = 1.0 # Prevent division by zero
            
        for j in range(25):
            # If the joint was masked (0.0, 0.0, 0.0), leave it as a strict zero
            if np.all(frame[j] == 0.0):
                norm_seq[i, j] = 0.0
            else:
                # 3. Shift the center to (0,0,0) and scale the size
                norm_seq[i, j] = (frame[j] - center) / shoulder_dist
                
    return norm_seq

def generate_augmented_bad_wall_pushup(sequence):
    bad_seq = np.copy(sequence)
    sag_amount = np.random.uniform(0.10, 0.25)
    for joint in [0, 1, 12, 16]:
        bad_seq[:, joint, 1] -= sag_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_pullup(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.30)
    for joint in [2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_sit_up(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    for joint in [1, 2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= drop_amount  
        bad_seq[:, joint, 2] += drop_amount  
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_dumbbell_floor_press(sequence):
    bad_seq = np.copy(sequence)
    flare_amount = np.random.uniform(0.15, 0.30)
    for joint in [5, 9, 6, 10]:
        bad_seq[:, joint, 1] += flare_amount 
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_decline_pushup(sequence):
    bad_seq = np.copy(sequence)
    sag_amount = np.random.uniform(0.15, 0.30)
    for joint in [0, 1, 12, 16]:
        bad_seq[:, joint, 1] -= sag_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_dumbbell_floor_flye(sequence):
    bad_seq = np.copy(sequence)
    bend_amount = np.random.uniform(0.15, 0.30)
    for joint in [6, 10]:
        bad_seq[:, joint, 1] -= bend_amount 
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_renegade_row(sequence):
    bad_seq = np.copy(sequence)
    twist_amount = np.random.uniform(0.15, 0.35)
    bad_seq[:, 12, 1] -= twist_amount  
    bad_seq[:, 16, 1] += twist_amount  
    bad_seq[:, 12, 2] += twist_amount  
    bad_seq[:, 16, 2] -= twist_amount  
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_good_morning(sequence):
    bad_seq = np.copy(sequence)
    round_amount = np.random.uniform(0.15, 0.35)
    for joint in [1, 2, 3]:
        bad_seq[:, joint, 1] -= round_amount
        bad_seq[:, joint, 2] += round_amount / 2
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_standing_overhead_extension(sequence):
    bad_seq = np.copy(sequence)
    drift_amount = np.random.uniform(0.15, 0.35)
    for joint in [5, 9]:
        bad_seq[:, joint, 1] -= drift_amount  
        bad_seq[:, joint, 2] += drift_amount  
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_dumbbell_kickback(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.30)
    for joint in [5, 9]:
        bad_seq[:, joint, 1] -= drop_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_floor_skull_crusher(sequence):
    bad_seq = np.copy(sequence)
    drift_amount = np.random.uniform(0.15, 0.35)
    for joint in [5, 9]:
        bad_seq[:, joint, 1] -= drift_amount
        bad_seq[:, joint, 2] -= drift_amount / 2
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_plank_triceps_extension(sequence):
    bad_seq = np.copy(sequence)
    sag_amount = np.random.uniform(0.15, 0.35)
    for joint in [0, 1, 12, 16]:
        bad_seq[:, joint, 1] -= sag_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_seated_knee_tucks(sequence):
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    lean_amount = np.random.uniform(0.10, 0.25)
    for joint in [2, 3]:
        bad_seq[:, joint, 1] -= lean_amount
        bad_seq[:, joint, 2] += lean_amount
    for joint in [15, 16, 19, 20, 21, 22]: 
        if joint < bad_seq.shape[1]: 
            bad_seq[:, joint, 1] += drop_amount 
    return bad_seq + np.random.normal(0, 0.025, sequence.shape)

def generate_augmented_bad_tricep_dips(sequence):
    bad_seq = np.copy(sequence)
    shrug_amount = np.random.uniform(0.10, 0.25)
    for joint in [2, 3, 4, 8]:
        if joint < bad_seq.shape[1]:
            bad_seq[:, joint, 1] += shrug_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_lying_leg_raise(sequence):
    bad_seq = np.copy(sequence)
    arch_amount = np.random.uniform(0.10, 0.25)
    for joint in [0, 1]:
        bad_seq[:, joint, 1] -= arch_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_russian_twist(sequence):
    bad_seq = np.copy(sequence)
    upright_amount = np.random.uniform(0.15, 0.35)
    for joint in [1, 2, 3]:
        bad_seq[:, joint, 1] -= upright_amount
        bad_seq[:, joint, 2] -= upright_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_reverse_curl(sequence):
    bad_seq = np.copy(sequence)
    drift_amount = np.random.uniform(0.15, 0.30)
    for joint in [5, 9]:
        bad_seq[:, joint, 2] -= drift_amount
        bad_seq[:, joint, 1] -= drift_amount / 2
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_concentration_curl(sequence):
    bad_seq = np.copy(sequence)
    lift_amount = np.random.uniform(0.15, 0.30)
    for joint in [5, 9]:
        bad_seq[:, joint, 1] -= lift_amount
        bad_seq[:, joint, 2] -= lift_amount / 2
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def extract_and_augment(exercise):
    good_folder = "train"
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
                        
                        # 1. Base real YouTube bad rep
                        dataset_x.append(sampled_bad)
                        dataset_y.append(0) 
                        real_bad_count += 1
                        
                        # 2. Add 2 augmented variations (x3 multiplier) just like Good Form!
                        for _ in range(2):
                            # We can reuse the good_form function here because it just adds generic camera noise
                            dataset_x.append(generate_augmented_good_form(sampled_bad)) 
                            dataset_y.append(0)
                            real_bad_count += 1

    # 2. LOAD GOOD FORM & CALCULATE TOTAL VOLUME
    if not os.path.exists(good_folder):
        print(f"[{exercise.upper()}] Error: '{good_folder}' directory not found in the current path.")
        return

    good_reps_list = [] 
    found_files = False

    for subject in sorted(os.listdir(good_folder)):
        subject_dir = os.path.join(good_folder, subject)
        if not os.path.isdir(subject_dir): continue
        
        joints_path = os.path.join(subject_dir, "joints3d_25", f"{exercise}.json")
        ann_path = os.path.join(subject_dir, "rep_ann.json")
        
        if not os.path.exists(joints_path) or not os.path.exists(ann_path):
            continue

        found_files = True
        
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
                
                # Base rep + 2 augmented variations = 3 total good reps
                for _ in range(2):
                    dataset_x.append(generate_augmented_good_form(sampled_good))
                    dataset_y.append(1)
                    good_reps_list.append(sampled_good)

    if not found_files:
        print(f"[{exercise.upper()}] Warning: No JSON files found in {good_folder}.")
        return
        
    total_good_count = len(good_reps_list)
    if total_good_count == 0:
        return
        
    # 3. THE SAFETY NET (Fill deficit with synthetic data to achieve 50/50 balance)
    deficit = total_good_count - real_bad_count
    
    if deficit > 0:
        print(f"[{exercise.upper()}] Safety Net Triggered: Generating {deficit} synthetic bad reps...")
        for i in range(deficit):
            base_good_rep = good_reps_list[i % len(good_reps_list)] 
            
            if exercise == "wall_pushup":
                dataset_x.append(generate_augmented_bad_wall_pushup(base_good_rep))
            elif exercise == "pullup":
                dataset_x.append(generate_augmented_bad_pullup(base_good_rep))
            elif exercise == "sit_up":
                dataset_x.append(generate_augmented_bad_sit_up(base_good_rep))
            elif exercise == "dumbbell_floor_press":
                dataset_x.append(generate_augmented_bad_dumbbell_floor_press(base_good_rep))
            elif exercise == "decline_pushup":
                dataset_x.append(generate_augmented_bad_decline_pushup(base_good_rep))
            elif exercise == "dumbbell_floor_flye":
                dataset_x.append(generate_augmented_bad_dumbbell_floor_flye(base_good_rep))
            elif exercise == "renegade_row":
                dataset_x.append(generate_augmented_bad_renegade_row(base_good_rep))
            elif exercise == "good_morning":
                dataset_x.append(generate_augmented_bad_good_morning(base_good_rep))
            elif exercise == "standing_overhead_extension":
                dataset_x.append(generate_augmented_bad_standing_overhead_extension(base_good_rep))
            elif exercise == "dumbbell_kickback":
                dataset_x.append(generate_augmented_bad_dumbbell_kickback(base_good_rep))
            elif exercise == "floor_skull_crusher":
                dataset_x.append(generate_augmented_bad_floor_skull_crusher(base_good_rep))
            elif exercise == "plank_triceps_extension":
                dataset_x.append(generate_augmented_bad_plank_triceps_extension(base_good_rep))
            elif exercise == "seated_knee_tucks":
                dataset_x.append(generate_augmented_bad_seated_knee_tucks(base_good_rep))
            elif exercise == "lying_leg_raise":
                dataset_x.append(generate_augmented_bad_lying_leg_raise(base_good_rep))
            elif exercise == "russian_twist":
                dataset_x.append(generate_augmented_bad_russian_twist(base_good_rep))
            elif exercise == "reverse_curl":
                dataset_x.append(generate_augmented_bad_reverse_curl(base_good_rep))
            elif exercise == "concentration_curl":
                dataset_x.append(generate_augmented_bad_concentration_curl(base_good_rep))
            elif exercise == "tricep_dips":
                dataset_x.append(generate_augmented_bad_tricep_dips(base_good_rep))
            else:
                dataset_x.append(generate_augmented_good_form(base_good_rep)) # Safe fallback
            
            dataset_y.append(0)

    elif deficit < 0:
        good_deficit = abs(deficit)
        print(f"[{exercise.upper()}] WARNING: You have more YouTube Bad Form than Good Form!")
        print(f"Upsampling Good Form: Generating {good_deficit} more good reps to maintain perfect 50/50 balance...")
        for i in range(good_deficit):
            base_good_rep = good_reps_list[i % len(good_reps_list)] 
            dataset_x.append(generate_augmented_good_form(base_good_rep))
            dataset_y.append(1)
            total_good_count += 1
        
    else:
        print(f"[{exercise.upper()}] Dataset naturally balanced with real videos. No synthetic data needed!")

    # 4. APPLY MASK, NORMALIZE, AND SAVE ARRAYS
    os.makedirs("dataset", exist_ok=True)
    out_x = f"dataset/train_{exercise}_X.npy"
    out_y = f"dataset/train_{exercise}_y.npy"
    
    # Loop through, mask the cheat-code joints, and normalize the skeleton scale
    processed_dataset_x = []
    for seq in dataset_x:
        masked_seq = apply_joint_mask(seq, exercise)
        normalized_seq = normalize_skeleton(masked_seq)
        processed_dataset_x.append(normalized_seq)
    
    np.save(out_x, np.array(processed_dataset_x, dtype=np.float32))
    np.save(out_y, np.array(dataset_y, dtype=np.int64))
    
    final_bad_count = len(dataset_y) - total_good_count
    print(f"✅ Generated Native TF {exercise.title()} Dataset: {total_good_count} Good | {final_bad_count} Bad (Total: {len(dataset_y)})\n")


if __name__ == "__main__":
    extract_and_augment("wall_pushup")
    extract_and_augment("pullup")
    extract_and_augment("sit_up")
    extract_and_augment("dumbbell_floor_press")
    extract_and_augment("decline_pushup")
    extract_and_augment("dumbbell_floor_flye")
    extract_and_augment("renegade_row")#
    extract_and_augment("good_morning")#
    extract_and_augment("standing_overhead_extension")#
    extract_and_augment("dumbbell_kickback")
    extract_and_augment("floor_skull_crusher")
    extract_and_augment("plank_triceps_extension")
    extract_and_augment("seated_knee_tucks")#
    extract_and_augment("lying_leg_raise")
    extract_and_augment("russian_twist")
    extract_and_augment("reverse_curl")
    extract_and_augment("concentration_curl")
    extract_and_augment("tricep_dips")