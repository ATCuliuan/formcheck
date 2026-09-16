import os
import json
import numpy as np

SEQUENCE_LENGTH = 30

def sample_frames(frames_data, target_length):
    current_length = len(frames_data)
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return np.array(frames_data)[indices]

# ==========================================
# --- REPLACED: STATIC BAD FORM FUNCTIONS ---
# These are now dynamic to create multiple variations per video
# ==========================================

def generate_augmented_good_form(sequence):
    """Injects slight 3D noise to simulate different body types and camera jitter."""
    noise = np.random.normal(0, 0.01, sequence.shape) 
    return sequence + noise

def generate_augmented_bad_pushup(sequence):
    """Randomizes the hip drop severity between 15cm and 40cm."""
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.40)
    for joint in [0, 1, 4]:
        bad_seq[:, joint, 1] -= drop_amount  
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_squat(sequence):
    """Randomizes the knee valgus (cave-in) severity."""
    bad_seq = np.copy(sequence)
    cave_amount = np.random.uniform(0.05, 0.15)
    bad_seq[:, 13, 0] -= cave_amount
    bad_seq[:, 17, 0] += cave_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_burpees(sequence):
    """
    Synthesizes a 'lazy' burpee:
    Simulates a hunched posture and failure to fully extend/jump at the top of the movement.
    """
    bad_seq = np.copy(sequence)
    hunch_amount = np.random.uniform(0.15, 0.35)
    for joint in [2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= hunch_amount
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_mule_kick(sequence):
    """
    Synthesizes a bad mule kick:
    Simulates a collapsed core and excessive lower back arching (lumbar extension).
    """
    bad_seq = np.copy(sequence)
    arch_amount = np.random.uniform(0.10, 0.25)
    for joint in [0, 20]:
        bad_seq[:, joint, 1] -= arch_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_dumbbell_reverse_lunge(sequence):
    """
    Synthesizes a bad dumbbell reverse lunge:
    Simulates the chest collapsing forward (excessive torso lean) 
    instead of keeping the back straight.
    """
    bad_seq = np.copy(sequence)
    lean_amount = np.random.uniform(0.15, 0.30)
    
    # Drop the Y-axis of the neck(2), head(3), shoulders(4, 8), and upper spine(20)
    for joint in [2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= lean_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_dumbbell_overhead_shoulder_press(sequence):
    """
    Synthesizes a bad dumbbell overhead press:
    Simulates incomplete extension (failing to press the weight all the way up) 
    and bent elbows at the top of the movement.
    """
    bad_seq = np.copy(sequence)
    # The amount the arms fail to extend
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Elbow (14), Left Elbow (15), Right Wrist (16), and Left Wrist (17)
    for joint in [14, 15, 16, 17]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_dumbbell_hammer_curls(sequence):
    """
    Synthesizes a bad dumbbell hammer curl:
    Simulates an incomplete range of motion (half-rep) where the user 
    fails to curl the weight all the way up to full elbow flexion.
    """
    bad_seq = np.copy(sequence)
    # The distance the wrists fail to travel upwards
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Wrist (16) and Left Wrist (17)
    for joint in [16, 17]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_band_pull_apart(sequence):
    """
    Synthesizes a bad band pull-apart:
    Simulates the 'T-Rex' arm by severely dropping the elbows 
    instead of maintaining a wide, straight-arm arc.
    """
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Elbow (5) and Left Elbow (9)
    for joint in [5, 9]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_barbell_row(sequence):
    """
    Synthesizes a bad barbell row:
    Simulates a rounded back (loss of thoracic extension) and dumped shoulders 
    by dropping the Y-axis of the upper spine, neck, and shoulder joints.
    """
    bad_seq = np.copy(sequence)
    hunch_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the upper body to simulate a hunched/rounded posture
    for joint in [2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= hunch_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_deadlift(sequence):
    """
    Synthesizes a bad deadlift:
    Simulates the dangerous 'Fishing Pole' (rounded back) by collapsing 
    the upper spine, neck, and shoulders downward.
    """
    bad_seq = np.copy(sequence)
    hunch_amount = np.random.uniform(0.15, 0.40)
    
    # Drop the Y-axis of the upper body to simulate a rounded, unsafe spine
    for joint in [2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= hunch_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_dumbbell_biceps_curls(sequence):
    """
    Synthesizes a bad dumbbell biceps curl:
    Simulates an incomplete range of motion (half-rep) where the user 
    fails to curl the weight all the way up to full elbow flexion.
    """
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Wrist (16) and Left Wrist (17)
    for joint in [16, 17]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_one_arm_row(sequence):
    """
    Synthesizes a bad one arm row:
    Simulates the "Shoulder Dump" where the user loses core tension, 
    letting their working shoulder and upper spine round toward the floor.
    """
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the upper body to simulate a rounded, dumped shoulder posture
    for joint in [2, 3, 4, 8, 20]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_overhead_trap_raises(sequence):
    """
    Synthesizes a bad overhead trap raise:
    Simulates the "T-Rex Raise" (bent elbows and incomplete vertical extension) 
    by dropping the Y-axis of the wrists and elbows.
    """
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Elbow (14), Left Elbow (15), Right Wrist (16), and Left Wrist (17)
    for joint in [14, 15, 16, 17]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_side_lateral_raise(sequence):
    """
    Synthesizes a bad side lateral raise:
    Simulates an incomplete range of motion (failing to raise arms to shoulder height) 
    by dropping the Y-axis of the elbows and wrists.
    """
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Elbow (14), Left Elbow (15), Right Wrist (16), and Left Wrist (17)
    for joint in [14, 15, 16, 17]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

def generate_augmented_bad_standing_ab_twists(sequence):
    """
    Synthesizes a bad standing ab twist:
    Simulates the "Arm Drop" where the user's shoulders tire and they let 
    their elbows and wrists sag toward their waist instead of keeping them elevated.
    """
    bad_seq = np.copy(sequence)
    drop_amount = np.random.uniform(0.15, 0.35)
    
    # Drop the Y-axis of the Right Elbow (14), Left Elbow (15), Right Wrist (16), and Left Wrist (17)
    for joint in [14, 15, 16, 17]:
        bad_seq[:, joint, 1] -= drop_amount
        
    return bad_seq + np.random.normal(0, 0.005, sequence.shape)

# ==========================================
# --- ADD NEW FUNCTIONS HERE ---
# When you add new exercises for your thesis, write their bad form 
# generators right here (e.g., generate_augmented_bad_lunge).
# ==========================================


def extract_and_augment(exercise="dumbbell_hammer_curls"):
    base_folder = "train"
    folder_path = os.path.join("..", base_folder) if os.path.exists(os.path.join("..", base_folder)) else base_folder
    
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' not found.")
        return

    dataset_x = []
    dataset_y = []

    for subject in sorted(os.listdir(folder_path)):
        subject_dir = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_dir):
            continue
        
        joints_path = os.path.join(subject_dir, "joints3d_25", f"{exercise}.json")
        ann_path = os.path.join(subject_dir, "rep_ann.json")
        if not os.path.exists(joints_path) or not os.path.exists(ann_path):
            continue

        with open(joints_path, 'r') as f:
            motion_data = json.load(f)["joints3d_25"]
        with open(ann_path, 'r') as f:
            rep_boundaries = json.load(f).get(exercise, [])

        for i in range(len(rep_boundaries) - 1):
            start_f = rep_boundaries[i]
            end_f = rep_boundaries[i + 1]
            sampled_good = sample_frames(motion_data[start_f:end_f], SEQUENCE_LENGTH)
            
            # ==========================================
            # --- REPLACED: APPENDING LOGIC ---
            # We now append the base rep, PLUS 5 augmented variations
            # ==========================================
            
            # 1. Base Real Rep (Class 1)
            dataset_x.append(sampled_good)
            dataset_y.append(1)
            
            # 2. Augment Good Form (Create 2 extra variations)
            for _ in range(2):
                dataset_x.append(generate_augmented_good_form(sampled_good))
                dataset_y.append(1)
            
            # 3. Augment Bad Form (Create 3 randomized bad variations)
            for _ in range(3):
                if exercise == "squat":
                    dataset_x.append(generate_augmented_bad_squat(sampled_good))
                elif exercise == "pushup":
                    dataset_x.append(generate_augmented_bad_pushup(sampled_good))
                elif exercise == "burpees":
                    dataset_x.append(generate_augmented_bad_burpees(sampled_good))
                elif exercise == "mule_kick":
                    dataset_x.append(generate_augmented_bad_mule_kick(sampled_good))
                elif exercise == "dumbbell_reverse_lunge":
                    dataset_x.append(generate_augmented_bad_dumbbell_reverse_lunge(sampled_good))
                elif exercise == "dumbbell_overhead_shoulder_press":
                    dataset_x.append(generate_augmented_bad_dumbbell_overhead_shoulder_press(sampled_good))
                elif exercise == "dumbbell_hammer_curls":
                    dataset_x.append(generate_augmented_bad_dumbbell_hammer_curls(sampled_good))
                elif exercise == "band_pull_apart":
                    dataset_x.append(generate_augmented_bad_band_pull_apart(sampled_good))
                elif exercise == "barbell_row":
                    dataset_x.append(generate_augmented_bad_barbell_row(sampled_good))
                elif exercise == "deadlift":
                    dataset_x.append(generate_augmented_bad_deadlift(sampled_good))
                elif exercise == "dumbbell_biceps_curls":
                    dataset_x.append(generate_augmented_bad_dumbbell_biceps_curls(sampled_good))
                elif exercise == "one_arm_row":
                    dataset_x.append(generate_augmented_bad_one_arm_row(sampled_good))
                elif exercise == "overhead_trap_raises":
                    dataset_x.append(generate_augmented_bad_overhead_trap_raises(sampled_good))
                elif exercise == "side_lateral_raise":
                    dataset_x.append(generate_augmented_bad_side_lateral_raise(sampled_good))
                elif exercise == "standing_ab_twists":
                    dataset_x.append(generate_augmented_bad_standing_ab_twists(sampled_good))
                
                # --- ADD NEW EXERCISE ROUTING HERE ---
                # elif exercise == "lunge":
                #     dataset_x.append(generate_augmented_bad_lunge(sampled_good))
                
                dataset_y.append(0)

    os.makedirs("dataset", exist_ok=True)
    out_x = f"dataset/train_{exercise}_X.npy"
    out_y = f"dataset/train_{exercise}_y.npy"
    
    np.save(out_x, np.array(dataset_x, dtype=np.float32))
    np.save(out_y, np.array(dataset_y, dtype=np.int64))
    
    good_count = sum(dataset_y)
    bad_count = len(dataset_y) - good_count
    print(f" Generated Native TF {exercise.title()} Dataset: {good_count} Good | {bad_count} Bad (Total: {len(dataset_y)})")

if __name__ == "__main__":
    # Generate both datasets
    extract_and_augment("squat")
    extract_and_augment("pushup")
    extract_and_augment("burpees")
    extract_and_augment("mule_kick")
    extract_and_augment("dumbbell_reverse_lunge")
    extract_and_augment("dumbbell_overhead_shoulder_press")
    extract_and_augment("dumbbell_hammer_curls")
    extract_and_augment("band_pull_apart")
    extract_and_augment("barbell_row")
    extract_and_augment("deadlift")
    extract_and_augment("dumbbell_biceps_curls")
    extract_and_augment("one_arm_row")
    extract_and_augment("overhead_trap_raises")
    extract_and_augment("side_lateral_raise")
    extract_and_augment("standing_ab_twists")