"""
Centralized Path Configuration
Handles all directory references for the FormCheck project.
Works regardless of where scripts are run from.
"""

import os

# Get the Backend directory (where this file is located)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Get formcheck directory
FORMCHECK_DIR = os.path.dirname(BACKEND_DIR)

# Get the root Thesis directory
THESIS_ROOT = os.path.dirname(FORMCHECK_DIR)

# Data directories
TRAIN_DATA_DIR = os.path.join(THESIS_ROOT, "train")
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

# Ensure dataset and models directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset files
STANDING_AB_TWISTS_X = os.path.join(DATASET_DIR, "standing_ab_twists_x.npy")
STANDING_AB_TWISTS_Y = os.path.join(DATASET_DIR, "standing_ab_twists_y.npy")

# Model files
STANDING_AB_TWISTS_MODEL = os.path.join(MODELS_DIR, "standing_ab_twists.pt")
ST_GCN_PUSHUP_MODEL = os.path.join(MODELS_DIR, "st_gcn_pushup.pt")
ST_GCN_SQUAT_MODEL = os.path.join(MODELS_DIR, "st_gcn_squat.pt")
ST_GCN_HEEL_RAISE_MODEL = os.path.join(MODELS_DIR, "st_gcn_heel_raise.pt")
ST_GCN_ARM_CIRCLES_MODEL = os.path.join(MODELS_DIR, "st_gcn_arm_circles.pt")
ST_GCN_CALF_STRETCH_MODEL = os.path.join(MODELS_DIR, "st_gcn_calf_stretch.pt")
ST_GCN_WOODCHOP_MODEL = os.path.join(MODELS_DIR, "st_gcn_woodchop.pt")
ST_GCN_CHEST_EXPANSION_MODEL = os.path.join(MODELS_DIR, "st_gcn_chest_expansion.pt")
ST_GCN_HAMSTRING_SWEEP_MODEL = os.path.join(MODELS_DIR, "st_gcn_hamstring_sweep.pt")
ST_GCN_HINGE_CHOP_MODEL = os.path.join(MODELS_DIR, "st_gcn_hinge_chop.pt")
ST_GCN_HIP_CIRCLES_MODEL = os.path.join(MODELS_DIR, "st_gcn_hip_circles.pt")
ST_GCN_LUNGE_TWIST_MODEL = os.path.join(MODELS_DIR, "st_gcn_lunge_twist.pt")
ST_GCN_NECK_ROTATION_MODEL = os.path.join(MODELS_DIR, "st_gcn_neck_rotation.pt")
ST_GCN_PLYO_CHOP_MODEL = os.path.join(MODELS_DIR, "st_gcn_plyo_chop.pt")
ST_GCN_REACH_TO_TOE_MODEL = os.path.join(MODELS_DIR, "st_gcn_reach_to_toe.pt")
ST_GCN_SHOULDER_ROLLS_MODEL = os.path.join(MODELS_DIR, "st_gcn_shoulder_rolls.pt")
ST_GCN_SIDE_LUNGE_REACH_MODEL = os.path.join(MODELS_DIR, "st_gcn_side_lunge_reach.pt")
ST_GCN_TOE_TOUCH_MODEL = os.path.join(MODELS_DIR, "st_gcn_toe_touch.pt")
ST_GCN_WRIST_CIRCLES_MODEL = os.path.join(MODELS_DIR, "st_gcn_wrist_circles.pt")

def print_config():
    """Debug function to verify all paths"""
    print("=" * 60)
    print("FormCheck Path Configuration")
    print("=" * 60)
    print(f"Backend Directory:    {BACKEND_DIR}")
    print(f"FormCheck Directory:  {FORMCHECK_DIR}")
    print(f"Thesis Root:          {THESIS_ROOT}")
    print(f"Train Data:           {TRAIN_DATA_DIR} (exists: {os.path.exists(TRAIN_DATA_DIR)})")
    print(f"Dataset Directory:    {DATASET_DIR} (exists: {os.path.exists(DATASET_DIR)})")
    print(f"Models Directory:     {MODELS_DIR} (exists: {os.path.exists(MODELS_DIR)})")
    print("=" * 60)
