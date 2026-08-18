import os
import cv2
import joblib
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize MediaPipe Pose for static images
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_landmarks(image_path):
    """Reads an image, extracts pose landmarks, and returns a flat list of features."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [Warning] Could not read image: {image_path}")
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
        
    # Flatten the 33 landmarks (x, y, z, visibility) into a single row of 132 features
    row = []
    for landmark in results.pose_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return row

def process_dataset_folder(base_folder, exercise_name):
    """Scans the positive/negative folders and builds a dataframe."""
    data = []
    labels = []
    
    exercise_path = os.path.join(base_folder, exercise_name)
    
    for label_dir, label_val in [("positive", 1), ("negative", 0)]:
        folder_path = os.path.join(exercise_path, label_dir)
        
        if not os.path.exists(folder_path):
            print(f"[Skip] Folder not found: {folder_path}")
            continue
            
        print(f"Scanning {label_dir} images in {folder_path}...")
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(folder_path, file_name)
            features = extract_landmarks(image_path)
            
            if features:
                data.append(features)
                labels.append(label_val)
                
    return pd.DataFrame(data), pd.Series(labels)

def train_exercise_model(exercise_name):
    print(f"=== Training Model for: {exercise_name.upper()} ===")
    
    # 1. Load Training Data
    print("\n--- Extracting Training Data ---")
    X_train, y_train = process_dataset_folder("dataset/train", exercise_name)
    
    if X_train.empty:
        print("Error: No training data found or no poses detected.")
        return
        
    # 2. Load Testing Data
    print("\n--- Extracting Testing Data ---")
    X_test, y_test = process_dataset_folder("dataset/test", exercise_name)
    
    # 3. Train Random Forest Model
    print("\n--- Training Random Forest Classifier ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Model
    if not X_test.empty:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative (Bad Form)', 'Positive (Good Form)']))
    else:
        print("\nNo test data found, skipping evaluation step.")
    
    # 5. Save the Model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{exercise_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Model successfully saved to: {model_path}\n")

if __name__ == "__main__":
    # You can change this to "pushup", "lunge", etc., when you have those datasets ready
    EXERCISE_TO_TRAIN = "pushup"
    train_exercise_model(EXERCISE_TO_TRAIN)