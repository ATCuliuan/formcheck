import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from st_gcn_model import STGCN

# 1. LOCK THE SEED: This stops the random slot-machine fluctuation.
np.random.seed(42)
tf.random.set_seed(42)

# Change this to whatever exercise you are currently training
EXERCISE = "standing_shoulder_internal_external_rotation" 

def twin_safe_split(X, y, test_size=0.2):
    """
    Groups indices into chunks of 3 (twins) and randomly shuffles the chunks 
    BEFORE splitting. This ensures the test set gets a fair mix of Real YouTube 
    videos and Synthetic videos, preventing artificial 100% scores.
    """
    good_idx = np.where(y == 1)[0]
    bad_idx = np.where(y == 0)[0]

    def shuffle_in_chunks(indices, chunk_size=6):
        # Group indices into chunks of 6 (locking base reps and their twins together)
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        # Shuffle the chunks so real YouTube data and synthetic data are mixed
        np.random.shuffle(chunks)
        # Flatten back into a single array
        return np.concatenate(chunks)

    # Mix the data BEFORE slicing
    good_idx_mixed = shuffle_in_chunks(good_idx)
    bad_idx_mixed = shuffle_in_chunks(bad_idx)

    # Now calculate the 80/20 split on the properly mixed arrays
    good_split = int(len(good_idx_mixed) * (1 - test_size))
    bad_split = int(len(bad_idx_mixed) * (1 - test_size))

    train_idx = np.concatenate([good_idx_mixed[:good_split], bad_idx_mixed[:bad_split]])
    test_idx = np.concatenate([good_idx_mixed[good_split:], bad_idx_mixed[bad_split:]])

    # Final shuffle so the training batches aren't ordered by class
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_model():
    print(f"Booting FormCheck Native Training Pipeline for: {EXERCISE.upper()}...")
    
    x_path = f"dataset/train_{EXERCISE}_X.npy"
    y_path = f"dataset/train_{EXERCISE}_y.npy"
    
    if not os.path.exists(x_path):
        x_path = f"dataset/{EXERCISE}_X.npy"
        y_path = f"dataset/{EXERCISE}_y.npy"

    if not os.path.exists(x_path):
        print(f"Error: Could not find dataset files for {EXERCISE} at {x_path}")
        return
        
    X_full = np.load(x_path)
    y_full = np.load(y_path)
    
    if X_full.ndim == 4 and X_full.shape[1] == 3:
        print("Detected PyTorch array format. Auto-transposing to (Batch, Time, Vertices, Channels)...")
        X_full = np.transpose(X_full, (0, 2, 3, 1))

    # Apply the custom Twin-Safe Split to destroy the Data Leak
    X_train, X_test, y_train, y_test = twin_safe_split(X_full, y_full, test_size=0.2)
    
    train_counts = np.bincount(y_train, minlength=2)
    test_counts = np.bincount(y_test, minlength=2)
    print(f"📊 Training Set: {len(X_train)} samples ({train_counts[0]} Bad, {train_counts[1]} Good)")
    print(f"📊 Testing Set:  {len(X_test)} samples ({test_counts[0]} Bad, {test_counts[1]} Good)")

    model = STGCN(num_classes=2)
    
    model.compile(
        # LOWER LEARNING RATE: Forces the AI to take smaller, more precise steps
        optimizer=Adam(learning_rate=0.0005),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    total = len(y_train)
    class_weight = {
        0: (1 / train_counts[0]) * (total / 2.0),
        1: (1 / train_counts[1]) * (total / 2.0)
    }

    # EARLY STOPPING: Increased patience to account for the slower learning rate
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=8,
        epochs=200,
        class_weight=class_weight,
        callbacks=[early_stop]
    )

    # ==========================================
    # EXPORT PIPELINE: Mobile TFLite & Backend TF
    # ==========================================
    os.makedirs("models/tflite", exist_ok=True)
    os.makedirs("models/backend", exist_ok=True)
    
    # 1. Export heavy-duty model for backend servers
    # Custom subclassed models require the TF SavedModel format (directory) instead of .h5
    backend_path = f"models/backend/st_gcn_{EXERCISE}"
    model.save(backend_path, save_format="tf")
    print(f"✅ Backend TF model exported -> {backend_path}")

    # 2. Export lightweight TFLite model for Mobile Frontend
    tflite_path = f"models/tflite/st_gcn_{EXERCISE}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"📱 Mobile TFLite model exported -> {tflite_path}")

    # ==========================================
    # FINAL SNAPSHOT EVALUATION METRICS
    # ==========================================
    actual_epochs = len(history.history['accuracy'])
    avg_train_acc = np.mean(history.history['accuracy'])
    avg_val_acc = np.mean(history.history['val_accuracy'])
    
    print("\n" + "="*40)
    print(f" METRICS: AVERAGE EPOCH PERFORMANCE")
    print("="*40)
    print(f"Average Training Accuracy (over {actual_epochs} epochs):   {avg_train_acc:.4f} ({avg_train_acc*100:.1f}%)")
    print(f"Average Validation Accuracy (over {actual_epochs} epochs): {avg_val_acc:.4f} ({avg_val_acc*100:.1f}%)")
    
    print("\n" + "="*40)
    print(f"HONEST EVALUATION METRICS (Twin Leak Patched)")
    print("="*40)
    
    y_pred_probs = model.predict(X_test)
    preds = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print("\nConfusion Matrix (2x2):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, preds, labels=[0, 1], target_names=["Bad Form (0)", "Good Form (1)"], zero_division=0))

if __name__ == "__main__":
    train_model()