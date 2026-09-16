import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from st_gcn_model import STGCN

# Change this to whatever exercise you are currently training
EXERCISE = "standing_ab_twists" 

def train_model():
    print(f"Booting Form-Tech Native Training Pipeline for: {EXERCISE.upper()}...")
    
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

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    train_counts = np.bincount(y_train, minlength=2)
    test_counts = np.bincount(y_test, minlength=2)
    print(f"📊 Training Set: {len(X_train)} samples ({train_counts[0]} Bad, {train_counts[1]} Good)")
    print(f"📊 Testing Set:  {len(X_test)} samples ({test_counts[0]} Bad, {test_counts[1]} Good)")

    model = STGCN(num_classes=2)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    total = len(y_train)
    class_weight = {
        0: (1 / train_counts[0]) * (total / 2.0),
        1: (1 / train_counts[1]) * (total / 2.0)
    }

    # Reduced epochs so the model doesn't perfectly memorize the synthetic data
    epochs = 50 
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=8,
        epochs=epochs,
        class_weight=class_weight
    )

    os.makedirs("models/tflite", exist_ok=True)
    tflite_path = f"models/tflite/st_gcn_{EXERCISE}.tflite"
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"🎉 SUCCESS! Native TFLite model generated directly -> {tflite_path}")

    # ==========================================
    # QA METRICS: Calculate the Average Across All Epochs
    # ==========================================
    avg_train_acc = np.mean(history.history['accuracy'])
    avg_val_acc = np.mean(history.history['val_accuracy'])
    
    print("\n" + "="*40)
    print(f" METRICS: AVERAGE EPOCH PERFORMANCE")
    print("="*40)
    print(f"Average Training Accuracy (over {epochs} epochs):   {avg_train_acc:.4f} ({avg_train_acc*100:.1f}%)")
    print(f"Average Validation Accuracy (over {epochs} epochs): {avg_val_acc:.4f} ({avg_val_acc*100:.1f}%)")
    
    print("\n" + "="*40)
    print(f"FINAL SNAPSHOT EVALUATION METRICS")
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