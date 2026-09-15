import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from st_gcn_model import STGCN

def train_model():
    print("🚀 Booting Form-Tech Native TensorFlow Training & TFLite Pipeline...")
    
    if not os.path.exists("dataset/train_pushup_X.npy"):
        print("⚠️ Error: Run prepare_fit3d_data.py first to generate datasets!")
        return
        
    # 1. Load extracted sequences
    X_full = np.load("dataset/train_pushup_X.npy")
    y_full = np.load("dataset/train_pushup_y.npy")
    
    # 2. Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    train_counts = np.bincount(y_train, minlength=2)
    test_counts = np.bincount(y_test, minlength=2)
    print(f"📊 Training Set: {len(X_train)} samples ({train_counts[0]} Bad, {train_counts[1]} Good)")
    print(f"📊 Testing Set:  {len(X_test)} samples ({test_counts[0]} Bad, {test_counts[1]} Good)")

    # 3. Model & Balanced Class Weights
    model = STGCN(num_classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    total = len(y_train)
    class_weight = {
        0: (1 / train_counts[0]) * (total / 2.0),
        1: (1 / train_counts[1]) * (total / 2.0)
    }

    # 4. Training Loop
    epochs = 40
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=8,
        epochs=epochs,
        class_weight=class_weight
    )

    # 5. Direct Native TFLite Export (Zero bridges required!)
    os.makedirs("models/tflite", exist_ok=True)
    tflite_path = "models/tflite/st_gcn_pushup.tflite"
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"🎉 SUCCESS! Native TFLite model generated directly -> {tflite_path}")

    # 6. Final Evaluation Metrics
    print("\n" + "="*40)
    print("📈 FINAL MODEL EVALUATION METRICS")
    print("="*40)
    
    y_pred_probs = model.predict(X_test)
    preds = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print("\nConfusion Matrix (2x2):")
    print(cm)
    print(f"\nTrue Negatives  (Correctly flagged bad form): {cm[0][0]}")
    print(f"False Positives (Good form flagged as bad):   {cm[0][1]}")
    print(f"False Negatives (Bad form missed by AI):      {cm[1][0]}")
    print(f"True Positives  (Correctly flagged good form): {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds, labels=[0, 1], target_names=["Bad Form (0)", "Good Form (1)"], zero_division=0))

if __name__ == "__main__":
    train_model()