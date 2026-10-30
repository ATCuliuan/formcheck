import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from ai_edge_litert.interpreter import Interpreter

# Exercises with verified trained models and .npy datasets
EXERCISES = [
    "squat",
    "pushup",
    "burpees",
    "mule_kick",
    "dumbbell_reverse_lunge",
    "dumbbell_overhead_shoulder_press",
    "dumbbell_hammer_curls",
    "band_pull_apart",
    "barbell_row",
    "deadlift",
    "dumbbell_biceps_curls",
    "one_arm_row",
    "overhead_trap_raises",
    "side_lateral_raise",
    "standing_ab_twists",
    "hurdle_step",
    "inline_lunge",
    "side_lunge",
    "standing_active_straight_leg_raise",
    "standing_shoulder_abduction",
    "standing_shoulder_extension",
    "standing_shoulder_scaption"
]

OUTPUT_DIR = "documentation/confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_and_plot(exercise):
    model_path = os.path.join("models", "tflite", f"st_gcn_{exercise}.tflite")
    data_x_path = os.path.join("dataset", f"train_{exercise}_X.npy")
    data_y_path = os.path.join("dataset", f"train_{exercise}_y.npy")

    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        return
    if not os.path.exists(data_x_path) or not os.path.exists(data_y_path):
        print(f"⚠️ Dataset missing for {exercise}")
        return

    # Load preprocessed arrays
    X = np.load(data_x_path)
    y = np.load(data_y_path)

    # Use the same split ratio used during training (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load TFLite Model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 1. Evaluate on Validation Data
    val_preds = []
    for sample in X_val:
        tensor_in = np.expand_dims(sample.astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], tensor_in)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        val_preds.append(np.argmax(output_data, axis=1)[0])

    # 2. Evaluate on Training Data to get Training Accuracy
    train_preds = []
    for sample in X_train:
        tensor_in = np.expand_dims(sample.astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], tensor_in)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        train_preds.append(np.argmax(output_data, axis=1)[0])

    # Calculate exact accuracy scores
    val_acc = accuracy_score(y_val, val_preds)
    train_acc = accuracy_score(y_train, train_preds)

    cm = confusion_matrix(y_val, val_preds)
    report = classification_report(y_val, val_preds, target_names=["Bad Form (0)", "Good Form (1)"])

    print(f"\n==================== {exercise.upper()} ====================")
    print(f"Training Accuracy:   {train_acc:.2%}")
    print(f"Validation Accuracy: {val_acc:.2%}")
    print("-" * 40)
    print(report)

    # Generate graphical Heatmap plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=["Bad Form", "Good Form"], 
        yticklabels=["Bad Form", "Good Form"],
        cbar=False
    )
    
    # Inject both metrics directly into the image title
    plot_title = f"{exercise.replace('_', ' ').title()}\nTrain Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}"
    plt.title(plot_title, fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()

    # Save figure as high-resolution image for documentation
    save_path = os.path.join(OUTPUT_DIR, f"{exercise}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Save classification metrics to text file
    with open(os.path.join(OUTPUT_DIR, f"{exercise}_report.txt"), "w") as f:
        f.write(f"Exercise: {exercise.replace('_', ' ').title()}\n")
        f.write(f"Training Accuracy:   {train_acc:.2%}\n")
        f.write(f"Validation Accuracy: {val_acc:.2%}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"✅ Saved plot to: {save_path}")

if __name__ == "__main__":
    for exercise in EXERCISES:
        evaluate_and_plot(exercise)