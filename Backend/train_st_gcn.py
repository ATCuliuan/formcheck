import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from st_gcn_model import STGCN
from path_config import STANDING_AB_TWISTS_X, STANDING_AB_TWISTS_Y, STANDING_AB_TWISTS_MODEL

def train_model():
    print("🚀 Booting Form-Tech ST-GCN Training & Validation Pipeline...")
    
    if not os.path.exists(STANDING_AB_TWISTS_X):
        print("⚠️ Error: Run prepare_fit3d_data.py first to generate datasets!")
        return
        
    # 1. Load extracted sequences
    X_full = np.load(STANDING_AB_TWISTS_X)
    y_full = np.load(STANDING_AB_TWISTS_Y)
    
    # 2. Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    train_counts = np.bincount(y_train, minlength=2)
    test_counts = np.bincount(y_test, minlength=2)
    print(f"📊 Training Set: {len(X_train)} samples ({train_counts[0]} Bad, {train_counts[1]} Good)")
    print(f"📊 Testing Set:  {len(X_test)} samples ({test_counts[0]} Bad, {test_counts[1]} Good)")

    # PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 3. Model & Balanced Loss Function
    class_weights = torch.tensor([len(y_train) / (2.0 * c) for c in train_counts], dtype=torch.float32)
    model = STGCN(num_classes=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_correct += (torch.argmax(outputs, 1) == batch_y).sum().item()
        
        train_acc = 100 * train_correct / len(y_train)

        # Validation Phase
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                test_correct += (torch.argmax(outputs, 1) == batch_y).sum().item()
        
        test_acc = 100 * test_correct / len(y_test)
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 5. Save Model
    try:
        torch.save(model.state_dict(), STANDING_AB_TWISTS_MODEL)
        print(f"✅ Training Complete! Model saved to {STANDING_AB_TWISTS_MODEL}")
    except Exception as e:
        print(f"❌ ERROR saving model: {e}")

    # 6. Evaluation Metrics
    print("\n" + "="*40)
    print("📈 FINAL MODEL EVALUATION METRICS")
    print("="*40)
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("\nConfusion Matrix (2x2):")
    print(cm)
    print(f"\nTrue Negatives  (Correctly flagged bad form): {cm[0][0]}")
    print(f"False Positives (Good form flagged as bad):   {cm[0][1]}")
    print(f"False Negatives (Bad form missed by AI):      {cm[1][0]}")
    print(f"True Positives  (Correctly flagged good form): {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=["Bad Form (0)", "Good Form (1)"], zero_division=0))

if __name__ == "__main__":
    train_model()