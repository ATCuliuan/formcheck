import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from st_gcn_model import STGCN

def train_model():
    print("🚀 Booting Form-Tech ST-GCN Training & Validation Pipeline...")
    
    if not os.path.exists("dataset/train_X.npy"):
        print("⚠️ Error: Run prepare_fit3d_data.py first to generate datasets!")
        return
        
    # 1. Load the 40 extracted sequences
    X_full = np.load("dataset/train_X.npy")
    y_full = np.load("dataset/train_y.npy")
    
    # 2. Perform an 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )
    
    print(f"📊 Dataset Split: {len(X_train)} Training sequences | {len(X_test)} Testing sequences")

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 3. Initialize Model
    model = STGCN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training & Validation Loop
    epochs = 20
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
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

        # --- TESTING PHASE (Unseen Data) ---
        model.eval()
        test_correct = 0
        with torch.no_grad(): # Turn off learning so it doesn't cheat
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                test_correct += (torch.argmax(outputs, 1) == batch_y).sum().item()
        
        # Prevent division by zero if test set is too small
        test_acc = (100 * test_correct / len(y_test)) if len(y_test) > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 5. Save the weights for Form-Tech integration
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/st_gcn_squat.pt")
    print("✅ Training & Validation Complete! Engine saved to models/st_gcn_squat.pt")

if __name__ == "__main__":
    train_model()