import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOCK THE SEED: Stops the random slot-machine fluctuation
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Change this to whatever exercise you are currently training
EXERCISE = "side_lateral_raise" 

# ==========================================
# PYTORCH ST-GCN ARCHITECTURE
# ==========================================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        # 9x1 Temporal convolution
        self.temporal = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        # Learnable Adjacency Matrix for the 25 joints
        self.adj = nn.Parameter(torch.ones(25, 25) / 25) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.spatial(x)
        # Apply the graph connection (Spatial)
        x = torch.einsum('nctv,vw->nctw', x, self.adj)
        # Apply time sequence (Temporal)
        x = self.temporal(x)
        x = self.bn(x)
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.l1 = STGCNBlock(3, 64)
        self.l2 = STGCNBlock(64, 64)
        self.l3 = STGCNBlock(64, 128, stride=2)
        self.l4 = STGCNBlock(128, 128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (Batch, Channels, Time, Vertices) -> (N, 3, 30, 25)
        N = x.size(0)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        
        # Global Average Pooling
        x = F.avg_pool2d(x, x.size()[2:]) 
        x = x.view(N, -1)
        return self.fc(x)

# ==========================================
# DATA SPLIT PIPELINE
# ==========================================
def twin_safe_split(X, y, test_size=0.2):
    """
    Groups indices into chunks of 6 (twins) and randomly shuffles the chunks 
    BEFORE splitting to prevent the Data Leak.
    """
    good_idx = np.where(y == 1)[0]
    bad_idx = np.where(y == 0)[0]

    def shuffle_in_chunks(indices, chunk_size=6):
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        np.random.shuffle(chunks)
        return np.concatenate(chunks) if len(chunks) > 0 else np.array([], dtype=int)

    good_idx_mixed = shuffle_in_chunks(good_idx)
    bad_idx_mixed = shuffle_in_chunks(bad_idx)

    good_split = int(len(good_idx_mixed) * (1 - test_size))
    bad_split = int(len(bad_idx_mixed) * (1 - test_size))

    train_idx = np.concatenate([good_idx_mixed[:good_split], bad_idx_mixed[:bad_split]])
    test_idx = np.concatenate([good_idx_mixed[good_split:], bad_idx_mixed[bad_split:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def train_model():
    print(f"Booting PyTorch Native Training Pipeline for: {EXERCISE.upper()}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

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
    
    # Keras is (B, T, V, C). PyTorch requires (B, C, T, V). 
    if X_full.ndim == 4 and X_full.shape[-1] == 3:
        print("Detected TensorFlow array format. Auto-transposing to (Batch, Channels, Time, Vertices)...")
        X_full = np.transpose(X_full, (0, 3, 1, 2))

    X_train, X_test, y_train, y_test = twin_safe_split(X_full, y_full, test_size=0.2)
    
    # Replicate Keras validation_split=0.2
    val_split_idx = int(len(X_train) * 0.8)
    X_train_sub, y_train_sub = X_train[:val_split_idx], y_train[:val_split_idx]
    X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]

    train_counts = np.bincount(y_train_sub, minlength=2)
    print(f"📊 Training Set: {len(X_train_sub)} samples ({train_counts[0]} Bad, {train_counts[1]} Good)")

    # Convert to PyTorch Tensors
    train_data = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), torch.tensor(y_train_sub, dtype=torch.long))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    model = STGCN(num_classes=2).to(device)
    
    # LOWER LEARNING RATE matches your TF script
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    total = len(y_train_sub)
    weight_0 = (1 / (train_counts[0] + 1e-6)) * (total / 2.0)
    weight_1 = (1 / (train_counts[1] + 1e-6)) * (total / 2.0)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Early Stopping Variables
    patience = 25
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_state = None

    epochs = 200
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        train_acc = train_correct.double() / len(train_data)
        
        # Validation Phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        val_acc = val_correct.double() / len(val_data)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss/len(train_data):.4f} - accuracy: {train_acc:.4f} - val_accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nRestoring model weights from the end of the best epoch: {epoch+1 - patience:03d}.")
                print(f"Epoch {epoch+1:05d}: early stopping")
                model.load_state_dict(best_model_state)
                break
                
    if best_model_state and epochs_no_improve < patience:
        model.load_state_dict(best_model_state)

    # ==========================================
    # EXPORT PIPELINE: PyTorch .pt File
    # ==========================================
    os.makedirs("models/pt", exist_ok=True)
    pt_path = f"models/pt/st_gcn_{EXERCISE}.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"\n✅ PyTorch native model exported -> {pt_path}")

    # ==========================================
    # FINAL SNAPSHOT EVALUATION METRICS
    # ==========================================
    print("\n" + "="*40)
    print(f"HONEST EVALUATION METRICS (Twin Leak Patched)")
    print("="*40)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("\nConfusion Matrix (2x2):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=["Bad Form (0)", "Good Form (1)"], zero_division=0))

if __name__ == "__main__":
    train_model()