import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """Spatial convolution treating the 25 joints as a physical graph."""
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
        # A basic adjacency matrix for 25 joints (identity graph for now)
        # We will map the exact bone connections later to increase accuracy.
        self.A = nn.Parameter(torch.eye(25) + torch.randn(25, 25) * 0.01, requires_grad=True)

    def forward(self, x):
        # x shape: (Batch, Channels, Temporal, Nodes)
        x = self.conv(x)
        # Multiply by adjacency matrix to share data across connected bones
        x = torch.einsum('nctw,vw->nctv', x, self.A)
        return x

class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for Form-Tech."""
    def __init__(self, num_classes=2):
        super(STGCN, self).__init__()
        
        # 3 input channels (x, y, z) -> 32 temporal features
        self.spatial_gcn1 = GraphConvolution(3, 32)
        # Looks across 9 consecutive frames to understand motion over time
        self.temporal_conv1 = nn.Conv2d(32, 64, kernel_size=(9, 1), padding=(4, 0))
        
        self.spatial_gcn2 = GraphConvolution(64, 64)
        self.temporal_conv2 = nn.Conv2d(64, 128, kernel_size=(9, 1), padding=(4, 0))
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Spatial-Temporal Block 1
        x = F.relu(self.spatial_gcn1(x))
        x = F.relu(self.temporal_conv1(x))
        
        # Spatial-Temporal Block 2
        x = F.relu(self.spatial_gcn2(x))
        x = F.relu(self.temporal_conv2(x))
        
        # Global average pooling (compress the 30 frames and 25 joints into 1 decision)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        # Final classification (Good Form vs Bad Form)
        out = self.fc(x)
        return out