import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_spatial_graph(num_node=25):
    """
    Constructs the physical adjacency matrix for 25 body joints.
    Maps the spine, arms, and legs so the AI understands human anatomy.
    """
    A = np.zeros((num_node, num_node))
    
    # Standard 25-Joint Skeletal Connections (Base -> Extremities)
    # Spine & Head
    edges = [(0, 1), (1, 20), (20, 2), (2, 3),
    # Left Arm
             (20, 4), (4, 5), (5, 6), (6, 7), (6, 21), (7, 22),
    # Right Arm
             (20, 8), (8, 9), (9, 10), (10, 11), (10, 23), (11, 24),
    # Left Leg
             (0, 12), (12, 13), (13, 14), (14, 15),
    # Right Leg
             (0, 16), (16, 17), (17, 18), (18, 19)]
             
    # Create bidirectional connections
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
        
    # Add self-loops (a joint is connected to itself)
    for i in range(num_node):
        A[i, i] = 1
        
    # Normalize the matrix to prevent weight explosion
    rowsum = A.sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    D = np.diag(r_inv)
    A_normalized = D.dot(A)
    
    return torch.tensor(A_normalized, dtype=torch.float32)

class GraphConvolution(nn.Module):
    """Spatial convolution treating the 25 joints as a physical graph."""
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
        # Load the physical human skeleton map
        self.A = nn.Parameter(get_spatial_graph(25), requires_grad=False)

    def forward(self, x):
        x = self.conv(x)
        # Multiply by adjacency matrix to share data across connected bones
        x = torch.einsum('nctw,vw->nctv', x, self.A)
        return x

class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for Form-Tech."""
    def __init__(self, num_classes=2):
        super(STGCN, self).__init__()
        
        self.spatial_gcn1 = GraphConvolution(3, 32)
        self.temporal_conv1 = nn.Conv2d(32, 64, kernel_size=(9, 1), padding=(4, 0))
        
        self.spatial_gcn2 = GraphConvolution(64, 64)
        self.temporal_conv2 = nn.Conv2d(64, 128, kernel_size=(9, 1), padding=(4, 0))
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.spatial_gcn1(x))
        x = F.relu(self.temporal_conv1(x))
        
        x = F.relu(self.spatial_gcn2(x))
        x = F.relu(self.temporal_conv2(x))
        
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        out = self.fc(x)
        return out