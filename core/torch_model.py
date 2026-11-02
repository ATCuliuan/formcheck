import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.register_buffer("adj", torch.zeros(25, 25))
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            padding=(4, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.spatial(x)
        x = torch.einsum("nctv,vw->nctw", x, self.adj)
        x = self.temporal(x)
        return F.relu(self.bn(x))


class _STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = _STGCNLayer(3, 64)
        self.l2 = _STGCNLayer(64, 64)
        self.l3 = _STGCNLayer(64, 128)
        self.l4 = _STGCNLayer(128, 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.fc(x.mean(dim=(2, 3)))


class TorchSTGCN:
    def __init__(self, state_dict, device):
        self.device = device
        self.model = _STGCN().to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, sequence):
        inputs = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(inputs).cpu().numpy()


def load_st_gcn_model(model_path):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        warnings.warn(f"PyTorch exercise model not found at {model_path}", RuntimeWarning)
        return None

    return TorchSTGCN(state_dict, device)
