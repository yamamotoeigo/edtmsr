import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the ResidualBlock as per EDTMSR architecture
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.scale = 0.1

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual * self.scale

# Define the EDTMSR model
class EDTMSR(nn.Module):
    def __init__(self, num_residual_blocks=8, scale_factor=2):
        super(EDTMSR, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor)
        )
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        x = self.conv2(x)
        # x = torch.clamp(x, min=0.0)
        return x