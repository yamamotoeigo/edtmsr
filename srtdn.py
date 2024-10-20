# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np

# # Dense Block
# class DenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate=8, num_layers=6): #rate=8, num_layers=4
#         super(DenseBlock, self).__init__()
#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True)
#             ) for i in range(num_layers)
#         ])

#     def forward(self, x):
#         for layer in self.layers:
#             out = layer(x)
#             x = torch.cat([x, out], dim=1)
#         return x

# # Bottleneck Layer
# class BottleneckLayer(nn.Module):
#     def __init__(self, in_channels, out_channels=256):
#         super(BottleneckLayer, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.conv(x))

# # Define the SRDenseNet model
# class SRDenseNet(nn.Module):
#     def __init__(self, growth_rate=8, num_dense_blocks=4, scale_factor=2): # blocks=4
#         super(SRDenseNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
#         # Create Dense Blocks
#         self.dense_blocks = nn.ModuleList([
#             DenseBlock(64 + i * growth_rate * num_dense_blocks, growth_rate, num_dense_blocks) for i in range(num_dense_blocks)
#         ])
        
#         # Bottleneck layer to reduce the number of feature maps
#         # self.bottleneck = BottleneckLayer(64 + num_dense_blocks * growth_rate * num_dense_blocks, 256)
#         self.bottleneck = BottleneckLayer(64 + num_dense_blocks * growth_rate * num_dense_blocks, 64)
        
#         # Deconvolution layers for upsampling
#         self.upsample = nn.Sequential(
#             # nn.Conv2d(256, 256 * scale_factor**2, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(64, 64 * scale_factor**2, kernel_size=3, stride=1, padding=1),
#             nn.PixelShuffle(upscale_factor=scale_factor),
#             nn.ReLU(inplace=True)
#         )
        
#         # Reconstruction layer
#         # self.conv2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.conv1(x)
        
#         # Apply Dense Blocks
#         for block in self.dense_blocks:
#             x = block(x)
        
#         x = self.bottleneck(x)
#         x = self.upsample(x)
#         x = self.conv2(x)
        
#         return x

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Dense Block with Depthwise Separable Convolutions
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=4, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels + i * growth_rate, in_channels + i * growth_rate, kernel_size=3, padding=1, groups=in_channels + i * growth_rate),
                # Pointwise Convolution
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=1),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

# Bottleneck Layer
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(BottleneckLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

# Define the SRDenseNet model with Depthwise Separable Convolutions
class SRDenseNet(nn.Module):
    def __init__(self, growth_rate=4, num_dense_blocks=3, scale_factor=2):
        super(SRDenseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # First convolution layer with 32 output channels
        
        # Create Dense Blocks
        self.dense_blocks = nn.ModuleList([
            DenseBlock(32 + i * growth_rate * num_dense_blocks, growth_rate, num_dense_blocks) for i in range(num_dense_blocks)
        ])
        
        # Bottleneck layer to reduce the number of feature maps
        self.bottleneck = BottleneckLayer(32 + num_dense_blocks * growth_rate * num_dense_blocks, 64)
        
        # Deconvolution layers for upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction layer
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        
        # Apply Dense Blocks
        for block in self.dense_blocks:
            x = block(x)
        
        x = self.bottleneck(x)
        x = self.upsample(x)
        x = self.conv2(x)
        
        return x