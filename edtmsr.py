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
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor)
        )
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return x

# Check if CUDA is available and use GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to the device
model = EDTMSR().to(device)

# Create high resolution traffic matrix
high_res_size = (18, 18)
low_res_size = (9, 9)
scale_factor = 2
num_samples = 1000

# Generate random high resolution traffic matrix and move to device
high_res_matrix = torch.rand(num_samples, 1, *high_res_size).to(device)

# Downsample to create low resolution traffic matrix using bicubic interpolation and move to device
low_res_matrix = F.interpolate(high_res_matrix, size=low_res_size, mode='bicubic')

# Create DataLoader for training
train_dataset = TensorDataset(low_res_matrix, high_res_matrix)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Generate random high resolution traffic matrix and move to device
high_res_matrix = torch.rand(1, 1, *high_res_size).to(device)

# Downsample to create low resolution traffic matrix using bicubic interpolation and move to device
low_res_matrix = F.interpolate(high_res_matrix, size=low_res_size, mode='bicubic')

# Test the model with a sample input
model.eval()
with torch.no_grad():
    sample_input = low_res_matrix[0:1]
    output = model(sample_input)

print(f"Input (Low Resolution) size: {sample_input.size()}")
print(f"Output (High Resolution) size: {output.size()}")
print(f"Expected High Resolution size: {high_res_matrix[0:1].size()}")

# Move data back to CPU and save to CSV files
high_res_matrix_cpu = high_res_matrix[0, 0].cpu().numpy()
low_res_matrix_cpu = low_res_matrix[0, 0].cpu().numpy()
output_cpu = output[0, 0].cpu().numpy()

# np.savetxt("high_res_matrix.csv", high_res_matrix_cpu, delimiter=",")
# np.savetxt("low_res_matrix.csv", low_res_matrix_cpu, delimiter=",")
# np.savetxt("output.csv", output_cpu, delimiter=",")
