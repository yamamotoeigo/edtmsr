import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
from tqdm import tqdm

"""
画像の超解像のためのニューラルネットワークの実装
`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf) に記載された
（単一スケールのベースラインスタイルのモデル）
"""

class EDSR(nn.Module):
    def __init__(self, img_size=32, num_layers=32, feature_size=256, scale=2, output_channels=3):
        super(EDSR, self).__init__()
        print("EDSRの構築...")
        self.img_size = img_size
        self.scale = scale
        self.output_channels = output_channels

        # 特徴マップの深さを変換するための最初の畳み込み層
        self.conv1 = nn.Conv2d(output_channels, feature_size, kernel_size=3, padding=1)

        # 残差ブロックのリスト
        self.res_blocks = nn.ModuleList([ResBlock(feature_size) for _ in range(num_layers)])

        # 最後の畳み込み層
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        
        # アップスケーリング用の転置畳み込み層
        self.upscale = nn.ConvTranspose2d(feature_size, output_channels, kernel_size=3, stride=scale, padding=1, output_padding=scale-1)
    
    def forward(self, x):
        mean_x = 127
        image_input = x - mean_x

        x = self.conv1(image_input)
        conv_1 = x

        for block in self.res_blocks:
            x = block(x)

        x = self.conv2(x)
        x += conv_1

        x = self.upscale(x)
        mean_y = 127
        x += mean_y

        return x

class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.scale = 0.1  # 論文に記載のスケーリング

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual * self.scale

def train(model, data_loader, test_loader=None, iterations=1000, save_dir="saved_models"):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    writer = SummaryWriter(save_dir + "/train")
    if test_loader is not None:
        test_writer = SummaryWriter(save_dir + "/test")
    
    model.train()
    for epoch in tqdm(range(iterations)):
        for i, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + i)
        
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                test_loss /= len(test_loader)
                test_writer.add_scalar('Loss/test', test_loss, epoch)
            model.train()
    
    torch.save(model.state_dict(), os.path.join(save_dir, "edsr.pth"))
    writer.close()
    if test_loader is not None:
        test_writer.close()

# 使用例
model = EDSR()
print(model, '\n')
# train(model, train_loader, test_loader, iterations=1000)
