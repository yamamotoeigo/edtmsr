import torch.nn.functional as F
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

config = {
    'dataset_names': {'high': 'high_TM', 'low': 'low_TM'},
    'filenames': {'high': 'data/high_matrices.h5', 'low': 'data/low_matrices.h5'},
    'normalize': False,
    'standardize': False,
    'log_normalize': False,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'model_type': 'srtdn',  # 'edtmsr' or 'srtdn'
    'output_model_filename': 'model/srtdn_model.pth',
    'results_filename': 'results/srtdn_2result.txt',
    'seed': 42  # ランダムシードを追加
}

# Check if CUDA is available and use GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ランダムシードを設定
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config['seed'])

# h5ファイルからテンソルを読み込む関数
def load_tensors_h5py(dataset_name, filename):
    with h5py.File(filename, 'r') as f:
        tensor = torch.tensor(f[dataset_name][:])
    return tensor

# バイキュービック補間による画像復元関数
def bicubic_interpolation(low_res_tensor, scale_factor=2):
    return F.interpolate(low_res_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=True)

# h5ファイルを読み込む
high_res_matrix = load_tensors_h5py(config['dataset_names']['high'], config['filenames']['high'])
low_res_matrix = load_tensors_h5py(config['dataset_names']['low'], config['filenames']['low'])

# データセットを作成
dataset = TensorDataset(low_res_matrix.unsqueeze(1), high_res_matrix.unsqueeze(1))
dataset_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# バイキュービック法による全データの復元とMAE計算
def evaluate_bicubic_all_data(dataset_loader, scale_factor=2):
    total_loss = 0.0
    total_count = 0
    for inputs, targets in dataset_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = bicubic_interpolation(inputs, scale_factor)
        loss = nn.L1Loss()(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_count += inputs.size(0)
    average_loss = total_loss / total_count
    return average_loss

# バイキュービック法による復元の評価
bicubic_loss = evaluate_bicubic_all_data(dataset_loader)
print(f"Bicubic Interpolation MAE (All Data): {bicubic_loss:.4f}")

# 実験結果をファイルに追加保存
with open('results/bicubic.txt', 'a') as f:
    f.write(f"Bicubic Interpolation MAE (All Data): {bicubic_loss:.4f}\n")
