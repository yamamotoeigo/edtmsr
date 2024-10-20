import edtmsr
import srtdn
import torch
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import time

# 設定パラメータ
config = {
    'dataset_names': {'high': 'high_TM', 'low': 'low_TM'},
    'filenames': {'high': 'data/high_matrices.h5', 'low': 'data/low_matrices.h5'},
    'normalize': False,
    'standardize': False,
    'log_normalize': False,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'model_type': 'edtmsr',  # 'edtmsr' or 'srtdn'
    'num_clients': 2,
    'num_rounds': 50,
    # 'output_model_filename': 'model/edtmsr_model.pth',
    'results_filename': 'results/edtmsr_fl_2clients_results.txt'
}

# CUDAの確認と使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ読み込み関数
def load_tensors_h5py(dataset_name, filename):
    with h5py.File(filename, 'r') as f:
        tensor = torch.tensor(f[dataset_name][:])
    return tensor

# データ前処理関数
def preprocess_data(high_res_matrix, low_res_matrix, config):
    if config['normalize']:
        high_res_matrix = normalize_tensor(high_res_matrix)
        low_res_matrix = normalize_tensor(low_res_matrix)

    if config['standardize']:
        high_res_matrix = standardize_tensor(high_res_matrix)
        low_res_matrix = standardize_tensor(low_res_matrix)

    if config['log_normalize']:
        high_res_matrix = log_normalize_tensor(high_res_matrix)
        low_res_matrix = log_normalize_tensor(low_res_matrix)
    
    return high_res_matrix, low_res_matrix

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def standardize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def log_normalize_tensor(tensor):
    return torch.log(tensor + 1.0)

# モデル初期化関数
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# 平均化関数
def average_weights(weight_list):
    avg_weights = weight_list[0].copy()
    for key in avg_weights.keys():
        for i in range(1, len(weight_list)):
            avg_weights[key] += weight_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weight_list)
    return avg_weights

# モデル訓練関数
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# メイン関数
def main(config):
    # データ読み込み
    high_res_matrix = load_tensors_h5py(config['dataset_names']['high'], config['filenames']['high'])
    low_res_matrix = load_tensors_h5py(config['dataset_names']['low'], config['filenames']['low'])

    # データ前処理
    high_res_matrix, low_res_matrix = preprocess_data(high_res_matrix, low_res_matrix, config)

    # CUDAの確認と使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの作成
    if config['model_type'] == 'edtmsr':
        global_model = edtmsr.EDTMSR(scale_factor=2).to(device)
    elif config['model_type'] == 'srtdn':
        global_model = srtdn.SRDenseNet(scale_factor=2).to(device)
    else:
        raise ValueError("Invalid model type specified in config.")

    initialize_weights(global_model)

    # データセットの作成と分割
    # dataset = TensorDataset(low_res_matrix.unsqueeze(1), high_res_matrix.unsqueeze(1))
    # dataset_size = len(dataset)
    # data_per_client = dataset_size // config['num_clients']
    # client_datasets = random_split(dataset, [data_per_client] * config['num_clients'])

    # データセットの作成と分割
    dataset = TensorDataset(low_res_matrix.unsqueeze(1), high_res_matrix.unsqueeze(1))
    dataset_size = len(dataset)
    data_per_client = dataset_size // config['num_clients']
    remaining_data = dataset_size % config['num_clients']
    split_lengths = [data_per_client] * config['num_clients']
    for i in range(remaining_data):
        split_lengths[i] += 1
    client_datasets = random_split(dataset, split_lengths)
    
    # データローダーの作成
    client_loaders = [DataLoader(client_data, batch_size=config['batch_size'], shuffle=True) for client_data in client_datasets]
    test_loader = DataLoader(client_datasets[-1], batch_size=config['batch_size'], shuffle=False)  # 最後のクライアントをテストデータとして使用

    # 損失関数と最適化手法の定義
    criterion = nn.L1Loss()
    optimizer = optim.Adam(global_model.parameters(), lr=config['learning_rate'])

    start = time.time()

    # 連合学習ラウンドの実行
    slowest_time = []
    round_losses = []
    for round in tqdm(range(config['num_rounds']), desc="Federated Learning Rounds"):
        local_weights = []
        max_time = 0
        for client_loader in tqdm(client_loaders, desc="Clients", leave=False):
            start_time = time.time()
            if config['model_type'] == 'edtmsr':
                local_model = edtmsr.EDTMSR(scale_factor=2).to(device)
            elif config['model_type'] == 'srtdn':
                local_model = srtdn.SRDenseNet(scale_factor=2).to(device)
            
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=config['learning_rate'])
            
            train_model(local_model, client_loader, criterion, optimizer, num_epochs=config['num_epochs'])
            local_weights.append(local_model.state_dict())
            
            temp_time = time.time() - start_time
            max_time = max(max_time, temp_time)
        slowest_time.append(max_time)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        # 各ラウンド終了後のグローバルモデルの損失を計算
        global_model.eval()
        round_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = global_model(inputs)
                loss = criterion(outputs, targets)
                round_loss += loss.item() * inputs.size(0)
        round_loss /= len(test_loader.dataset)
        round_losses.append(round_loss)

    # グローバルモデルのテスト
    global_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = global_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)

    end = time.time()

    # モデルの保存
    torch.save(global_model.state_dict(), config['output_model_filename'])
    print("Model has been saved successfully.")
    print(f"Training and evaluation took {end-start:.2f} seconds.")
    print(f"Test Loss: {test_loss:.4f}")

    # テスト結果の保存
    test_inputs, test_outputs, test_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = global_model(inputs)
            test_inputs.append(inputs)
            test_outputs.append(outputs)
            test_targets.append(targets)

    test_inputs = torch.cat(test_inputs)
    test_outputs = torch.cat(test_outputs)
    test_targets = torch.cat(test_targets)

    # モデルのパラメータ数を表示
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    torch.save(test_inputs, 'test_inputs.pt')
    torch.save(test_outputs, 'test_outputs.pt')
    torch.save(test_targets, 'test_targets.pt')
    print("Test results have been saved successfully.")

    # 実験結果をファイルに保存
    with open(config['results_filename'], 'w') as f:
        f.write(f"Num clients: {config['num_clients']}\n")
        f.write(f"Training and evaluation took {end-start:.2f} seconds.\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Number of parameters in the model: {num_params}\n")
        f.write("Round losses:\n")
        for i in range(config['num_rounds']):
            f.write(f"Round {i+1}: {round_losses[i]:.4f}\n")
        # configの内容を保存
        f.write("\nConfig:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print("Experiment results have been saved.")

if __name__ == "__main__":
    main(config)
