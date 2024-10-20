import argparse
import edtmsr
import srtdn
import torch
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import os
import webhook

# コマンドライン引数を解析する関数
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning for Image Super-Resolution")
    parser.add_argument('--high_res_file', type=str, default='data/high_matrices.h5', help="Path to high resolution h5 file")
    parser.add_argument('--low_res_file', type=str, default='data/low_matrices.h5', help="Path to low resolution h5 file")
    parser.add_argument('--model_type', type=str, required=True, choices=['edtmsr', 'srtdn'], help="Model type to use")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs per round")
    parser.add_argument('--normalize', action='store_true', help="Normalize the data")
    parser.add_argument('--standardize', action='store_true', help="Standardize the data")
    parser.add_argument('--log_normalize', action='store_true', help="Log normalize the data")
    parser.add_argument('--output_model_filename', type=str, default=None, help="Path to save the output model")
    parser.add_argument('--results_filename', type=str, default=None, help="Path to save the results")
    return parser.parse_args()


# CUDAの確認と使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# h5ファイルからテンソルを読み込む関数
def load_tensors_h5py(dataset_name, filename):
    with h5py.File(filename, 'r') as f:
        tensor = torch.tensor(f[dataset_name][:])
    return tensor

# PSNRを計算する関数
def calculate_psnr(output, target):
    mse = nn.functional.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# モデルをトレーニングする関数
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
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
        train_losses.append(epoch_loss)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)
        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {valid_loss:.4f}")
    return train_losses, valid_losses

# 正規化と標準化の関数
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

def main(args):
    # h5ファイルを読み込む
    high_res_matrix = load_tensors_h5py('high_TM', args.high_res_file)
    low_res_matrix = load_tensors_h5py('low_TM', args.low_res_file)

    # データの前処理
    if args.normalize:
        high_res_matrix = normalize_tensor(high_res_matrix)
        low_res_matrix = normalize_tensor(low_res_matrix)

    if args.standardize:
        high_res_matrix = standardize_tensor(high_res_matrix)
        low_res_matrix = standardize_tensor(low_res_matrix)

    if args.log_normalize:
        high_res_matrix = log_normalize_tensor(high_res_matrix)
        low_res_matrix = log_normalize_tensor(low_res_matrix)

    print(high_res_matrix.size())
    print(low_res_matrix.size())

    # Check if CUDA is available and use GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    if args.model_type == 'edtmsr':
        model = edtmsr.EDTMSR(scale_factor=2).to(device)
    elif args.model_type == 'srtdn':
        model = srtdn.SRDenseNet(scale_factor=2).to(device)
    else:
        raise ValueError("Invalid model type specified in args.")

    initialize_weights(model)

    # データセットを作成
    dataset = TensorDataset(low_res_matrix.unsqueeze(1), high_res_matrix.unsqueeze(1))
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    valid_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start = time.time()
    # Train the model
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, criterion, optimizer, args.num_epochs)

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    test_psnr = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            psnr = calculate_psnr(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            test_psnr += psnr.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    test_psnr /= len(test_loader.dataset)

    end = time.time()

    # Save the model
    if args.output_model_filename is None:
        args.output_model_filename = f'model/{args.model_type}_model.pth'
    torch.save(model.state_dict(), args.output_model_filename)
    print("Model has been saved successfully.")

    print(f"Training and evaluation took {end-start:.2f} seconds.")
    print(f"Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}")

    # Save the test results
    test_inputs = []
    test_outputs = []
    test_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_inputs.append(inputs)
            test_outputs.append(outputs)
            test_targets.append(targets)

    test_inputs = torch.cat(test_inputs)
    test_outputs = torch.cat(test_outputs)
    test_targets = torch.cat(test_targets)

    torch.save(test_inputs, 'test_inputs.pt')
    torch.save(test_outputs, 'test_outputs.pt')
    torch.save(test_targets, 'test_targets.pt')

    print("Test results have been saved successfully.")

    # モデルのパラメータ数を表示
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # resultsディレクトリを作成
    if args.results_filename is None:
        results_dir = os.path.join('results', args.model_type)
        os.makedirs(results_dir, exist_ok=True)
        args.results_filename = os.path.join(results_dir, f'{args.model_type}_result.txt')

    # 実験結果をファイルに保存
    with open(args.results_filename, 'w') as f:
        f.write(f"Training and evaluation took {end-start:.2f} seconds.\n")
        f.write(f"Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}\n")
        f.write(f"Number of parameters in the model: {num_params}\n")
        f.write("Train Losses per epoch:\n")
        for i, loss in enumerate(train_losses):
            f.write(f"Epoch {i+1}: {loss:.4f}\n")
        f.write("Validation Losses per epoch:\n")
        for i, loss in enumerate(valid_losses):
            f.write(f"Epoch {i+1}: {loss:.4f}\n")
        f.write("Config parameters:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print("Experiment results have been saved")
    webhook.send_webhook(args.results_filename)

if __name__ == "__main__":
    args = parse_args()
    main(args)
