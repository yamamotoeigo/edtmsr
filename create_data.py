import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

topology_file_path = '/home/yamamoto/workspace/research/edtmsr/origin_data/topology-anonymised.xml'
traffic_file_path = '/home/yamamoto/workspace/research/edtmsr/origin_data/traffic-matrices/IntraTM-2005-01-01-00-30.xml'

# データセットで使用されているノードの隣接行列を作成
def parse_topology(file_path) -> np.ndarray:
    """
    XMLトポロジファイルを解析し、ソートされた隣接行列をNumPy配列として返します。
    
    Args:
    - file_path (str): XMLトポロジファイルのパス。
    
    Returns:
    - adj_matrix_sorted_array (np.ndarray): ソートされた隣接行列のNumPy配列。
    """
    # XMLファイルを解析
    tree = ET.parse(file_path)
    root = tree.getroot()

    # ノードIDをインデックスにマッピングする辞書を作成
    nodes = {}
    index = 0

    # ノードIDをインデックスにマッピング
    for node in root.findall('.//node'):
        node_id = node.get('id')
        if node_id not in nodes:
            nodes[node_id] = index
            index += 1

    # 隣接行列を作成
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # リンクで隣接行列を埋める
    for link in root.findall('.//link'):
        source_element = link.find('./from')
        target_element = link.find('./to')
        if source_element is not None and target_element is not None:
            source = source_element.get('node')
            target = target_element.get('node')
            if source in nodes and target in nodes:
                adj_matrix[nodes[source], nodes[target]] = 1
                adj_matrix[nodes[target], nodes[source]] = 1  # 無向リンクを想定

    # ノードIDを数値順にソートし、隣接行列を再配置
    sorted_node_ids = sorted(nodes.keys(), key=int)
    sorted_indices = [nodes[node_id] for node_id in sorted_node_ids]
    adj_matrix_sorted = adj_matrix[np.ix_(sorted_indices, sorted_indices)]

    # ソートされた隣接行列をDataFrameに変換
    adj_matrix_sorted_df = pd.DataFrame(adj_matrix_sorted, index=sorted_node_ids, columns=sorted_node_ids)

    # DataFrameをNumPy配列に変換
    adj_matrix_sorted_array = adj_matrix_sorted_df.values
    return adj_matrix_sorted_array

# トラフィックデータからトラフィックマトリックス(tensor)を作成
def parse_traffic_matrix(file_path) -> torch.Tensor:
    """
    XMLトラフィックマトリックスファイルを解析し、トラフィックマトリックスをNumPy配列およびテンソルとして返します。
    
    Args:
    - file_path (str): XMLトラフィックマトリックスファイルのパス。
    
    Returns:
    - traffic_matrix_tensor (torch.Tensor): ソートされたトラフィックマトリックスのテンソル。
    """
    # XMLファイルを解析
    tree = ET.parse(file_path)
    root = tree.getroot()

    # トラフィックデータを保持するリストを初期化
    traffic_data = []

    # 構造に基づいてトラフィックデータを抽出
    for src_node in root.findall('.//src'):
        source = src_node.get('id')
        for dst_node in src_node.findall('dst'):
            destination = dst_node.get('id')
            traffic = float(dst_node.text)
            traffic_data.append([source, destination, traffic])

    # トラフィックデータをpandas DataFrameに変換
    traffic_df = pd.DataFrame(traffic_data, columns=['Source', 'Destination', 'Traffic'])

    # 一意のソースとデスティネーションを取得
    sources = set(traffic_df['Source'])
    destinations = set(traffic_df['Destination'])
    all_nodes = sorted(sources.union(destinations), key=int)

    # ピボットテーブルを作成してトラフィックマトリックスを表現
    traffic_matrix = traffic_df.pivot(index='Source', columns='Destination', values='Traffic').reindex(index=all_nodes, columns=all_nodes).fillna(0)

    # ソートされたトラフィックマトリックスを表示
    traffic_matrix_sorted = traffic_matrix.loc[all_nodes, all_nodes]

    # DataFrameをNumPy配列に変換
    traffic_matrix_sorted_array = traffic_matrix_sorted.values

    # NumPy配列をテンソルに変換
    traffic_matrix_tensor = torch.tensor(traffic_matrix_sorted_array, dtype=torch.float32)

    return traffic_matrix_tensor

# スライディングウィンドウを適用してサブマトリックスを生成
def sliding_window(tensor, window_size=18, step_size=1):
    """
    スライディングウィンドウを適用してサブマトリックスを生成する関数。
    
    Args:
    - tensor (torch.Tensor): 入力テンソル (2D)。
    - window_size (int): ウィンドウのサイズ。
    - step_size (int): ステップサイズ。
    
    Returns:
    - torch.Tensor: 生成されたサブマトリックスのテンソル。
    """
    if tensor.dim() != 2:
        raise ValueError("入力テンソルは2次元である必要があります (H, W)")

    rows, cols = tensor.shape
    windows = []

    for i in range(0, rows - window_size + 1, step_size):
        for j in range(0, cols - window_size + 1, step_size):
            window = tensor[i:i + window_size, j:j + window_size]
            windows.append(window)

    return torch.stack(windows)

# 2次元テンソルに対してバイキュービック補間を行う関数
def bicubic_interpolation_2d(tensor, output_size):
    """
    2次元テンソルに対してバイキュービック補間を行う関数。
    
    Args:
    - tensor (torch.Tensor): 入力テンソル (2D)。
    - output_size (tuple): 補間後のサイズ (高さ, 幅)。
    
    Returns:
    - torch.Tensor: 補間後のテンソル。
    """
    if tensor.dim() != 2:
        raise ValueError("入力テンソルは2次元である必要があります (H, W)")

    tensor_3d = tensor.unsqueeze(0).unsqueeze(0)  # バッチ次元とチャネル次元を追加
    interpolated_tensor_3d = F.interpolate(tensor_3d, size=output_size, mode='bicubic')
    interpolated_tensor_2d = interpolated_tensor_3d.squeeze(0).squeeze(0)  # 追加した次元を削除

    return interpolated_tensor_2d

# ヒートマップを生成して保存する関数
def save_heatmap(tensor, filename):
    filename= '/home/yamamoto/workspace/research/edtmsr/traffic_heatmap/' + filename
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor.numpy(), cmap='viridis', annot=False, fmt='g')
    plt.title('Traffic Matrix Heatmap')
    plt.xlabel('Destination')
    plt.ylabel('Source')
    plt.savefig(filename)
    plt.close()

    
# 隣接行列を取得
adj_matrix_sorted_array = parse_topology(topology_file_path)

# トラフィックデータを取得
traffic_matrix_tensor = parse_traffic_matrix(traffic_file_path)
print(traffic_matrix_tensor.size())

sub_matrices = sliding_window(traffic_matrix_tensor)
print(f'sub_matrices size: {sub_matrices.size()}')
print(f'sub_matrices[0] size{sub_matrices[0].size()}')

# バイキュービック補間を行って低解像度のマトリックスを得る
low_res_matrix = bicubic_interpolation_2d(sub_matrices[0], output_size=(9, 9))
print(f'low_res_matrix size: {low_res_matrix.size()}')


# 高解像度のトラフィックマトリックスのヒートマップを保存
save_heatmap(sub_matrices[0], 'high_res_heatmap.png')
# 低解像度のトラフィックマトリックスのヒートマップを保存
save_heatmap(low_res_matrix, 'low_res_heatmap.png')