#!/usr/bin/env python
# coding: utf-8

# In[12]:


# pyファイルに変換
get_ipython().system('jupyter nbconvert --to script /home/yamamoto/workspace/research/edtmsr/data_visualize.ipynb')
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import pprint


# 本研究で使用するデータセットのトポロジを取得 \
# 隣接行列とグラフ構造で表現

# In[13]:


import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# Parse the XML file
tree = ET.parse('/home/yamamoto/workspace/research/edtmsr/origin_data/topology-anonymised.xml')
root = tree.getroot()

# Adjust the XML parsing and adjacency matrix generation with additional checks
nodes = {}
index = 0

# Create a mapping for node IDs to indices
for node in root.findall('.//node'):
    node_id = node.get('id')
    if node_id not in nodes:
        nodes[node_id] = index
        index += 1

# Create adjacency matrix
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# Fill adjacency matrix with links
for link in root.findall('.//link'):
    source_element = link.find('./from')
    target_element = link.find('./to')
    if source_element is not None and target_element is not None:
        source = source_element.get('node')
        target = target_element.get('node')
        if source in nodes and target in nodes:
            adj_matrix[nodes[source], nodes[target]] = 1
            adj_matrix[nodes[target], nodes[source]] = 1  # Assuming undirected links

# Sort nodes by numerical order and rearrange the adjacency matrix accordingly
sorted_node_ids = sorted(nodes.keys(), key=int)
sorted_indices = [nodes[node_id] for node_id in sorted_node_ids]
adj_matrix_sorted = adj_matrix[np.ix_(sorted_indices, sorted_indices)]

# Convert the sorted adjacency matrix to a DataFrame for better readability
adj_matrix_sorted_df = pd.DataFrame(adj_matrix_sorted, index=sorted_node_ids, columns=sorted_node_ids)

# Display the sorted adjacency matrix as a DataFrame
adj_matrix_sorted_df

# dfをアレイに変換
adj_matrix_sorted_array = adj_matrix_sorted_df.values
print(adj_matrix_sorted_array)


# In[14]:


# 隣接行列からグラフを作成
G = nx.from_numpy_array(adj_matrix_sorted_array)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True)
plt.show()


# In[15]:


# # グラフの作成
# G = nx.Graph()
# G.add_edges_from(edges)

# # グラフの描画
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, seed=12)  # ノードのレイアウト
# nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=3000, font_size=12, font_weight='bold')
# plt.title('Network Graph')
# plt.show()


# In[16]:


# 繋がってるノードだけを表示
print(G[21])


# In[17]:


import xml.etree.ElementTree as ET
import pandas as pd
import pprint

# Parse the XML file
tree = ET.parse('/home/yamamoto/workspace/research/edtmsr/origin_data/traffic-matrices/IntraTM-2005-01-01-00-30.xml')
root = tree.getroot()

# Initialize an empty list to hold the traffic data
traffic_data = []

# Extract traffic data based on the observed structure
for src_node in root.findall('.//src'):
    source = src_node.get('id')
    for dst_node in src_node.findall('dst'):
        destination = dst_node.get('id')
        traffic = float(dst_node.text)
        traffic_data.append([source, destination, traffic])

# Convert the traffic data to a pandas DataFrame
traffic_df = pd.DataFrame(traffic_data, columns=['Source', 'Destination', 'Traffic'])

# Convert the traffic data to a pandas DataFrame
traffic_df = pd.DataFrame(traffic_data, columns=['Source', 'Destination', 'Traffic'])

# Get unique sources and destinations
sources = set(traffic_df['Source'])
destinations = set(traffic_df['Destination'])
all_nodes = sorted(sources.union(destinations), key=int)

# Create a pivot table to represent the traffic matrix
traffic_matrix = traffic_df.pivot(index='Source', columns='Destination', values='Traffic').reindex(index=all_nodes, columns=all_nodes).fillna(0)

# Display the sorted traffic matrix
traffic_matrix_sorted = traffic_matrix.loc[all_nodes, all_nodes]
# print(traffic_matrix_sorted)

# dfをアレイに変換
adj_matrix_sorted_array = traffic_matrix_sorted.values
print(adj_matrix_sorted_array.shape)

# dfをtensorに変換
# traffic_matrix_tensor = traffic_matrix_sorted.values
# traffic_matrix_tensor = traffic_matrix_tensor.reshape(1, traffic_matrix_tensor.shape[0], traffic_matrix_tensor.shape[1])


# csvに保存
traffic_matrix_sorted.to_csv('traffic_matrix.csv')


# In[18]:


print(adj_matrix_sorted_array[0][0])


# In[19]:


# スライディングウィンドウを適用してサブマトリックスを生成する関数（ステップサイズを調整）
def sliding_window(matrix, window_size, step_size):
    windows = []
    rows, cols = matrix.shape
    for i in range(0, rows - window_size + 1, step_size):
        for j in range(0, cols - window_size + 1, step_size):
            window = matrix[i:i + window_size, j:j + window_size]
            windows.append(window)
    return np.array(windows)


# In[20]:


# 18x18のスライディングウィンドウを適用
window_size = 18
# step_size = window_size // 2
step_size = 1

sub_matrices = sliding_window(adj_matrix_sorted_array, window_size, step_size)

# 結果の確認
sub_matrices.shape, sub_matrices[0]

# print(len(sub_matrices))
print(f'sub_matrix.shape: {sub_matrices.shape}')
print(sub_matrices)


# In[21]:


import cv2
import numpy as np

def resize_with_opencv(matrix, scale_factor):
    height, width = matrix.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    resized_matrix = cv2.resize(matrix, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    resized_matrix = np.clip(resized_matrix, 0, None)
    return resized_matrix

# Example usage
resized_matrix = resize_with_opencv(sub_matrices[0], 0.5)
print(resized_matrix.shape)
print(resized_matrix)


# In[22]:


# # あるディレクトリに含まれているファイルの数を表示
import os

path = '/home/yamamoto/workspace/research/edtmsr/origin_data/traffic-matrices/'
datasets = []

# # 18x18のスライディングウィンドウを適用
# window_size = 18
# step_size = 1

files = os.listdir(path)
for file in files:
    # print(file)
    datasets.append(file)
    # sub_matrices = sliding_window(adj_matrix_sorted_array, window_size, step_size)
print(len(datasets))

