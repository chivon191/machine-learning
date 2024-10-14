import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import pandas as pd

def compute_entropy(y):
    if len(y) == 0:
        return 0
    
    p = np.mean(y)
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []

    for i in node_indices:
        if i < len(X):  # Ensure the index is within the bounds of X
            if X[i][feature] == 1:
                left_indices.append(i)
            else:
                right_indices.append(i)

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    node_entropy = compute_entropy(y_node)

    # Kiểm tra xem X_left và X_right có rỗng không
    if len(X_left) == 0 or len(X_right) == 0:
        return 0  # Trả về 0 nếu không thể chia

    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy
    
    return information_gain

def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    
    best_feature = -1
    max_info_gain = 0
    
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
            
    return best_feature

def build_tree_recursive(X, y, node_indices, current_depth, max_depth, tree):
    if current_depth == max_depth or len(node_indices) == 0:
        return

    best_feature = get_best_split(X, y, node_indices) 
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    if len(left_indices) == 0 and len(right_indices) == 0:  # Kiểm tra nếu không có chỉ số nào
        return
    
    tree.append((left_indices, right_indices, best_feature))
    
    build_tree_recursive(X, y, left_indices, current_depth + 1, max_depth, tree)
    build_tree_recursive(X, y, right_indices, current_depth + 1, max_depth, tree)

def generate_tree_viz(tree, X, y):
    G = nx.DiGraph()
    
    # Create the root node
    root_indices = list(range(len(X)))
    G.add_node(0, label='Root')
    
    idx = 1
    for i, (left_indices, right_indices, feature) in enumerate(tree):
        G.add_node(idx, label=f'Feature {feature} = 1')
        G.add_edge(0, idx)  # Edge from root to left child
        
        idx += 1
        G.add_node(idx, label=f'Feature {feature} = 0')
        G.add_edge(0, idx)  # Edge from root to right child

        # Add edges to children
        if len(left_indices) > 0:
            G.add_edge(i + 1, idx - 2)  # Edge from parent to left child
        if len(right_indices) > 0:
            G.add_edge(i + 1, idx - 1)  # Edge from parent to right child

    pos = graphviz_layout(G, prog="dot")

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, arrows=True, labels=nx.get_node_attributes(G, 'label'))
    plt.title("Decision Tree Visualization")
    plt.show()

# Đọc dữ liệu
df = pd.read_csv('D:\\IUH\\Nam tu\\HK1\\Machine Learning\\machine-learning\\decision-tree-graded\\Lab6_Decision_Trees_graded\\data_heart.csv')
X_train = df.drop(columns=['HeartDisease']).head(20).to_numpy()
y_train = df['HeartDisease'].head(20).to_numpy()

node_indices = list(range(len(X_train)))
tree = []
build_tree_recursive(X_train, y_train, node_indices, 0, 2, tree)  # Max depth = 2

generate_tree_viz(tree, X_train, y_train)
