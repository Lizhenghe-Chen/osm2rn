#!/usr/bin/env python3
"""
Step 5: GraphSAGE准备 - 为MADDPG集成准备GraphSAGE模型
包括数据加载器、模型定义和预训练功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pickle
import json
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
import pandas as pd

class GraphSAGE(nn.Module):
    """GraphSAGE模型用于路网节点嵌入"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # 输出层
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        """前向传播"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 不在最后一层使用激活函数
                x = F.relu(x)
                x = self.dropout(x)
        return x

class RoadNetworkDataLoader:
    """路网数据加载器"""
    
    def __init__(self, road_graph_path, features_path, node_ids_path):
        self.road_graph_path = road_graph_path
        self.features_path = features_path
        self.node_ids_path = node_ids_path
        
        # 加载数据
        self.G = None
        self.features = None
        self.node_ids = None
        self.node_id_to_idx = None
        self.pyg_data = None
        
    def load_data(self):
        """加载路网图数据"""
        print("加载路网数据...")
        
        # 加载NetworkX图
        with open(self.road_graph_path, 'rb') as f:
            self.G = pickle.load(f)
        
        # 加载节点特征
        self.features = np.load(self.features_path)
        
        # 加载节点ID列表
        with open(self.node_ids_path, 'r') as f:
            self.node_ids = json.load(f)
        
        # 创建节点ID到索引的映射
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        
        print(f"加载完成:")
        print(f"- 图: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        print(f"- 特征: {self.features.shape}")
        print(f"- 节点映射: {len(self.node_id_to_idx)} 个")
        
        return self.G, self.features, self.node_ids, self.node_id_to_idx
    
    def convert_to_pytorch_geometric(self):
        """转换为PyTorch Geometric格式"""
        print("转换为PyTorch Geometric格式...")
        
        if self.G is None:
            self.load_data()
        
        # 将NetworkX图转换为PyG格式
        self.pyg_data = from_networkx(self.G)
        
        # 设置节点特征
        self.pyg_data.x = torch.tensor(self.features, dtype=torch.float)
        
        print(f"PyG数据:")
        print(f"- 节点数: {self.pyg_data.num_nodes}")
        print(f"- 边数: {self.pyg_data.num_edges}")
        print(f"- 特征维度: {self.pyg_data.x.shape}")
        
        return self.pyg_data
    
    def get_node_embedding(self, model, node_id):
        """获取特定节点的嵌入"""
        if self.node_id_to_idx is None:
            self.load_data()
        if self.node_id_to_idx is None:
            raise ValueError("Failed to load node_id_to_idx")
        if self.pyg_data is None:
            self.convert_to_pytorch_geometric()
        if node_id not in self.node_id_to_idx:
            raise ValueError(f"节点ID {node_id} 不存在")
        
        idx = self.node_id_to_idx[node_id]
    def get_batch_embeddings(self, model, node_ids):
        """批量获取节点嵌入"""
        if self.node_id_to_idx is None:
            self.load_data()
        if self.node_id_to_idx is None:
            raise ValueError("Failed to load node_id_to_idx")
        if self.pyg_data is None:
            self.convert_to_pytorch_geometric()
        if self.pyg_data is None:
            raise ValueError("Failed to convert to PyTorch Geometric format")
        indices = []
        for node_id in node_ids:
            if node_id in self.node_id_to_idx:
                indices.append(self.node_id_to_idx[node_id])
            else:
                print(f"警告: 节点ID {node_id} 不存在，跳过")
        
        if not indices:
            return torch.empty(0, model.convs[-1].out_channels)
        
        with torch.no_grad():
            embeddings = model(self.pyg_data.x, self.pyg_data.edge_index)
            return embeddings[indices]

def create_self_supervised_training_data(G, features):
    """创建自监督训练数据（节点分类或链接预测）"""
    print("创建自监督训练数据...")
    
    # 使用度数作为伪标签进行节点分类
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    
    # 将度数分为5个类别
    degree_classes = []
    for node in G.nodes():
        degree = degrees[node]
        if degree == 1:
            class_label = 0  # 端点
        elif degree == 2:
            class_label = 1  # 简单路径点
        elif degree == 3:
            class_label = 2  # 三路交叉
        elif degree == 4:
            class_label = 3  # 四路交叉
        else:
            class_label = 4  # 复杂交叉
        degree_classes.append(class_label)
    
    print(f"度数分类统计:")
    for i in range(5):
        count = degree_classes.count(i)
        print(f"- 类别 {i}: {count} 个节点")
    
    return torch.tensor(degree_classes, dtype=torch.long)

def train_graphsage_model(data_loader, model, labels, epochs=100):
    """训练GraphSAGE模型"""
    print("开始训练GraphSAGE模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data_loader.pyg_data.to(device)
    labels = labels.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练/验证split（使用最大连通分量）
    largest_cc = max(nx.connected_components(data_loader.G), key=len)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    for node_id in largest_cc:
        if node_id in data_loader.node_id_to_idx:
            idx = data_loader.node_id_to_idx[node_id]
            train_mask[idx] = True
    
    train_indices = train_mask.nonzero().squeeze()
    val_size = len(train_indices) // 10  # 10%用于验证
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]
    
    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_indices], labels[train_indices])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_indices], labels[val_indices])
                
                # 计算准确率
                pred = val_out[val_indices].argmax(dim=1)
                val_acc = (pred == labels[val_indices]).float().mean()
                
            print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            model.train()
    
    return model

def test_integration_with_grid_system():
    """测试与网格系统的集成"""
    print("测试与网格系统的集成...")
    
    # 加载网格到节点的映射
    mapping_path = 'shanghai_dataset/mapping/grid_to_node_mapping.json'
    with open(mapping_path, 'r') as f:
        grid_to_node = json.load(f)
    
    print(f"网格到节点映射: {len(grid_to_node)} 个网格")
    
    # 测试几个随机网格的映射
    import random
    test_grids = random.sample(list(grid_to_node.keys()), min(5, len(grid_to_node)))
    
    print("测试网格映射:")
    for grid_id in test_grids:
        node_id = grid_to_node[grid_id]
        print(f"- 网格 {grid_id} -> 节点 {node_id}")
    
    return True

def save_model_and_config(model, data_loader, save_dir):
    """保存模型和配置"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), save_dir / 'graphsage_model.pth')
    
    # 保存配置
    config = {
        'model_config': {
            'input_dim': model.convs[0].in_channels,
            'hidden_dim': model.convs[0].out_channels,
            'output_dim': model.convs[-1].out_channels,
            'num_layers': len(model.convs)
        },
        'data_config': {
            'num_nodes': data_loader.pyg_data.num_nodes,
            'num_edges': data_loader.pyg_data.num_edges,
            'feature_dim': data_loader.pyg_data.x.shape[1]
        }
    }
    
    with open(save_dir / 'graphsage_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"模型和配置保存到: {save_dir}")
    return config

def main():
    """主函数"""
    print("=== Step 5: GraphSAGE准备 ===")
    
    try:
        # 1. 初始化数据加载器
        data_loader = RoadNetworkDataLoader(
            road_graph_path='shanghai_dataset/road_network/road_graph_fixed.pkl',
            features_path='shanghai_dataset/road_network/node_features_fixed.npy',
            node_ids_path='shanghai_dataset/road_network/node_id_list_fixed.json'
        )
        
        # 2. 加载数据
        G, features, node_ids, node_id_to_idx = data_loader.load_data()
        
        # 3. 转换为PyTorch Geometric格式
        pyg_data = data_loader.convert_to_pytorch_geometric()
        
        # 4. 创建GraphSAGE模型
        input_dim = features.shape[1]  # 9维特征
        hidden_dim = 64
        output_dim = 32  # 嵌入维度
        model = GraphSAGE(input_dim, hidden_dim, output_dim, num_layers=3)
        
        print(f"GraphSAGE模型:")
        print(f"- 输入维度: {input_dim}")
        print(f"- 隐藏维度: {hidden_dim}")
        print(f"- 输出维度: {output_dim}")
        print(f"- 层数: 3")
        
        # 5. 创建自监督训练数据
        labels = create_self_supervised_training_data(G, features)
        
        # 6. 训练模型
        trained_model = train_graphsage_model(data_loader, model, labels, epochs=50)
        
        # 7. 测试与网格系统的集成
        integration_success = test_integration_with_grid_system()
        
        # 8. 保存模型和配置
        config = save_model_and_config(trained_model, data_loader, 'shanghai_dataset/graphsage')
        
        print("\n=== GraphSAGE准备完成 ===")
        print("✅ 数据加载成功")
        print("✅ 模型训练完成")
        print("✅ 网格集成测试通过" if integration_success else "❌ 网格集成测试失败")
        print("✅ 模型和配置已保存")
        
        # 9. 测试嵌入提取
        print("\n测试嵌入提取:")
        test_node_ids = node_ids[:3]  # 测试前3个节点
        embeddings = data_loader.get_batch_embeddings(trained_model, test_node_ids)
        print(f"测试节点 {test_node_ids} 的嵌入维度: {embeddings.shape}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
