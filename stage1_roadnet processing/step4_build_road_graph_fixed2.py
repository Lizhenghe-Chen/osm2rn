#!/usr/bin/env python3
"""
Step 4.2: 修复路网图构建 - 正确处理边连接信息和列名
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import json
import time
from pathlib import Path
import shapefile

DATASET_DIR = Path('shanghai_dataset')


def dataset_path(*parts):
    return str(DATASET_DIR.joinpath(*parts))

def load_processed_nodes():
    """加载处理过的节点数据"""
    print("正在加载节点数据...")
    nodes_df = pd.read_csv(dataset_path('road_network', 'nodes_processed.csv'))
    print(f"加载了 {len(nodes_df)} 个节点")
    print(f"节点数据列名: {list(nodes_df.columns)}")
    return nodes_df

def extract_edge_connectivity_from_shapefile():
    """从原始shapefile中提取边的连接信息"""
    print("正在从shapefile提取边连接信息...")
    
    # 读取边的shapefile
    shp_path = dataset_path('edges')
    edges_sf = shapefile.Reader(shp_path)
    
    edges_data = []
    processed_count = 0
    
    for shape_record in edges_sf.shapeRecords():
        try:
            # 获取边的几何信息
            shape = shape_record.shape
            record = shape_record.record
            
            # 获取边ID（从记录中）
            eid = record[0] if len(record) > 0 else processed_count
            
            # 获取线段的起始和结束坐标
            if shape.shapeType == 3:  # 线段类型
                coords = shape.points
                if len(coords) >= 2:
                    start_coord = coords[0]  # 起始点坐标
                    end_coord = coords[-1]   # 结束点坐标
                    
                    edges_data.append({
                        'eid': eid,
                        'start_x': start_coord[0],
                        'start_y': start_coord[1],
                        'end_x': end_coord[0],
                        'end_y': end_coord[1],
                        'geometry_points': len(coords)
                    })
            
            processed_count += 1
            if processed_count % 50000 == 0:
                print(f"已处理 {processed_count} 条边...")
                
        except Exception as e:
            print(f"处理边 {processed_count} 时出错: {e}")
            continue
    
    print(f"从shapefile提取了 {len(edges_data)} 条边的连接信息")
    return pd.DataFrame(edges_data)

def find_nearest_node_for_coord(coord, nodes_df, coord_to_node_map):
    """为给定坐标找到最近的节点"""
    coord_key = f"{coord[0]:.6f},{coord[1]:.6f}"
    
    if coord_key in coord_to_node_map:
        return coord_to_node_map[coord_key]
    
    # 计算到所有节点的距离
    distances = np.sqrt((nodes_df['longitude'] - coord[0])**2 + (nodes_df['latitude'] - coord[1])**2)
    nearest_idx = distances.idxmin()
    nearest_node_id = nodes_df.loc[nearest_idx, 'node_id']
    
    # 缓存结果
    coord_to_node_map[coord_key] = nearest_node_id
    return nearest_node_id

def build_node_coordinate_map(nodes_df):
    """构建节点坐标到节点ID的映射"""
    print("构建节点坐标映射...")
    coord_to_node = {}
    
    for _, row in nodes_df.iterrows():
        coord_key = f"{row['longitude']:.6f},{row['latitude']:.6f}"
        coord_to_node[coord_key] = row['node_id']
    
    print(f"构建了 {len(coord_to_node)} 个坐标映射")
    return coord_to_node

def build_road_graph_with_connectivity(nodes_df, edges_df):
    """构建包含正确连接信息的路网图"""
    print("构建路网图...")
    start_time = time.time()
    
    # 创建图
    G = nx.Graph()
    
    # 添加所有节点
    node_ids = nodes_df['node_id'].tolist()
    G.add_nodes_from(node_ids)
    print(f"添加了 {len(node_ids)} 个节点")
    
    # 构建坐标到节点ID的映射
    coord_to_node_map = build_node_coordinate_map(nodes_df)
    
    # 添加边
    edges_added = 0
    edges_failed = 0
    
    for idx, edge_row in edges_df.iterrows():
        try:
            # 找到起始和结束节点
            start_coord = (edge_row['start_x'], edge_row['start_y'])
            end_coord = (edge_row['end_x'], edge_row['end_y'])
            
            start_node = find_nearest_node_for_coord(start_coord, nodes_df, coord_to_node_map)
            end_node = find_nearest_node_for_coord(end_coord, nodes_df, coord_to_node_map)
            
            if start_node != end_node:  # 避免自环
                # 计算边的长度作为权重
                distance = np.sqrt((edge_row['end_x'] - edge_row['start_x'])**2 + 
                                 (edge_row['end_y'] - edge_row['start_y'])**2)
                
                G.add_edge(start_node, end_node, 
                          weight=distance,
                          eid=edge_row['eid'],
                          geometry_points=edge_row['geometry_points'])
                edges_added += 1
            else:
                edges_failed += 1
                
        except Exception as e:
            edges_failed += 1
            if edges_failed % 1000 == 0:
                print(f"边添加失败数: {edges_failed}")
        
        # 每10000条边显示进度
        if (idx + 1) % 10000 == 0:
            print(f"已处理 {idx + 1} 条边，成功添加 {edges_added} 条...")
    
    build_time = time.time() - start_time
    print(f"图构建完成，用时: {build_time:.2f}秒")
    print(f"成功添加 {edges_added} 条边")
    print(f"失败 {edges_failed} 条边")
    print(f"图统计: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    print(f"连通分量数: {nx.number_connected_components(G)}")
    
    return G

def calculate_node_features(G, nodes_df):
    """计算节点特征"""
    print("计算节点特征...")
    
    features = []
    node_ids = []
    processed = 0
    
    for node_id in G.nodes():
        # 找到该节点在nodes_df中的信息
        node_info = nodes_df[nodes_df['node_id'] == node_id].iloc[0]
        
        # 基本坐标特征
        x, y = node_info['longitude'], node_info['latitude']
        
        # 图结构特征
        degree = G.degree(node_id)
        
        # 邻居统计
        neighbors = list(G.neighbors(node_id))
        if neighbors:
            neighbor_coords = []
            for nid in neighbors:
                neighbor_info = nodes_df[nodes_df['node_id'] == nid].iloc[0]
                neighbor_coords.append((neighbor_info['longitude'], neighbor_info['latitude']))
            
            # 到邻居的平均距离
            distances = [np.sqrt((x - nx)**2 + (y - ny)**2) for nx, ny in neighbor_coords]
            avg_neighbor_distance = np.mean(distances)
            max_neighbor_distance = np.max(distances)
            min_neighbor_distance = np.min(distances)
        else:
            avg_neighbor_distance = 0
            max_neighbor_distance = 0
            min_neighbor_distance = 0
        
        # 聚类系数
        clustering = nx.clustering(G, node_id)
        
        # 特征向量 [x, y, degree, avg_dist, max_dist, min_dist, clustering, betweenness, closeness]
        feature_vector = [
            x, y, degree, 
            avg_neighbor_distance, max_neighbor_distance, min_neighbor_distance,
            clustering, 0, 0  # betweenness和closeness暂时设为0（计算量大）
        ]
        
        features.append(feature_vector)
        node_ids.append(node_id)
        
        processed += 1
        if processed % 50000 == 0:
            print(f"已处理 {processed} 个节点的特征...")
    
    features_array = np.array(features, dtype=np.float32)
    print(f"计算了 {len(features)} 个节点的特征，每个特征维度: {features_array.shape[1]}")
    
    return features_array, node_ids

def save_results(G, features_array, node_ids):
    """保存结果"""
    print("保存结果...")
    
    # 确保目录存在
    output_dir = DATASET_DIR / 'road_network'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图
    with open(output_dir / 'road_graph_fixed.pkl', 'wb') as f:
        pickle.dump(G, f)
    
    # 保存特征
    np.save(output_dir / 'node_features_fixed.npy', features_array)
    
    # 保存节点ID列表
    with open(output_dir / 'node_id_list_fixed.json', 'w') as f:
        json.dump(node_ids, f)
    
    # 保存连通性报告
    report = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'connected_components': nx.number_connected_components(G),
        'largest_component_size': len(max(nx.connected_components(G), key=len)),
        'average_degree': np.mean([d for n, d in G.degree()]),
        'feature_dimensions': features_array.shape[1]
    }
    
    with open(output_dir / 'connectivity_report_fixed.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("结果保存完成:")
    print(f"- 图文件: road_graph_fixed.pkl")
    print(f"- 特征文件: node_features_fixed.npy")
    print(f"- 节点ID列表: node_id_list_fixed.json")
    print(f"- 连通性报告: connectivity_report_fixed.json")
    
    return report

def main():
    """主函数"""
    print("=== Step 4.2: 修复路网图构建 ===")
    
    try:
        # 1. 加载节点数据
        nodes_df = load_processed_nodes()
        
        # 2. 从shapefile提取边连接信息
        edges_df = extract_edge_connectivity_from_shapefile()
        
        # 3. 构建包含正确连接的路网图
        G = build_road_graph_with_connectivity(nodes_df, edges_df)
        
        # 4. 计算节点特征
        features_array, node_ids = calculate_node_features(G, nodes_df)
        
        # 5. 保存结果
        report = save_results(G, features_array, node_ids)
        
        print("\n=== 修复完成 ===")
        print(f"图统计:")
        print(f"- 节点数: {report['total_nodes']:,}")
        print(f"- 边数: {report['total_edges']:,}")
        print(f"- 连通分量数: {report['connected_components']:,}")
        print(f"- 最大连通分量大小: {report['largest_component_size']:,}")
        print(f"- 平均度数: {report['average_degree']:.2f}")
        
        if report['connected_components'] == 1:
            print("✅ 路网图完全连通！")
        else:
            print(f"⚠️  路网图有 {report['connected_components']} 个连通分量")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
