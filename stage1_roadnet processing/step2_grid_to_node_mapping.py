#!/usr/bin/env python3
"""
Step 2.1: 格子到节点映射算法实现
为6400个格子中心点找到最近的路网节点，建立映射表
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

def load_grid_data():
    """加载格子数据"""
    print("📊 加载格子数据...")
    
    try:
        grid_file = "./shanghai_dataset/shanghai_grid_chengdu_format.csv"
        grids = pd.read_csv(grid_file)
        
        print(f"✅ 成功加载格子数据")
        print(f"  格子总数: {len(grids):,}")
        print(f"  字段: {list(grids.columns)}")
        
        # 提取格子中心坐标
        grid_centers = grids[['grid_id', 'center_lng', 'center_lat']].copy()
        grid_centers.columns = ['grid_id', 'longitude', 'latitude']
        
        print(f"  坐标范围:")
        print(f"    经度: {grid_centers['longitude'].min():.6f} ~ {grid_centers['longitude'].max():.6f}")
        print(f"    纬度: {grid_centers['latitude'].min():.6f} ~ {grid_centers['latitude'].max():.6f}")
        
        return grid_centers
        
    except Exception as e:
        print(f"❌ 加载格子数据失败: {e}")
        return None

def load_road_nodes():
    """加载路网节点数据"""
    print("\n🛣️ 加载路网节点数据...")
    
    try:
        nodes_file = "./shanghai_dataset/road_network/nodes_processed.csv"
        nodes = pd.read_csv(nodes_file)
        
        print(f"✅ 成功加载路网节点")
        print(f"  节点总数: {len(nodes):,}")
        print(f"  字段: {list(nodes.columns)}")
        
        print(f"  坐标范围:")
        print(f"    经度: {nodes['longitude'].min():.6f} ~ {nodes['longitude'].max():.6f}")
        print(f"    纬度: {nodes['latitude'].min():.6f} ~ {nodes['latitude'].max():.6f}")
        
        return nodes
        
    except Exception as e:
        print(f"❌ 加载路网节点失败: {e}")
        return None

def create_nearest_neighbor_mapping(grid_centers, road_nodes):
    """使用最近邻算法创建格子到节点的映射"""
    print("\n🔍 执行最近邻映射算法...")
    
    try:
        start_time = time.time()
        
        # 准备坐标数据
        print("  📍 准备坐标数据...")
        grid_coords = grid_centers[['longitude', 'latitude']].values
        node_coords = road_nodes[['longitude', 'latitude']].values
        
        print(f"    格子坐标: {grid_coords.shape}")
        print(f"    节点坐标: {node_coords.shape}")
        
        # 构建KD-Tree最近邻模型
        print("  🌲 构建KD-Tree空间索引...")
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
        nn_model.fit(node_coords)
        
        # 批量查找最近邻
        print("  🎯 执行最近邻搜索...")
        distances, indices = nn_model.kneighbors(grid_coords)
        
        # 生成映射表
        print("  📝 生成映射表...")
        mapping_list = []
        
        for i, grid_id in enumerate(grid_centers['grid_id']):
            nearest_node_idx = indices[i][0]
            distance = distances[i][0]
            
            # 获取最近节点信息
            nearest_node = road_nodes.iloc[nearest_node_idx]
            
            mapping_entry = {
                'grid_id': int(grid_id),
                'node_id': int(nearest_node['node_id']),
                'distance': float(distance),
                'grid_lng': float(grid_centers.iloc[i]['longitude']),
                'grid_lat': float(grid_centers.iloc[i]['latitude']),
                'node_lng': float(nearest_node['longitude']),
                'node_lat': float(nearest_node['latitude'])
            }
            
            mapping_list.append(mapping_entry)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 映射完成，用时: {elapsed_time:.2f}秒")
        
        return mapping_list
        
    except Exception as e:
        print(f"❌ 最近邻映射失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_mapping_quality(mapping_list):
    """分析映射质量"""
    print("\n📊 分析映射质量...")
    
    try:
        # 转换为DataFrame便于分析
        mapping_df = pd.DataFrame(mapping_list)
        
        # 距离统计
        distances = mapping_df['distance']
        
        print(f"📏 映射距离分析:")
        print(f"  平均距离: {distances.mean():.6f}° ({distances.mean() * 111000:.1f}米)")
        print(f"  中位距离: {distances.median():.6f}° ({distances.median() * 111000:.1f}米)")
        print(f"  最小距离: {distances.min():.6f}° ({distances.min() * 111000:.1f}米)")
        print(f"  最大距离: {distances.max():.6f}° ({distances.max() * 111000:.1f}米)")
        print(f"  标准差: {distances.std():.6f}° ({distances.std() * 111000:.1f}米)")
        
        # 距离分布分析
        print(f"\n📈 距离分布:")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            dist_p = np.percentile(distances, p)
            print(f"  {p:2d}% 格子距离 < {dist_p:.6f}° ({dist_p * 111000:.1f}米)")
        
        # 检查异常值
        print(f"\n⚠️ 异常值检查:")
        threshold = distances.mean() + 3 * distances.std()
        outliers = mapping_df[mapping_df['distance'] > threshold]
        
        print(f"  异常阈值: {threshold:.6f}° ({threshold * 111000:.1f}米)")
        print(f"  异常格子: {len(outliers)} 个 ({len(outliers)/len(mapping_df)*100:.1f}%)")
        
        if len(outliers) > 0:
            print(f"  异常样例 (前3个):")
            for i, (_, row) in enumerate(outliers.head(3).iterrows(), 1):
                print(f"    {i}. 格子{int(row['grid_id'])}: 距离{row['distance']:.6f}° ({row['distance']*111000:.1f}米)")
        
        # 节点覆盖分析
        unique_nodes = mapping_df['node_id'].nunique()
        total_grids = len(mapping_df)
        
        print(f"\n🎯 节点覆盖分析:")
        print(f"  唯一节点: {unique_nodes:,}")
        print(f"  总格子数: {total_grids:,}")
        print(f"  平均每节点: {total_grids/unique_nodes:.1f} 个格子")
        
        # 节点使用频率
        node_usage = mapping_df['node_id'].value_counts()
        print(f"  最多使用节点: {node_usage.iloc[0]} 个格子")
        print(f"  最少使用节点: {node_usage.iloc[-1]} 个格子")
        
        return {
            'total_mappings': len(mapping_list),
            'distance_stats': {
                'mean': float(distances.mean()),
                'median': float(distances.median()),
                'min': float(distances.min()),
                'max': float(distances.max()),
                'std': float(distances.std())
            },
            'outliers': len(outliers),
            'unique_nodes_used': unique_nodes,
            'node_usage_max': int(node_usage.iloc[0]),
            'node_usage_min': int(node_usage.iloc[-1])
        }
        
    except Exception as e:
        print(f"❌ 映射质量分析失败: {e}")
        return None

def save_mapping_results(mapping_list, quality_stats):
    """保存映射结果"""
    print("\n💾 保存映射结果...")
    
    try:
        # 创建输出目录
        Path("shanghai_dataset/mapping").mkdir(parents=True, exist_ok=True)
        
        # 保存完整映射表（JSON格式）
        mapping_path = "shanghai_dataset/mapping/grid_to_node_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_list, f, ensure_ascii=False, indent=2)
        print(f"✅ 完整映射表已保存: {mapping_path}")
        
        # 保存简化映射表（仅ID对应关系）
        simple_mapping = {str(item['grid_id']): item['node_id'] for item in mapping_list}
        simple_path = "shanghai_dataset/mapping/grid_to_node_simple.json"
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(simple_mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ 简化映射表已保存: {simple_path}")
        
        # 保存CSV格式（便于查看）
        mapping_df = pd.DataFrame(mapping_list)
        csv_path = "shanghai_dataset/mapping/grid_to_node_mapping.csv"
        mapping_df.to_csv(csv_path, index=False)
        print(f"✅ CSV映射表已保存: {csv_path}")
        
        # 保存质量统计报告
        stats_path = "shanghai_dataset/mapping/mapping_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(quality_stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 质量统计已保存: {stats_path}")
        
        # 生成统计报告（文本格式）
        report_path = "shanghai_dataset/mapping/mapping_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 格子到节点映射质量报告\n\n")
            f.write(f"生成时间: 2025-09-08\n")
            f.write(f"映射总数: {quality_stats['total_mappings']:,}\n\n")
            
            f.write("## 距离统计\n")
            f.write(f"平均距离: {quality_stats['distance_stats']['mean']:.6f}° ({quality_stats['distance_stats']['mean']*111000:.1f}米)\n")
            f.write(f"中位距离: {quality_stats['distance_stats']['median']:.6f}° ({quality_stats['distance_stats']['median']*111000:.1f}米)\n")
            f.write(f"最大距离: {quality_stats['distance_stats']['max']:.6f}° ({quality_stats['distance_stats']['max']*111000:.1f}米)\n")
            f.write(f"标准差: {quality_stats['distance_stats']['std']:.6f}° ({quality_stats['distance_stats']['std']*111000:.1f}米)\n\n")
            
            f.write("## 覆盖统计\n")
            f.write(f"使用的唯一节点: {quality_stats['unique_nodes_used']:,}\n")
            f.write(f"节点最大使用: {quality_stats['node_usage_max']} 个格子\n")
            f.write(f"节点最小使用: {quality_stats['node_usage_min']} 个格子\n")
            f.write(f"异常映射: {quality_stats['outliers']} 个\n\n")
            
            f.write("## 结论\n")
            if quality_stats['distance_stats']['mean'] * 111000 < 500:  # 平均距离小于500米
                f.write("✅ 映射质量优秀，平均距离在可接受范围内\n")
            else:
                f.write("⚠️ 映射质量需要关注，平均距离较大\n")
            
            if quality_stats['outliers'] / quality_stats['total_mappings'] < 0.05:  # 异常值小于5%
                f.write("✅ 异常值比例在合理范围内\n")
            else:
                f.write("⚠️ 异常值比例较高，需要进一步检查\n")
        
        print(f"✅ 质量报告已保存: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存映射结果失败: {e}")
        return False

def main():
    """主函数"""
    print("🗺️ Step 2.1: 格子到节点映射算法")
    print("=" * 60)
    
    # 1. 加载格子数据
    grid_centers = load_grid_data()
    if grid_centers is None:
        print("❌ 无法继续，格子数据加载失败")
        return
    
    # 2. 加载路网节点数据
    road_nodes = load_road_nodes()
    if road_nodes is None:
        print("❌ 无法继续，路网节点加载失败")
        return
    
    # 3. 执行最近邻映射
    mapping_list = create_nearest_neighbor_mapping(grid_centers, road_nodes)
    if mapping_list is None:
        print("❌ 无法继续，映射算法失败")
        return
    
    # 4. 分析映射质量
    quality_stats = analyze_mapping_quality(mapping_list)
    if quality_stats is None:
        print("❌ 映射质量分析失败")
        return
    
    # 5. 保存映射结果
    success = save_mapping_results(mapping_list, quality_stats)
    if not success:
        print("❌ 保存映射结果失败")
        return
    
    print(f"\n🎉 Step 2.1 完成总结:")
    print("=" * 60)
    print(f"✅ 成功映射: {quality_stats['total_mappings']:,} 个格子")
    print(f"✅ 平均距离: {quality_stats['distance_stats']['mean']*111000:.1f} 米")
    print(f"✅ 使用节点: {quality_stats['unique_nodes_used']:,} 个")
    print(f"✅ 异常映射: {quality_stats['outliers']} 个")
    
    print(f"\n📂 输出文件:")
    print("- shanghai_dataset/mapping/grid_to_node_mapping.json")
    print("- shanghai_dataset/mapping/grid_to_node_simple.json")
    print("- shanghai_dataset/mapping/grid_to_node_mapping.csv")
    print("- shanghai_dataset/mapping/mapping_statistics.json")
    print("- shanghai_dataset/mapping/mapping_report.txt")
    
    print(f"\n🎯 **准备就绪，可以开始 Step 3.1: 订单格式转换！**")

if __name__ == "__main__":
    main()
