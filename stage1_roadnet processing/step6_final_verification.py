#!/usr/bin/env python3
"""
Step 6: 最终集成验证 - 简化版本
验证数据流水线和MADDPG集成准备
"""

import numpy as np
import pickle
import json
import pandas as pd
import networkx as nx
from pathlib import Path

def verify_data_pipeline():
    """验证数据流水线完整性"""
    print("=== 验证数据流水线 ===")
    
    results = {}
    
    # 1. 验证网格到节点映射
    try:
        with open('shanghai_dataset/mapping/grid_to_node_mapping.json', 'r') as f:
            mapping_list = json.load(f)
        
        grid_to_node = {item['grid_id']: item['node_id'] for item in mapping_list}
        results['grid_mapping'] = {
            'total_grids': len(grid_to_node),
            'status': 'success'
        }
        print(f"✅ 网格映射: {len(grid_to_node)} 个网格")
    except Exception as e:
        results['grid_mapping'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 网格映射失败: {e}")
    
    # 2. 验证路网图
    try:
        with open('shanghai_dataset/road_network/road_graph_fixed.pkl', 'rb') as f:
            road_graph = pickle.load(f)
        
        results['road_graph'] = {
            'nodes': road_graph.number_of_nodes(),
            'edges': road_graph.number_of_edges(),
            'connected_components': nx.number_connected_components(road_graph),
            'largest_cc_size': len(max(nx.connected_components(road_graph), key=len)),
            'status': 'success'
        }
        print(f"✅ 路网图: {road_graph.number_of_nodes()} 节点, {road_graph.number_of_edges()} 边")
        print(f"   连通分量: {results['road_graph']['connected_components']}")
    except Exception as e:
        results['road_graph'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 路网图失败: {e}")
    
    # 3. 验证节点特征
    try:
        features = np.load('shanghai_dataset/road_network/node_features_fixed.npy')
        results['node_features'] = {
            'shape': features.shape,
            'status': 'success'
        }
        print(f"✅ 节点特征: {features.shape}")
    except Exception as e:
        results['node_features'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 节点特征失败: {e}")
    
    # 4. 验证订单数据转换
    try:
        order_files = list(Path('shanghai_dataset/processed_orders_road').glob('Orders_Dataset_shanghai_road_day_*'))
        
        # 测试加载一个订单文件
        if order_files:
            with open(order_files[0], 'rb') as f:
                orders = pickle.load(f)
            
            results['order_conversion'] = {
                'total_days': len(order_files),
                'timesteps_per_day': len(orders),
                'sample_orders': len(orders[0]) if orders else 0,
                'status': 'success'
            }
            print(f"✅ 订单转换: {len(order_files)} 天, 每天 {len(orders)} 个时间步")
        else:
            results['order_conversion'] = {'status': 'failed', 'error': 'No order files found'}
            print("❌ 找不到订单文件")
    except Exception as e:
        results['order_conversion'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 订单转换失败: {e}")
    
    return results

def test_path_planning():
    """测试路径规划功能"""
    print("\n=== 测试路径规划 ===")
    
    try:
        # 加载数据
        with open('shanghai_dataset/road_network/road_graph_fixed.pkl', 'rb') as f:
            G = pickle.load(f)
        
        with open('shanghai_dataset/mapping/grid_to_node_mapping.json', 'r') as f:
            mapping_list = json.load(f)
        grid_to_node = {item['grid_id']: item['node_id'] for item in mapping_list}
        
        # 找到最大连通分量中的节点
        largest_cc = max(nx.connected_components(G), key=len)
        
        # 找到对应的网格
        valid_grids = []
        for grid_id, node_id in grid_to_node.items():
            if node_id in largest_cc:
                valid_grids.append(grid_id)
        
        if len(valid_grids) < 2:
            print("❌ 找不到足够的连通网格")
            return False
        
        # 选择两个测试网格
        import random
        test_grids = random.sample(valid_grids, 2)
        start_node = grid_to_node[test_grids[0]]
        end_node = grid_to_node[test_grids[1]]
        
        # 计算最短路径
        if nx.has_path(G, start_node, end_node):
            path = nx.shortest_path(G, start_node, end_node, weight='weight')
            path_length = nx.shortest_path_length(G, start_node, end_node, weight='weight')
            
            print(f"✅ 路径规划成功:")
            print(f"   从网格 {test_grids[0]} (节点{start_node}) 到网格 {test_grids[1]} (节点{end_node})")
            print(f"   路径长度: {len(path)} 节点, 总距离: {path_length:.4f}")
            return True
        else:
            print(f"❌ 节点不连通")
            return False
            
    except Exception as e:
        print(f"❌ 路径规划测试失败: {e}")
        return False

def generate_final_report(verification_results, path_planning_success):
    """生成最终报告"""
    print("\n=== 生成最终报告 ===")
    
    # 计算成功的组件数
    successful_components = sum(1 for result in verification_results.values() 
                              if result.get('status') == 'success')
    total_components = len(verification_results)
    
    report = {
        'verification_summary': {
            'successful_components': successful_components,
            'total_components': total_components,
            'success_rate': successful_components / total_components,
            'path_planning_test': path_planning_success
        },
        'component_details': verification_results,
        'ready_for_maddpg': successful_components >= 3 and path_planning_success,
        'next_steps': [
            "修改 air_ground_simulation.py 集成路网导航",
            "为载具智能体添加GraphSAGE嵌入",
            "实现混合导航系统（载具用路网，无人机用网格）",
            "开始MADDPG训练"
        ]
    }
    
    # 保存报告
    with open('shanghai_dataset/final_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("最终验证报告:")
    print(f"✅ 成功组件: {successful_components}/{total_components}")
    print(f"✅ 成功率: {report['verification_summary']['success_rate']:.2%}")
    print(f"✅ 路径规划: {'通过' if path_planning_success else '失败'}")
    print(f"✅ MADDPG就绪: {'是' if report['ready_for_maddpg'] else '否'}")
    
    if report['ready_for_maddpg']:
        print("\n🎉 数据流水线验证完成！可以开始MADDPG集成了。")
        print("\n下一步建议:")
        for step in report['next_steps']:
            print(f"  • {step}")
    
    return report

def main():
    """主函数"""
    print("=== Step 6: 最终集成验证 ===")
    
    try:
        # 1. 验证数据流水线
        verification_results = verify_data_pipeline()
        
        # 2. 测试路径规划
        path_planning_success = test_path_planning()
        
        # 3. 生成最终报告
        final_report = generate_final_report(verification_results, path_planning_success)
        
        print(f"\n报告已保存: shanghai_dataset/final_verification_report.json")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
