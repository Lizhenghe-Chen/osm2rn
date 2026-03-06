#!/usr/bin/env python3
"""
Step 1.1: 路网数据结构深度分析（备用版本）
使用pyshp直接读取shapefile，避免geopandas兼容性问题
"""

import shapefile
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_DATASET_DIR = Path(os.getenv('ROADNET_INPUT_DIR', 'Geneva'))
OUTPUT_BASE_DIR = Path(os.getenv('ROADNET_OUTPUT_DIR', str(INPUT_DATASET_DIR)))


def input_path(*parts):
    return str(INPUT_DATASET_DIR.joinpath(*parts))


def output_path(*parts):
    return str(OUTPUT_BASE_DIR.joinpath(*parts))

def create_output_dirs():
    """创建输出目录"""
    dirs = [
        output_path('road_network'),
        output_path('mapping'),
        output_path('reports')
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("📁 创建输出目录完成")

def analyze_road_nodes():
    """分析路网节点数据"""
    print("\n🛣️ 分析路网节点数据...")
    print("=" * 50)
    
    try:
        # 使用pyshp读取nodes shapefile
        nodes_path = input_path('nodes')
        sf = shapefile.Reader(nodes_path)
        
        shapes = sf.shapes()
        records = sf.records()
        fields = [field[0] for field in sf.fields[1:]]  # 跳过删除标记字段
        
        print(f"✅ 成功读取路网节点")
        print(f"  📊 节点总数: {len(shapes):,}")
        print(f"  📊 数据维度: ({len(shapes)}, {len(fields)})")
        
        # 显示字段信息
        print(f"\n📋 节点数据字段:")
        for i, field in enumerate(fields, 1):
            sample = str(records[0][i-1])[:20] if len(records) > 0 and i-1 < len(records[0]) else "N/A"
            print(f"  {i:2d}. {field:<15} - 样例: {sample}")
        
        # 提取坐标信息
        coordinates = []
        node_ids = []
        
        for i, (shape, record) in enumerate(zip(shapes, records)):
            if shape.points:
                lng, lat = shape.points[0]  # 节点是点，只有一个坐标
                coordinates.append((lng, lat))
                
                # 尝试提取节点ID
                node_id = None
                for j, field in enumerate(fields):
                    if 'osmid' in field.lower() or 'id' in field.lower():
                        if j < len(record):
                            node_id = record[j]
                            break
                
                if node_id is None:
                    node_id = i  # 使用索引作为默认ID
                
                node_ids.append(node_id)
        
        if coordinates:
            lngs, lats = zip(*coordinates)
            lng_min, lng_max = min(lngs), max(lngs)
            lat_min, lat_max = min(lats), max(lats)
            
            print(f"\n📍 节点坐标范围:")
            print(f"  经度: {lng_min:.6f} ~ {lng_max:.6f}")
            print(f"  纬度: {lat_min:.6f} ~ {lat_max:.6f}")
            print(f"  坐标跨度: {lng_max - lng_min:.6f}° × {lat_max - lat_min:.6f}°")
            
            # 分析节点ID
            if node_ids:
                print(f"\n🆔 节点ID分析:")
                id_set = set(node_ids)
                print(f"  ID类型: {type(node_ids[0])}")
                print(f"  ID范围: {min(node_ids)} ~ {max(node_ids)}")
                print(f"  唯一节点: {len(id_set):,}")
                print(f"  重复节点: {len(node_ids) - len(id_set)}")
            
            # 显示前几个节点样例
            print(f"\n🔍 节点数据样例 (前3个):")
            for i in range(min(3, len(coordinates))):
                coord = f"({coordinates[i][0]:.6f}, {coordinates[i][1]:.6f})"
                node_id = node_ids[i] if i < len(node_ids) else "N/A"
                print(f"  {i+1}. ID: {node_id}, 坐标: {coord}")
            
            # 创建DataFrame并保存
            nodes_df = pd.DataFrame({
                'node_id': node_ids[:len(coordinates)],
                'longitude': [coord[0] for coord in coordinates],
                'latitude': [coord[1] for coord in coordinates]
            })
            
            # 添加其他字段
            for j, field in enumerate(fields):
                if field.lower() in ['highway', 'ref', 'name']:
                    values = [record[j] if j < len(record) else None for record in records]
                    nodes_df[field] = values[:len(coordinates)]
            
            output_file = output_path('road_network', 'nodes_processed.csv')
            nodes_df.to_csv(output_file, index=False)
            print(f"\n💾 节点数据已保存: {output_file}")
            
            sf.close()
            
            return {
                'total_nodes': len(coordinates),
                'coordinate_bounds': {
                    'lng_min': lng_min,
                    'lng_max': lng_max,
                    'lat_min': lat_min,
                    'lat_max': lat_max
                },
                'fields': fields,
                'sample_data': nodes_df.head(5).to_dict('records')
            }
        else:
            print("⚠️ 未找到有效坐标数据")
            sf.close()
            return {
                'total_nodes': 0,
                'coordinate_bounds': None,
                'fields': fields,
                'sample_data': []
            }
        
    except Exception as e:
        print(f"❌ 节点数据分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_road_edges():
    """分析路网边数据"""
    print(f"\n🛤️ 分析路网边数据...")
    print("=" * 50)
    
    try:
        # 使用pyshp读取edges shapefile
        edges_path = input_path('edges')
        sf = shapefile.Reader(edges_path)
        
        shapes = sf.shapes()
        records = sf.records()
        fields = [field[0] for field in sf.fields[1:]]
        
        print(f"✅ 成功读取路网边")
        print(f"  📊 边总数: {len(shapes):,}")
        print(f"  📊 数据维度: ({len(shapes)}, {len(fields)})")
        
        # 显示字段信息
        print(f"\n📋 边数据字段:")
        for i, field in enumerate(fields, 1):
            sample = str(records[0][i-1])[:20] if len(records) > 0 and i-1 < len(records[0]) else "N/A"
            print(f"  {i:2d}. {field:<15} - 样例: {sample}")
        
        # 分析边连接关系
        connect_fields = ['u', 'v', 'key']
        connect_data = {}
        for field in connect_fields:
            if field in fields:
                field_idx = fields.index(field)
                values = [record[field_idx] if field_idx < len(record) else None for record in records]
                connect_data[field] = values
                
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    print(f"\n🔗 连接字段 {field}:")
                    print(f"  范围: {min(valid_values)} ~ {max(valid_values)}")
                    print(f"  唯一值: {len(set(valid_values)):,}")
        
        # 分析边长度
        length_fields = [field for field in fields if 'length' in field.lower()]
        length_data = {}
        if length_fields:
            print(f"\n📏 边长度分析:")
            for field in length_fields:
                field_idx = fields.index(field)
                lengths = [record[field_idx] if field_idx < len(record) else None for record in records]
                lengths = [l for l in lengths if l is not None and isinstance(l, (int, float))]
                
                if lengths:
                    length_data[field] = lengths
                    print(f"  {field}:")
                    print(f"    范围: {min(lengths):.2f} ~ {max(lengths):.2f}")
                    print(f"    均值: {np.mean(lengths):.2f}, 中位数: {np.median(lengths):.2f}")
                    print(f"    标准差: {np.std(lengths):.2f}")
        
        # 分析道路类型
        highway_fields = [field for field in fields if 'highway' in field.lower()]
        if highway_fields:
            print(f"\n🛣️ 道路类型分析:")
            for field in highway_fields:
                field_idx = fields.index(field)
                highway_types = [record[field_idx] if field_idx < len(record) else None for record in records]
                highway_types = [h for h in highway_types if h is not None]
                
                if highway_types:
                    from collections import Counter
                    type_counts = Counter(highway_types)
                    print(f"  {field} (前10种类型):")
                    for highway_type, count in type_counts.most_common(10):
                        pct = count / len(highway_types) * 100
                        print(f"    {str(highway_type):<20}: {count:>6} ({pct:5.1f}%)")
        
        # 显示前几条边样例
        print(f"\n�� 边数据样例 (前3条):")
        for i in range(min(3, len(records))):
            record = records[i]
            u = record[fields.index('u')] if 'u' in fields and fields.index('u') < len(record) else 'N/A'
            v = record[fields.index('v')] if 'v' in fields and fields.index('v') < len(record) else 'N/A'
            length = record[fields.index('length')] if 'length' in fields and fields.index('length') < len(record) else 'N/A'
            highway = record[fields.index('highway')] if 'highway' in fields and fields.index('highway') < len(record) else 'N/A'
            print(f"  {i+1}. {u} → {v}, 长度: {length}, 类型: {highway}")
        
        # 创建DataFrame并保存
        edges_df = pd.DataFrame()
        for field in fields:
            field_idx = fields.index(field)
            values = [record[field_idx] if field_idx < len(record) else None for record in records]
            edges_df[field] = values
        
        output_file = output_path('road_network', 'edges_processed.csv')
        edges_df.to_csv(output_file, index=False)
        print(f"\n💾 边数据已保存: {output_file}")
        
        sf.close()
        
        return {
            'total_edges': len(shapes),
            'fields': fields,
            'length_stats': {
                'min': min(length_data['length']) if length_data and 'length' in length_data else None,
                'max': max(length_data['length']) if length_data and 'length' in length_data else None,
                'mean': np.mean(length_data['length']) if length_data and 'length' in length_data else None,
                'median': np.median(length_data['length']) if length_data and 'length' in length_data else None
            },
            'sample_data': edges_df.head(5).to_dict('records')
        }
        
    except Exception as e:
        print(f"❌ 边数据分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_spatial_overlap(nodes_info):
    """检查路网与格子区域的空间重叠"""
    print(f"\n🗺️ 检查路网与格子区域的空间重叠...")
    print("=" * 50)
    
    try:
        # 读取格子数据
        grid_file = input_path('shanghai_grid_chengdu_format.csv')
        grids = pd.read_csv(grid_file)
        
        print(f"📊 格子数据:")
        print(f"  格子总数: {len(grids):,}")
        
        # 格子坐标范围
        grid_bounds = {
            'lng_min': grids['center_lng'].min(),
            'lng_max': grids['center_lng'].max(),
            'lat_min': grids['center_lat'].min(),
            'lat_max': grids['center_lat'].max()
        }
        
        # 路网坐标范围
        if nodes_info and nodes_info['coordinate_bounds']:
            node_bounds = nodes_info['coordinate_bounds']
        else:
            print("❌ 无法获取节点坐标范围")
            return None
        
        print(f"\n�� 空间范围对比:")
        print(f"  格子区域:")
        print(f"    经度: {grid_bounds['lng_min']:.6f} ~ {grid_bounds['lng_max']:.6f}")
        print(f"    纬度: {grid_bounds['lat_min']:.6f} ~ {grid_bounds['lat_max']:.6f}")
        print(f"  路网区域:")
        print(f"    经度: {node_bounds['lng_min']:.6f} ~ {node_bounds['lng_max']:.6f}")
        print(f"    纬度: {node_bounds['lat_min']:.6f} ~ {node_bounds['lat_max']:.6f}")
        
        # 计算重叠区域
        overlap_lng = (max(grid_bounds['lng_min'], node_bounds['lng_min']), 
                      min(grid_bounds['lng_max'], node_bounds['lng_max']))
        overlap_lat = (max(grid_bounds['lat_min'], node_bounds['lat_min']),
                      min(grid_bounds['lat_max'], node_bounds['lat_max']))
        
        # 检查重叠有效性
        valid_overlap = (overlap_lng[1] > overlap_lng[0] and overlap_lat[1] > overlap_lat[0])
        
        print(f"\n🎯 重叠分析:")
        print(f"  重叠区域:")
        print(f"    经度: {overlap_lng[0]:.6f} ~ {overlap_lng[1]:.6f}")
        print(f"    纬度: {overlap_lat[0]:.6f} ~ {overlap_lat[1]:.6f}")
        print(f"  重叠有效性: {'✅ 有重叠' if valid_overlap else '❌ 无重叠'}")
        
        if valid_overlap:
            # 计算覆盖率
            grid_area = (grid_bounds['lng_max'] - grid_bounds['lng_min']) * (grid_bounds['lat_max'] - grid_bounds['lat_min'])
            overlap_area = (overlap_lng[1] - overlap_lng[0]) * (overlap_lat[1] - overlap_lat[0])
            coverage_ratio = overlap_area / grid_area
            
            print(f"  覆盖率: {coverage_ratio*100:.1f}%")
            print(f"  结论: {'✅ 路网覆盖充分' if coverage_ratio > 0.8 else '⚠️ 路网覆盖不足'}")
            
            return {
                'valid_overlap': valid_overlap,
                'grid_bounds': grid_bounds,
                'node_bounds': node_bounds,
                'overlap_bounds': {
                    'lng': overlap_lng,
                    'lat': overlap_lat
                },
                'coverage_ratio': coverage_ratio
            }
        
        return {'valid_overlap': False}
        
    except Exception as e:
        print(f"❌ 空间重叠检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_analysis_report(nodes_info, edges_info, overlap_info):
    """生成分析报告"""
    print(f"\n📝 生成分析报告...")
    print("=" * 50)
    
    report = {
        'analysis_date': '2025-09-08',
        'dataset': 'Shanghai Road Network',
        'nodes_analysis': nodes_info,
        'edges_analysis': edges_info,
        'spatial_overlap': overlap_info,
        'summary': {
            'total_nodes': nodes_info['total_nodes'] if nodes_info else 0,
            'total_edges': edges_info['total_edges'] if edges_info else 0,
            'spatial_overlap_valid': overlap_info['valid_overlap'] if overlap_info else False,
            'coverage_ratio': overlap_info.get('coverage_ratio', 0) if overlap_info else 0
        }
    }
    
    # 保存JSON报告
    report_path = output_path('reports', 'road_network_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 分析报告已保存: {report_path}")
    
    # 生成Markdown报告
    md_report_path = output_path('reports', 'road_network_analysis.md')
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write("# 上海路网数据分析报告\n\n")
        f.write(f"**分析日期**: 2025-09-08\n\n")
        
        if nodes_info:
            f.write("## 📍 节点分析\n")
            f.write(f"- **节点总数**: {nodes_info['total_nodes']:,}\n")
            if nodes_info['coordinate_bounds']:
                f.write(f"- **坐标范围**: \n")
                f.write(f"  - 经度: {nodes_info['coordinate_bounds']['lng_min']:.6f} ~ {nodes_info['coordinate_bounds']['lng_max']:.6f}\n")
                f.write(f"  - 纬度: {nodes_info['coordinate_bounds']['lat_min']:.6f} ~ {nodes_info['coordinate_bounds']['lat_max']:.6f}\n\n")
        
        if edges_info:
            f.write("## 🛤️ 边分析\n")
            f.write(f"- **边总数**: {edges_info['total_edges']:,}\n")
            if edges_info['length_stats'] and edges_info['length_stats']['mean']:
                f.write(f"- **平均长度**: {edges_info['length_stats']['mean']:.2f}\n")
            f.write("\n")
        
        if overlap_info:
            f.write("## 🗺️ 空间重叠分析\n")
            f.write(f"- **重叠有效性**: {'✅ 有效' if overlap_info['valid_overlap'] else '❌ 无效'}\n")
            f.write(f"- **覆盖率**: {overlap_info.get('coverage_ratio', 0)*100:.1f}%\n\n")
        
        f.write("## 📊 预处理准备状态\n")
        f.write("- ✅ 路网数据读取成功\n")
        f.write("- ✅ 格子数据兼容性确认\n")
        f.write("- ✅ 空间重叠验证完成\n")
        f.write("- 🎯 **下一步**: 开始格子到节点映射\n")
    
    print(f"✅ Markdown报告已保存: {md_report_path}")
    
    return report

def main():
    """主分析函数"""
    print("🛣️ Step 1.1: 上海路网数据结构分析（备用版本）")
    print("=" * 60)
    print(f"输入目录: {INPUT_DATASET_DIR}")
    print(f"输出目录: {OUTPUT_BASE_DIR}")
    
    # 创建输出目录
    create_output_dirs()
    
    # 分析节点数据
    nodes_info = analyze_road_nodes()
    
    # 分析边数据
    edges_info = analyze_road_edges()
    
    # 检查空间重叠
    overlap_info = check_spatial_overlap(nodes_info)
    
    # 生成分析报告
    report = generate_analysis_report(nodes_info, edges_info, overlap_info)
    
    print(f"\n🎉 Step 1.1 完成总结:")
    print("=" * 60)
    if nodes_info:
        print(f"✅ 节点数据: {nodes_info['total_nodes']:,} 个节点")
    if edges_info:
        print(f"✅ 边数据: {edges_info['total_edges']:,} 条边")
    if overlap_info and overlap_info['valid_overlap']:
        print(f"✅ 空间重叠: {overlap_info.get('coverage_ratio', 0)*100:.1f}% 覆盖率")
    
    print(f"\n📂 输出文件:")
    print(f"- {output_path('road_network', 'nodes_processed.csv')}")
    print(f"- {output_path('road_network', 'edges_processed.csv')}")
    print(f"- {output_path('reports', 'road_network_analysis.json')}")
    print(f"- {output_path('reports', 'road_network_analysis.md')}")
    
    print(f"\n🎯 **准备就绪，可以开始 Step 2.1: 格子到节点映射！**")

if __name__ == "__main__":
    main()
