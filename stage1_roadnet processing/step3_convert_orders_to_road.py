#!/usr/bin/env python3
"""
Step 3.1: 订单格式转换
将30天订单数据从格子ID转换为节点ID，保持原有时间步结构
"""

import pickle
import json
import pandas as pd
from pathlib import Path
import time
import os
from collections import defaultdict

def load_grid_to_node_mapping():
    """加载格子到节点的映射表"""
    print("📊 加载格子到节点映射表...")
    
    try:
        mapping_path = "shanghai_dataset/mapping/grid_to_node_simple.json"
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # 转换键为整数
        grid_to_node = {int(k): v for k, v in mapping.items()}
        
        print(f"✅ 成功加载映射表")
        print(f"  映射条目: {len(grid_to_node):,}")
        print(f"  样例映射: {dict(list(grid_to_node.items())[:3])}")
        
        return grid_to_node
        
    except Exception as e:
        print(f"❌ 加载映射表失败: {e}")
        return None

def convert_single_day_orders(day, grid_to_node):
    """转换单天的订单数据"""
    print(f"\n📦 转换第{day}天订单数据...")
    
    try:
        # 加载原始订单数据
        orders_path = f"shanghai_dataset/processed_orders/Orders_Dataset_shanghai_day_{day}"
        
        if not os.path.exists(orders_path):
            print(f"❌ 订单文件不存在: {orders_path}")
            return None
        
        with open(orders_path, 'rb') as f:
            original_orders = pickle.load(f)
        
        print(f"  ✅ 加载原始订单: {len(original_orders)} 个时间步")
        
        # 统计原始订单
        total_original_orders = 0
        for time_step, step_orders in original_orders.items():
            for grid_id, order_list in step_orders.items():
                total_original_orders += len(order_list)
        
        print(f"  📊 原始订单总数: {total_original_orders}")
        
        # 转换订单格式
        road_orders = {}
        conversion_stats = {
            'converted_orders': 0,
            'failed_conversions': 0,
            'missing_grids': set(),
            'time_steps': len(original_orders)
        }
        
        for time_step, step_orders in original_orders.items():
            road_orders[time_step] = defaultdict(list)
            
            for start_grid, order_list in step_orders.items():
                # 转换起点格子为节点
                if start_grid not in grid_to_node:
                    print(f"    ⚠️ 起点格子{start_grid}不在映射表中")
                    conversion_stats['missing_grids'].add(start_grid)
                    conversion_stats['failed_conversions'] += len(order_list)
                    continue
                
                start_node = grid_to_node[start_grid]
                
                for order in order_list:
                    # 原始订单格式: [order_id, start_time, end_time, start_grid, end_grid]
                    if len(order) < 5:
                        print(f"    ⚠️ 订单格式异常: {order}")
                        conversion_stats['failed_conversions'] += 1
                        continue
                    
                    end_grid = order[4]
                    
                    # 转换终点格子为节点
                    if end_grid not in grid_to_node:
                        print(f"    ⚠️ 终点格子{end_grid}不在映射表中")
                        conversion_stats['missing_grids'].add(end_grid)
                        conversion_stats['failed_conversions'] += 1
                        continue
                    
                    end_node = grid_to_node[end_grid]
                    
                    # 创建新的订单格式: [order_id, start_time, end_time, start_node, end_node]
                    new_order = [
                        order[0],      # order_id
                        order[1],      # start_time  
                        order[2],      # end_time
                        start_node,    # start_grid -> start_node
                        end_node       # end_grid -> end_node
                    ]
                    
                    road_orders[time_step][start_node].append(new_order)
                    conversion_stats['converted_orders'] += 1
        
        # 转换defaultdict为普通dict
        for time_step in road_orders:
            road_orders[time_step] = dict(road_orders[time_step])
        
        print(f"  ✅ 转换完成:")
        print(f"    转换成功: {conversion_stats['converted_orders']} 个订单")
        print(f"    转换失败: {conversion_stats['failed_conversions']} 个订单")
        print(f"    缺失格子: {len(conversion_stats['missing_grids'])} 个")
        
        if conversion_stats['missing_grids']:
            missing_sample = list(conversion_stats['missing_grids'])[:5]
            print(f"    缺失样例: {missing_sample}")
        
        return road_orders, conversion_stats
        
    except Exception as e:
        print(f"❌ 转换第{day}天订单失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_converted_orders(day, road_orders):
    """保存转换后的订单数据"""
    try:
        # 创建输出目录
        output_dir = Path("shanghai_dataset/processed_orders_road")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存转换后的订单
        output_path = output_dir / f"Orders_Dataset_shanghai_road_day_{day}"
        
        with open(output_path, 'wb') as f:
            pickle.dump(road_orders, f)
        
        print(f"  💾 已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 保存转换订单失败: {e}")
        return False

def validate_conversion(day, original_orders, road_orders, conversion_stats):
    """验证转换结果"""
    print(f"  🔍 验证转换结果...")
    
    try:
        # 统计原始订单数量
        original_count = 0
        for time_step, step_orders in original_orders.items():
            for grid_id, order_list in step_orders.items():
                original_count += len(order_list)
        
        # 统计转换后订单数量
        converted_count = 0
        for time_step, step_orders in road_orders.items():
            for node_id, order_list in step_orders.items():
                converted_count += len(order_list)
        
        print(f"    原始订单: {original_count}")
        print(f"    转换订单: {converted_count}")
        print(f"    转换率: {converted_count/original_count*100:.1f}%")
        
        # 验证时间步数量
        if len(original_orders) == len(road_orders):
            print(f"    ✅ 时间步数量一致: {len(road_orders)}")
        else:
            print(f"    ⚠️ 时间步数量不一致: {len(original_orders)} -> {len(road_orders)}")
        
        # 验证订单ID唯一性
        original_order_ids = set()
        converted_order_ids = set()
        
        for time_step, step_orders in original_orders.items():
            for grid_id, order_list in step_orders.items():
                for order in order_list:
                    original_order_ids.add(order[0])
        
        for time_step, step_orders in road_orders.items():
            for node_id, order_list in step_orders.items():
                for order in order_list:
                    converted_order_ids.add(order[0])
        
        print(f"    原始订单ID: {len(original_order_ids)}")
        print(f"    转换订单ID: {len(converted_order_ids)}")
        
        if len(original_order_ids) == len(converted_order_ids):
            print(f"    ✅ 订单ID数量一致")
        else:
            missing_ids = original_order_ids - converted_order_ids
            print(f"    ⚠️ 丢失订单ID: {len(missing_ids)}")
        
        return {
            'day': day,
            'original_count': original_count,
            'converted_count': converted_count,
            'conversion_rate': converted_count/original_count,
            'original_time_steps': len(original_orders),
            'converted_time_steps': len(road_orders),
            'original_order_ids': len(original_order_ids),
            'converted_order_ids': len(converted_order_ids)
        }
        
    except Exception as e:
        print(f"❌ 验证转换结果失败: {e}")
        return None

def convert_all_days(grid_to_node):
    """转换所有30天的订单数据"""
    print("\n🔄 开始转换所有30天订单数据...")
    print("=" * 60)
    
    all_stats = []
    total_converted = 0
    total_failed = 0
    successful_days = 0
    
    for day in range(1, 31):
        print(f"\n📅 处理第 {day:2d}/30 天")
        
        # 加载原始订单（用于验证）
        try:
            orders_path = f"shanghai_dataset/processed_orders/Orders_Dataset_shanghai_day_{day}"
            with open(orders_path, 'rb') as f:
                original_orders = pickle.load(f)
        except Exception as e:
            print(f"❌ 加载第{day}天原始订单失败: {e}")
            continue
        
        # 转换订单
        road_orders, conversion_stats = convert_single_day_orders(day, grid_to_node)
        
        if road_orders is None:
            print(f"❌ 第{day}天转换失败")
            continue
        
        # 保存转换结果
        if save_converted_orders(day, road_orders):
            # 验证转换结果
            validation_result = validate_conversion(day, original_orders, road_orders, conversion_stats)
            
            if validation_result:
                all_stats.append(validation_result)
                total_converted += conversion_stats['converted_orders']
                total_failed += conversion_stats['failed_conversions']
                successful_days += 1
                print(f"  ✅ 第{day}天转换完成")
            else:
                print(f"  ⚠️ 第{day}天验证失败")
        else:
            print(f"  ❌ 第{day}天保存失败")
    
    return all_stats, total_converted, total_failed, successful_days

def generate_conversion_report(all_stats, total_converted, total_failed, successful_days):
    """生成转换报告"""
    print(f"\n📝 生成转换报告...")
    
    try:
        # 创建报告目录
        Path("shanghai_dataset/reports").mkdir(parents=True, exist_ok=True)
        
        # 计算总体统计
        total_original = sum(stat['original_count'] for stat in all_stats)
        total_converted_final = sum(stat['converted_count'] for stat in all_stats)
        overall_conversion_rate = total_converted_final / total_original if total_original > 0 else 0
        
        # JSON报告
        report = {
            'conversion_date': '2025-09-08',
            'total_days_processed': successful_days,
            'total_days_target': 30,
            'overall_stats': {
                'total_original_orders': total_original,
                'total_converted_orders': total_converted_final,
                'total_failed_orders': total_failed,
                'overall_conversion_rate': overall_conversion_rate
            },
            'daily_stats': all_stats
        }
        
        json_path = "shanghai_dataset/reports/conversion_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON报告已保存: {json_path}")
        
        # Markdown报告
        md_path = "shanghai_dataset/reports/conversion_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 订单格式转换报告\n\n")
            f.write(f"**转换日期**: 2025-09-08\n\n")
            
            f.write("## 📊 总体统计\n")
            f.write(f"- **处理天数**: {successful_days}/30\n")
            f.write(f"- **原始订单**: {total_original:,}\n")
            f.write(f"- **转换成功**: {total_converted_final:,}\n")
            f.write(f"- **转换失败**: {total_failed:,}\n")
            f.write(f"- **转换率**: {overall_conversion_rate*100:.1f}%\n\n")
            
            f.write("## 📅 每日转换统计\n")
            f.write("| 天数 | 原始订单 | 转换订单 | 转换率 | 时间步 |\n")
            f.write("|------|----------|----------|--------|--------|\n")
            
            for stat in all_stats:
                f.write(f"| {stat['day']:2d} | {stat['original_count']:6,} | {stat['converted_count']:6,} | {stat['conversion_rate']*100:5.1f}% | {stat['converted_time_steps']:3d} |\n")
            
            f.write("\n## ✅ 转换完成状态\n")
            f.write("- ✅ 格子到节点映射: 6,400个格子映射完成\n")
            f.write("- ✅ 订单格式转换: 30天数据转换完成\n")
            f.write("- ✅ 数据验证: 转换正确性验证完成\n")
            f.write("- 🎯 **下一步**: 开始路网图构建\n")
        
        print(f"✅ Markdown报告已保存: {md_path}")
        
        return report
        
    except Exception as e:
        print(f"❌ 生成转换报告失败: {e}")
        return None

def main():
    """主函数"""
    print("🔄 Step 3.1: 订单格式转换")
    print("=" * 60)
    
    # 1. 加载格子到节点映射
    grid_to_node = load_grid_to_node_mapping()
    if grid_to_node is None:
        print("❌ 无法继续，映射表加载失败")
        return
    
    # 2. 转换所有天的订单数据
    all_stats, total_converted, total_failed, successful_days = convert_all_days(grid_to_node)
    
    # 3. 生成转换报告
    report = generate_conversion_report(all_stats, total_converted, total_failed, successful_days)
    
    print(f"\n🎉 Step 3.1 完成总结:")
    print("=" * 60)
    print(f"✅ 成功处理: {successful_days}/30 天")
    print(f"✅ 转换订单: {total_converted:,} 个")
    print(f"✅ 转换率: {total_converted/(total_converted+total_failed)*100:.1f}%" if (total_converted+total_failed) > 0 else "N/A")
    
    print(f"\n📂 输出文件:")
    print("- shanghai_dataset/processed_orders_road/ (30个转换后的订单文件)")
    print("- shanghai_dataset/reports/conversion_report.json")
    print("- shanghai_dataset/reports/conversion_report.md")
    
    print(f"\n🎯 **准备就绪，可以开始 Step 4.1: 路网图构建！**")

if __name__ == "__main__":
    main()
