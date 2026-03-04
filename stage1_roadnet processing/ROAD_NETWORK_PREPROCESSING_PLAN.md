# 🛣️ 上海路网集成预处理计划

## 📋 预处理目标
将上海数据集从**格子导航系统**扩展为**格子+路网混合导航系统**：
- 🚁 **UAV**: 继续使用格子系统（欧氏距离）
- 🚛 **地面载体**: 改用路网系统（图最短路径 + GraphSAGE）

---

## 🔍 第一阶段：数据探索与验证

### Step 1.1: 路网数据结构分析
```python
# 文件: analyze_road_network_structure.py
```
**目标**: 
- 读取 `shanghai_dataset/nodes.shp` 和 `shanghai_dataset/edges.shp`
- 分析节点数量、字段结构、坐标范围
- 分析边的连接关系、道路类型、长度分布
- 验证与格子区域的空间重叠

**输出**:
- 节点数据报告（数量、坐标范围、字段说明）
- 边数据报告（数量、类型分布、长度统计）
- 空间重叠分析（格子区域vs路网覆盖）

### Step 1.2: 格子-路网空间关系验证
```python
# 文件: validate_spatial_overlap.py
```
**目标**:
- 确认6400个格子与路网节点的空间对应关系
- 识别无路网覆盖的格子（如果有）
- 计算格子密度vs节点密度

**输出**:
- 空间重叠报告
- 问题格子列表（无路网覆盖）
- 格子-节点密度对比

---

## 🗺️ 第二阶段：格子到节点映射

### Step 2.1: 最近邻映射算法实现
```python
# 文件: grid_to_node_mapping.py
```
**目标**:
- 为每个格子中心点找到最近的路网节点
- 使用KD-Tree优化搜索效率
- 生成格子ID → 节点ID映射表

**算法**:
```python
def create_grid_to_node_mapping():
    # 1. 读取格子中心坐标
    grid_centers = load_grid_centers()  # 6400个点
    
    # 2. 读取路网节点坐标  
    road_nodes = load_road_nodes()      # ~590K个点
    
    # 3. 构建KD-Tree空间索引
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(road_nodes[['lng', 'lat']])
    
    # 4. 批量查找最近邻
    distances, indices = nn_model.kneighbors(grid_centers[['lng', 'lat']])
    
    # 5. 生成映射表
    mapping = {
        grid_id: road_nodes.iloc[indices[i][0]]['node_id']
        for i, grid_id in enumerate(grid_centers['grid_id'])
    }
    
    return mapping
```

**输出**:
- `grid_to_node_mapping.json`: 完整映射表
- `mapping_statistics.txt`: 映射质量报告
- `mapping_visualization.png`: 空间分布可视化

### Step 2.2: 映射质量验证
```python
# 文件: validate_mapping_quality.py
```
**目标**:
- 统计映射距离分布
- 识别异常映射（距离过远）
- 验证映射的合理性

**验证指标**:
- 平均映射距离
- 最大映射距离
- 距离分布直方图
- 异常案例分析

---

## 🔄 第三阶段：订单数据转换

### Step 3.1: 订单格式转换
```python
# 文件: convert_orders_to_road.py
```
**目标**:
- 将30天订单数据从格子ID转换为节点ID
- 保持原有时间步结构
- 验证转换正确性

**转换流程**:
```python
def convert_orders_to_road():
    # 1. 加载映射表
    grid_to_node = load_mapping('grid_to_node_mapping.json')
    
    # 2. 处理每天的订单数据
    for day in range(1, 31):
        orders = load_orders(f'Orders_Dataset_shanghai_day_{day}')
        road_orders = {}
        
        # 3. 转换每个时间步的订单
        for time_step, step_orders in orders.items():
            road_orders[time_step] = {}
            
            for start_grid, order_list in step_orders.items():
                # 转换起点格子为节点
                start_node = grid_to_node[start_grid]
                road_orders[time_step][start_node] = []
                
                for order in order_list:
                    # [order_id, start_time, end_time, start_grid, end_grid]
                    # 转换为: [order_id, start_time, end_time, start_node, end_node]
                    new_order = [
                        order[0],  # order_id
                        order[1],  # start_time  
                        order[2],  # end_time
                        grid_to_node[order[3]],  # start_grid -> start_node
                        grid_to_node[order[4]]   # end_grid -> end_node
                    ]
                    road_orders[time_step][start_node].append(new_order)
        
        # 4. 保存转换后的数据
        save_orders(road_orders, f'Orders_Dataset_shanghai_road_day_{day}')
```

**输出**:
- `shanghai_dataset/processed_orders_road/`: 30个转换后的订单文件
- `conversion_report.txt`: 转换统计报告
- `conversion_validation.txt`: 数据完整性验证

### Step 3.2: 转换验证与统计
```python
# 文件: validate_conversion.py
```
**目标**:
- 验证转换前后订单数量一致性
- 检查节点ID有效性
- 生成转换质量报告

---

## 🗂️ 第四阶段：路网图构建

### Step 4.1: 路网图数据结构
```python
# 文件: build_road_graph.py
```
**目标**:
- 将shapefile转换为NetworkX图结构
- 添加节点特征（坐标、度数等）
- 添加边特征（长度、道路类型等）

**图结构**:
```python
import networkx as nx

def build_road_graph():
    G = nx.Graph()
    
    # 添加节点
    for node in nodes_data:
        G.add_node(node['osmid'], 
                  lng=node['lng'],
                  lat=node['lat'],
                  highway_type=node.get('highway', 'unknown'))
    
    # 添加边
    for edge in edges_data:
        G.add_edge(edge['u'], edge['v'],
                  length=edge['length'],
                  highway=edge.get('highway', 'unknown'))
    
    return G
```

### Step 4.2: 最短路径预计算
```python
# 文件: precompute_shortest_paths.py
```
**目标**:
- 为常用节点对预计算最短路径
- 构建距离矩阵或索引
- 优化运行时查询效率

---

## 🧠 第五阶段：GraphSAGE准备

### Step 5.1: 节点特征工程
```python
# 文件: extract_node_features.py
```
**目标**:
- 提取节点基础特征（经纬度、度数）
- 编码道路类型特征
- 计算邻域统计特征

**特征设计**:
```python
node_features = [
    'longitude',      # 经度
    'latitude',       # 纬度  
    'degree',         # 节点度数
    'highway_primary',     # 主要道路 (0/1)
    'highway_secondary',   # 次要道路 (0/1) 
    'highway_residential', # 住宅道路 (0/1)
    'neighborhood_density', # 邻域密度
    'avg_edge_length'      # 平均边长度
]
```

### Step 5.2: GraphSAGE模型框架
```python
# 文件: graphsage_model.py
```
**目标**:
- 实现GraphSAGE网络结构
- 定义聚合函数和更新函数
- 准备训练接口

---

## 📊 第六阶段：集成验证

### Step 6.1: 数据完整性检查
```python
# 文件: final_data_validation.py
```
**验证项目**:
- ✅ 格子映射覆盖率：6400/6400
- ✅ 订单转换正确性：20,664订单完整转换
- ✅ 路网连通性：主要区域连通
- ✅ 节点特征完整性：无缺失值

### Step 6.2: 性能基准测试
```python
# 文件: benchmark_road_distance.py
```
**目标**:
- 对比格子距离vs路网距离
- 测量路径查询性能
- 评估GraphSAGE推理速度

---

## 📅 实施时间表

### 本周（第1周）
- ✅ **周一-周二**: Step 1.1-1.2 数据探索
- ✅ **周三-周四**: Step 2.1-2.2 格子映射
- ✅ **周五**: Step 3.1 订单转换

### 下周（第2周）  
- 🎯 **周一-周二**: Step 3.2-4.1 转换验证+图构建
- 🎯 **周三-周四**: Step 4.2-5.1 路径预计算+特征工程
- 🎯 **周五**: Step 5.2 GraphSAGE模型

### 第3周
- 🎯 **周一-周三**: Step 6.1-6.2 集成验证
- 🎯 **周四-周五**: 与MADDPG集成测试

---

## 🎯 预期输出文件结构

```
shanghai_dataset/
├── road_network/
│   ├── nodes_processed.csv              # 处理后的节点数据
│   ├── edges_processed.csv              # 处理后的边数据
│   ├── road_graph.pkl                   # NetworkX图对象
│   └── node_features.npy                # 节点特征矩阵
├── mapping/
│   ├── grid_to_node_mapping.json        # 格子→节点映射
│   ├── mapping_statistics.txt           # 映射质量报告
│   └── spatial_analysis.png             # 空间分布可视化
├── processed_orders_road/               # 路网格式订单
│   ├── Orders_Dataset_shanghai_road_day_1
│   ├── Orders_Dataset_shanghai_road_day_2
│   └── ... (30个文件)
└── reports/
    ├── road_network_analysis.md         # 路网分析报告
    ├── conversion_report.md             # 转换报告
    └── validation_report.md             # 验证报告
```

---

## ❓ 需要确认的关键问题

1. **映射策略**: 使用最近邻映射是否合理？是否需要考虑道路可达性？

2. **GraphSAGE集成点**: 是否在距离计算时就集成GraphSAGE，还是在训练时集成？

3. **混合系统**: UAV和地面载体的交接机制如何设计？

4. **性能权衡**: 图最短路径计算vs欧氏距离的性能差异是否可接受？

5. **特征选择**: 除了位置和道路类型，还需要哪些节点特征？

---

请确认这个预处理计划是否符合您的预期，我们可以立即开始实施！
