# 图谱探索 API (Graph Explorer API)

提供强大的知识图谱可视化和交互功能，支持实体关系查询、路径分析、社区发现等高级功能。

## 目录

- [概述](#概述)
- [API 端点](#api-端点)
- [使用示例](#使用示例)
- [数据格式](#数据格式)
- [前端集成](#前端集成)
- [性能优化](#性能优化)

## 概述

Graph Explorer API 复用了 `kg_service.py` 中经过验证的强大逻辑，提供以下核心功能：

1. **图谱概览** - 获取整体知识图谱
2. **子图查询** - 以实体为中心的局部图谱
3. **路径分析** - 实体间的最短路径和所有路径
4. **社区发现** - 实体所属的社区结构
5. **影响力分析** - 实体的影响范围
6. **原文溯源** - 查看原始文档片段

## API 端点

### 基础信息

**Base URL**: `http://localhost:8000/graph`

**认证**: 无需认证（开发环境）

**响应格式**: JSON

### 端点列表

#### 1. 获取图谱概览

```http
GET /graph/overview
```

获取知识图谱的整体视图。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| limit | int | 否 | 100 | 返回节点数（1-1000） |
| query | string | 否 | null | 搜索关键词 |

**示例请求**:
```bash
curl "http://localhost:8000/graph/overview?limit=50"
curl "http://localhost:8000/graph/overview?query=学生"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "nodes": [
      {
        "id": "国家奖学金",
        "label": "国家奖学金",
        "description": "最高荣誉的奖学金",
        "group": "奖学金类型"
      }
    ],
    "links": [
      {
        "source": "学生",
        "target": "国家奖学金",
        "label": "申请",
        "weight": 1
      }
    ]
  },
  "meta": {
    "node_count": 50,
    "link_count": 75
  }
}
```

---

#### 2. 获取实体子图

```http
GET /graph/subgraph
```

获取指定实体周围的局部图谱。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| entity_id | string | 是 | - | 中心实体ID |
| hops | int | 否 | 1 | 扩展跳数（1-3） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/subgraph?entity_id=国家奖学金&hops=2"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "nodes": [...],
    "links": [...],
    "influence_stats": {
      "direct_connections": 5,
      "total_connections": 12,
      "connection_types": [
        {"type": "申请", "count": 5},
        {"type": "评选", "count": 3}
      ]
    }
  },
  "meta": {
    "center_entity": "国家奖学金",
    "hops": 2,
    "node_count": 13,
    "link_count": 18
  }
}
```

---

#### 3. 获取实体详细信息

```http
GET /graph/entity/{entity_id}
```

获取单个实体的详细信息。

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| entity_id | string | 是 | 实体ID |

**查询参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| include_neighbors | bool | 否 | false | 是否包含邻居 |

**示例请求**:
```bash
curl "http://localhost:8000/graph/entity/国家奖学金?include_neighbors=true"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "entity": {
      "id": "国家奖学金",
      "description": "最高荣誉的奖学金，要求成绩优异",
      "labels": ["奖学金类型"],
      "properties": {
        "id": "国家奖学金",
        "description": "...",
        "amount": "8000元"
      }
    },
    "neighbors": {
      "nodes": [...],
      "links": [...],
      "stats": {
        "direct_connections": 5,
        "total_connections": 5
      }
    }
  }
}
```

---

#### 4. 查询最短路径

```http
GET /graph/shortest-path
```

查询两个实体之间的最短路径。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| source | string | 是 | - | 起始实体ID |
| target | string | 是 | - | 目标实体ID |
| max_hops | int | 否 | 3 | 最大跳数（1-5） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/shortest-path?source=学生&target=国家奖学金&max_hops=3"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "nodes": [
      {"id": "学生", "label": "学生", "group": "学生类型"},
      {"id": "成绩优秀", "label": "成绩优秀", "group": "条件"},
      {"id": "国家奖学金", "label": "国家奖学金", "group": "奖学金类型"}
    ],
    "links": [
      {"source": "学生", "target": "成绩优秀", "label": "要求"},
      {"source": "成绩优秀", "target": "国家奖学金", "label": "申请"}
    ],
    "path_info": "从 学生 到 国家奖学金 的最短路径",
    "path_length": 2
  },
  "meta": {
    "source": "学生",
    "target": "国家奖学金",
    "path_length": 2,
    "max_hops": 3
  }
}
```

---

#### 5. 查询所有路径

```http
GET /graph/path
```

查询两个实体之间的所有路径（最多10条）。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| source | string | 是 | - | 起始实体ID |
| target | string | 是 | - | 目标实体ID |
| max_depth | int | 否 | 3 | 最大深度（1-5） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/path?source=学生&target=国家奖学金&max_depth=3"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "nodes": [...],
    "links": [...],
    "paths_info": [
      "学生 -[要求]-> 成绩优秀 -> 成绩优秀 -[申请]-> 国家奖学金",
      "学生 -[参加]-> 学生会 -> 学生会 -[推荐]-> 国家奖学金"
    ],
    "path_count": 2
  },
  "meta": {
    "source": "学生",
    "target": "国家奖学金",
    "path_count": 2,
    "max_depth": 3
  }
}
```

---

#### 6. 查询共同邻居

```http
GET /graph/common-neighbors
```

查询两个实体的共同邻居。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| entity_a | string | 是 | 实体A的ID |
| entity_b | string | 是 | 实体B的ID |

**示例请求**:
```bash
curl "http://localhost:8000/graph/common-neighbors?entity_a=国家奖学金&entity_b=励志奖学金"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "nodes": [
      {"id": "国家奖学金", "label": "国家奖学金", "group": "Source"},
      {"id": "励志奖学金", "label": "励志奖学金", "group": "Target"},
      {"id": "学生", "label": "学生", "group": "学生类型"},
      {"id": "成绩优秀", "label": "成绩优秀", "group": "条件"}
    ],
    "links": [
      {"source": "国家奖学金", "target": "学生", "label": "连接"},
      {"source": "学生", "target": "励志奖学金", "label": "连接"}
    ],
    "common_neighbors": ["学生", "成绩优秀"],
    "neighbor_count": 2
  },
  "meta": {
    "entity_a": "国家奖学金",
    "entity_b": "励志奖学金",
    "neighbor_count": 2
  }
}
```

---

#### 7. 查询实体影响范围

```http
GET /graph/influence
```

分析实体的影响范围（N跳邻居）。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| entity_id | string | 是 | - | 实体ID |
| max_depth | int | 否 | 2 | 扩展深度（1-3） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/influence?entity_id=国家奖学金&max_depth=2"
```

---

#### 8. 查询实体社区

```http
GET /graph/community
```

查询实体所属的社区。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| entity_id | string | 是 | - | 实体ID |
| max_depth | int | 否 | 2 | 扩展深度（1-3） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/community?entity_id=国家奖学金&max_depth=2"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "nodes": [...],
    "links": [...],
    "communities": [
      {
        "id": 1,
        "size": 8,
        "density": 0.75,
        "contains_center": true,
        "sample_members": ["国家奖学金", "学生", "成绩优秀", "学生会", "奖学金评审"]
      }
    ],
    "community_count": 1,
    "entity_community": 1
  },
  "meta": {
    "entity_id": "国家奖学金",
    "max_depth": 2,
    "community_count": 1
  }
}
```

---

#### 9. 查询实体环路

```http
GET /graph/cycles
```

查询从实体出发又回到自身的环路。

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| entity_id | string | 是 | - | 实体ID |
| max_depth | int | 否 | 4 | 最大深度（1-4） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/cycles?entity_id=学生&max_depth=4"
```

---

#### 10. 获取原文片段

```http
GET /graph/source/chunk
```

获取文本块的原文内容。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| chunk_id | string | 是 | 文本块ID（SHA1） |

**示例请求**:
```bash
curl "http://localhost:8000/graph/source/chunk?chunk_id=abc123..."
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "chunk_id": "abc123...",
    "content": "文件名: 学生手册.pdf\n\n国家奖学金是最高荣誉的奖学金...",
    "file_name": "学生手册.pdf"
  }
}
```

---

#### 11. 获取文件信息

```http
GET /graph/source/file-info
```

获取源ID对应的文件信息。

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| source_id | string | 是 | 源ID |

---

#### 12. 获取图谱统计信息

```http
GET /graph/stats
```

获取知识图谱的统计信息。

**示例请求**:
```bash
curl "http://localhost:8000/graph/stats"
```

**示例响应**:
```json
{
  "status": "success",
  "data": {
    "total_entities": 245,
    "total_relationships": 387,
    "entity_types": [
      {"type": "学生类型", "count": 45},
      {"type": "奖学金类型", "count": 12},
      {"type": "处分类型", "count": 8}
    ],
    "relationship_types": [
      {"type": "申请", "count": 89},
      {"type": "评选", "count": 56},
      {"type": "管理", "count": 42}
    ]
  }
}
```

## 使用示例

### Python 示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 获取图谱概览
response = requests.get(f"{BASE_URL}/graph/overview", params={"limit": 50})
data = response.json()
print(f"节点数: {data['meta']['node_count']}")

# 2. 查询实体子图
response = requests.get(
    f"{BASE_URL}/graph/subgraph",
    params={"entity_id": "国家奖学金", "hops": 2}
)
subgraph = response.json()["data"]

# 3. 查询最短路径
response = requests.get(
    f"{BASE_URL}/graph/shortest-path",
    params={
        "source": "学生",
        "target": "国家奖学金",
        "max_hops": 3
    }
)
path = response.json()["data"]
print(f"路径长度: {path['path_length']}")

# 4. 获取图谱统计
response = requests.get(f"{BASE_URL}/graph/stats")
stats = response.json()["data"]
print(f"实体总数: {stats['total_entities']}")
print(f"关系总数: {stats['total_relationships']}")
```

### JavaScript 示例

```javascript
const BASE_URL = "http://localhost:8000";

// 1. 获取图谱概览
async function getGraphOverview() {
  const response = await fetch(`${BASE_URL}/graph/overview?limit=50`);
  const data = await response.json();
  return data.data;
}

// 2. 查询实体子图
async function getSubgraph(entityId, hops = 1) {
  const response = await fetch(
    `${BASE_URL}/graph/subgraph?entity_id=${entityId}&hops=${hops}`
  );
  const data = await response.json();
  return data.data;
}

// 3. 查询最短路径
async function getShortestPath(source, target) {
  const response = await fetch(
    `${BASE_URL}/graph/shortest-path?source=${source}&target=${target}`
  );
  const data = await response.json();
  return data.data;
}

// 使用示例
getSubgraph("国家奖学金", 2).then(subgraph => {
  console.log("节点数:", subgraph.nodes.length);
  console.log("边数:", subgraph.links.length);
});
```

## 数据格式

### 节点 (Node) 格式

```json
{
  "id": "国家奖学金",
  "label": "国家奖学金",
  "description": "最高荣誉的奖学金",
  "group": "奖学金类型"
}
```

**字段说明**:
- `id`: 唯一标识符
- `label`: 显示名称
- `description`: 描述信息
- `group`: 节点分组（用于可视化着色）

### 边 (Link) 格式

```json
{
  "source": "学生",
  "target": "国家奖学金",
  "label": "申请",
  "weight": 1
}
```

**字段说明**:
- `source`: 源节点ID
- `target`: 目标节点ID
- `label`: 关系类型
- `weight`: 边权重

## 前端集成

### 推荐可视化库

1. **D3.js** - 强大灵活的图可视化库
2. **Cytoscape.js** - 专业的图论库
3. **G6 (AntV)** - 蚂蚁集团图可视化引擎
4. **ECharts Graph** - 简单易用
5. **Vis.js Network** - 快速上手

### D3.js 集成示例

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    .node { stroke: #fff; stroke-width: 1.5px; }
    .link { stroke: #999; stroke-opacity: 0.6; }
  </style>
</head>
<body>
  <svg width="960" height="600"></svg>
  <script>
    // 获取数据
    fetch('http://localhost:8000/graph/overview?limit=100')
      .then(res => res.json())
      .then(data => {
        const graph = data.data;

        // 创建力导向图
        const simulation = d3.forceSimulation(graph.nodes)
          .force("link", d3.forceLink(graph.links).id(d => d.id))
          .force("charge", d3.forceManyBody())
          .force("center", d3.forceCenter(480, 300));

        const svg = d3.select("svg");

        // 绘制边
        const link = svg.append("g")
          .selectAll("line")
          .data(graph.links)
          .enter().append("line")
          .attr("class", "link");

        // 绘制节点
        const node = svg.append("g")
          .selectAll("circle")
          .data(graph.nodes)
          .enter().append("circle")
          .attr("class", "node")
          .attr("r", 5)
          .attr("fill", d => colorByGroup(d.group));

        // 节点标签
        const label = svg.append("g")
          .selectAll("text")
          .data(graph.nodes)
          .enter().append("text")
          .text(d => d.label)
          .attr("font-size", 10);

        // 更新位置
        simulation.on("tick", () => {
          link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

          node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

          label
            .attr("x", d => d.x + 8)
            .attr("y", d => d.y + 3);
        });
      });
  </script>
</body>
</html>
```

### React 集成示例

```jsx
import React, { useEffect, useState } from 'react';
import Graph from 'react-graph-vis';

function GraphExplorer({ entityId }) {
  const [graph, setGraph] = useState({ nodes: [], edges: [] });

  useEffect(() => {
    fetch(`http://localhost:8000/graph/subgraph?entity_id=${entityId}&hops=2`)
      .then(res => res.json())
      .then(data => {
        const { nodes, links } = data.data;

        setGraph({
          nodes: nodes.map(n => ({
            id: n.id,
            label: n.label,
            title: n.description,
            group: n.group
          })),
          edges: links.map(l => ({
            from: l.source,
            to: l.target,
            label: l.label
          }))
        });
      });
  }, [entityId]);

  const options = {
    layout: { hierarchical: false },
    edges: { color: "#000000" },
    nodes: { shape: "dot", size: 16 }
  };

  return <Graph graph={graph} options={options} />;
}
```

## 性能优化

### 1. 限制返回数据量

```python
# 不推荐：获取全部数据
response = requests.get(f"{BASE_URL}/graph/overview?limit=1000")

# 推荐：按需获取
response = requests.get(f"{BASE_URL}/graph/overview?limit=100")
```

### 2. 使用子图而非全图

```python
# 不推荐：加载整个图谱
full_graph = requests.get(f"{BASE_URL}/graph/overview?limit=1000")

# 推荐：只加载需要的子图
subgraph = requests.get(
    f"{BASE_URL}/graph/subgraph",
    params={"entity_id": "国家奖学金", "hops": 1}
)
```

### 3. 控制跳数

```python
# 跳数越大，返回数据越多
# 1 跳: 适合详情页 (~10-50 节点)
# 2 跳: 适合局部探索 (~50-200 节点)
# 3 跳: 适合深度分析 (~200-1000 节点)

response = requests.get(
    f"{BASE_URL}/graph/subgraph",
    params={"entity_id": "国家奖学金", "hops": 1}  # 从小开始
)
```

### 4. 缓存结果

```python
import functools
from datetime import datetime, timedelta

@functools.lru_cache(maxsize=100)
def get_cached_subgraph(entity_id, hops):
    response = requests.get(
        f"{BASE_URL}/graph/subgraph",
        params={"entity_id": entity_id, "hops": hops}
    )
    return response.json()
```

## 错误处理

### 常见错误码

| 状态码 | 说明 | 处理方式 |
|--------|------|----------|
| 404 | 实体不存在 | 检查实体ID是否正确 |
| 500 | 服务器错误 | 查看服务器日志 |
| 400 | 参数错误 | 检查参数格式 |

### 错误响应格式

```json
{
  "detail": "实体 '不存在的实体' 不存在"
}
```

### 错误处理示例

```python
try:
    response = requests.get(
        f"{BASE_URL}/graph/subgraph",
        params={"entity_id": "不存在的实体", "hops": 1}
    )
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print(f"实体不存在: {e.response.json()['detail']}")
    else:
        print(f"请求失败: {e}")
```

## 最佳实践

1. **渐进式加载**: 先加载概览，点击节点后再加载详细信息
2. **按需查询**: 只查询用户当前需要的数据
3. **结果缓存**: 缓存常用查询结果
4. **错误提示**: 友好的错误提示信息
5. **加载状态**: 显示加载动画
6. **参数验证**: 前端验证参数范围
7. **降级策略**: 查询失败时显示缓存数据

## 交互式 API 文档

启动服务后，访问以下地址查看交互式 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

可以直接在浏览器中测试所有API端点。

## 相关资源

- [kg_service.py 源码](../services/kg_service.py)
- [Neo4j Cypher 文档](https://neo4j.com/docs/cypher-manual/current/)
- [D3.js Force Layout](https://d3js.org/d3-force)
- [Cytoscape.js 文档](https://js.cytoscape.org/)

## 常见问题

### Q: 如何获取某个实体的完整邻居？

使用 `/graph/influence` 端点：

```bash
curl "http://localhost:8000/graph/influence?entity_id=国家奖学金&max_depth=1"
```

### Q: 如何查找两个实体的关系？

使用 `/graph/shortest-path` 或 `/graph/common-neighbors`：

```bash
curl "http://localhost:8000/graph/common-neighbors?entity_a=学生&entity_b=国家奖学金"
```

### Q: 如何限制返回的数据量？

使用 `limit` 参数或降低 `hops`/`max_depth`：

```bash
curl "http://localhost:8000/graph/overview?limit=50"
```

### Q: 前端如何实现节点点击交互？

```javascript
// 点击节点时加载其子图
node.on("click", async (event) => {
  const entityId = event.target.data("id");
  const subgraph = await fetch(
    `http://localhost:8000/graph/subgraph?entity_id=${entityId}&hops=1`
  ).then(res => res.json());

  // 更新图谱显示
  updateGraph(subgraph.data);
});
```
