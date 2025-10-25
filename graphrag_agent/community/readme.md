# 社区检测与摘要模块

## 文件结构

```
graphrag_agent/community/
├── __init__.py                    # 模块入口，导出工厂类
├── readme.md                      # 模块说明文档
├── detector/                      # 社区检测器目录
│   ├── __init__.py                # 检测器工厂类
│   ├── base.py                    # 基础检测器抽象类
│   ├── leiden.py                  # Leiden算法实现
│   ├── projections.py             # 图投影混入类
│   └── sllpa.py                   # SLLPA算法实现
└── summary/                       # 社区摘要目录
    ├── __init__.py                # 摘要工厂类
    ├── base.py                    # 基础摘要生成器抽象类
    ├── leiden.py                  # Leiden社区摘要实现
    └── sllpa.py                   # SLLPA社区摘要实现
```

## 模块概述

本模块为基于 Neo4j 图数据库的社区检测与摘要功能提供支持，是 GraphRAG 知识图谱项目的核心组件之一。主要功能包括：

1. **社区检测**：利用图社区发现算法识别知识图谱中的聚类结构
2. **社区摘要**：基于 LLM 为每个社区生成语义化的摘要描述，用于全局搜索场景

### 应用场景

- **全局搜索（Global Search）**：社区摘要提供高层次的语义检索入口
- **知识图谱分析**：揭示实体间的隐式聚类关系
- **主题发现**：自动识别文档集合中的主题领域
- **推荐系统**：基于社区结构进行相似实体推荐

## 设计思路与实现

### 设计模式

本模块采用多种设计模式确保代码的可维护性和可扩展性：

1. **工厂模式**：通过 `CommunityDetectorFactory` 和 `CommunitySummarizerFactory` 创建不同类型的检测器和摘要生成器，隐藏实现细节，便于算法切换
2. **混入类（Mixin）**：使用 `GraphProjectionMixin` 提供共享的图投影功能，避免代码重复
3. **上下文管理器**：在 `BaseCommunityDetector` 中使用 `_graph_projection_context` 管理 GDS 投影资源的生命周期，确保资源正确释放
4. **模板方法模式**：在基类中定义算法骨架，由子类实现具体步骤，保证流程一致性
5. **策略模式**：通过可配置的算法类型（leiden/sllpa）实现不同的社区检测策略

### 核心组件与流程

#### 1. 社区检测

**核心类**：`BaseCommunityDetector`
**实现算法**：
- **Leiden 算法** (`LeidenDetector`)：基于模块度优化的分层聚类算法，适用于大规模图
- **SLLPA 算法** (`SLLPADetector`)：Speaker-Listener Label Propagation Algorithm，基于标签传播，适合检测重叠社区

**关键流程**：
1. **图投影**：通过 `create_projection()` 将 Neo4j 原生图投影到 GDS（Graph Data Science）库的内存图结构
   - 支持节点标签过滤（仅处理实体节点）
   - 支持关系类型过滤
   - 包含三级降级策略：标准模式 → 过滤模式 → 保守模式
2. **社区检测**：执行 `detect_communities()` 调用特定算法识别社区结构
   - Leiden：优化模块度、分辨率、随机种子等参数
   - SLLPA：自动回退到 Leiden（如果未检测到社区）
3. **结果保存**：通过 `save_communities()` 将社区 ID 持久化到图数据库节点属性
   - 属性名：`leidenCommunity` 或 `sllpaCommunity`
   - 批量写入优化
4. **资源清理**：使用 `cleanup()` 释放 GDS 投影占用的内存资源

**自适应优化**：
- **资源感知**：根据系统可用内存（通过 `GDS_MEMORY_LIMIT` 配置）自动调整算法参数
- **并发控制**：通过 `GDS_CONCURRENCY` 环境变量控制 GDS 操作的并行度
- **错误恢复**：多层错误处理和备用方案（如 SLLPA → Leiden 自动回退）
- **性能监控**：记录投影时间、检测时间、写入时间等统计数据

#### 2. 社区摘要

**核心类**：`BaseSummarizer`
**辅助组件**：
- `BaseCommunityDescriber`：负责生成社区的自然语言描述
- `BaseCommunityRanker`：计算社区重要性排名
- `BaseCommunityStorer`：将摘要结果持久化到图数据库

**关键流程**：
1. **社区排名**：通过 `calculate_ranks()` 计算社区重要性
   - 基于社区规模（节点数量）
   - 基于社区内关系密度
   - 支持自定义排名策略
2. **信息收集**：通过 `collect_community_info()` 批量获取社区内容
   - 收集社区内所有实体节点及其属性
   - 收集社区内所有关系及其属性
   - 分批处理大规模社区，避免内存溢出
3. **摘要生成**：使用 LLM 模型为每个社区生成语义摘要
   - 结构化提示词设计（实体列表 + 关系列表 → 摘要）
   - 支持自定义摘要长度和风格
   - 并行处理多个社区，提升生成效率
4. **结果存储**：将摘要信息保存回图数据库
   - 创建或更新社区摘要节点（`CommunitySummary` 标签）
   - 建立社区与摘要的关联关系
   - 记录摘要生成时间戳和元数据

**性能优化**：
- **并行处理**：利用 `ThreadPoolExecutor` 多线程生成摘要，并发度可通过 `MAX_WORKERS` 配置
- **分批处理**：对大规模社区数据分批获取（批量大小：`BATCH_SIZE`），降低单次查询负载
- **缓存机制**：避免重复生成已存在的社区摘要（可选）
- **性能统计**：记录排名计算、信息收集、摘要生成、存储各阶段耗时

## 算法选择指南

### Leiden vs SLLPA

| 特性 | Leiden | SLLPA |
|------|--------|-------|
| **算法类型** | 基于模块度优化的分层聚类 | 基于标签传播的社区检测 |
| **适用场景** | 大规模图、清晰的社区边界 | 重叠社区、动态图 |
| **时间复杂度** | O(n log n) | O(m + n) |
| **社区类型** | 非重叠社区 | 可检测重叠社区 |
| **参数敏感度** | 中等（分辨率参数影响社区粒度）| 较高（迭代次数、阈值）|
| **稳定性** | 高（确定性结果） | 中等（可能需要多次运行）|
| **推荐使用** | 默认选择，适合大多数场景 | 需要检测重叠社区时 |

**配置方式**：
```bash
# 在 .env 文件中设置
GRAPH_COMMUNITY_ALGORITHM=leiden  # 或 sllpa
```

或在 `graphrag_agent/config/settings.py` 中设置：
```python
community_algorithm = "leiden"  # 或 "sllpa"
```

## 配置参数说明

### 社区检测相关

```bash
# GDS 内存限制（GB）
GDS_MEMORY_LIMIT=6

# GDS 并发度
GDS_CONCURRENCY=4

# 社区检测算法
GRAPH_COMMUNITY_ALGORITHM=leiden

# Leiden 特定参数（可选，使用默认值即可）
# LEIDEN_MAX_LEVELS=10
# LEIDEN_GAMMA=1.0
# LEIDEN_THETA=0.01
```

### 社区摘要相关

```bash
# 摘要生成并发度
MAX_WORKERS=4

# 社区信息批量获取大小
BATCH_SIZE=100

# LLM 配置（用于生成摘要）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=http://localhost:13000/v1
OPENAI_LLM_MODEL=gpt-4o
```

## 核心函数

### 社区检测模块

**`BaseCommunityDetector.process()`**

执行完整的社区检测流程，包括投影、检测和保存。

```python
def process(self) -> Dict[str, Any]:
    """
    执行完整的社区检测流程

    Returns:
        Dict: 包含统计信息的字典
            - community_count: 检测到的社区数量
            - node_count: 处理的节点数量
            - projection_time: 图投影耗时（秒）
            - detection_time: 社区检测耗时（秒）
            - save_time: 结果保存耗时（秒）
            - total_time: 总耗时（秒）
    """
```

**`GraphProjectionMixin.create_projection()`**

创建图投影，支持多种降级策略应对内存或数据问题。

```python
def create_projection(self) -> Tuple[Any, Dict]:
    """
    创建图投影，支持标准、过滤和保守多种模式

    降级策略：
    1. 标准模式：投影所有节点和关系
    2. 过滤模式：仅投影实体节点（Entity 标签）
    3. 保守模式：使用 Cypher 投影，最小化内存占用

    Returns:
        Tuple[Any, Dict]: (GDS 图对象, 投影统计信息)
    """
```

**`LeidenDetector.detect_communities()`**

执行 Leiden 算法社区检测，含参数优化和失败降级。

```python
def detect_communities(self) -> Dict[str, Any]:
    """
    执行 Leiden 算法社区检测

    关键参数：
    - includeIntermediateCommunities: 是否包含中间层级社区
    - randomSeed: 随机种子，确保结果可复现
    - maxLevels: 最大层级数
    - gamma: 分辨率参数，控制社区粒度

    Returns:
        Dict: 检测结果统计
    """
```

**`SLLPADetector.detect_communities()`**

执行 SLLPA 算法，如果未检测到社区则自动回退到 Leiden。

```python
def detect_communities(self) -> Dict[str, Any]:
    """
    执行 SLLPA 算法社区检测

    自动回退机制：
    - 如果 SLLPA 未检测到社区，自动切换到 Leiden
    - 保证始终能输出可用的社区划分结果

    Returns:
        Dict: 检测结果统计
    """
```

### 社区摘要模块

**`BaseSummarizer.process_communities()`**

处理所有社区的摘要生成流程，包括权重计算、信息收集、摘要生成和存储。

```python
def process_communities(self) -> List[Dict]:
    """
    处理所有社区的摘要生成流程

    流程步骤：
    1. 计算社区重要性排名
    2. 批量收集社区内实体和关系信息
    3. 并行调用 LLM 生成摘要
    4. 将摘要持久化到 Neo4j

    Returns:
        List[Dict]: 所有社区摘要列表
            - community_id: 社区 ID
            - summary: 摘要文本
            - node_count: 社区内节点数
            - relationship_count: 社区内关系数
            - rank: 社区重要性排名
    """
```

**`BaseSummarizer._process_communities_parallel()`**

并行处理社区摘要生成，利用多线程提升效率。

```python
def _process_communities_parallel(self, community_info: List[Dict], workers: int) -> List[Dict]:
    """
    利用多线程并行生成社区摘要

    Args:
        community_info: 社区信息列表
        workers: 线程池大小（默认使用 MAX_WORKERS）

    Returns:
        List[Dict]: 生成的摘要列表

    注意事项：
    - 每个线程独立调用 LLM API
    - 自动处理 API 限流和重试
    - 支持进度追踪
    """
```

**`LeidenSummarizer.collect_community_info()`**

收集 Leiden 社区的详细信息，支持大规模批量处理。

```python
def collect_community_info(self) -> List[Dict]:
    """
    收集社区信息，支持大规模批量处理

    收集内容：
    - 社区 ID 和统计信息（节点数、关系数）
    - 社区内所有实体节点（包括名称、类型、属性）
    - 社区内所有关系（包括类型、属性、源/目标实体）

    优化策略：
    - 分批查询（BATCH_SIZE），避免单次查询过大
    - 使用 UNWIND 批量处理
    - 过滤空社区

    Returns:
        List[Dict]: 社区信息列表
    """
```

**`BaseCommunityStorer.store_summary()`**

将社区摘要持久化到 Neo4j 图数据库。

```python
def store_summary(self, community_id: int, summary: str, metadata: Dict) -> bool:
    """
    将社区摘要存储到图数据库

    存储策略：
    1. 创建或更新 CommunitySummary 节点
    2. 设置摘要文本和元数据（节点数、关系数、生成时间等）
    3. 建立 HAS_SUMMARY 关系连接社区节点

    Args:
        community_id: 社区 ID
        summary: 摘要文本
        metadata: 元数据字典

    Returns:
        bool: 存储是否成功
    """
```

## 使用示例

### 1. 基础社区检测

```python
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from graphrag_agent.community import CommunityDetectorFactory

# 初始化图连接
graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="password"
)
gds = GraphDataScience(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# 创建 Leiden 社区检测器
detector = CommunityDetectorFactory.create('leiden', gds, graph)

# 执行社区检测
results = detector.process()

print(f"检测到 {results['community_count']} 个社区")
print(f"处理了 {results['node_count']} 个节点")
print(f"总耗时: {results['total_time']:.2f} 秒")
```

### 2. 社区摘要生成

```python
from graphrag_agent.community import CommunitySummarizerFactory

# 创建对应的 Leiden 摘要生成器
summarizer = CommunitySummarizerFactory.create_summarizer('leiden', graph)

# 生成社区摘要
summaries = summarizer.process_communities()

print(f"已生成 {len(summaries)} 个社区摘要")

# 查看摘要详情
for summary in summaries[:3]:  # 显示前3个
    print(f"\n社区 {summary['community_id']}:")
    print(f"  节点数: {summary['node_count']}")
    print(f"  关系数: {summary['relationship_count']}")
    print(f"  摘要: {summary['summary'][:100]}...")
```

### 3. 完整流程（检测 + 摘要）

```python
from graphrag_agent.community import CommunityDetectorFactory, CommunitySummarizerFactory

# 步骤1：社区检测
algorithm = 'leiden'  # 或 'sllpa'
detector = CommunityDetectorFactory.create(algorithm, gds, graph)
detection_results = detector.process()

print(f"社区检测完成: {detection_results['community_count']} 个社区")

# 步骤2：生成摘要
summarizer = CommunitySummarizerFactory.create_summarizer(algorithm, graph)
summaries = summarizer.process_communities()

print(f"摘要生成完成: {len(summaries)} 个摘要")
```

### 4. 使用 SLLPA 算法

```python
# SLLPA 算法适合检测重叠社区
detector = CommunityDetectorFactory.create('sllpa', gds, graph)
results = detector.process()

# 如果 SLLPA 未检测到社区，会自动回退到 Leiden
if results.get('fallback_to_leiden'):
    print("SLLPA 未检测到社区，已自动切换到 Leiden 算法")

# 生成 SLLPA 社区摘要
summarizer = CommunitySummarizerFactory.create_summarizer('sllpa', graph)
summaries = summarizer.process_communities()
```

### 5. 在完整构建流程中使用

在 `graphrag_agent/integrations/build/main.py` 中，社区检测和摘要是知识图谱构建的一部分：

```python
from graphrag_agent.integrations.build.builders import IndexCommunityBuilder

# 索引社区构建器内部会调用社区检测和摘要
builder = IndexCommunityBuilder()
builder.build()  # 自动执行社区检测 + 摘要生成
```

## 性能考量

### 内存管理

- **GDS 投影内存**：图投影会将 Neo4j 图数据加载到内存，内存占用与图规模成正比
  - 通过 `GDS_MEMORY_LIMIT` 限制最大内存使用（单位：GB）
  - 提供三级降级策略：标准 → 过滤（仅实体节点）→ 保守（Cypher 投影）
  - 投影完成后自动清理，释放内存

### 并行处理

- **社区检测并发**：GDS 算法内部并行，通过 `GDS_CONCURRENCY` 控制线程数
- **摘要生成并发**：使用 `ThreadPoolExecutor` 并行调用 LLM，并发度通过 `MAX_WORKERS` 配置
  - 建议值：4-8（取决于 LLM API 限流策略）
  - 过高并发可能触发 API 限流

### 批处理优化

- **社区信息收集**：分批查询（`BATCH_SIZE`），避免单次查询数据量过大
  - 默认批量大小：100
  - 大规模图建议调整为 50-200

### 性能统计

所有操作都会记录详细的性能指标：

```python
results = detector.process()
# 输出示例：
# {
#     'community_count': 42,
#     'node_count': 1523,
#     'projection_time': 2.34,
#     'detection_time': 5.67,
#     'save_time': 1.23,
#     'total_time': 9.24
# }
```

### 大规模图优化建议

针对超大规模图（节点数 > 100万）：

1. **降低并发度**：避免内存压力过大
2. **增加批量大小**：减少查询次数
3. **使用 SLLPA**：时间复杂度更优（O(m+n) vs O(n log n)）
4. **分阶段处理**：先检测，后摘要，避免资源争抢

## 扩展性

### 添加新的社区检测算法

通过继承 `BaseCommunityDetector` 添加自定义算法：

```python
from graphrag_agent.community.detector.base import BaseCommunityDetector

class CustomDetector(BaseCommunityDetector):
    """自定义社区检测算法"""

    def detect_communities(self) -> Dict[str, Any]:
        """实现自定义检测逻辑"""
        # 使用 self.gds 调用 GDS 库函数
        # 或实现自己的算法逻辑
        pass

    def save_communities(self, results: Dict) -> None:
        """将结果保存到图数据库"""
        # 实现保存逻辑
        pass

# 注册到工厂
from graphrag_agent.community import CommunityDetectorFactory
CommunityDetectorFactory.register('custom', CustomDetector)

# 使用
detector = CommunityDetectorFactory.create('custom', gds, graph)
```

### 自定义社区摘要生成

通过继承 `BaseSummarizer` 实现自定义摘要逻辑：

```python
from graphrag_agent.community.summary.base import BaseSummarizer

class CustomSummarizer(BaseSummarizer):
    """自定义摘要生成器"""

    def collect_community_info(self) -> List[Dict]:
        """自定义信息收集策略"""
        pass

    def _generate_single_summary(self, community_info: Dict) -> str:
        """自定义摘要生成逻辑"""
        # 可以使用不同的 LLM 模型
        # 或实现基于模板的摘要生成
        pass

# 注册到工厂
from graphrag_agent.community import CommunitySummarizerFactory
CommunitySummarizerFactory.register('custom', CustomSummarizer)
```

### 扩展点清单

1. **自定义排名策略**：继承 `BaseCommunityRanker`，实现不同的社区重要性计算方法
2. **自定义描述生成**：继承 `BaseCommunityDescriber`，使用不同的提示词或模型
3. **自定义存储逻辑**：继承 `BaseCommunityStorer`，支持其他数据库或存储方式
4. **图投影策略**：扩展 `GraphProjectionMixin`，支持更多过滤和优化策略

## 数据结构说明

### 社区属性存储

社区检测结果以节点属性形式存储在 Neo4j 中：

```cypher
// Leiden 算法结果
MATCH (e:Entity)
RETURN e.name, e.leidenCommunity

// SLLPA 算法结果
MATCH (e:Entity)
RETURN e.name, e.sllpaCommunity
```

### 社区摘要节点结构

社区摘要以独立节点形式存储：

```cypher
(:CommunitySummary {
    community_id: 42,              // 社区 ID
    summary: "这是一个关于...",     // 摘要文本
    node_count: 156,                // 社区内节点数
    relationship_count: 423,        // 社区内关系数
    rank: 0.85,                     // 重要性排名分数
    created_at: "2025-10-25T10:30:00",  // 生成时间
    algorithm: "leiden"             // 使用的算法
})
```

### 社区与摘要关系

```cypher
// 查询社区及其摘要
MATCH (e:Entity {leidenCommunity: 42})<-[:BELONGS_TO]-(s:CommunitySummary)
RETURN e, s
```