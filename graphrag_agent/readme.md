# GraphRAG Agent 核心包

GraphRAG Agent 是一个基于知识图谱的综合检索增强生成（RAG）系统核心包，提供了智能体、搜索能力、图处理和缓存机制等核心功能。

## 📦 包结构

```
graphrag_agent/
├── agents/              # 🤖 智能体模块
│   ├── base.py          # Agent基类
│   ├── graph_agent.py   # 基于图结构的Agent
│   ├── hybrid_agent.py  # 混合搜索Agent
│   ├── naive_rag_agent.py  # 简单向量检索Agent
│   ├── deep_research_agent.py  # 深度研究Agent
│   ├── fusion_agent.py  # Fusion GraphRAG Agent
│   └── multi_agent/     # Plan-Execute-Report 多智能体编排栈
│       ├── planner/     # 规划器：澄清、任务分解、计划审校
│       ├── executor/    # 执行器：检索、研究、反思执行器及协调器
│       ├── reporter/    # 报告器：纲要生成、章节写作、一致性检查
│       ├── core/        # 核心模型：PlanSpec、State、ExecutionRecord
│       ├── tools/       # 工具：证据追踪、检索适配器、JSON解析
│       └── integration/ # 集成：工厂类、兼容门面
│
├── cache_manager/       # 📦 缓存管理模块
│   ├── manager.py       # 统一缓存管理器
│   ├── backends/        # 存储后端实现
│   │   ├── base.py      # 后端基类
│   │   ├── memory.py    # 内存缓存
│   │   ├── disk.py      # 磁盘缓存
│   │   ├── hybrid.py    # 混合缓存
│   │   └── thread_safe.py  # 线程安全包装器
│   ├── models/          # 数据模型
│   │   └── cache_item.py  # 缓存项模型
│   ├── strategies/      # 缓存键生成策略
│   │   ├── base.py      # 策略基类
│   │   ├── simple.py    # 简单策略
│   │   └── context_aware.py  # 上下文感知策略
│   ├── vector_similarity/  # 向量相似度匹配
│   │   ├── matcher.py   # 相似度匹配器
│   │   └── embeddings.py  # 嵌入提供器
│   └── model_cache.py   # 模型缓存初始化
│
├── community/           # 🔍 社区检测与摘要
│   ├── detector/        # 社区检测算法
│   │   ├── base.py      # 检测器基类
│   │   ├── leiden.py    # Leiden算法
│   │   └── sllpa.py     # SLLPA算法
│   └── summary/         # 社区摘要生成
│       ├── base.py      # 摘要器基类
│       ├── leiden.py    # Leiden摘要器
│       └── sllpa.py     # SLLPA摘要器
│
├── config/              # ⚙️ 配置模块
│   ├── neo4jdb.py       # Neo4j数据库连接管理
│   ├── prompt.py        # 提示模板
│   └── settings.py      # 全局配置
│
├── evaluation/          # 📊 评估系统
│   ├── core/            # 评估核心组件
│   │   ├── base_metric.py     # 指标基类
│   │   ├── base_evaluator.py  # 评估器基类
│   │   └── evaluation_data.py # 评估数据模型
│   ├── metrics/         # 评估指标实现
│   │   ├── answer_metrics.py      # 答案质量指标
│   │   ├── retrieval_metrics.py   # 检索性能指标
│   │   ├── graph_metrics.py       # 图谱评估指标
│   │   ├── llm_metrics.py         # LLM评估指标
│   │   └── deep_search_metrics.py # 深度搜索指标
│   ├── evaluators/      # 评估器实现
│   │   ├── answer_evaluator.py    # 答案评估器
│   │   ├── retrieval_evaluator.py # 检索评估器
│   │   └── composite_evaluator.py # 组合评估器
│   ├── evaluator_config/  # 评估器配置
│   ├── preprocessing/   # 预处理工具
│   ├── utils/           # 评估工具函数
│   └── test/            # 评估测试脚本
│
├── graph/               # 📈 图谱构建模块
│   ├── core/            # 核心组件
│   │   ├── graph_connection.py  # 图数据库连接管理
│   │   ├── base_indexer.py      # 索引器基类
│   │   └── utils.py             # 工具函数
│   ├── extraction/      # 实体关系提取
│   │   ├── entity_extractor.py  # 实体关系提取器
│   │   └── graph_writer.py      # 图写入器
│   ├── indexing/        # 索引管理
│   │   ├── chunk_indexer.py     # 文本块索引
│   │   └── entity_indexer.py    # 实体索引
│   ├── processing/      # 实体处理
│   │   ├── entity_merger.py         # 实体合并
│   │   ├── similar_entity.py        # 相似实体检测
│   │   ├── entity_disambiguation.py # 实体消歧
│   │   ├── entity_alignment.py      # 实体对齐
│   │   └── entity_quality.py        # 实体质量处理
│   └── structure/       # 图结构构建
│       └── struct_builder.py    # 结构构建器
│
├── integrations/        # 🔌 集成模块
│   └── build/           # 🏗️ 知识图谱构建
│       ├── main.py      # 构建入口
│       ├── build_graph.py  # 基础图谱构建
│       ├── build_index_and_community.py  # 索引和社区构建
│       ├── build_chunk_index.py  # 文本块索引构建
│       └── incremental/  # 增量更新子模块
│           └── incremental_update.py  # 增量更新管理
│
├── models/              # 🧩 模型管理
│   └── get_models.py    # 模型初始化
│
├── pipelines/           # 🔄 数据管道
│   └── ingestion/       # 📄 文档摄取处理
│       ├── document_processor.py  # 文档处理核心
│       ├── file_reader.py         # 多格式文件读取
│       └── text_chunker.py        # 文本分块
│
└── search/              # 🔎 搜索模块
    ├── local_search.py  # 本地搜索
    ├── global_search.py # 全局搜索
    └── tool/            # 搜索工具集
        ├── base.py                  # 搜索工具基类
        ├── local_search_tool.py     # 本地搜索工具
        ├── global_search_tool.py    # 全局搜索工具
        ├── hybrid_tool.py           # 混合搜索工具
        ├── naive_search_tool.py     # 简单搜索工具
        ├── deep_research_tool.py    # 深度研究工具
        ├── deeper_research_tool.py  # 更深度研究工具
        └── reasoning/               # 推理组件
            ├── nlp.py               # NLP工具
            ├── prompts.py           # 提示模板
            ├── thinking.py          # 思考引擎
            ├── validator.py         # 答案验证器
            ├── search.py            # 双路径搜索
            ├── community_enhance.py # 社区感知增强
            ├── kg_builder.py        # 动态知识图谱构建
            └── evidence.py          # 证据链追踪
```

## 🚀 核心功能

### 1. 智能体系统（agents/）

提供多种类型的智能体实现，支持不同复杂度的问答场景：

- **NaiveRagAgent**: 基础向量检索Agent，适合简单问题
- **GraphAgent**: 基于图结构的Agent，支持关系推理
- **HybridAgent**: 混合多种检索方式的Agent
- **DeepResearchAgent**: 深度研究Agent，支持多步推理
- **FusionGraphRAGAgent**: 最先进的Agent，采用Plan-Execute-Report多智能体架构

**多智能体编排栈（multi_agent/）**：

新一代Plan-Execute-Report架构，提供智能化任务规划与执行能力：

- **Planner**: 规划器，包含澄清（Clarifier）、任务分解（TaskDecomposer）、计划审校（PlanReviewer）
- **Executor**: 执行器，包含检索执行器（RetrievalExecutor）、研究执行器（ResearchExecutor）、反思器（Reflector）和工作协调器（WorkerCoordinator）
- **Reporter**: 报告生成器，采用Map-Reduce模式生成结构化长文档
- **Core**: 核心数据模型，包括PlanSpec（计划规范）、State（状态管理）、ExecutionRecord（执行记录）
- **Tools**: 工具组件，包括证据追踪、检索适配器、JSON解析器
- **Integration**: 集成层，提供工厂类和兼容门面

### 2. 缓存管理（cache_manager/）

高效的多层缓存系统：

- **多种存储后端**: 内存、磁盘、混合缓存
- **智能键策略**: 简单键、上下文感知键、关键词感知键
- **向量相似度匹配**: 支持语义相似的缓存查询
- **线程安全**: 提供线程安全的缓存包装器

### 3. 社区检测（community/）

支持多种社区检测算法：

- **Leiden算法**: 高质量社区发现
- **SLLPA算法**: 标签传播算法
- **社区摘要**: 自动生成社区摘要文本

### 4. 图谱构建（graph/）

完整的知识图谱构建流程：

- **实体关系提取**: LLM驱动的实体关系识别
- **实体处理**: 消歧、对齐、合并、质量提升
- **索引管理**: 文本块索引、实体索引
- **增量更新**: 支持动态增量构建

### 5. 评估系统（evaluation/）

20+ 种评估指标：

- **答案质量**: EM、F1 Score
- **检索性能**: Precision、Utilization、Latency
- **图谱评估**: Entity Coverage、Graph Coverage、Community Relevance
- **LLM评估**: Coherence、Factual Consistency、Comprehensiveness
- **深度搜索**: Reasoning Coherence、Reasoning Depth、Iterative Improvement

### 6. 搜索工具（search/）

多种搜索策略：

- **LocalSearch**: 本地邻域搜索
- **GlobalSearch**: 全局社区摘要搜索
- **HybridSearch**: 混合搜索策略
- **DeepResearch**: 深度推理搜索，支持思考链
- **Chain of Exploration**: 知识图谱上的多步探索

### 7. 数据处理（pipelines/）

灵活的文档处理管道：

- **多格式支持**: TXT、PDF、MD、DOCX、CSV、JSON、YAML等
- **智能分块**: 支持多种文本分块策略
- **文档预处理**: 清洗、标准化、元数据提取

## 💡 使用示例

### 基础使用

```python
from graphrag_agent import __version__
from graphrag_agent.agents import FusionGraphRAGAgent
from graphrag_agent.search import LocalSearch, GlobalSearch

# 创建Agent
agent = FusionGraphRAGAgent()

# 执行查询
result = agent.query("你的问题")
print(result)
```

### 使用缓存

```python
from graphrag_agent.cache_manager import (
    CacheManager,
    HybridCacheBackend,
    ContextAwareCacheKeyStrategy
)

# 创建缓存管理器
cache_manager = CacheManager(
    backend=HybridCacheBackend(),
    key_strategy=ContextAwareCacheKeyStrategy()
)

# 使用缓存
cached_result = cache_manager.get(query)
if not cached_result:
    result = agent.query(query)
    cache_manager.set(query, result)
```

### 评估系统

```python
from graphrag_agent.evaluation import (
    AnswerEvaluator,
    GraphRAGRetrievalEvaluator
)
from graphrag_agent.evaluation.core import AnswerEvaluationData

# 创建评估器
evaluator = AnswerEvaluator(config)

# 评估答案
eval_data = AnswerEvaluationData(samples=[...])
results = evaluator.evaluate(eval_data)
```

### 图谱构建

```python
from graphrag_agent.graph import (
    EntityRelationExtractor,
    GraphWriter,
    EntityDisambiguator,
    EntityAligner
)
from graphrag_agent.pipelines.ingestion import DocumentProcessor

# 处理文档
processor = DocumentProcessor()
chunks = processor.process_file("document.pdf")

# 提取实体关系
extractor = EntityRelationExtractor()
entities, relations = extractor.extract(chunks)

# 实体消歧和对齐
disambiguator = EntityDisambiguator()
aligned_entities = disambiguator.disambiguate(entities)

# 写入图谱
writer = GraphWriter()
writer.write(aligned_entities, relations)
```

## 🔧 配置

### 数据库配置

在 `config/neo4jdb.py` 中配置 Neo4j 连接：

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### 模型配置

在 `config/settings.py` 中配置 LLM 和 Embedding 模型：

```python
LLM_MODEL = "your_llm_model"
EMBEDDING_MODEL = "your_embedding_model"
```

## 📊 性能特性

- **增量更新**: 支持知识图谱的动态增量构建
- **智能缓存**: 多层缓存减少重复计算
- **并行处理**: 批处理和并行提取提升效率
- **实体质量**: 消歧和对齐机制提升实体准确性
