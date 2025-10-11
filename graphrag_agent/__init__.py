"""
GraphRAG Agent - 基于图的综合RAG系统

本包提供了用于高级RAG应用的智能体、搜索能力、图处理和缓存机制。

核心使用流程：
1. 准备文档 (files/) - 使用 pipelines.ingestion 模块
2. 建图 (全量/增量) - 使用 integrations.build 模块
3. 搜索问答 - 使用 search.tool 或 agents 模块
"""

__version__ = "0.1.0"

# ============ 1. 文档处理模块 ============
from graphrag_agent.pipelines.ingestion import (
    DocumentProcessor,
    FileReader,
    TextChunker
)

# ============ 2. 图谱构建模块 ============
# 核心组件
from graphrag_agent.graph import (
    GraphConnectionManager,
    connection_manager,
    EntityRelationExtractor,
    GraphWriter,
    GraphStructureBuilder
)

# 实体处理
from graphrag_agent.graph import (
    EntityMerger,
    SimilarEntityDetector,
    EntityDisambiguator,
    EntityAligner,
    EntityQualityProcessor
)

# 索引管理
from graphrag_agent.graph import (
    ChunkIndexManager,
    EntityIndexManager
)

# 社区检测
from graphrag_agent.community import (
    CommunityDetectorFactory,
    CommunitySummarizerFactory
)

# ============ 3. 搜索模块 ============
# 基础搜索
from graphrag_agent.search import (
    LocalSearch,
    GlobalSearch
)

# 搜索工具
from graphrag_agent.search import (
    LocalSearchTool,
    GlobalSearchTool,
    HybridSearchTool,
    NaiveSearchTool,
    DeepResearchTool
)

# ============ 4. Agent 模块 ============
# 注意：需要时从 graphrag_agent.agents 导入具体的 Agent 类
# 例如：from graphrag_agent.agents import FusionGraphRAGAgent

# ============ 5. 缓存管理模块 ============
from graphrag_agent.cache_manager import (
    CacheManager,
    # 存储后端
    MemoryCacheBackend,
    DiskCacheBackend,
    HybridCacheBackend,
    # 键策略
    SimpleCacheKeyStrategy,
    ContextAwareCacheKeyStrategy,
    ContextAndKeywordAwareCacheKeyStrategy,
    # 向量相似度
    VectorSimilarityMatcher
)

# ============ 6. 评估模块 ============
from graphrag_agent.evaluation.core import (
    BaseMetric,
    BaseEvaluator,
    AnswerEvaluationSample,
    AnswerEvaluationData,
    RetrievalEvaluationSample,
    RetrievalEvaluationData
)

from graphrag_agent.evaluation.evaluators import (
    AnswerEvaluator,
    GraphRAGRetrievalEvaluator,
    CompositeGraphRAGEvaluator
)

# ============ 7. 配置模块 ============
# 注意：配置通常在 graphrag_agent.config.settings 中设置
# 例如：from graphrag_agent.config.settings import theme, entity_types

__all__ = [
    # 版本
    '__version__',

    # 文档处理
    'DocumentProcessor',
    'FileReader',
    'TextChunker',

    # 图谱构建 - 核心
    'GraphConnectionManager',
    'connection_manager',
    'EntityRelationExtractor',
    'GraphWriter',
    'GraphStructureBuilder',

    # 图谱构建 - 实体处理
    'EntityMerger',
    'SimilarEntityDetector',
    'EntityDisambiguator',
    'EntityAligner',
    'EntityQualityProcessor',

    # 图谱构建 - 索引
    'ChunkIndexManager',
    'EntityIndexManager',

    # 社区检测
    'CommunityDetectorFactory',
    'CommunitySummarizerFactory',

    # 搜索
    'LocalSearch',
    'GlobalSearch',
    'LocalSearchTool',
    'GlobalSearchTool',
    'HybridSearchTool',
    'NaiveSearchTool',
    'DeepResearchTool',

    # 缓存
    'CacheManager',
    'MemoryCacheBackend',
    'DiskCacheBackend',
    'HybridCacheBackend',
    'SimpleCacheKeyStrategy',
    'ContextAwareCacheKeyStrategy',
    'ContextAndKeywordAwareCacheKeyStrategy',
    'VectorSimilarityMatcher',

    # 评估
    'BaseMetric',
    'BaseEvaluator',
    'AnswerEvaluationSample',
    'AnswerEvaluationData',
    'RetrievalEvaluationSample',
    'RetrievalEvaluationData',
    'AnswerEvaluator',
    'GraphRAGRetrievalEvaluator',
    'CompositeGraphRAGEvaluator',
]
