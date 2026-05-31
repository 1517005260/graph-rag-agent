# 答案评估指标
from graphrag_agent.evaluation.metrics.answer_metrics import (
    ExactMatch, F1Score
)

# 检索评估指标
from graphrag_agent.evaluation.metrics.retrieval_metrics import (
    RetrievalPrecision, RetrievalUtilization, 
    RetrievalLatency, ChunkUtilization
)

# 图评估指标
from graphrag_agent.evaluation.metrics.graph_metrics import (
    EntityCoverageMetric, GraphCoverageMetric,
    RelationshipUtilizationMetric, CommunityRelevanceMetric,
    SubgraphQualityMetric
)

# LLM评估指标
from graphrag_agent.evaluation.metrics.llm_metrics import (
    ResponseCoherence, FactualConsistency,
    ComprehensiveAnswerMetric, LLMGraphRagEvaluator
)

# 深度研究指标
from graphrag_agent.evaluation.metrics.deep_search_metrics import (
    ReasoningCoherence, ReasoningDepth,
    IterativeImprovementMetric, KnowledgeGraphUtilizationMetric
)

# 定义所有可用的指标
__all_metrics__ = {
    # 答案评估指标
    'em': 'evaluator.metrics.answer_metrics.ExactMatch',
    'f1': 'evaluator.metrics.answer_metrics.F1Score',
    
    # 检索评估指标
    'retrieval_precision': 'evaluator.metrics.retrieval_metrics.RetrievalPrecision',
    'retrieval_utilization': 'evaluator.metrics.retrieval_metrics.RetrievalUtilization',
    'retrieval_latency': 'evaluator.metrics.retrieval_metrics.RetrievalLatency',
    'chunk_utilization': 'evaluator.metrics.retrieval_metrics.ChunkUtilization',
    
    # 图评估指标
    'entity_coverage': 'evaluator.metrics.graph_metrics.EntityCoverageMetric',
    'graph_coverage': 'evaluator.metrics.graph_metrics.GraphCoverageMetric',
    'relationship_utilization': 'evaluator.metrics.graph_metrics.RelationshipUtilizationMetric',
    'community_relevance': 'evaluator.metrics.graph_metrics.CommunityRelevanceMetric',
    'subgraph_quality': 'evaluator.metrics.graph_metrics.SubgraphQualityMetric',
    
    # LLM评估指标
    'response_coherence': 'evaluator.metrics.llm_metrics.ResponseCoherence',
    'factual_consistency': 'evaluator.metrics.llm_metrics.FactualConsistency',
    'answer_comprehensiveness': 'evaluator.metrics.llm_metrics.ComprehensiveAnswerMetric',
    'llm_evaluation': 'evaluator.metrics.llm_metrics.LLMGraphRagEvaluator',
    
    # 深度研究指标
    'reasoning_coherence': 'evaluator.metrics.deep_search_metrics.ReasoningCoherence',
    'reasoning_depth': 'evaluator.metrics.deep_search_metrics.ReasoningDepth',
    'iterative_improvement': 'evaluator.metrics.deep_search_metrics.IterativeImprovementMetric',
    'knowledge_graph_utilization': 'evaluator.metrics.deep_search_metrics.KnowledgeGraphUtilizationMetric'
}

def list_available_metrics():
    """
    列出所有可用的指标
    
    Returns:
        List[str]: 指标名称列表
    """
    return list(__all_metrics__.keys())

def get_metric_class(metric_name: str):
    """
    获取指标类
    
    Args:
        metric_name: 指标名称
        
    Returns:
        指标类
    """
    metric_name = metric_name.lower()
    if metric_name not in __all_metrics__:
        return None

    import importlib
    module_path, class_name = __all_metrics__[metric_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_metric_instance(metric_name: str, config):
    """
    获取指标实例

    Args:
        metric_name: 指标名称
        config: 配置对象

    Returns:
        指标实例
    """
    metric_cls = get_metric_class(metric_name)
    if metric_cls:
        return metric_cls(config)
    return None

__all__ = [
    # 答案评估指标
    'ExactMatch',
    'F1Score',

    # 检索评估指标
    'RetrievalPrecision',
    'RetrievalUtilization',
    'RetrievalLatency',
    'ChunkUtilization',

    # 图评估指标
    'EntityCoverageMetric',
    'GraphCoverageMetric',
    'RelationshipUtilizationMetric',
    'CommunityRelevanceMetric',
    'SubgraphQualityMetric',

    # LLM评估指标
    'ResponseCoherence',
    'FactualConsistency',
    'ComprehensiveAnswerMetric',
    'LLMGraphRagEvaluator',

    # 深度研究指标
    'ReasoningCoherence',
    'ReasoningDepth',
    'IterativeImprovementMetric',
    'KnowledgeGraphUtilizationMetric',

    # 工具函数
    'list_available_metrics',
    'get_metric_class',
    'get_metric_instance',

    # 指标字典
    '__all_metrics__'
]