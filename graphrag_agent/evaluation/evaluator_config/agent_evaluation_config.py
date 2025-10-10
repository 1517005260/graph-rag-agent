from typing import List
from graphrag_agent.evaluation.metrics import list_available_metrics

# 获取所有可用指标
available_metrics = list_available_metrics()

# 不同Agent类型的默认评估指标配置
AGENT_EVALUATION_CONFIG = {
    "graph": {
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization',
                        'community_relevance', 'subgraph_quality']
            if m in available_metrics
        ]
    },
    
    "hybrid": {
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization',
                        'community_relevance', 'subgraph_quality']
            if m in available_metrics
        ]
    },
    
    "fusion": {
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization',
                        'community_relevance', 'subgraph_quality']
            if m in available_metrics
        ],
        "reasoning_metrics": [
            m for m in ['reasoning_coherence', 'reasoning_depth', 'iterative_improvement']
            if m in available_metrics
        ]
    },
    
    "naive": {
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'chunk_utilization']  # 只使用与传统向量检索相关的指标
            if m in available_metrics
        ]
    },
    
    "deep": {
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization']
            if m in available_metrics
        ],
        "reasoning_metrics": [
            m for m in ['reasoning_coherence', 'reasoning_depth', 'iterative_improvement']
            if m in available_metrics
        ],
        "deeper_metrics": [
            m for m in ['knowledge_graph_utilization']  # 仅适用于DeeperResearchTool
            if m in available_metrics
        ]
    }
}

def get_agent_metrics(agent_type: str, metric_type: str = None) -> List[str]:
    """
    获取特定Agent类型的评估指标列表
    
    Args:
        agent_type: Agent类型 (graph, hybrid, naive, deep)
        metric_type: 指标类型 (answer, retrieval, reasoning, deeper, None表示全部)
        
    Returns:
        List[str]: 指标列表
    """
    if agent_type not in AGENT_EVALUATION_CONFIG:
        raise ValueError(f"不支持的Agent类型: {agent_type}")
    
    if metric_type == "answer":
        return AGENT_EVALUATION_CONFIG[agent_type]["answer_metrics"]
    elif metric_type == "retrieval":
        return AGENT_EVALUATION_CONFIG[agent_type]["retrieval_metrics"]
    elif metric_type == "reasoning" and "reasoning_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
        return AGENT_EVALUATION_CONFIG[agent_type]["reasoning_metrics"]
    elif metric_type == "deeper" and "deeper_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
        return AGENT_EVALUATION_CONFIG[agent_type]["deeper_metrics"]
    else:
        # 返回所有指标
        metrics = []
        metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["answer_metrics"])
        metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["retrieval_metrics"])
        
        if "reasoning_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
            metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["reasoning_metrics"])
        
        if "deeper_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
            metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["deeper_metrics"])
            
        return metrics