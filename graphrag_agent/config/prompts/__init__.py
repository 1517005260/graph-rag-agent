"""
提示模板统一注册入口。

所有提示模板常量集中在此包中，确保配置一致。
"""

from typing import Optional

from graphrag_agent.config.prompts.graph_prompts import (
    system_template_build_graph,
    human_template_build_graph,
    system_template_build_index,
    user_template_build_index,
    community_template,
    COMMUNITY_SUMMARY_PROMPT,
    entity_alignment_prompt,
)
from graphrag_agent.config.prompts.search_prompts import (
    LOCAL_SEARCH_CONTEXT_PROMPT,
    LOCAL_SEARCH_KEYWORD_PROMPT,
    GLOBAL_SEARCH_MAP_PROMPT,
    GLOBAL_SEARCH_REDUCE_PROMPT,
    GLOBAL_SEARCH_KEYWORD_PROMPT,
    HYBRID_TOOL_QUERY_PROMPT,
    NAIVE_SEARCH_QUERY_PROMPT,
)
from graphrag_agent.config.prompts.agent_prompts import (
    GRAPH_AGENT_KEYWORD_PROMPT,
    GRAPH_AGENT_GENERATE_PROMPT,
    GRAPH_AGENT_REDUCE_PROMPT,
    DEEP_RESEARCH_THINKING_SUMMARY_PROMPT,
    EXPLORATION_SUMMARY_PROMPT,
    CONTRADICTION_IMPACT_PROMPT,
    HYBRID_AGENT_GENERATE_PROMPT,
    NAIVE_RAG_HUMAN_PROMPT,
)
from graphrag_agent.config.prompts.qa_prompts import (
    NAIVE_PROMPT,
    LC_SYSTEM_PROMPT,
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    contextualize_q_system_prompt,
    MINERU_IMAGE_DESCRIPTION_PROMPT,
)
from graphrag_agent.config.prompts.reasoning_prompts import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
    MAX_SEARCH_LIMIT,
    REASON_PROMPT,
    RELEVANT_EXTRACTION_PROMPT,
    SUB_QUERY_PROMPT,
    FOLLOWUP_QUERY_PROMPT,
    FINAL_ANSWER_PROMPT,
    INITIAL_THINKING_PROMPT,
    HYPOTHESIS_GENERATION_PROMPT,
    HYPOTHESIS_VERIFICATION_PROMPT,
    VERIFICATION_STATUS_PROMPT,
    UPDATE_THINKING_PROMPT,
    COUNTERFACTUAL_ANALYSIS_PROMPT,
    COUNTERFACTUAL_COMPARISON_PROMPT,
    SEARCH_RESULT_COMPARISON_PROMPT,
    SEARCH_MULTI_HYPOTHESIS_PROMPT,
)
from graphrag_agent.config.prompts.planner_prompts import (
    TASK_DECOMPOSE_PROMPT,
    CLARIFY_PROMPT,
    PLAN_REVIEW_PROMPT,
)
from graphrag_agent.config.prompts.executor_prompts import (
    EXECUTE_PROMPT,
    REFLECT_PROMPT,
    REPLAN_PROMPT,
)
from graphrag_agent.config.prompts.reporter_prompts import (
    OUTLINE_PROMPT,
    SECTION_WRITE_PROMPT,
    CONSISTENCY_CHECK_PROMPT,
    CITATION_FORMAT_PROMPT,
    EVIDENCE_MAP_PROMPT,
    SECTION_REDUCE_PROMPT,
    INTERMEDIATE_SUMMARY_PROMPT,
    MERGE_PROMPT,
    REFINE_PROMPT,
    INTRO_PROMPT,
    CONCLUSION_PROMPT,
    TERMINOLOGY_PROMPT,
)

__all__ = [
    # 图谱构建相关模板
    "system_template_build_graph",
    "human_template_build_graph",
    "system_template_build_index",
    "user_template_build_index",
    "community_template",
    "COMMUNITY_SUMMARY_PROMPT",
    "entity_alignment_prompt",
    # 搜索工具模板
    "LOCAL_SEARCH_CONTEXT_PROMPT",
    "LOCAL_SEARCH_KEYWORD_PROMPT",
    "GLOBAL_SEARCH_MAP_PROMPT",
    "GLOBAL_SEARCH_REDUCE_PROMPT",
    "GLOBAL_SEARCH_KEYWORD_PROMPT",
    "HYBRID_TOOL_QUERY_PROMPT",
    "NAIVE_SEARCH_QUERY_PROMPT",
    # Agent层模板
    "GRAPH_AGENT_KEYWORD_PROMPT",
    "GRAPH_AGENT_GENERATE_PROMPT",
    "GRAPH_AGENT_REDUCE_PROMPT",
    "DEEP_RESEARCH_THINKING_SUMMARY_PROMPT",
    "EXPLORATION_SUMMARY_PROMPT",
    "CONTRADICTION_IMPACT_PROMPT",
    "HYBRID_AGENT_GENERATE_PROMPT",
    "NAIVE_RAG_HUMAN_PROMPT",
    # 问答阶段模板
    "NAIVE_PROMPT",
    "LC_SYSTEM_PROMPT",
    "MAP_SYSTEM_PROMPT",
    "REDUCE_SYSTEM_PROMPT",
    "contextualize_q_system_prompt",
    "MINERU_IMAGE_DESCRIPTION_PROMPT",
    # 深度推理模板
    "BEGIN_SEARCH_QUERY",
    "END_SEARCH_QUERY",
    "BEGIN_SEARCH_RESULT",
    "END_SEARCH_RESULT",
    "MAX_SEARCH_LIMIT",
    "REASON_PROMPT",
    "RELEVANT_EXTRACTION_PROMPT",
    "SUB_QUERY_PROMPT",
    "FOLLOWUP_QUERY_PROMPT",
    "FINAL_ANSWER_PROMPT",
    "INITIAL_THINKING_PROMPT",
    "HYPOTHESIS_GENERATION_PROMPT",
    "HYPOTHESIS_VERIFICATION_PROMPT",
    "VERIFICATION_STATUS_PROMPT",
    "UPDATE_THINKING_PROMPT",
    "COUNTERFACTUAL_ANALYSIS_PROMPT",
    "COUNTERFACTUAL_COMPARISON_PROMPT",
    "SEARCH_RESULT_COMPARISON_PROMPT",
    "SEARCH_MULTI_HYPOTHESIS_PROMPT",
    # 计划层模板
    "TASK_DECOMPOSE_PROMPT",
    "CLARIFY_PROMPT",
    "PLAN_REVIEW_PROMPT",
    # 执行层模板
    "EXECUTE_PROMPT",
    "REFLECT_PROMPT",
    "REPLAN_PROMPT",
    # 报告层模板
    "OUTLINE_PROMPT",
    "SECTION_WRITE_PROMPT",
    "CONSISTENCY_CHECK_PROMPT",
    "CITATION_FORMAT_PROMPT",
    "EVIDENCE_MAP_PROMPT",
    "SECTION_REDUCE_PROMPT",
    "INTERMEDIATE_SUMMARY_PROMPT",
    "MERGE_PROMPT",
    "REFINE_PROMPT",
    "INTRO_PROMPT",
    "CONCLUSION_PROMPT",
    "TERMINOLOGY_PROMPT",
]

PROMPT_REGISTRY = {
    name.lower(): value
    for name, value in globals().items()
    if name.isupper() and isinstance(value, str)
}


def get_prompt_by_name(name: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """根据名称（不区分大小写）获取提示模板。"""
    if not name:
        return default
    return PROMPT_REGISTRY.get(name.lower(), default)


__all__.extend(
    [
        "PROMPT_REGISTRY",
        "get_prompt_by_name",
    ]
)
