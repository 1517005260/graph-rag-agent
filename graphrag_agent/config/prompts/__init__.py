"""
Prompts统一管理模块

所有多Agent相关的Prompt模板集中管理
"""

from graphrag_agent.config.prompts.planner_prompts import (
    TASK_DECOMPOSE_PROMPT,
    CLARIFY_PROMPT,
    PLAN_REVIEW_PROMPT
)

from graphrag_agent.config.prompts.executor_prompts import (
    EXECUTE_PROMPT,
    REFLECT_PROMPT,
    REPLAN_PROMPT
)

from graphrag_agent.config.prompts.reporter_prompts import (
    OUTLINE_PROMPT,
    SECTION_WRITE_PROMPT,
    CONSISTENCY_CHECK_PROMPT,
    CITATION_FORMAT_PROMPT
)

__all__ = [
    # Planner prompts
    "TASK_DECOMPOSE_PROMPT",
    "CLARIFY_PROMPT",
    "PLAN_REVIEW_PROMPT",
    # Executor prompts
    "EXECUTE_PROMPT",
    "REFLECT_PROMPT",
    "REPLAN_PROMPT",
    # Reporter prompts
    "OUTLINE_PROMPT",
    "SECTION_WRITE_PROMPT",
    "CONSISTENCY_CHECK_PROMPT",
    "CITATION_FORMAT_PROMPT",
]
