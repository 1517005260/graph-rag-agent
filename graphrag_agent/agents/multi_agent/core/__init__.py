"""
核心数据模型和状态定义
"""

from graphrag_agent.agents.multi_agent.core.state import (
    PlanExecuteState,
    PlanContext,
    ExecutionContext,
    ReportContext
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanSpec,
    ProblemStatement,
    TaskNode,
    TaskGraph,
    AcceptanceCriteria,
    PlanExecutionSignal,
)
from graphrag_agent.agents.multi_agent.core.execution_record import (
    ExecutionRecord,
    ToolCall,
    ReflectionResult,
    ExecutionMetadata
)
from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalResult,
    RetrievalMetadata
)

__all__ = [
    "PlanExecuteState",
    "PlanContext",
    "ExecutionContext",
    "ReportContext",
    "PlanSpec",
    "ProblemStatement",
    "TaskNode",
    "TaskGraph",
    "AcceptanceCriteria",
    "PlanExecutionSignal",
    "ExecutionRecord",
    "ToolCall",
    "ReflectionResult",
    "ExecutionMetadata",
    "RetrievalResult",
    "RetrievalMetadata",
]
