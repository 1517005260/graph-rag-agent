"""
多Agent协作系统模块

实现Plan-Execute-Report三层架构的多Agent协同框架
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
    AcceptanceCriteria
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
    # State models
    "PlanExecuteState",
    "PlanContext",
    "ExecutionContext",
    "ReportContext",
    # Plan models
    "PlanSpec",
    "ProblemStatement",
    "TaskNode",
    "TaskGraph",
    "AcceptanceCriteria",
    # Execution models
    "ExecutionRecord",
    "ToolCall",
    "ReflectionResult",
    "ExecutionMetadata",
    # Retrieval models
    "RetrievalResult",
    "RetrievalMetadata",
]
