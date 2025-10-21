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
    AcceptanceCriteria,
    PlanExecutionSignal
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
from graphrag_agent.agents.multi_agent.executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
    RetrievalExecutor,
    ResearchExecutor,
    ReflectionExecutor,
    WorkerCoordinator,
)
from graphrag_agent.agents.multi_agent.reporter import (
    BaseReporter,
    ReporterConfig,
    ReportResult,
    SectionContent,
    OutlineBuilder,
    ReportOutline,
    SectionOutline,
    SectionWriter,
    SectionWriterConfig,
    SectionDraft,
    ConsistencyChecker,
    ConsistencyCheckResult,
    CitationFormatter,
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
    "PlanExecutionSignal",
    # Execution models
    "ExecutionRecord",
    "ToolCall",
    "ReflectionResult",
    "ExecutionMetadata",
    # Retrieval models
    "RetrievalResult",
    "RetrievalMetadata",
    # Executors
    "BaseExecutor",
    "ExecutorConfig",
    "TaskExecutionResult",
    "RetrievalExecutor",
    "ResearchExecutor",
    "ReflectionExecutor",
    "WorkerCoordinator",
    # Reporter models
    "BaseReporter",
    "ReporterConfig",
    "ReportResult",
    "SectionContent",
    "OutlineBuilder",
    "ReportOutline",
    "SectionOutline",
    "SectionWriter",
    "SectionWriterConfig",
    "SectionDraft",
    "ConsistencyChecker",
    "ConsistencyCheckResult",
    "CitationFormatter",
]
