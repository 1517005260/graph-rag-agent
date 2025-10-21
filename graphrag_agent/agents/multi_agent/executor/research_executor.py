"""
研究执行器

负责处理深度研究类任务，调用DeepResearch/DeeperResearch工具并生成结构化ExecutionRecord。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import time
import logging

from graphrag_agent.agents.multi_agent.core.execution_record import (
    ExecutionMetadata,
    ExecutionRecord,
    ToolCall,
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
)
from graphrag_agent.search.tool_registry import TOOL_REGISTRY

_LOGGER = logging.getLogger(__name__)


class ResearchExecutor(BaseExecutor):
    """
    深度研究任务执行器
    """

    worker_type: str = "research_executor"
    SUPPORTED_TASKS = {"deep_research", "deeper_research"}

    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        super().__init__(config)
        self._tool_cache: Dict[str, Any] = {}

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        tool_name = task.task_type
        payload = self.build_default_inputs(task)
        _LOGGER.info("[ResearchExecutor] 执行任务 %s (%s)", task.task_id, tool_name)

        tool = self._get_tool_instance(tool_name)
        start_time = time.perf_counter()
        success = True
        error_message: Optional[str] = None
        result_payload: Any = None

        try:
            result_payload = tool.search(payload)
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)
            _LOGGER.exception("研究任务执行失败 task=%s error=%s", task.task_id, exc)

        latency = time.perf_counter() - start_time

        tool_call = ToolCall(
            tool_name=tool_name,
            args=payload,
            result=result_payload if success else None,
            status="success" if success else "failed",
            error=error_message,
            latency_ms=round(latency * 1000, 3),
        )

        evidence = self._wrap_research_evidence(task, tool_name, result_payload) if success else []

        metadata = ExecutionMetadata(
            worker_type=self.worker_type,
            latency_seconds=latency,
            tool_calls_count=1,
            evidence_count=len(evidence),
            environment={
                "execution_mode": signal.execution_mode,
            },
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            session_id=state.session_id,
            worker_type=self.worker_type,
            inputs={
                "payload": payload,
                "task": task.model_dump(),
            },
            tool_calls=[tool_call],
            evidence=evidence,
            metadata=metadata,
        )

        self._update_state(state, task, record, success, error_message, result_payload)

        return TaskExecutionResult(record=record, success=success, error=error_message)

    def _get_tool_instance(self, task_type: str) -> Any:
        if task_type not in self._tool_cache:
            if task_type not in TOOL_REGISTRY:
                raise KeyError(f"未注册的研究工具: {task_type}")
            self._tool_cache[task_type] = TOOL_REGISTRY[task_type]()
        return self._tool_cache[task_type]

    def _wrap_research_evidence(
        self,
        task: TaskNode,
        tool_name: str,
        result_payload: Any,
    ) -> List[RetrievalResult]:
        """
        将研究结果包装成RetrievalResult，便于Reporter引用。
        """
        if isinstance(result_payload, dict):
            # 若工具已经返回结构化数据，则尝试直接读取
            textual = result_payload.get("answer") or result_payload.get("summary") or str(result_payload)
        else:
            textual = str(result_payload or "")

        metadata = RetrievalMetadata(
            source_id=f"{task.task_id}:{tool_name}",
            source_type="document",
            confidence=0.65,
        )
        result = RetrievalResult(
            granularity="DO",
            evidence=textual,
            metadata=metadata,
            source=tool_name,  # 与RetrievalResult枚举保持一致
            score=0.65,
        )
        return [result]

    def _update_state(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        record: ExecutionRecord,
        success: bool,
        error: Optional[str],
        result_payload: Any,
    ) -> None:
        exec_context = state.execution_context
        if exec_context is None:
            return

        exec_context.current_task_id = task.task_id
        exec_context.tool_call_history.append(
            {
                "task_id": task.task_id,
                "tool_name": record.tool_calls[0].tool_name if record.tool_calls else "",
                "status": record.tool_calls[0].status if record.tool_calls else "unknown",
                "latency_ms": record.metadata.latency_seconds * 1000,
            }
        )

        if success:
            if task.task_id not in exec_context.completed_task_ids:
                exec_context.completed_task_ids.append(task.task_id)
            exec_context.intermediate_results[task.task_id] = {
                "research_result": result_payload,
            }
        else:
            exec_context.errors.append(
                {
                    "task_id": task.task_id,
                    "error": error,
                    "worker_type": self.worker_type,
                }
            )

        state.execution_records.append(record)

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "completed" if success else "failed")
