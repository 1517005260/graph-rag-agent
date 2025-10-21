"""
检索执行器

负责解析PlanExecutionSignal中的检索类任务并调用既有搜索工具（local/global/chain等）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
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
from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
)
from graphrag_agent.search.tool_registry import (
    TOOL_REGISTRY,
    EXTRA_TOOL_FACTORIES,
    create_extra_tool,
)

_LOGGER = logging.getLogger(__name__)


class RetrievalExecutor(BaseExecutor):
    """
    检索任务执行器
    """

    worker_type: str = "retrieval_executor"

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        *,
        tool_registry: Optional[Dict[str, Any]] = None,
        extra_tool_factories: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self._tool_registry = tool_registry or TOOL_REGISTRY
        self._extra_factories = extra_tool_factories or EXTRA_TOOL_FACTORIES
        self._tool_cache: Dict[str, Any] = {}
        self._extra_cache: Dict[str, Any] = {}

    def can_handle(self, task_type: str) -> bool:
        return task_type in self._tool_registry or task_type in self._extra_factories

    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        """
        执行检索类任务，默认调用search工具的structured_search接口。
        """
        tool_name = task.task_type
        payload = self.build_default_inputs(task)
        cache_key = f"{task.task_id}:{tool_name}"

        _LOGGER.info(
            "[RetrievalExecutor] 执行任务 %s (%s) payload=%s",
            task.task_id,
            tool_name,
            payload,
        )

        tool_instance = self._get_tool_instance(tool_name)
        start_time = time.perf_counter()
        success = True
        error_message: Optional[str] = None
        structured_output: Dict[str, Any] = {}

        try:
            structured_output = self._invoke_tool(tool_instance, tool_name, payload)
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)
            _LOGGER.exception("检索任务执行失败 task=%s error=%s", task.task_id, exc)

        latency = time.perf_counter() - start_time
        tool_call = ToolCall(
            tool_name=tool_name,
            args=payload,
            result=structured_output if success else None,
            status="success" if success else "failed",
            error=error_message,
            latency_ms=round(latency * 1000, 3),
        )

        evidence = self._extract_evidence(structured_output) if success else []

        metadata = ExecutionMetadata(
            worker_type=self.worker_type,
            latency_seconds=latency,
            tool_calls_count=1,
            evidence_count=len(evidence),
            environment={
                "execution_mode": signal.execution_mode,
                "cache_key": cache_key,
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

        self._update_state(state, task, record, success, error_message)

        return TaskExecutionResult(
            record=record,
            success=success,
            error=error_message,
        )

    def _get_tool_instance(self, task_type: str) -> Any:
        if task_type in self._tool_registry:
            if task_type not in self._tool_cache:
                self._tool_cache[task_type] = self._tool_registry[task_type]()
            return self._tool_cache[task_type]
        if task_type in self._extra_factories:
            if task_type not in self._extra_cache:
                self._extra_cache[task_type] = create_extra_tool(task_type)
            return self._extra_cache[task_type]
        raise KeyError(f"未找到任务类型 {task_type} 对应的检索工具")

    def _invoke_tool(self, tool: Any, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用具体的检索工具。

        优先使用structured_search，其次使用search，特殊工具（如chain_exploration）调用自定义方法。
        """
        if hasattr(tool, "structured_search"):
            return tool.structured_search(payload)
        if hasattr(tool, "search"):
            result = tool.search(payload)
            if isinstance(result, dict):
                return result
            return {"answer": result, "retrieval_results": []}
        if task_type == "chain_exploration" and hasattr(tool, "explore"):
            query = payload.get("query")
            start_entities = payload.get("start_entities") or payload.get("entities")
            max_steps = payload.get("max_steps")
            exploration_width = payload.get("exploration_width")
            return tool.explore(
                query=query,
                start_entities=start_entities,
                max_steps=max_steps,
                exploration_width=exploration_width,
            )
        raise ValueError(f"任务类型 {task_type} 未提供可用的执行方法")

    def _extract_evidence(self, output: Dict[str, Any]) -> List[RetrievalResult]:
        results_payload = output.get("retrieval_results") if isinstance(output, dict) else None
        evidence: List[RetrievalResult] = []
        if not isinstance(results_payload, list):
            return evidence

        for item in results_payload:
            try:
                if isinstance(item, RetrievalResult):
                    evidence.append(item)
                elif isinstance(item, dict):
                    evidence.append(RetrievalResult.from_dict(item))
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("无法解析retrieval_result: %s error=%s", item, exc)
        return evidence

    def _update_state(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        record: ExecutionRecord,
        success: bool,
        error: Optional[str],
    ) -> None:
        """
        将执行结果写回PlanExecuteState，更新任务状态、上下文与记录。
        """
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
            exec_context.retrieval_cache[task.task_id] = [res.to_dict() for res in record.evidence]
            tool_result_payload: Dict[str, Any] = {}
            if record.tool_calls and isinstance(record.tool_calls[0].result, dict):
                tool_result_payload = record.tool_calls[0].result  # type: ignore[assignment]
            exec_context.intermediate_results[task.task_id] = {
                "answer": tool_result_payload.get("answer"),
                "raw_result": tool_result_payload,
                "metadata": record.metadata.model_dump(),
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
