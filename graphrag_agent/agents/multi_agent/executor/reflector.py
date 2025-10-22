"""
反思执行器

基于 AnswerValidationTool 对既有任务输出进行质量校验，并产出反思结果。
"""
from typing import Any, Dict, Optional, Tuple, List
import time

from graphrag_agent.agents.multi_agent.core.execution_record import (
    ExecutionMetadata,
    ExecutionRecord,
    ReflectionResult,
    ToolCall,
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
)
from graphrag_agent.search.tool.validation_tool import AnswerValidationTool


class ReflectionExecutor(BaseExecutor):
    """
    反思任务执行器
    """

    worker_type: str = "reflection_executor"

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        *,
        validation_tool: Optional[AnswerValidationTool] = None,
    ) -> None:
        super().__init__(config)
        self._validation_tool = validation_tool or AnswerValidationTool()

    def can_handle(self, task_type: str) -> bool:
        return task_type == "reflection"

    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        """
        对指定任务结果进行反思与验证。
        """
        payload = self.build_default_inputs(task)
        query, answer, target_task_id = self._resolve_query_answer(state, payload)

        start_time = time.perf_counter()
        validation_payload: Optional[Dict[str, Any]] = None
        success = False
        error: Optional[str] = None
        suggestions: List[str] = []

        if not query:
            error = "缺少用于验证的查询内容"
        elif answer is None:
            error = "未找到可验证的答案"
        else:
            try:
                validation_payload = self._validation_tool.validate(query, answer)
                validation_result = validation_payload.get("validation", {})
                success = bool(validation_result.get("passed", False))
                for key, passed in validation_result.items():
                    if key == "passed":
                        continue
                    if not passed:
                        suggestions.append(f"验证项 {key} 未通过")
            except Exception as exc:  # noqa: BLE001
                error = f"答案验证失败: {exc}"

        latency = time.perf_counter() - start_time

        tool_calls = []
        if validation_payload is not None:
            tool_calls.append(
                ToolCall(
                    tool_name="answer_validator",
                    args={"query": query, "answer": answer},
                    result=validation_payload,
                    status="success" if success else "failed",
                    latency_ms=round(latency * 1000, 3),
                    error=None if success else "验证未通过",
                )
            )

        metadata = ExecutionMetadata(
            worker_type=self.worker_type,
            latency_seconds=latency,
            tool_calls_count=len(tool_calls),
            evidence_count=0,
            environment={
                "execution_mode": signal.execution_mode,
                "target_task_id": target_task_id,
            },
        )

        reflection = ReflectionResult(
            success=success,
            confidence=0.85 if success else 0.4,
            suggestions=suggestions if not success else [],
            needs_retry=not success,
            reasoning=(
                None
                if success
                else error
                or "验证未通过，建议回滚或追加检索"
            ),
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            session_id=state.session_id,
            worker_type=self.worker_type,
            inputs={"payload": payload, "query": query, "answer": answer},
            tool_calls=tool_calls,
            reflection=reflection,
            metadata=metadata,
        )

        self._update_state(
            state,
            task,
            record,
            success=success,
            error=error,
            target_task_id=target_task_id,
        )

        return TaskExecutionResult(record=record, success=success, error=error)

    def _resolve_query_answer(
        self,
        state: PlanExecuteState,
        payload: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        确定用于校验的 query 与 answer。
        """
        query = str(payload.get("query") or state.plan_context.refined_query or state.plan_context.original_query or state.input or "")
        answer: Optional[str] = payload.get("answer")

        target_task_id = payload.get("target_task_id")
        if answer is None and target_task_id:
            answer = self._lookup_answer_from_state(state, target_task_id)

        if answer is None and state.execution_records:
            last_record = state.execution_records[-1]
            answer = self._extract_answer_from_record(last_record)

        return query, answer, target_task_id

    def _lookup_answer_from_state(
        self,
        state: PlanExecuteState,
        target_task_id: str,
    ) -> Optional[str]:
        """
        从ExecutionContext中查找目标任务的答案。
        """
        context = state.execution_context
        if context is None:
            return None

        intermediate = context.intermediate_results.get(target_task_id)
        if not intermediate:
            return None

        if isinstance(intermediate, dict):
            answer = intermediate.get("answer") or intermediate.get("research_result")
            if isinstance(answer, str):
                return answer
            if isinstance(answer, dict):
                return answer.get("answer") or answer.get("summary")
        return None

    def _extract_answer_from_record(
        self,
        record: ExecutionRecord,
    ) -> Optional[str]:
        """
        从已有的执行记录中提取可能的答案。
        """
        if record.tool_calls:
            result = record.tool_calls[0].result
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("summary")
                if isinstance(answer, str):
                    return answer
        return None

    def _update_state(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        record: ExecutionRecord,
        success: bool,
        error: Optional[str],
        target_task_id: Optional[str],
    ) -> None:
        """
        将执行结果写回状态，维护任务完成情况。
        """
        state.execution_records.append(record)

        exec_context = state.execution_context
        if exec_context is not None:
            exec_context.current_task_id = task.task_id
            exec_context.tool_call_history.append(
                {
                    "task_id": task.task_id,
                    "tool_name": "answer_validator",
                    "status": "success" if success else "failed",
                    "latency_ms": record.metadata.latency_seconds * 1000,
                }
            )
            exec_context.intermediate_results[task.task_id] = {
                "reflection": record.reflection.model_dump(),
                "target_task_id": target_task_id,
                "action": "confirm" if success else "retry",
            }

            if success:
                if task.task_id not in exec_context.completed_task_ids:
                    exec_context.completed_task_ids.append(task.task_id)
            else:
                if target_task_id and target_task_id in exec_context.completed_task_ids:
                    exec_context.completed_task_ids.remove(target_task_id)
                exec_context.errors.append(
                    {
                        "task_id": task.task_id,
                        "error": error or "unknown",
                        "worker_type": self.worker_type,
                        "target_task_id": target_task_id,
                    }
                )
                if target_task_id:
                    retry_bucket = exec_context.intermediate_results.setdefault(
                        "__reflection_retry__", []
                    )
                    if isinstance(retry_bucket, list):
                        retry_bucket.append(
                            {
                                "target_task_id": target_task_id,
                                "reason": error or "validation_failed",
                                "reflection_task_id": task.task_id,
                            }
                        )

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "completed" if success else "failed")
            if not success and target_task_id:
                state.plan.update_task_status(target_task_id, "pending")
