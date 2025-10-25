"""
执行调度器

根据 PlanExecutionSignal 调度不同类型的 Worker 执行任务，支持串行与并行模式。
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Tuple
import logging

from graphrag_agent.agents.multi_agent.core.execution_record import (
    ExecutionMetadata,
    ExecutionRecord,
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    TaskExecutionResult,
)
from graphrag_agent.agents.multi_agent.executor.research_executor import ResearchExecutor
from graphrag_agent.agents.multi_agent.executor.retrieval_executor import RetrievalExecutor
from graphrag_agent.agents.multi_agent.executor.reflector import ReflectionExecutor
from graphrag_agent.config.settings import (
    MULTI_AGENT_REFLECTION_ALLOW_RETRY,
    MULTI_AGENT_REFLECTION_MAX_RETRIES,
    MULTI_AGENT_WORKER_EXECUTION_MODE,
    MULTI_AGENT_WORKER_MAX_CONCURRENCY,
)

_LOGGER = logging.getLogger(__name__)


class WorkerCoordinator:
    """
    Worker协调器

    负责解析计划信号、选择合适的执行器并串行或并行地执行任务。
    默认模式可通过环境变量或构造参数进行配置。
    """

    def __init__(
        self,
        executors: Optional[List[BaseExecutor]] = None,
        *,
        execution_mode: Optional[str] = None,
        max_parallel_workers: Optional[int] = None,
    ) -> None:
        if executors is None:
            executors = [
                RetrievalExecutor(),
                ResearchExecutor(),
                ReflectionExecutor(),
            ]
        self.executors = executors
        configured_mode = (
            execution_mode.strip().lower()
            if isinstance(execution_mode, str)
            else MULTI_AGENT_WORKER_EXECUTION_MODE
        )
        if configured_mode not in {"sequential", "parallel"}:
            raise ValueError(
                f"WorkerCoordinator execution_mode 必须为 sequential 或 parallel，当前为 {configured_mode}"
            )
        workers = max_parallel_workers or MULTI_AGENT_WORKER_MAX_CONCURRENCY
        if workers < 1:
            raise ValueError("max_parallel_workers 必须大于等于 1")
        self.execution_mode = configured_mode
        self.max_parallel_workers = workers

    def register_executor(self, executor: BaseExecutor) -> None:
        """注册额外的执行器"""
        self.executors.append(executor)

    def execute_plan(
        self,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> List[ExecutionRecord]:
        """根据计划信号执行所有任务，返回执行记录列表。"""
        task_map = self._prepare_tasks(signal)

        if state.plan is not None:
            state.plan.status = "executing"

        effective_mode = self._resolve_execution_mode(signal.execution_mode)
        if effective_mode == "parallel":
            results = self._execute_parallel(state, signal, task_map)
        else:
            results = self._execute_sequential(state, signal, task_map)

        if state.plan is not None:
            node_status = [node.status for node in state.plan.task_graph.nodes]
            if node_status and all(status == "completed" for status in node_status):
                state.plan.status = "completed"
            elif any(status == "failed" for status in node_status):
                state.plan.status = "failed"

        return results

    def _resolve_execution_mode(self, requested_mode: str) -> str:
        requested = (requested_mode or "sequential").lower()
        if requested not in {"sequential", "parallel"}:
            if requested == "adaptive":
                _LOGGER.warning("Planner 请求 adaptive 模式，当前仅支持串行或并行，将使用 %s", self.execution_mode)
            else:
                _LOGGER.warning(
                    "Planner 请求的执行模式 %s 无效，将使用 %s",
                    requested_mode,
                    self.execution_mode,
                )
            requested = "sequential"

        effective = self.execution_mode
        if effective != requested:
            _LOGGER.info(
                "WorkerCoordinator 使用 %s 模式执行计划（覆盖 planner 请求 %s）",
                effective,
                requested_mode,
            )
        else:
            _LOGGER.debug("WorkerCoordinator 以 %s 模式执行计划", effective)
        return effective

    def _execute_sequential(
        self,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
        task_map: Dict[str, TaskNode],
    ) -> List[ExecutionRecord]:
        results: List[ExecutionRecord] = []
        sequence = signal.execution_sequence or list(task_map.keys())
        for task_id in sequence:
            task = task_map.get(task_id)
            if task is None:
                _LOGGER.warning("计划信号中包含未知任务: %s", task_id)
                continue
            self._execute_single_task(
                state=state,
                signal=signal,
                task=task,
                task_map=task_map,
                results=results,
                skip_dependency_check=False,
            )
        return results

    def _execute_parallel(
        self,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
        task_map: Dict[str, TaskNode],
    ) -> List[ExecutionRecord]:
        results: List[ExecutionRecord] = []
        sequence = signal.execution_sequence or list(task_map.keys())
        missing = [task_id for task_id in sequence if task_id not in task_map]
        for task_id in missing:
            _LOGGER.warning("计划信号中包含未知任务: %s", task_id)

        pending: List[str] = [task_id for task_id in sequence if task_id in task_map]
        if not pending:
            return results

        max_workers = min(self.max_parallel_workers, max(1, len(pending)))
        inflight: Dict[object, str] = {}
        task_status: Dict[str, str] = {task_id: "pending" for task_id in pending}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending or inflight:
                scheduled_this_round = False

                for task_id in list(pending):
                    if len(inflight) >= max_workers:
                        break

                    status = task_status.get(task_id)
                    if status != "pending":
                        pending.remove(task_id)
                        continue

                    task = task_map[task_id]
                    dependency_ok, dependency_error, failure_reason = self._check_dependencies(task, state)
                    if dependency_ok:
                        future = executor.submit(
                            self._execute_single_task,
                            state=state,
                            signal=signal,
                            task=task,
                            task_map=task_map,
                            results=results,
                            skip_dependency_check=True,
                        )
                        inflight[future] = task_id
                        task_status[task_id] = "running"
                        pending.remove(task_id)
                        scheduled_this_round = True
                        continue

                    if failure_reason in {"dependency_failed", "dependency_missing"}:
                        _LOGGER.error(
                            "任务依赖未满足，跳过执行: task_id=%s reason=%s",
                            task.task_id,
                            dependency_error,
                        )
                        failure_record = self._create_failure_record(
                            state,
                            task,
                            dependency_error or "依赖未满足",
                            failure_reason=failure_reason,
                        )
                        results.append(failure_record)
                        task_status[task_id] = "failed"
                        pending.remove(task_id)
                        scheduled_this_round = True
                    # dependency 未完成，等待下一轮

                if inflight:
                    done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        task_id = inflight.pop(future)
                        task = task_map[task_id]
                        try:
                            success, _ = future.result()
                        except Exception as exc:  # noqa: BLE001
                            _LOGGER.exception("任务执行异常: task_id=%s error=%s", task_id, exc)
                            failure_record = self._create_failure_record(
                                state,
                                task,
                                f"任务执行异常: {exc}",
                                failure_reason="execution_exception",
                            )
                            results.append(failure_record)
                            success = False

                        task_status[task_id] = "completed" if success else "failed"
                        scheduled_this_round = True
                    continue

                if not scheduled_this_round:
                    if pending:
                        for task_id in list(pending):
                            task = task_map[task_id]
                            failure_record = self._create_failure_record(
                                state,
                                task,
                                "任务依赖未解析或存在循环依赖",
                                failure_reason="dependency_unresolved",
                            )
                            results.append(failure_record)
                            task_status[task_id] = "failed"
                        pending.clear()
                    break

        return results

    def _execute_single_task(
        self,
        *,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
        task: TaskNode,
        task_map: Dict[str, TaskNode],
        results: List[ExecutionRecord],
        skip_dependency_check: bool,
    ) -> Tuple[bool, Optional[str]]:
        if not skip_dependency_check:
            dependency_ok, dependency_error, failure_reason = self._check_dependencies(task, state)
            if not dependency_ok:
                _LOGGER.error(
                    "任务依赖未满足，跳过执行: task_id=%s reason=%s",
                    task.task_id,
                    dependency_error,
                )
                failure_record = self._create_failure_record(
                    state,
                    task,
                    dependency_error or "依赖未满足",
                    failure_reason=failure_reason,
                )
                results.append(failure_record)
                return False, failure_reason

        executor = self._select_executor(task.task_type)
        if executor is None:
            _LOGGER.error("未找到匹配的执行器: task_id=%s type=%s", task.task_id, task.task_type)
            failure_record = self._create_failure_record(
                state,
                task,
                f"未找到任务类型 {task.task_type} 对应的执行器",
                failure_reason="executor_not_found",
            )
            results.append(failure_record)
            return False, "executor_not_found"

        try:
            if state.plan is not None:
                state.plan.update_task_status(task.task_id, "running")

            exec_result = executor.execute_task(task, state, signal)
            results.append(exec_result.record)

            if (
                task.task_type == "reflection"
                and MULTI_AGENT_REFLECTION_ALLOW_RETRY
            ):
                retry_result = self._handle_reflection_retry(
                    task=task,
                    initial_result=exec_result,
                    state=state,
                    signal=signal,
                    task_map=task_map,
                    executor=executor,
                    results=results,
                )
                if retry_result is not None:
                    exec_result = retry_result

            if exec_result.success:
                return True, None
            return False, exec_result.error or "execution_failed"

        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("任务执行异常: task_id=%s error=%s", task.task_id, exc)
            failure_record = self._create_failure_record(
                state,
                task,
                f"任务执行异常: {exc}",
                failure_reason="execution_exception",
            )
            results.append(failure_record)
            return False, "execution_exception"

    def _prepare_tasks(self, signal: PlanExecutionSignal) -> Dict[str, TaskNode]:
        """将信号中的任务恢复为TaskNode对象"""
        task_map: Dict[str, TaskNode] = {}
        for task_payload in signal.tasks:
            try:
                task = TaskNode(**task_payload)
                task_map[task.task_id] = task
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("任务解析失败: payload=%s error=%s", task_payload, exc)
        return task_map

    def _select_executor(self, task_type: str) -> Optional[BaseExecutor]:
        for executor in self.executors:
            if executor.can_handle(task_type):
                return executor
        return None

    def _create_failure_record(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        error: str,
        *,
        failure_reason: str = "unknown",
    ) -> ExecutionRecord:
        """
        当无可用执行器时创建失败记录，并更新状态。
        """
        metadata = ExecutionMetadata(
            worker_type="worker_coordinator",
            latency_seconds=0.0,
            tool_calls_count=0,
            evidence_count=0,
            environment={"reason": failure_reason},
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            session_id=state.session_id,
            worker_type="worker_coordinator",
            inputs={
                "task": task.model_dump(),
            },
            tool_calls=[],
            evidence=[],
            metadata=metadata,
        )

        if state.execution_context is not None:
            state.execution_context.errors.append(
                {
                    "task_id": task.task_id,
                    "error": error,
                    "worker_type": "worker_coordinator",
                    "reason": failure_reason,
                }
            )

        state.execution_records.append(record)

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "failed")

        return record

    def _check_dependencies(
        self,
        task: TaskNode,
        state: PlanExecuteState,
    ) -> tuple[bool, Optional[str], str]:
        """
        检查任务依赖是否满足。

        返回 (是否可执行, 错误信息, 失败原因标签)。
        """
        if not task.depends_on:
            return True, None, "none"

        plan = state.plan
        status_map: Dict[str, str] = {}
        if plan is not None:
            status_map = {node.task_id: node.status for node in plan.task_graph.nodes}

        exec_context = state.execution_context
        completed_ids = set(exec_context.completed_task_ids if exec_context else [])

        failed_dependencies = []
        pending_dependencies = []
        missing_dependencies = []

        for dep_id in task.depends_on:
            status = status_map.get(dep_id)
            if status == "failed":
                failed_dependencies.append(dep_id)
            elif status == "completed" or dep_id in completed_ids:
                continue
            elif status is None:
                missing_dependencies.append(dep_id)
            else:
                pending_dependencies.append(dep_id)

        if failed_dependencies:
            return (
                False,
                f"依赖任务失败: {', '.join(failed_dependencies)}",
                "dependency_failed",
            )

        if missing_dependencies:
            return (
                False,
                f"依赖任务缺失: {', '.join(missing_dependencies)}",
                "dependency_missing",
            )

        if pending_dependencies:
            return (
                False,
                f"依赖任务未完成: {', '.join(pending_dependencies)}",
                "dependency_unfinished",
            )

        if not all(dep in completed_ids for dep in task.depends_on):
            remaining = [dep for dep in task.depends_on if dep not in completed_ids]
            return (
                False,
                f"依赖任务尚未标记完成: {', '.join(remaining)}",
                "dependency_unfinished",
            )

        return True, None, "ready"

    def _handle_reflection_retry(
        self,
        *,
        task: TaskNode,
        initial_result: TaskExecutionResult,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
        task_map: Dict[str, TaskNode],
        executor: BaseExecutor,
        results: List[ExecutionRecord],
    ) -> Optional[TaskExecutionResult]:
        """
        处理反思任务的自动重试逻辑，必要时重新执行目标任务并再次运行反思。
        """
        reflection = getattr(initial_result.record, "reflection", None)  # type: ignore[attr-defined]
        if reflection is None or not reflection.needs_retry:
            return None

        exec_context = state.execution_context
        if exec_context is None:
            return None

        target_task_id = initial_result.record.metadata.environment.get("target_task_id")
        if not target_task_id:
            _LOGGER.warning(
                "反思任务缺失 target_task_id，无法执行重试: task_id=%s",
                task.task_id,
            )
            return None

        final_result: Optional[TaskExecutionResult] = None
        retry_counts = exec_context.reflection_retry_counts

        while (
            reflection is not None
            and reflection.needs_retry
            and retry_counts.get(target_task_id, 0)
            < MULTI_AGENT_REFLECTION_MAX_RETRIES
        ):
            attempt_index = retry_counts.get(target_task_id, 0) + 1
            retry_counts[target_task_id] = attempt_index

            target_task = task_map.get(target_task_id)
            if target_task is None:
                _LOGGER.warning(
                    "无法找到反思重试的目标任务: target_task_id=%s",
                    target_task_id,
                )
                break

            target_executor = self._select_executor(target_task.task_type)
            if target_executor is None:
                _LOGGER.warning(
                    "缺少处理目标任务的执行器，终止反思重试: task_type=%s",
                    target_task.task_type,
                )
                break

            _LOGGER.info(
                "触发反思重试: target_task_id=%s attempt=%s/%s",
                target_task_id,
                attempt_index,
                MULTI_AGENT_REFLECTION_MAX_RETRIES,
            )
            if state.plan is not None:
                state.plan.update_task_status(target_task_id, "running")
            retry_result = target_executor.execute_task(target_task, state, signal)
            results.append(retry_result.record)

            if not retry_result.success:
                _LOGGER.warning(
                    "目标任务重试失败，终止后续反思: target_task_id=%s",
                    target_task_id,
                )
                break

            if state.plan is not None:
                state.plan.update_task_status(task.task_id, "running")
            updated_result = executor.execute_task(task, state, signal)
            results.append(updated_result.record)
            final_result = updated_result
            reflection = getattr(updated_result.record, "reflection", None)

        if (
            reflection is not None
            and reflection.needs_retry
            and retry_counts.get(target_task_id, 0)
            >= MULTI_AGENT_REFLECTION_MAX_RETRIES
        ):
            _LOGGER.info(
                "反思重试已达上限，仍未通过验证: target_task_id=%s",
                target_task_id,
            )

        if final_result is None:
            return None
        return final_result
