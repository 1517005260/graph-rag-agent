"""
执行调度器

根据PlanExecutionSignal调度不同类型的Worker执行任务。
"""

from __future__ import annotations

from typing import Dict, List, Optional
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
from graphrag_agent.agents.multi_agent.executor.base_executor import BaseExecutor
from graphrag_agent.agents.multi_agent.executor.research_executor import ResearchExecutor
from graphrag_agent.agents.multi_agent.executor.retrieval_executor import RetrievalExecutor
from graphrag_agent.agents.multi_agent.executor.reflector import ReflectionExecutor

_LOGGER = logging.getLogger(__name__)


class WorkerCoordinator:
    """
    Worker协调器

    负责解析计划信号、选择合适的执行器并串行/并行地执行任务。
    当前实现以串行为主，其他执行模式将降级为串行执行。
    """

    def __init__(self, executors: Optional[List[BaseExecutor]] = None) -> None:
        if executors is None:
            executors = [
                RetrievalExecutor(),
                ResearchExecutor(),
                ReflectionExecutor(),
            ]
        self.executors = executors

    def register_executor(self, executor: BaseExecutor) -> None:
        """注册额外的执行器"""
        self.executors.append(executor)

    def execute_plan(
        self,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> List[ExecutionRecord]:
        """
        根据计划信号执行所有任务，返回执行记录列表。
        """
        task_map = self._prepare_tasks(signal)

        if state.plan is not None:
            state.plan.status = "executing"

        results: List[ExecutionRecord] = []
        if signal.execution_mode in ("parallel", "adaptive"):
            _LOGGER.info(
                "当前执行器暂不支持%s模式，自动降级为串行执行",
                signal.execution_mode,
            )

        sequence = signal.execution_sequence or list(task_map.keys())
        for task_id in sequence:
            task = task_map.get(task_id)
            if task is None:
                _LOGGER.warning("计划信号中包含未知任务: %s", task_id)
                continue

            executor = self._select_executor(task.task_type)
            if executor is None:
                _LOGGER.error("未找到匹配的执行器: task_id=%s type=%s", task_id, task.task_type)
                failure_record = self._create_failure_record(state, task, f"未找到任务类型 {task.task_type} 对应的执行器")
                results.append(failure_record)
                continue

            if state.plan is not None:
                state.plan.update_task_status(task.task_id, "running")

            exec_result = executor.execute_task(task, state, signal)
            results.append(exec_result.record)

        if state.plan is not None:
            node_status = [node.status for node in state.plan.task_graph.nodes]
            if node_status and all(status == "completed" for status in node_status):
                state.plan.status = "completed"
            elif any(status == "failed" for status in node_status):
                state.plan.status = "failed"

        return results

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
    ) -> ExecutionRecord:
        """
        当无可用执行器时创建失败记录，并更新状态。
        """
        metadata = ExecutionMetadata(
            worker_type="worker_coordinator",
            latency_seconds=0.0,
            tool_calls_count=0,
            evidence_count=0,
            environment={"reason": "executor_not_found"},
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
                }
            )

        state.execution_records.append(record)

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "failed")

        return record
