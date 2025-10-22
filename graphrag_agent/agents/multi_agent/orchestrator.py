"""
多Agent总编排器

负责串联 Planner → WorkerCoordinator → Reporter，形成完整的
Plan-Execute-Report 生命周期。
"""
from typing import List, Optional, Sequence, Literal
import logging
import time
import json

from pydantic import BaseModel, Field

from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.core.execution_record import ExecutionRecord
from graphrag_agent.agents.multi_agent.planner.base_planner import (
    BasePlanner,
    PlannerResult,
)
from graphrag_agent.agents.multi_agent.executor.worker_coordinator import WorkerCoordinator
from graphrag_agent.agents.multi_agent.reporter.base_reporter import (
    BaseReporter,
    ReportResult,
)

_LOGGER = logging.getLogger(__name__)


class OrchestratorConfig(BaseModel):
    """
    编排器配置
    """

    auto_generate_report: bool = Field(
        default=True,
        description="执行完成后是否自动生成报告",
    )
    stop_on_clarification: bool = Field(
        default=True,
        description="当Planner需要澄清时是否立即停止后续流程",
    )
    strict_plan_signal: bool = Field(
        default=True,
        description="Planner未返回执行信号时是否视为失败",
    )


class OrchestratorMetrics(BaseModel):
    """
    编排器耗时指标
    """

    planning_seconds: float = Field(default=0.0, description="规划阶段耗时")
    execution_seconds: float = Field(default=0.0, description="执行阶段耗时")
    reporting_seconds: float = Field(default=0.0, description="报告生成耗时")


class OrchestratorResult(BaseModel):
    """
    编排结果
    """

    status: Literal["completed", "needs_clarification", "failed", "partial"] = Field(
        description="整体流程状态"
    )
    planner: Optional[PlannerResult] = Field(
        default=None, description="规划阶段的详细结果"
    )
    execution_records: List[ExecutionRecord] = Field(
        default_factory=list, description="执行阶段产生的记录"
    )
    report: Optional[ReportResult] = Field(
        default=None, description="Reporter 生成的报告"
    )
    errors: List[str] = Field(default_factory=list, description="流程中的错误列表")
    metrics: OrchestratorMetrics = Field(
        default_factory=OrchestratorMetrics, description="阶段耗时指标"
    )

    def requires_clarification(self) -> bool:
        """是否需要用户澄清"""
        if self.status != "needs_clarification":
            return False
        if self.planner is None:
            return False
        clarification = self.planner.clarification
        return clarification.needs_clarification and bool(clarification.questions)


class MultiAgentOrchestrator:
    """
    多Agent流程编排器
    """

    def __init__(
        self,
        *,
        planner: BasePlanner,
        worker_coordinator: WorkerCoordinator,
        reporter: BaseReporter,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self._planner = planner
        self._worker = worker_coordinator
        self._reporter = reporter
        self.config = config or OrchestratorConfig()

    def run(
        self,
        state: PlanExecuteState,
        *,
        assumptions: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        执行完整的 Plan-Execute-Report 流程
        """
        errors: List[str] = []
        metrics = OrchestratorMetrics()

        # --- Plan ---
        plan_start = time.perf_counter()
        try:
            planner_result = self._planner.generate_plan(
                state,
                assumptions=list(assumptions) if assumptions else None,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("Planner执行失败: %s", exc)
            errors.append(f"Planner执行失败: {exc}")
            metrics.planning_seconds = time.perf_counter() - plan_start
            return OrchestratorResult(
                status="failed",
                planner=None,
                execution_records=[],
                report=None,
                errors=errors,
                metrics=metrics,
            )
        metrics.planning_seconds = time.perf_counter() - plan_start

        self._print_plan_summary(planner_result)

        if planner_result.plan_spec is None:
            status = "needs_clarification"
            if not planner_result.clarification.needs_clarification:
                errors.append("Planner未生成PlanSpec且未提供澄清指引")
                status = "failed"
            if self.config.stop_on_clarification or status == "failed":
                return OrchestratorResult(
                    status=status,
                    planner=planner_result,
                    execution_records=[],
                    report=None,
                    errors=errors,
                    metrics=metrics,
                )

        signal = planner_result.executor_signal
        if signal is None:
            message = "Planner未提供执行信号，无法继续执行"
            _LOGGER.error(message)
            errors.append(message)
            if self.config.strict_plan_signal:
                return OrchestratorResult(
                    status="failed",
                    planner=planner_result,
                    execution_records=[],
                    report=None,
                    errors=errors,
                    metrics=metrics,
                )

        # --- Execute ---
        execution_records: List[ExecutionRecord] = []
        if signal is not None:
            exec_start = time.perf_counter()
            try:
                execution_records = self._worker.execute_plan(state, signal)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("执行阶段失败: %s", exc)
                errors.append(f"执行阶段失败: {exc}")
            finally:
                metrics.execution_seconds = time.perf_counter() - exec_start

        # --- Report ---
        report_result: Optional[ReportResult] = None
        if self.config.auto_generate_report and not errors:
            report_start = time.perf_counter()
            try:
                report_result = self._reporter.generate_report(
                    state,
                    report_type=report_type,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("报告生成失败: %s", exc)
                errors.append(f"报告生成失败: {exc}")
            finally:
                metrics.reporting_seconds = time.perf_counter() - report_start

        # --- Determine final status ---
        status = "completed"
        if errors:
            status = "failed"
        elif state.plan is not None:
            if state.plan.status == "failed":
                status = "failed"
            elif state.plan.status not in ("completed", "executing"):
                status = "partial"
            elif state.plan.status == "executing":
                status = "partial"

        state.update_timestamp()

        return OrchestratorResult(
            status=status,
            planner=planner_result,
            execution_records=execution_records,
            report=report_result,
            errors=errors,
            metrics=metrics,
        )

    def _print_plan_summary(self, planner_result: PlannerResult) -> None:
        """
        将计划摘要输出为JSON，便于在终端直接查看拆解后的任务列表。
        """
        plan = planner_result.plan_spec
        if plan is None:
            return
        summary = {
            "plan_id": plan.plan_id,
            "version": plan.version,
            "status": plan.status,
            "tasks": [
                {
                    "task_id": node.task_id,
                    "description": node.description,
                    "tool": node.task_type,
                    "parameters": node.parameters,
                    "priority": node.priority,
                    "depends_on": node.depends_on,
                }
                for node in plan.task_graph.nodes
            ],
        }
        try:
            encoded = json.dumps(summary, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("计划摘要序列化失败: %s", exc)
            return
        print(f"[PlanSpec] 规划结果:\n{encoded}")
