"""
多Agent总编排器

负责串联 Planner → WorkerCoordinator → Reporter，形成完整的
Plan-Execute-Report 生命周期。
"""
from typing import List, Optional, Sequence, Literal, Dict, Any
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
    enable_fast_path: bool = Field(
        default=True,
        description="是否启用简单查询的快速通道（跳过 Planner）",
    )
    fast_path_max_length: int = Field(
        default=30,
        description="快速通道的查询最大长度（字符数）",
    )
    fast_path_keywords: List[str] = Field(
        default_factory=lambda: ["对比", "分析", "详细", "深入", "研究", "解释", "为什么"],
        description="包含这些关键词的查询不走快速通道",
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

    def _determine_route(self, query: str) -> str:
        """
        [路由网关] 根据查询特征决定执行路径

        三层路由策略：
        1. FAST: 事实性/简单查询 -> Local Search（跳过 Planner）
        2. SLOW: 分析性/综合查询 -> Global Search（跳过 Planner，使用社区级搜索）
        3. HEAVY: 研究性/复杂任务 -> 完整 Plan-Execute-Report（包含 Planner）

        Args:
            query: 用户输入的查询文本

        Returns:
            路由类型: "FAST" | "SLOW" | "HEAVY"
        """
        if not self.config.enable_fast_path:
            return "HEAVY"

        # Heavy lane: 研究性/复杂任务关键词
        heavy_keywords = ["深度研究", "调研报告", "长文", "详细调查", "深入", "详细"]
        if any(kw in query for kw in heavy_keywords):
            _LOGGER.debug("检测到复杂研究关键词，路由到 HEAVY 通道")
            return "HEAVY"

        # Slow lane: 分析性/综合性关键词
        slow_keywords = ["总结", "概括", "全貌", "趋势", "对比", "分析"]
        if any(kw in query for kw in slow_keywords):
            _LOGGER.debug("检测到分析性关键词，路由到 SLOW 通道（Global Search）")
            return "SLOW"

        # Fast lane: 短的事实性查询
        if len(query) < 50:
            _LOGGER.debug("检测到简单事实性查询，路由到 FAST 通道（Local Search）")
            return "FAST"

        # 默认走完整流程
        _LOGGER.debug("查询较长且无明显特征，路由到 HEAVY 通道")
        return "HEAVY"

    def _create_direct_signal(
        self,
        query: str,
        route: str,
    ) -> "PlanExecutionSignal":
        """
        为直接通道（FAST/SLOW）构造执行信号

        Args:
            query: 用户查询
            route: 路由类型（"FAST" 或 "SLOW"）

        Returns:
            PlanExecutionSignal: 执行信号
        """
        from graphrag_agent.agents.multi_agent.core.plan_spec import (
            PlanExecutionSignal,
            TaskNode,
        )

        if route == "FAST":
            # FAST 通道: Local Search（实体级搜索）
            task = TaskNode(
                task_id="fast_local_search_1",
                task_type="local_search",
                description=f"快速检索回答查询：{query}",
                parameters={"query": query},
                priority=1,
                depends_on=[],
            )
        elif route == "SLOW":
            # SLOW 通道: Global Search（社区级搜索）
            task = TaskNode(
                task_id="slow_global_search_1",
                task_type="global_search",
                description=f"全局检索回答查询：{query}",
                parameters={"query": query},
                priority=1,
                depends_on=[],
            )
        else:
            # 兜底：使用 local_search
            task = TaskNode(
                task_id="fallback_search_1",
                task_type="local_search",
                description=f"检索回答查询：{query}",
                parameters={"query": query},
                priority=1,
                depends_on=[],
            )

        return PlanExecutionSignal(
            tasks=[task.model_dump()],
            execution_mode="sequential",
        )

    def _create_dummy_plan(self) -> Optional[PlannerResult]:
        """
        为直接通道创建一个占位的 PlannerResult，用于前端兼容性

        Returns:
            占位的 PlannerResult（plan_spec=None 表示跳过规划）
        """
        from graphrag_agent.agents.multi_agent.core.clarification import ClarificationResult

        return PlannerResult(
            plan_spec=None,  # 直接通道跳过规划
            executor_signal=None,  # 由 _create_direct_signal 单独构造
            clarification=ClarificationResult(
                needs_clarification=False,
                questions=[],
            ),
        )

    def _execute_direct_lane(
        self,
        state: PlanExecuteState,
        metrics: OrchestratorMetrics,
        route: str,
        report_type: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        直接通道：FAST/SLOW 查询跳过 Planner，直接执行搜索

        路由策略：
        - FAST: local_search（实体级检索）
        - SLOW: global_search（社区级聚合）

        流程：
        1. 构造直接执行信号（根据路由类型）
        2. 执行检索任务（复用 WorkerCoordinator）
        3. 可选：生成简单报告

        Args:
            state: 当前状态
            metrics: 性能指标
            route: 路由类型（"FAST" 或 "SLOW"）
            report_type: 报告类型

        Returns:
            编排结果
        """
        errors: List[str] = []

        # 构造执行信号
        signal = self._create_direct_signal(state.input, route)

        # 规划阶段耗时为 0（跳过 Planner）
        metrics.planning_seconds = 0.0

        # --- 执行阶段 ---
        execution_records: List[ExecutionRecord] = []
        exec_start = time.perf_counter()
        try:
            execution_records = self._worker.execute_plan(state, signal)
        except Exception as exc:  # noqa: BLE001
            lane_name = "FAST" if route == "FAST" else "SLOW"
            _LOGGER.exception("%s 通道执行失败: %s", lane_name, exc)
            errors.append(f"{lane_name} 通道执行失败: {exc}")
        finally:
            metrics.execution_seconds = time.perf_counter() - exec_start

        self._print_execution_summary(execution_records, state)

        # --- 报告生成（可选）---
        report_result: Optional[ReportResult] = None
        # 对于直接通道，可以选择不生成完整报告
        # 如果需要报告，可以解除以下注释：
        # if self.config.auto_generate_report and not errors:
        #     report_start = time.perf_counter()
        #     try:
        #         report_result = self._reporter.generate_report(
        #             state,
        #             report_type=report_type or "short_answer",
        #         )
        #         if report_result is not None:
        #             self._print_report_summary(report_result)
        #     except Exception as exc:  # noqa: BLE001
        #         _LOGGER.exception("报告生成失败: %s", exc)
        #         errors.append(f"报告生成失败: {exc}")
        #     finally:
        #         metrics.reporting_seconds = time.perf_counter() - report_start

        # 确定最终状态
        status = "completed" if not errors else "failed"
        state.update_timestamp()

        lane_name = "FAST (Local Search)" if route == "FAST" else "SLOW (Global Search)"
        _LOGGER.info(
            "🚀 %s 通道完成 | 耗时: %.2fs (节省 Planner 时间)",
            lane_name,
            metrics.execution_seconds,
        )

        return OrchestratorResult(
            status=status,
            planner=self._create_dummy_plan(),  # 占位，前端兼容性
            execution_records=execution_records,
            report=report_result,
            errors=errors,
            metrics=metrics,
        )

    def run(
        self,
        state: PlanExecuteState,
        *,
        assumptions: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        执行完整的 Plan-Execute-Report 流程

        三层自适应路由：
        1. FAST 通道: 简单查询 -> Local Search（跳过 Planner）
        2. SLOW 通道: 分析查询 -> Global Search（跳过 Planner）
        3. HEAVY 通道: 复杂任务 -> 完整 Plan-Execute-Report（包含 Planner）
        """
        errors: List[str] = []
        metrics = OrchestratorMetrics()

        # ========== 三层路由网关 ==========
        route = self._determine_route(state.input)

        if route in ["FAST", "SLOW"]:
            # FAST/SLOW 通道：跳过 Planner，直接执行搜索
            lane_type = "Local Search" if route == "FAST" else "Global Search"
            _LOGGER.info("🚀 进入 %s 通道：%s（跳过 Planner）", route, lane_type)
            return self._execute_direct_lane(state, metrics, route, report_type)

        # ========== HEAVY 通道：完整 Plan-Execute-Report ==========
        _LOGGER.info("🔬 进入 HEAVY 通道：完整 Plan-Execute-Report 流程")
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
        self._print_execution_summary(execution_records, state)

        # --- Report ---
        report_result: Optional[ReportResult] = None
        if self.config.auto_generate_report and not errors:
            report_start = time.perf_counter()
            try:
                report_result = self._reporter.generate_report(
                    state,
                    report_type=report_type,
                )
                if report_result is not None:
                    self._print_report_summary(report_result)
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

    def _print_execution_summary(
        self,
        execution_records: Sequence[ExecutionRecord],
        state: PlanExecuteState,
    ) -> None:
        """
        将执行阶段的核心指标打印为JSON，方便本地调试。
        """
        if not execution_records:
            print("[Execute] 未执行任务。")
            return

        task_status: Dict[str, str] = {}
        if state.plan is not None:
            task_status = {
                node.task_id: node.status for node in state.plan.task_graph.nodes
            }

        summary: List[Dict[str, Any]] = []
        for record in execution_records:
            env_payload = dict(getattr(record.metadata, "environment", {}) or {})
            error_messages = [
                call.error for call in record.tool_calls if getattr(call, "error", None)
            ]
            record_summary: Dict[str, Any] = {
                "task_id": record.task_id,
                "worker": record.worker_type,
                "status": task_status.get(record.task_id, "unknown"),
                "tool_calls": [call.tool_name for call in record.tool_calls],
                "evidence_count": len(record.evidence),
                "latency_seconds": round(record.metadata.latency_seconds, 3),
            }
            if error_messages:
                record_summary["errors"] = error_messages
                record_summary["error"] = error_messages[0]

            failure_reason = env_payload.get("reason") or env_payload.get("failure_reason")
            if failure_reason:
                record_summary["failure_reason"] = failure_reason
            if env_payload.get("target_task_id"):
                record_summary["target_task_id"] = env_payload["target_task_id"]
            if env_payload.get("validation_passed") is not None:
                record_summary["validation_passed"] = env_payload["validation_passed"]

            if record.reflection is not None:
                reflection_block: Dict[str, Any] = {
                    "success": record.reflection.success,
                    "needs_retry": record.reflection.needs_retry,
                    "reasoning": record.reflection.reasoning,
                }
                if record.reflection.suggestions:
                    reflection_block["suggestions"] = record.reflection.suggestions
                record_summary["reflection"] = reflection_block

            summary.append(record_summary)

        try:
            print("[Execute] 执行结果:")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("执行摘要序列化失败: %s", exc)

        exec_context = state.execution_context
        if exec_context is not None and exec_context.errors:
            # 打印执行阶段积累的节点错误，便于快速定位失败原因
            try:
                print("[Execute] 节点错误详情:")
                print(json.dumps(exec_context.errors, ensure_ascii=False, indent=2))
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("节点错误详情序列化失败: %s", exc)

    def _print_report_summary(self, report_result: ReportResult) -> None:
        """
        打印Reporter阶段的摘要信息。
        """
        sections = [
            {
                "section_id": section.section_id,
                "title": section.title,
                "word_count": len(section.content),
            }
            for section in report_result.sections
        ]
        payload: Dict[str, Any] = {
            "report_type": report_result.outline.report_type,
            "title": report_result.outline.title,
            "section_count": len(report_result.sections),
            "sections": sections,
            "has_references": bool(report_result.references),
        }
        if report_result.consistency_check is not None:
            check = report_result.consistency_check
            issues = check.issues
            corrections = check.corrections
            if issues and len(issues) > 5:
                # 控制台输出保留前5个问题，其余问题仍可在payload中查看完整结果
                issues = issues[:5] + [{"info": f"其余 {len(check.issues) - 5} 项已省略，可在payload中查看完整结果"}]
            if corrections and len(corrections) > 5:
                corrections = corrections[:5] + [{"info": f"其余 {len(check.corrections) - 5} 项已省略，可在payload中查看完整结果"}]
            payload["consistency_check"] = {
                "is_consistent": check.is_consistent,
                "issues": issues,
                "corrections": corrections or [],
            }
        try:
            print("[Report] 生成报告:")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("报告摘要序列化失败: %s", exc)
