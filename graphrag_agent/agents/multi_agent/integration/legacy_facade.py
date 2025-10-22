"""
兼容层：让旧有调用方式可以无缝使用新的多智能体编排器。

旧版协调器通过 ``process_query`` 返回答案和调试信息。本文件提供等价
接口，但实际执行委托给新的 Plan-Execute-Report 流程。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from langchain_core.messages import HumanMessage

from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.integration.multi_agent_factory import (
    MultiAgentFactory,
    OrchestratorBundle,
)
from graphrag_agent.agents.multi_agent.orchestrator import OrchestratorResult


class MultiAgentFacade:
    """对外暴露稳定接口，便于旧系统逐步迁移到新框架。"""

    def __init__(
        self,
        *,
        bundle: Optional[OrchestratorBundle] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        self.cache_manager = cache_manager
        self.bundle = bundle or MultiAgentFactory.create_default_bundle(
            cache_manager=cache_manager
        )

    def process_query(
        self,
        query: str,
        *,
        assumptions: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
        extra_messages: Optional[Iterable[HumanMessage]] = None,
    ) -> Dict[str, Any]:
        """执行多智能体流程，并返回与旧协调器兼容的结构化结果。"""
        state = self._build_state(query, extra_messages)
        result = self.bundle.orchestrator.run(
            state,
            assumptions=assumptions,
            report_type=report_type,
        )
        return self._format_result(state, result)

    def _build_state(
        self,
        query: str,
        extra_messages: Optional[Iterable[HumanMessage]],
    ) -> PlanExecuteState:
        messages = [HumanMessage(content=query)]
        if extra_messages:
            messages.extend(extra_messages)
        return PlanExecuteState(messages=messages, input=query)

    def _format_result(
        self,
        state: PlanExecuteState,
        orchestrator_result: OrchestratorResult,
    ) -> Dict[str, Any]:
        """将新编排器的输出整理为旧接口能够消费的字典结构。"""
        report = orchestrator_result.report
        planner = orchestrator_result.planner
        execution_records = [
            record.model_dump(mode="json") if hasattr(record, "model_dump") else record
            for record in orchestrator_result.execution_records
        ]

        response = state.response
        if report and report.final_report:
            response = report.final_report

        payload: Dict[str, Any] = {
            "status": orchestrator_result.status,
            "response": response,
            "planner": planner.model_dump(mode="json") if planner else None,
            "execution_records": execution_records,
            "errors": orchestrator_result.errors,
            "metrics": orchestrator_result.metrics.model_dump(mode="json"),
        }

        if report:
            payload["report"] = {
                "outline": report.outline.model_dump(mode="json"),
                "sections": [section.model_dump(mode="json") for section in report.sections],
                "references": report.references,
                "consistency_check": (
                    report.consistency_check.model_dump(mode="json")
                    if report.consistency_check
                    else None
                ),
            }
        if state.report_context is not None:
            payload["report_context"] = {
                "report_id": state.report_context.report_id,
                "cache_hit": state.report_context.cache_hit,
            }
        return payload


__all__ = ["MultiAgentFacade"]
