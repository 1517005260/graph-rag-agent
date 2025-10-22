"""
构建多智能体编排器所需组件的工厂工具。
"""
from dataclasses import dataclass
from typing import Optional

from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.agents.multi_agent.executor.worker_coordinator import (
    WorkerCoordinator,
)
from graphrag_agent.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
)
from graphrag_agent.agents.multi_agent.planner.base_planner import BasePlanner
from graphrag_agent.agents.multi_agent.reporter.base_reporter import BaseReporter
from graphrag_agent.agents.multi_agent.reporter.base_reporter import ReporterConfig


@dataclass
class OrchestratorBundle:
    """封装编排器所依赖的 Planner、Worker、Reporter 等组件。"""

    planner: BasePlanner
    worker: WorkerCoordinator
    reporter: BaseReporter
    orchestrator: MultiAgentOrchestrator


class MultiAgentFactory:
    """用于创建带默认配置的多智能体编排栈。"""

    @staticmethod
    def create_default_bundle(
        *,
        planner: Optional[BasePlanner] = None,
        worker: Optional[WorkerCoordinator] = None,
        reporter: Optional[BaseReporter] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        reporter_config: Optional[ReporterConfig] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> OrchestratorBundle:
        planner = planner or BasePlanner()
        worker = worker or WorkerCoordinator()
        reporter = reporter or BaseReporter(
            config=reporter_config,
            cache_manager=cache_manager,
        )
        orchestrator = MultiAgentOrchestrator(
            planner=planner,
            worker_coordinator=worker,
            reporter=reporter,
            config=orchestrator_config or OrchestratorConfig(),
        )
        return OrchestratorBundle(
            planner=planner,
            worker=worker,
            reporter=reporter,
            orchestrator=orchestrator,
        )


__all__ = ["MultiAgentFactory", "OrchestratorBundle"]
