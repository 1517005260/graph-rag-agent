"""
多智能体集成工具。

提供 Facade 及工厂方法，便于旧入口在不改外部契约的前提下切换到新的
Plan-Execute-Report 编排流程。
"""

from graphrag_agent.agents.multi_agent.integration.legacy_facade import (
    MultiAgentFacade,
)
from graphrag_agent.agents.multi_agent.integration.multi_agent_factory import (
    MultiAgentFactory,
    OrchestratorBundle,
)

__all__ = [
    "MultiAgentFacade",
    "MultiAgentFactory",
    "OrchestratorBundle",
]
