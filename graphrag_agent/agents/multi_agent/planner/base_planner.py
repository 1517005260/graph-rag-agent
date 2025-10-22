"""
Planner编排基类

整合Clarifier、TaskDecomposer、PlanReviewer，输出结构化的PlanSpec
"""
from typing import Optional, List
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from graphrag_agent.agents.multi_agent.core.state import (
    PlanExecuteState,
    PlanContext,
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanSpec,
    PlanExecutionSignal,
)
from graphrag_agent.agents.multi_agent.planner.clarifier import (
    Clarifier,
    ClarificationResult,
)
from graphrag_agent.agents.multi_agent.planner.task_decomposer import (
    TaskDecomposer,
    TaskDecompositionResult,
)
from graphrag_agent.agents.multi_agent.planner.plan_reviewer import (
    PlanReviewer,
    PlanReviewOutcome,
)
from graphrag_agent.models.get_models import get_llm_model

_LOGGER = logging.getLogger(__name__)


class PlannerConfig(BaseModel):
    """
    Planner配置
    """
    max_tasks: int = Field(default=6, description="单次任务分解允许的最大任务数")
    allow_unclarified_plan: bool = Field(
        default=True,
        description="若存在未解决的澄清问题，是否强制继续生成计划",
    )
    default_domain: str = Field(default="通用", description="默认领域背景，用于澄清提示")


class PlannerResult(BaseModel):
    """
    Planner综合输出
    """
    plan_spec: Optional[PlanSpec] = Field(default=None, description="最终生成的PlanSpec")
    clarification: ClarificationResult = Field(description="澄清结果")
    task_decomposition: Optional[TaskDecompositionResult] = Field(
        default=None,
        description="任务分解详细结果",
    )
    review_outcome: Optional[PlanReviewOutcome] = Field(
        default=None,
        description="计划审校产出详情",
    )
    executor_signal: Optional[PlanExecutionSignal] = Field(
        default=None,
        description="传递给Executor的计划信号"
    )

    def needs_clarification(self) -> bool:
        """是否仍需澄清"""
        if not self.clarification.needs_clarification:
            return False
        if not self.clarification.questions:
            return False
        # 未生成PlanSpec时视为待澄清
        return self.plan_spec is None

    def executor_signal_json(self) -> Optional[str]:
        """
        将执行信号转换为JSON字符串，便于跨组件传递
        """
        if self.executor_signal is None:
            return None
        # Pydantic v1/v2 兼容性处理
        import json
        return json.dumps(self.executor_signal.model_dump(mode="json"), ensure_ascii=False, indent=2)


class BasePlanner:
    """
    Planner基类

    提供生成PlanSpec的标准流程：澄清 → 任务分解 → 计划审校
    """

    def __init__(
        self,
        *,
        llm: Optional[BaseChatModel] = None,
        config: Optional[PlannerConfig] = None,
        clarifier: Optional[Clarifier] = None,
        task_decomposer: Optional[TaskDecomposer] = None,
        plan_reviewer: Optional[PlanReviewer] = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self._llm = llm or get_llm_model()

        # 所有子组件共享同一个LLM实例，便于缓存与限流
        self._clarifier = clarifier or Clarifier(self._llm, default_domain=self.config.default_domain)
        self._task_decomposer = task_decomposer or TaskDecomposer(self._llm, max_tasks=self.config.max_tasks)
        self._plan_reviewer = plan_reviewer or PlanReviewer(self._llm)

    def generate_plan(
        self,
        state: PlanExecuteState,
        *,
        assumptions: Optional[List[str]] = None,
    ) -> PlannerResult:
        """
        核心入口：生成任务执行计划

        参数:
            state: 当前会话的PlanExecuteState
            assumptions: 用户确认的前提条件

        返回:
            PlannerResult
        """
        # 确保PlanContext存在
        context = self._ensure_plan_context(state)

        # Step 1: 澄清分析
        clarification = self._clarifier.analyze(context)
        _LOGGER.info("Clarification result: %s", clarification.model_dump())

        if not clarification.is_satisfied(context) and not self.config.allow_unclarified_plan:
            _LOGGER.info("澄清问题尚未全部回答，暂停计划生成")
            return PlannerResult(
                plan_spec=None,
                clarification=clarification,
                task_decomposition=None,
                review_outcome=None,
            )

        # Step 2: 任务分解
        refined_query = context.refined_query or context.original_query
        task_decomposition = self._task_decomposer.decompose(refined_query)

        # Step 3: 计划审校
        review_outcome = self._plan_reviewer.review(
            original_query=context.original_query,
            refined_query=refined_query,
            task_graph=task_decomposition.task_graph,
            assumptions=assumptions or [],
            background_info=context.domain_context,
            user_intent=context.user_preferences.get("intent"),
        )

        plan_spec = review_outcome.plan_spec
        # 将生成的计划写回状态
        state.plan = plan_spec
        state.plan_context = context
        state.update_timestamp()

        executor_signal = plan_spec.to_execution_signal()

        return PlannerResult(
            plan_spec=plan_spec,
            clarification=clarification,
            task_decomposition=task_decomposition,
            review_outcome=review_outcome,
            executor_signal=executor_signal,
        )

    def _ensure_plan_context(self, state: PlanExecuteState) -> PlanContext:
        """
        确保PlanContext存在并回填基础字段
        """
        if state.plan_context is None:
            state.plan_context = PlanContext(original_query=state.input or "")

        context = state.plan_context
        if not context.original_query:
            context.original_query = state.input
        if not context.refined_query:
            # 若尚未有澄清后的查询，则默认等同于原始查询
            context.refined_query = context.original_query
        return context
