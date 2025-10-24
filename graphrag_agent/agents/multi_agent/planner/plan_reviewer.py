"""
计划审校节点模块

负责对任务图进行审校并生成完整的PlanSpec
"""
from typing import Optional, Dict, Any
import json
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.config.prompts import PLAN_REVIEW_PROMPT
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanSpec,
    ProblemStatement,
    AcceptanceCriteria,
    TaskGraph,
)
from graphrag_agent.agents.multi_agent.tools.json_parser import parse_json_text

_LOGGER = logging.getLogger(__name__)


class PlanValidationResult(BaseModel):
    """
    审校校验结果
    """
    is_valid: bool = Field(default=True, description="计划是否通过内置审校")
    issues: list[str] = Field(default_factory=list, description="发现的问题列表")
    suggestions: list[str] = Field(default_factory=list, description="改进建议")
    estimated_total_tokens: Optional[int] = Field(default=None, description="预估总token消耗")
    estimated_time_minutes: Optional[float] = Field(default=None, description="预估执行时间(分钟)")
    raw_response: Optional[str] = Field(default=None, description="LLM 原始输出")


class PlanReviewOutcome(BaseModel):
    """
    审校节点产出
    """
    plan_spec: PlanSpec = Field(description="最终PlanSpec")
    validation: PlanValidationResult = Field(description="审校校验结果")
    reviewed_task_graph: TaskGraph = Field(description="审校后的任务图")
    extra_data: Dict[str, Any] = Field(default_factory=dict, description="LLM 额外输出(如说明文本)")


class PlanReviewer:
    """
    计划审校节点
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self._llm = llm or get_llm_model()

    def review(
        self,
        *,
        original_query: str,
        refined_query: Optional[str],
        task_graph: TaskGraph,
        assumptions: list[str],
        background_info: Optional[str] = None,
        user_intent: Optional[str] = None,
    ) -> PlanReviewOutcome:
        """
        对任务图执行审校并输出PlanSpec
        """
        task_graph_json = json.dumps(task_graph.to_dict(), ensure_ascii=False, indent=2)
        assumptions_text = json.dumps(assumptions or [], ensure_ascii=False)

        prompt = PLAN_REVIEW_PROMPT.format(
            query=original_query,
            refined_query=refined_query or original_query,
            task_graph=task_graph_json,
            assumptions=assumptions_text,
        )
        _LOGGER.debug("PlanReviewer prompt: %s", prompt)
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)

        problem_statement = parsed.get("problem_statement") or {}
        # 如果LLM未填充背景信息，则尝试回写外部上下文
        problem_statement.setdefault("background_info", background_info)
        problem_statement.setdefault("user_intent", user_intent)

        acceptance_data = parsed.get("acceptance_criteria") or {}
        validation_data = parsed.get("validation_results") or {}

        reviewed_task_graph = self._resolve_task_graph(parsed.get("task_graph"), task_graph)

        plan_spec = PlanSpec(
            problem_statement=ProblemStatement(**problem_statement),
            assumptions=assumptions,
            task_graph=reviewed_task_graph,
            acceptance_criteria=AcceptanceCriteria(**acceptance_data) if acceptance_data else AcceptanceCriteria(),
            status="draft",
        )

        # 若计划通过自检则标记为draft->approved交由外部确认
        validation = PlanValidationResult(
            raw_response=response,
            **{
                key: value
                for key, value in validation_data.items()
                if key in PlanValidationResult.model_fields and key != "raw_response"
            },
        )

        try:
            plan_spec.validate()
        except ValueError as exc:
            validation.is_valid = False
            validation.issues.append(str(exc))

        extra_keys = {"problem_statement", "acceptance_criteria", "validation_results", "task_graph"}
        extra_data = {k: v for k, v in parsed.items() if k not in extra_keys}

        _LOGGER.debug("PlanReviewer plan_spec: %s", plan_spec.to_dict())
        _LOGGER.debug("PlanReviewer validation: %s", validation.model_dump())
        return PlanReviewOutcome(
            plan_spec=plan_spec,
            validation=validation,
            reviewed_task_graph=reviewed_task_graph,
            extra_data=extra_data,
        )

    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM获取输出"""
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM输出为JSON"""
        try:
            return parse_json_text(response)
        except ValueError as exc:
            _LOGGER.error("PlanReviewer JSON解析失败: %s | 原始输出: %s", exc, response)
            raise ValueError("无法解析计划审校输出为有效JSON") from exc

    def _resolve_task_graph(
        self,
        maybe_task_graph: Optional[Dict[str, Any]],
        original_graph: TaskGraph,
    ) -> TaskGraph:
        """
        优先采用LLM返回的task_graph（若结构合法），否则回退到原始图
        """
        if not maybe_task_graph:
            return original_graph

        try:
            resolved_graph = TaskGraph.from_dict(maybe_task_graph)
            resolved_graph.validate_dependencies()
            return resolved_graph
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("LLM返回的task_graph无效，使用原始图. 原因: %s", exc)
            return original_graph
