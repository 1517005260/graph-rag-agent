"""
澄清节点模块

负责根据用户原始查询识别模糊点并生成澄清问题
"""
from typing import Optional, Any, Dict, List
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.config.prompts import CLARIFY_PROMPT
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.agents.multi_agent.core.state import PlanContext
from graphrag_agent.agents.multi_agent.tools.json_parser import parse_json_text

_LOGGER = logging.getLogger(__name__)


class ClarificationResult(BaseModel):
    """
    澄清结果数据模型
    """
    needs_clarification: bool = Field(default=False, description="是否需要额外澄清")
    questions: List[str] = Field(default_factory=list, description="需要向用户提出的澄清问题列表")
    ambiguity_types: List[str] = Field(default_factory=list, description="检测到的模糊类型标签")
    raw_response: Optional[str] = Field(default=None, description="LLM 原始输出，便于调试")

    def is_satisfied(self, context: PlanContext) -> bool:
        """
        判断澄清问题是否已被回答

        规则：
            - 若无需澄清，直接返回 True
            - 若需要澄清，则需在上下文的澄清历史中找到对应问题且答案非空
        """
        if not self.needs_clarification:
            return True

        answered_questions = {
            item.get("question"): item.get("answer")
            for item in context.clarification_history
            if item.get("question")
        }
        for question in self.questions:
            answer = answered_questions.get(question)
            if not answer:
                return False
        return True


class Clarifier:
    """
    查询澄清节点

    调用 LLM 对用户查询进行模糊性分析，输出 ClarificationResult
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        *,
        default_domain: str = "通用",
    ) -> None:
        # 为便于单元测试支持注入自定义 LLM
        self._llm = llm or get_llm_model()
        self._default_domain = default_domain

    def analyze(self, context: PlanContext) -> ClarificationResult:
        """
        分析查询是否需要澄清，并生成具体问题

        参数:
            context: Planner 当前上下文

        返回:
            ClarificationResult
        """
        prompt = CLARIFY_PROMPT.format(
            query=context.refined_query or context.original_query,
            domain=context.domain_context or self._default_domain,
        )

        _LOGGER.debug("Clarifier prompt: %s", prompt)
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        result = ClarificationResult(**parsed, raw_response=response)
        _LOGGER.debug("Clarifier result: %s", result.model_dump())
        return result

    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM并提取文本内容"""
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """将LLM输出解析为JSON结构"""
        try:
            return parse_json_text(response)
        except ValueError as exc:
            _LOGGER.error("Clarifier JSON解析失败: %s | 原始输出: %s", exc, response)
            raise ValueError("无法解析澄清节点输出为有效JSON") from exc
