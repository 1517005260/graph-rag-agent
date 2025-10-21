"""
一致性校验模块
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import json
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.config.prompts import CONSISTENCY_CHECK_PROMPT
from graphrag_agent.models.get_models import get_llm_model

_LOGGER = logging.getLogger(__name__)


class ConsistencyCheckResult(BaseModel):
    """
    一致性检查结果
    """
    is_consistent: bool = Field(default=True, description="报告是否通过检查")
    issues: list[Dict[str, Any]] = Field(default_factory=list, description="问题列表")
    corrections: list[Dict[str, Any]] = Field(default_factory=list, description="修正建议")
    raw_response: Optional[str] = Field(default=None, description="LLM原始输出")


class ConsistencyChecker:
    """
    调用LLM进行事实与引用一致性校验
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm or get_llm_model()

    def check(self, report_content: str, evidence_list: str) -> ConsistencyCheckResult:
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            report_content=report_content,
            evidence_list=evidence_list,
        )
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        result = ConsistencyCheckResult(**parsed, raw_response=response)
        _LOGGER.debug("ConsistencyChecker 输出: %s", result.model_dump())
        return result

    def _invoke_llm(self, prompt: str) -> str:
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            segments = cleaned.split("```")
            if len(segments) >= 3:
                cleaned = segments[2] if segments[1].strip().startswith("json") else segments[1]
        if "{" in cleaned and "}" in cleaned:
            cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            _LOGGER.error("ConsistencyChecker JSON解析失败: %s | 原始输出: %s", exc, response)
            raise ValueError("一致性校验结果解析失败") from exc
