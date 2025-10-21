"""
纲要生成模块

根据PlanSpec与执行证据构建结构化报告纲要
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.config.prompts import OUTLINE_PROMPT
from graphrag_agent.models.get_models import get_llm_model

_LOGGER = logging.getLogger(__name__)


class SectionOutline(BaseModel):
    """
    单个章节纲要信息
    """
    section_id: str = Field(description="章节唯一标识符")
    title: str = Field(description="章节标题")
    summary: str = Field(description="章节摘要说明")
    evidence_ids: List[str] = Field(default_factory=list, description="章节引用的证据ID列表")
    estimated_words: int = Field(default=400, description="预估字数")


class ReportOutline(BaseModel):
    """
    报告纲要结构
    """
    report_type: str = Field(description="报告类型 short_answer | long_document")
    title: str = Field(description="报告标题")
    abstract: Optional[str] = Field(default=None, description="摘要（长文档特有）")
    sections: List[SectionOutline] = Field(default_factory=list, description="章节列表")
    total_estimated_words: Optional[int] = Field(default=None, description="预估总字数")


class OutlineBuilder:
    """
    纲要生成器，调用LLM根据证据生成结构化大纲
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm or get_llm_model()

    def build_outline(
        self,
        *,
        query: str,
        plan_summary: str,
        evidence_summary: str,
        evidence_count: int,
        report_type: str,
    ) -> ReportOutline:
        """
        核心入口：生成报告纲要
        """
        prompt = OUTLINE_PROMPT.format(
            query=query,
            plan_summary=plan_summary,
            evidence_summary=evidence_summary,
            evidence_count=evidence_count,
            report_type=report_type,
        )
        response = self._invoke_llm(prompt)
        outline_data = self._parse_response(response)
        outline = ReportOutline(**outline_data)
        _LOGGER.debug("OutlineBuilder 输出: %s", outline.model_dump())
        return outline

    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM生成大纲"""
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM返回的JSON字符串
        """
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
            _LOGGER.error("OutlineBuilder JSON解析失败: %s | 原始输出: %s", exc, response)
            raise ValueError("纲要生成结果解析失败") from exc
