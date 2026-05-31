"""
报告组装器 (ReportAssembler)

负责在Map-Reduce流程中对章节结果进行全局组装，生成最终报告。
"""
from typing import Any, Dict
import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.agents.multi_agent.reporter.outline_builder import ReportOutline
from graphrag_agent.config.prompts import (
    INTRO_PROMPT,
    CONCLUSION_PROMPT,
    TERMINOLOGY_PROMPT,
)
from graphrag_agent.models.get_models import get_llm_model

_LOGGER = logging.getLogger(__name__)


class ReportAssembler:
    """
    报告级组装器。
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm or get_llm_model()

    def assemble(
        self,
        outline: ReportOutline,
        section_contents: Dict[str, str],
        *,
        global_context: Dict[str, Any] | None = None,
    ) -> str:
        """
        根据章节内容生成完整报告。
        """
        global_context = global_context or {}

        terminology = self._extract_global_terminology(section_contents)
        introduction_needed = not self._has_intro_section(outline)
        conclusion_needed = not self._has_conclusion_section(outline)
        introduction = (
            self._generate_introduction(outline, section_contents, global_context)
            if introduction_needed
            else ""
        )
        conclusion = (
            self._generate_conclusion(outline, section_contents, global_context)
            if conclusion_needed
            else ""
        )

        parts = [f"# {outline.title}"]

        if outline.report_type == "long_document" and outline.abstract:
            parts.extend(["", "## 摘要", outline.abstract.strip()])

        if introduction_needed and introduction.strip():
            parts.extend(["", "## 引言", introduction.strip()])

        for section in outline.sections:
            content = section_contents.get(section.section_id, "").strip()
            if not content:
                continue
            parts.extend(["", f"## {section.title}", content])

        if conclusion_needed and conclusion.strip():
            parts.extend(["", "## 结论", conclusion.strip()])

        if terminology:
            parts.extend(["", "## 术语表", self._format_terminology(terminology)])

        return "\n".join(parts).strip()

    def _extract_global_terminology(
        self,
        section_contents: Dict[str, str],
    ) -> Dict[str, str]:
        text = "\n\n".join(section_contents.values())
        if not text.strip():
            return {}
        prompt = TERMINOLOGY_PROMPT.format(section_text=text[:6000])
        content = self._invoke_llm(prompt)
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return {
                    str(key): str(value)
                    for key, value in data.items()
                }
        except json.JSONDecodeError as exc:
            _LOGGER.debug("术语解析失败，忽略: %s error=%s", content, exc)
        return {}

    def _generate_introduction(
        self,
        outline: ReportOutline,
        section_contents: Dict[str, str],
        global_context: Dict[str, Any],
    ) -> str:
        section_summaries = "\n".join(
            f"- {section.title}: {section.summary}"
            for section in outline.sections
        )
        prompt = INTRO_PROMPT.format(
            report_title=outline.title,
            query=global_context.get("query", ""),
            section_summaries=section_summaries,
        )
        return self._invoke_llm(prompt)

    @staticmethod
    def _has_intro_section(outline: ReportOutline) -> bool:
        if not outline.sections:
            return False
        title = outline.sections[0].title.lower()
        return any(keyword in title for keyword in ("引言", "introduction"))

    @staticmethod
    def _has_conclusion_section(outline: ReportOutline) -> bool:
        if not outline.sections:
            return False
        title = outline.sections[-1].title.lower()
        return any(keyword in title for keyword in ("结论", "conclusion", "总结"))

    def _generate_conclusion(
        self,
        outline: ReportOutline,
        section_contents: Dict[str, str],
        global_context: Dict[str, Any],
    ) -> str:
        aggregated = "\n\n".join(
            f"## {section.title}\n{section_contents.get(section.section_id, '')}"
            for section in outline.sections
            if section_contents.get(section.section_id)
        )
        prompt = CONCLUSION_PROMPT.format(
            report_title=outline.title,
            section_content=aggregated[:6000],
            evidence_count=global_context.get("evidence_count", 0),
        )
        return self._invoke_llm(prompt)

    def _format_terminology(self, terminology: Dict[str, str]) -> str:
        lines = [
            f"- **{term}**: {meaning}"
            for term, meaning in terminology.items()
        ]
        return "\n".join(lines)

    def _invoke_llm(self, prompt: str) -> str:
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()
