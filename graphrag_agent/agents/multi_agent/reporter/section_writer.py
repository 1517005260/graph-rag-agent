"""
章节写作模块

实现分批写作能力，支持超过LLM上下文长度的长文档生成
"""
from typing import List, Dict, Any, Optional, Iterable
import logging
import textwrap
import json

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.config.prompts import SECTION_WRITE_PROMPT
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.agents.multi_agent.reporter.outline_builder import (
    ReportOutline,
    SectionOutline,
)

_LOGGER = logging.getLogger(__name__)


class SectionWriterConfig(BaseModel):
    """
    章节写作配置
    """
    max_evidence_per_call: int = Field(
        default=8,
        ge=1,
        description="单次写作调用可使用的最大证据数量"
    )
    max_previous_context_chars: int = Field(
        default=800,
        ge=200,
        description="多批写作时保留的前文摘要字符数"
    )
    enable_multi_pass: bool = Field(
        default=True,
        description="是否启用多批写作以支持超长章节"
    )


class SectionDraft(BaseModel):
    """
    章节写作结果
    """
    section_id: str = Field(description="章节ID")
    content: str = Field(description="章节Markdown内容")
    used_evidence_ids: List[str] = Field(default_factory=list, description="写作中引用的证据ID")


class SectionWriter:
    """
    章节写作器
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[SectionWriterConfig] = None,
    ) -> None:
        self._llm = llm or get_llm_model()
        self.config = config or SectionWriterConfig()

    def write_section(
        self,
        outline: ReportOutline,
        section: SectionOutline,
        evidence_map: Dict[str, RetrievalResult],
        fallback_evidence_ids: Optional[List[str]] = None,
    ) -> SectionDraft:
        """
        根据大纲与证据撰写章节内容
        """
        evidence_ids = self._select_evidence_ids(section, evidence_map, fallback_evidence_ids)
        evidence_entries = [evidence_map[eid] for eid in evidence_ids if eid in evidence_map]
        if not evidence_entries and fallback_evidence_ids:
            # 使用备用证据，避免完全无材料
            evidence_entries = [
                evidence_map[eid] for eid in fallback_evidence_ids[: self.config.max_evidence_per_call]
                if eid in evidence_map
            ]
            evidence_ids = [result.result_id for result in evidence_entries]

        batches = self._split_into_batches(evidence_entries, self.config.max_evidence_per_call)
        contents: List[str] = []
        used_ids: List[str] = []
        outline_context = self._build_outline_snapshot(outline, section)
        outline_context_text = json.dumps(outline_context, ensure_ascii=False)

        if not evidence_entries:
            placeholder = (
                f"⚠️ 当前章节《{section.title}》未检索到可引用的证据。"
                " 请补充检索或调整查询后重试。"
            )
            _LOGGER.warning(
                "No evidence available for section_id=%s title=%s",
                section.section_id,
                section.title,
            )
            return SectionDraft(
                section_id=section.section_id,
                content=placeholder,
                used_evidence_ids=[],
            )

        for batch_index, batch in enumerate(batches, start=1):
            evidence_list_text = self._format_evidence(batch)
            context_instruction = ""

            if self.config.enable_multi_pass and len(batches) > 1:
                context_instruction = textwrap.dedent(
                    f"""
                    **写作阶段**: 第{batch_index}/{len(batches)}批，请确保与前文衔接。
                    {'**前文摘要**: ' + self._extract_previous_summary(contents) if contents else ''}
                    """
                ).strip()

            prompt = SECTION_WRITE_PROMPT.format(
                outline=outline_context_text,
                section_id=section.section_id,
                section_title=section.title,
                section_summary=section.summary,
                estimated_words=section.estimated_words,
                evidence_list=evidence_list_text + ("\n\n" + context_instruction if context_instruction else "")
            )

            generated = self._invoke_llm(prompt)
            contents.append(generated.strip())
            used_ids.extend([item.result_id for item in batch])

        final_content = "\n\n".join(contents).strip()
        return SectionDraft(
            section_id=section.section_id,
            content=final_content,
            used_evidence_ids=used_ids,
        )

    def _select_evidence_ids(
        self,
        section: SectionOutline,
        evidence_map: Dict[str, RetrievalResult],
        fallback_evidence_ids: Optional[List[str]],
    ) -> List[str]:
        """按优先级选择证据ID"""
        if section.evidence_ids:
            return section.evidence_ids
        if fallback_evidence_ids:
            return fallback_evidence_ids
        return list(evidence_map.keys())

    def _split_into_batches(
        self,
        evidence_entries: List[RetrievalResult],
        batch_size: int,
    ) -> List[List[RetrievalResult]]:
        """按批次切分证据列表"""
        if not evidence_entries:
            return [[]]
        batches: List[List[RetrievalResult]] = []
        for i in range(0, len(evidence_entries), batch_size):
            batches.append(evidence_entries[i:i + batch_size])
        return batches

    def _format_evidence(self, entries: Iterable[RetrievalResult]) -> str:
        """
        将证据列表格式化为Prompt可读文本
        """
        lines = []
        for item in entries:
            snippet = ""
            if isinstance(item.evidence, str):
                snippet = item.evidence[:200].replace("\n", " ")
            elif isinstance(item.evidence, dict):
                snippet = str({k: v for k, v in list(item.evidence.items())[:4]})
            line = (
                f"- {item.result_id} | {item.granularity} | {item.source} | "
                f"置信度:{item.metadata.confidence:.2f} | 摘要:{snippet}"
            )
            lines.append(line)
        return "\n".join(lines) if lines else "（无可用证据）"

    def _extract_previous_summary(self, contents: List[str]) -> str:
        """
        从已有内容中截取摘要，用于多批写作的上下文提示
        """
        if not contents:
            return ""
        joined = "\n\n".join(contents)
        return joined[-self.config.max_previous_context_chars:]

    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM生成章节内容"""
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _build_outline_snapshot(
        self,
        outline: ReportOutline,
        section: SectionOutline,
    ) -> Dict[str, Any]:
        """
        构造精简的纲要上下文，减少提示长度但保留章节位置信息。
        """
        try:
            index = next(
                idx for idx, item in enumerate(outline.sections) if item.section_id == section.section_id
            )
        except StopIteration:
            index = 0

        previous_title = outline.sections[index - 1].title if index > 0 else None
        next_title = (
            outline.sections[index + 1].title if index + 1 < len(outline.sections) else None
        )

        snapshot: Dict[str, Any] = {
            "report_title": outline.title,
            "report_type": outline.report_type,
            "section_index": index + 1,
            "total_sections": len(outline.sections),
            "section_titles": [item.title for item in outline.sections],
        }
        if outline.abstract:
            snapshot["abstract"] = outline.abstract[:400]
        if previous_title:
            snapshot["previous_section"] = previous_title
        if next_title:
            snapshot["next_section"] = next_title
        return snapshot
