"""
证据映射器 (EvidenceMapper)

负责将原始检索证据批次压缩为结构化摘要，作为章节级Reduce的输入。
"""
from typing import Iterable, List, Optional
import asyncio
import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.agents.multi_agent.reporter.outline_builder import SectionOutline
from graphrag_agent.models.get_models import get_llm_model

_LOGGER = logging.getLogger(__name__)


class EvidenceSummary(BaseModel):
    """
    证据批次摘要结构，用于后续Reduce阶段。
    """

    batch_id: str = Field(description="批次ID")
    evidence_ids: List[str] = Field(default_factory=list, description="包含的证据ID")
    key_points: List[str] = Field(default_factory=list, description="关键论点列表")
    entities: List[str] = Field(default_factory=list, description="提及的实体列表")
    summary_text: str = Field(default="", description="摘要文本")
    token_count: int = Field(default=0, description="摘要估算token数量")


class EvidenceMapper:
    """
    证据级Map操作。
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        *,
        max_evidence_per_batch: int = 8,
    ) -> None:
        self._llm = llm or get_llm_model()
        self._batch_size = max(1, max_evidence_per_batch)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def map_evidence_batch(
        self,
        evidence_batch: List[RetrievalResult],
        section_context: SectionOutline,
        *,
        batch_index: int = 0,
    ) -> EvidenceSummary:
        """
        对单个证据批次执行Map操作。
        """
        if not evidence_batch:
            return EvidenceSummary(
                batch_id=f"batch-{batch_index}",
                summary_text="",
            )

        prompt = _build_evidence_prompt(
            section_context=section_context,
            evidence_list=self._format_evidence(evidence_batch),
        )

        response = self._invoke_llm(prompt)
        parsed = self._parse_summary(
            response,
            evidence_batch=evidence_batch,
            batch_index=batch_index,
        )
        return parsed

    async def map_parallel(
        self,
        evidence_batches: List[List[RetrievalResult]],
        section_context: SectionOutline,
    ) -> List[EvidenceSummary]:
        """
        并行Map多个批次。
        """

        async def _run(index: int, batch: List[RetrievalResult]) -> EvidenceSummary:
            return await asyncio.to_thread(
                self.map_evidence_batch,
                batch,
                section_context,
                batch_index=index,
            )

        tasks = [
            _run(idx, batch)
            for idx, batch in enumerate(evidence_batches)
        ]
        return await asyncio.gather(*tasks)

    def split_batches(
        self,
        evidence_entries: Iterable[RetrievalResult],
    ) -> List[List[RetrievalResult]]:
        """
        将证据切分为批次，便于Map阶段处理。
        """
        entries = list(evidence_entries)
        if not entries:
            return [[]]
        batches: List[List[RetrievalResult]] = []
        for start in range(0, len(entries), self._batch_size):
            batches.append(entries[start: start + self._batch_size])
        return batches

    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM并返回字符串内容。"""
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _parse_summary(
        self,
        raw_text: str,
        *,
        evidence_batch: List[RetrievalResult],
        batch_index: int,
    ) -> EvidenceSummary:
        """
        尝试解析LLM返回的JSON结构。
        """
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            _LOGGER.debug(
                "证据映射结果非JSON，使用退化结构: %s error=%s",
                raw_text,
                exc,
            )
            payload = {
                "key_points": [],
                "entities": [],
                "summary_text": raw_text.strip(),
            }

        evidence_ids = [item.result_id for item in evidence_batch]
        summary = EvidenceSummary(
            batch_id=f"batch-{batch_index}",
            evidence_ids=evidence_ids,
            key_points=payload.get("key_points") or [],
            entities=payload.get("entities") or [],
            summary_text=payload.get("summary_text", "").strip(),
        )

        if not summary.summary_text:
            summary.summary_text = "\n".join(summary.key_points)

        summary.token_count = _estimate_token_count(summary.summary_text)
        return summary

    def _format_evidence(
        self,
        entries: Iterable[RetrievalResult],
    ) -> str:
        """
        将证据列表格式化为Prompt需要的文本。
        """
        lines: List[str] = []
        for item in entries:
            snippet = ""
            if isinstance(item.evidence, str):
                snippet = item.evidence.replace("\n", " ")[:240]
            elif isinstance(item.evidence, dict):
                preview = {
                    key: item.evidence[key]
                    for key in list(item.evidence.keys())[:4]
                }
                snippet = json.dumps(preview, ensure_ascii=False)
            lines.append(
                f"- {item.result_id} | {item.granularity} | {item.source} | "
                f"置信度:{item.metadata.confidence:.2f} | 内容:{snippet}"
            )
        return "\n".join(lines) if lines else "（无可用证据）"


def _build_evidence_prompt(
    section_context: SectionOutline,
    evidence_list: str,
) -> str:
    return EVIDENCE_MAP_PROMPT.format(
        section_title=section_context.title,
        section_goal=section_context.summary,
        evidence_list=evidence_list,
    )


def _estimate_token_count(text: str) -> int:
    """
    粗略估算token数量，避免依赖外部tokenizer。
    """
    if not text:
        return 0
    # 使用近似换算：中文按字符数，英文按词数估算
    length = len(text)
    english_words = len(text.split())
    return max(length // 2, english_words)


EVIDENCE_MAP_PROMPT = """
你是一个信息提取专家。请从以下证据中提取关键信息，用于撰写章节「{section_title}」。

**章节目标**: {section_goal}

**证据列表**:
{evidence_list}

**任务**:
1. 列出3-5个关键论点（key_points）
2. 识别所有提及的实体（entities）
3. 生成200字以内的摘要文本（summary_text）

**输出格式**（JSON）:
{{
    "key_points": ["论点1", "论点2"],
    "entities": ["实体1", "实体2"],
    "summary_text": "摘要文本..."
}}

如果无法提取信息，请返回空的JSON对象。
""".strip()
