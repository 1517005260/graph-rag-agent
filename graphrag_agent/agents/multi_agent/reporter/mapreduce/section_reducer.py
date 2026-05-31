"""
章节归约器 (SectionReducer)

负责将证据摘要归约为章节内容，支持多种Reduce策略。
"""
from enum import Enum
from typing import Iterable, List, Optional
import json
import logging
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.agents.multi_agent.reporter.mapreduce.evidence_mapper import (
    EvidenceSummary,
)
from graphrag_agent.agents.multi_agent.reporter.outline_builder import SectionOutline
from graphrag_agent.config.prompts import (
    SECTION_REDUCE_PROMPT,
    INTERMEDIATE_SUMMARY_PROMPT,
    MERGE_PROMPT,
    REFINE_PROMPT,
)
from graphrag_agent.models.get_models import get_llm_model

_LOGGER = logging.getLogger(__name__)


class ReduceStrategy(str, Enum):
    """章节归约策略枚举。"""

    COLLAPSE = "collapse"
    REFINE = "refine"
    TREE = "tree"


class SectionReducer:
    """
    章节级Reduce操作，实现多种Map-Reduce归约策略。
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        *,
        strategy: ReduceStrategy = ReduceStrategy.TREE,
    ) -> None:
        self._llm = llm or get_llm_model()
        self._strategy = strategy

    def reduce(
        self,
        evidence_summaries: List[EvidenceSummary],
        section_context: SectionOutline,
        *,
        max_tokens: int = 4000,
    ) -> str:
        """
        将证据摘要归约为章节内容。
        """
        if not evidence_summaries:
            return ""

        summaries = [summary.copy() for summary in evidence_summaries]

        if self._strategy == ReduceStrategy.COLLAPSE:
            return self._collapse_reduce(
                summaries,
                section_context,
                max_tokens=max_tokens,
            )
        if self._strategy == ReduceStrategy.REFINE:
            return self._refine_reduce(
                summaries,
                section_context,
            )
        return self._tree_reduce(
            summaries,
            section_context,
            max_tokens=max_tokens,
        )

    def _collapse_reduce(
        self,
        summaries: List[EvidenceSummary],
        section_context: SectionOutline,
        *,
        max_tokens: int,
    ) -> str:
        total_tokens = sum(item.token_count for item in summaries)
        if total_tokens <= max_tokens:
            return self._generate_section_text(summaries, section_context)

        intermediate: List[EvidenceSummary] = []
        current: List[EvidenceSummary] = []
        current_tokens = 0

        for summary in summaries:
            if current_tokens + summary.token_count > max_tokens and current:
                intermediate.append(
                    self._generate_intermediate_summary(
                        current,
                        section_context,
                    )
                )
                current = [summary]
                current_tokens = summary.token_count
            else:
                current.append(summary)
                current_tokens += summary.token_count

        if current:
            intermediate.append(
                self._generate_intermediate_summary(
                    current,
                    section_context,
                )
            )

        return self._collapse_reduce(
            intermediate,
            section_context,
            max_tokens=max_tokens,
        )

    def _tree_reduce(
        self,
        summaries: List[EvidenceSummary],
        section_context: SectionOutline,
        *,
        max_tokens: int,
    ) -> str:
        if len(summaries) == 1:
            return self._generate_section_text(summaries, section_context)

        next_level: List[EvidenceSummary] = []
        iterator = iter(range(0, len(summaries), 2))
        for index in iterator:
            left = summaries[index]
            right = summaries[index + 1] if index + 1 < len(summaries) else None
            if right is None:
                next_level.append(left)
                continue
            next_level.append(
                self._merge_two_summaries(left, right, section_context)
            )

        if len(next_level) == 1 and next_level[0].token_count <= max_tokens:
            return self._generate_section_text(next_level, section_context)
        return self._tree_reduce(
            next_level,
            section_context,
            max_tokens=max_tokens,
        )

    def _refine_reduce(
        self,
        summaries: List[EvidenceSummary],
        section_context: SectionOutline,
    ) -> str:
        initial = self._generate_section_text(
            [summaries[0]],
            section_context,
        )
        current_draft = initial

        for summary in summaries[1:]:
            prompt = REFINE_PROMPT.format(
                current_draft=current_draft,
                new_evidence=summary.summary_text,
                section_title=section_context.title,
            )
            current_draft = self._invoke_llm(prompt)
        return current_draft.strip()

    def _generate_section_text(
        self,
        summaries: Iterable[EvidenceSummary],
        section_context: SectionOutline,
    ) -> str:
        text = "\n".join(
            _format_summary_for_prompt(item)
            for item in summaries
        )
        prompt = SECTION_REDUCE_PROMPT.format(
            section_title=section_context.title,
            section_goal=section_context.summary,
            estimated_words=section_context.estimated_words,
            evidence_summaries=text,
        )
        return self._invoke_llm(prompt)

    def _generate_intermediate_summary(
        self,
        summaries: List[EvidenceSummary],
        section_context: SectionOutline,
    ) -> EvidenceSummary:
        text = "\n".join(
            _format_summary_for_prompt(item)
            for item in summaries
        )
        prompt = INTERMEDIATE_SUMMARY_PROMPT.format(
            section_title=section_context.title,
            section_goal=section_context.summary,
            evidence_summaries=text,
        )
        content = self._invoke_llm(prompt)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            _LOGGER.debug(
                "中间归约结果非JSON，使用降级文本: %s error=%s",
                content,
                exc,
            )
            payload = {
                "summary_text": content.strip(),
                "key_points": [],
                "entities": [],
            }

        summary = EvidenceSummary(
            batch_id=f"reduce-{uuid.uuid4().hex[:8]}",
            evidence_ids=[
                eid
                for item in summaries
                for eid in item.evidence_ids
            ],
            key_points=payload.get("key_points") or [],
            entities=payload.get("entities") or [],
            summary_text=payload.get("summary_text", "").strip(),
        )
        if not summary.summary_text:
            summary.summary_text = "\n".join(summary.key_points)
        summary.token_count = max(
            _estimate_token_count(summary.summary_text),
            sum(item.token_count for item in summaries) // 2,
        )
        return summary

    def _merge_two_summaries(
        self,
        left: EvidenceSummary,
        right: EvidenceSummary,
        section_context: SectionOutline,
    ) -> EvidenceSummary:
        prompt = MERGE_PROMPT.format(
            section_title=section_context.title,
            left_summary=_format_summary_for_prompt(left),
            right_summary=_format_summary_for_prompt(right),
        )
        content = self._invoke_llm(prompt)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {
                "summary_text": content.strip(),
                "key_points": [],
                "entities": [],
            }

        summary = EvidenceSummary(
            batch_id=f"merge-{uuid.uuid4().hex[:8]}",
            evidence_ids=list({*left.evidence_ids, *right.evidence_ids}),
            key_points=payload.get("key_points") or [],
            entities=payload.get("entities") or [],
            summary_text=payload.get("summary_text", "").strip(),
        )
        if not summary.summary_text:
            summary.summary_text = "\n".join(filter(None, [
                left.summary_text,
                right.summary_text,
            ]))
        summary.token_count = _estimate_token_count(summary.summary_text)
        return summary

    def _invoke_llm(self, prompt: str) -> str:
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()


def _format_summary_for_prompt(summary: EvidenceSummary) -> str:
    points = "\n".join(f"  - {point}" for point in summary.key_points)
    entities = ", ".join(summary.entities) if summary.entities else "无"
    return (
        f"[{summary.batch_id}] 证据ID: {', '.join(summary.evidence_ids)}\n"
        f"摘要: {summary.summary_text}\n"
        f"关键论点:\n{points if points else '  - 无'}\n"
        f"实体: {entities}"
    )


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    length = len(text)
    english_words = len(text.split())
    return max(length // 2, english_words)
