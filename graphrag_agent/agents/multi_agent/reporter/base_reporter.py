"""
Reporter编排基类

负责串联纲要生成、章节写作、引用整理与一致性校验
"""
from typing import List, Dict, Any, Optional, Iterable, Tuple
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from graphrag_agent.agents.multi_agent.core.plan_spec import PlanSpec
from graphrag_agent.agents.multi_agent.core.execution_record import ExecutionRecord
from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.reporter.outline_builder import (
    OutlineBuilder,
    ReportOutline,
    SectionOutline,
)
from graphrag_agent.agents.multi_agent.reporter.section_writer import (
    SectionWriter,
    SectionWriterConfig,
    SectionDraft,
)
from graphrag_agent.agents.multi_agent.reporter.consistency_checker import (
    ConsistencyChecker,
    ConsistencyCheckResult,
)
from graphrag_agent.agents.multi_agent.reporter.formatter import CitationFormatter
from graphrag_agent.agents.multi_agent.reporter.mapreduce import (
    EvidenceMapper,
    EvidenceSummary,
    SectionReducer,
    ReduceStrategy,
    ReportAssembler,
)
from graphrag_agent.agents.multi_agent.tools.evidence_tracker import EvidenceTracker
from graphrag_agent.cache_manager.manager import CacheManager

_LOGGER = logging.getLogger(__name__)


class ReporterConfig(BaseModel):
    """
    Reporter层配置
    """
    default_report_type: str = Field(default="long_document", description="默认报告类型")
    citation_style: str = Field(default="default", description="引用格式类型")
    max_evidence_summary: int = Field(default=30, description="纲要生成时展示的最大证据条数")
    section_writer: SectionWriterConfig = Field(default_factory=SectionWriterConfig, description="章节写作配置")
    enable_consistency_check: bool = Field(default=True, description="是否启用一致性检查")
    enable_mapreduce: bool = Field(default=True, description="是否启用Map-Reduce写作模式")
    reduce_strategy: ReduceStrategy = Field(default=ReduceStrategy.TREE, description="章节Reduce策略")
    max_tokens_per_reduce: int = Field(default=4000, ge=1000, description="Reduce阶段单次调用的最大token估算")
    enable_parallel_map: bool = Field(default=True, description="证据Map阶段是否启用并行")
    mapreduce_evidence_threshold: int = Field(default=20, ge=0, description="触发Map-Reduce模式的证据数量阈值")


class SectionContent(BaseModel):
    """
    章节内容包装
    """
    section_id: str = Field(description="章节ID")
    title: str = Field(description="章节标题")
    content: str = Field(description="章节内容（Markdown）")
    used_evidence_ids: List[str] = Field(default_factory=list, description="引用的证据ID")


class ReportResult(BaseModel):
    """
    Reporter最终结果
    """
    outline: ReportOutline = Field(description="报告纲要")
    sections: List[SectionContent] = Field(description="章节内容列表")
    final_report: str = Field(description="最终报告Markdown文本")
    references: Optional[str] = Field(default=None, description="引用列表Markdown")
    consistency_check: Optional[ConsistencyCheckResult] = Field(default=None, description="一致性检查结果")


class BaseReporter:
    """
    Reporter基类，串联各个子组件
    """

    def __init__(
        self,
        config: Optional[ReporterConfig] = None,
        *,
        outline_builder: Optional[OutlineBuilder] = None,
        section_writer: Optional[SectionWriter] = None,
        consistency_checker: Optional[ConsistencyChecker] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        cache_manager: Optional[CacheManager] = None,
        evidence_mapper: Optional[EvidenceMapper] = None,
        section_reducer: Optional[SectionReducer] = None,
        report_assembler: Optional[ReportAssembler] = None,
    ) -> None:
        self.config = config or ReporterConfig()
        self._outline_builder = outline_builder or OutlineBuilder()
        self._section_writer = section_writer or SectionWriter(config=self.config.section_writer)
        self._consistency_checker = consistency_checker or ConsistencyChecker()
        self._citation_formatter = citation_formatter or CitationFormatter()
        self._cache_manager = cache_manager
        self._evidence_mapper = evidence_mapper
        self._section_reducer = section_reducer
        self._report_assembler = report_assembler

    def generate_report(
        self,
        state: PlanExecuteState,
        plan: Optional[PlanSpec] = None,
        execution_records: Optional[List[ExecutionRecord]] = None,
        report_type: Optional[str] = None,
    ) -> ReportResult:
        """生成报告主流程，支持report_id级缓存与分段复用"""

        plan = plan or state.plan
        if plan is None:
            raise ValueError("生成报告需要PlanSpec")

        execution_records = execution_records or state.execution_records

        resolved_report_type = report_type or (
            state.report_context.report_type if state.report_context else None
        )
        if not resolved_report_type:
            resolved_report_type = self.config.default_report_type

        report_id = self._resolve_report_id(plan, resolved_report_type)
        evidence_map = self._collect_evidence(state, execution_records)
        evidence_fingerprint = self._build_evidence_fingerprint(evidence_map)

        cached_payload = self._load_cached_payload(report_id)
        if (
            cached_payload
            and cached_payload.get("evidence_fingerprint") == evidence_fingerprint
        ):
            report_result = self._deserialize_report_result(cached_payload)
            self._update_state_report_context(
                state,
                report_result,
                report_id,
                cache_hit=True,
            )
            state.response = report_result.final_report
            state.update_timestamp()
            return report_result

        plan_summary = self._build_plan_summary(plan)
        evidence_summary, limited_ids = self._build_evidence_summary(evidence_map)

        outline = self._outline_builder.build_outline(
            query=plan.problem_statement.original_query,
            plan_summary=plan_summary,
            evidence_summary=evidence_summary,
            evidence_count=len(evidence_map),
            report_type=resolved_report_type,
        )

        section_cache_index = self._build_section_cache_index(cached_payload)
        use_mapreduce = self._should_use_mapreduce(evidence_map)

        if use_mapreduce:
            section_contents, used_evidence_ids = self._generate_sections_mapreduce(
                outline=outline,
                evidence_map=evidence_map,
                section_cache_index=section_cache_index,
                evidence_fingerprint=evidence_fingerprint,
                fallback_evidence_ids=limited_ids,
            )
            final_report = self._assemble_report_mapreduce(
                outline=outline,
                section_contents=section_contents,
                query=plan.problem_statement.original_query,
                evidence_count=len(evidence_map),
            )
        else:
            section_contents, used_evidence_ids = self._generate_sections_traditional(
                outline=outline,
                evidence_map=evidence_map,
                section_cache_index=section_cache_index,
                evidence_fingerprint=evidence_fingerprint,
                fallback_evidence_ids=limited_ids,
            )
            final_report = self._assemble_report(outline, section_contents)

        consistency_result: Optional[ConsistencyCheckResult] = None
        if self.config.enable_consistency_check and evidence_map:
            evidence_text = self._format_evidence_for_check(evidence_map.values())
            try:
                consistency_result = self._consistency_checker.check(
                    final_report, evidence_text
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("一致性检查失败: %s", exc)

        references = self._format_references(evidence_map, used_evidence_ids)

        report_result = ReportResult(
            outline=outline,
            sections=section_contents,
            final_report=final_report,
            references=references,
            consistency_check=consistency_result,
        )

        self._save_report_cache(
            report_id,
            evidence_fingerprint,
            report_result,
            outline,
            evidence_map,
        )

        self._update_state_report_context(
            state,
            report_result,
            report_id,
            cache_hit=False,
        )
        state.response = final_report
        state.update_timestamp()
        return report_result

    def _collect_evidence(
        self,
        state: PlanExecuteState,
        execution_records: Iterable[ExecutionRecord],
    ) -> Dict[str, RetrievalResult]:
        """从执行记录与执行上下文中聚合标准化的RetrievalResult"""

        evidence_map: Dict[str, RetrievalResult] = {}

        tracker = _extract_tracker_from_state(state)
        if tracker is not None:
            for result in tracker.all_results():
                evidence_map[result.result_id] = result

        for record in execution_records:
            for item in record.evidence:
                try:
                    if isinstance(item, RetrievalResult):
                        result = item
                    elif isinstance(item, dict):
                        result = RetrievalResult.from_dict(item)
                    else:
                        continue
                    evidence_map[result.result_id] = result
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.debug("忽略无法解析的证据: %s error=%s", item, exc)

        return evidence_map

    def _build_plan_summary(self, plan: PlanSpec) -> str:
        """
        根据PlanSpec生成用于提示纲要的任务摘要
        """
        lines = [f"计划ID: {plan.plan_id}, 版本: {plan.version}, 状态: {plan.status}"]
        for node in plan.task_graph.nodes:
            lines.append(
                f"- {node.task_id} | 类型:{node.task_type} | 状态:{node.status} | "
                f"优先级:{node.priority} | 描述:{node.description}"
            )
        return "\n".join(lines)

    def _build_evidence_summary(
        self,
        evidence_map: Dict[str, RetrievalResult],
    ) -> Tuple[str, List[str]]:
        """
        构建用于纲要生成的证据摘要，限制最大条数，防止Prompt过长
        """
        lines: List[str] = []
        limited_ids: List[str] = []
        for idx, result in enumerate(evidence_map.values()):
            if idx >= self.config.max_evidence_summary:
                lines.append("...（其余证据省略）")
                break
            snippet = ""
            if isinstance(result.evidence, str):
                snippet = result.evidence[:160].replace("\n", " ")
            elif isinstance(result.evidence, dict):
                snippet = str({k: result.evidence[k] for k in list(result.evidence.keys())[:4]})
            line = f"{result.result_id} | {result.granularity} | {result.source} | {snippet}"
            lines.append(line)
            limited_ids.append(result.result_id)
        summary_text = "\n".join(lines) if lines else "无结构化证据"
        return summary_text, limited_ids

    def _should_use_mapreduce(
        self,
        evidence_map: Dict[str, RetrievalResult],
    ) -> bool:
        """
        判断是否启用Map-Reduce模式。
        """
        if not self.config.enable_mapreduce:
            return False
        return len(evidence_map) >= self.config.mapreduce_evidence_threshold

    def _generate_sections_traditional(
        self,
        outline: ReportOutline,
        evidence_map: Dict[str, RetrievalResult],
        section_cache_index: Dict[str, Dict[str, Any]],
        evidence_fingerprint: Dict[str, str],
        fallback_evidence_ids: List[str],
    ) -> Tuple[List[SectionContent], List[str]]:
        section_contents: List[SectionContent] = []
        used_evidence_ids: List[str] = []

        for section in outline.sections:
            cached_section = section_cache_index.get(section.section_id)
            if cached_section and self._can_reuse_section(
                section,
                cached_section,
                evidence_fingerprint,
            ):
                section_contents.append(
                    SectionContent(
                        section_id=section.section_id,
                        title=section.title,
                        content=cached_section["content"],
                        used_evidence_ids=list(cached_section["used_evidence_ids"]),
                    )
                )
                used_evidence_ids.extend(cached_section["used_evidence_ids"])
                continue

            draft = self._section_writer.write_section(
                outline=outline,
                section=section,
                evidence_map=evidence_map,
                fallback_evidence_ids=fallback_evidence_ids,
            )
            section_contents.append(
                SectionContent(
                    section_id=section.section_id,
                    title=section.title,
                    content=draft.content,
                    used_evidence_ids=draft.used_evidence_ids,
                )
            )
            used_evidence_ids.extend(draft.used_evidence_ids)

        return section_contents, used_evidence_ids

    def _generate_sections_mapreduce(
        self,
        outline: ReportOutline,
        evidence_map: Dict[str, RetrievalResult],
        section_cache_index: Dict[str, Dict[str, Any]],
        evidence_fingerprint: Dict[str, str],
        fallback_evidence_ids: List[str],
    ) -> Tuple[List[SectionContent], List[str]]:
        self._ensure_mapreduce_components()
        assert self._evidence_mapper is not None
        assert self._section_reducer is not None

        section_contents: List[SectionContent] = []
        all_used_ids: List[str] = []

        for section in outline.sections:
            cached_section = section_cache_index.get(section.section_id)
            if cached_section and self._can_reuse_section(
                section,
                cached_section,
                evidence_fingerprint,
            ):
                section_contents.append(
                    SectionContent(
                        section_id=section.section_id,
                        title=section.title,
                        content=cached_section["content"],
                        used_evidence_ids=list(cached_section["used_evidence_ids"]),
                    )
                )
                all_used_ids.extend(cached_section["used_evidence_ids"])
                continue

            evidence_entries = self._collect_section_evidence(
                section,
                evidence_map,
                fallback_evidence_ids,
            )
            if not evidence_entries:
                section_contents.append(
                    SectionContent(
                        section_id=section.section_id,
                        title=section.title,
                        content="",
                        used_evidence_ids=[],
                    )
                )
                continue
            evidence_batches = self._evidence_mapper.split_batches(evidence_entries)
            evidence_summaries = self._map_evidence_batches(evidence_batches, section)

            section_text = self._section_reducer.reduce(
                evidence_summaries,
                section,
                max_tokens=self.config.max_tokens_per_reduce,
            )

            used_ids: List[str] = []
            for summary in evidence_summaries:
                for evidence_id in summary.evidence_ids:
                    if evidence_id not in used_ids:
                        used_ids.append(evidence_id)

            section_contents.append(
                SectionContent(
                    section_id=section.section_id,
                    title=section.title,
                    content=section_text.strip(),
                    used_evidence_ids=used_ids,
                )
            )
            all_used_ids.extend(used_ids)

        return section_contents, all_used_ids

    def _collect_section_evidence(
        self,
        section: SectionOutline,
        evidence_map: Dict[str, RetrievalResult],
        fallback_evidence_ids: List[str],
    ) -> List[RetrievalResult]:
        if section.evidence_ids:
            candidate_ids = [eid for eid in section.evidence_ids if eid in evidence_map]
        elif fallback_evidence_ids:
            candidate_ids = [eid for eid in fallback_evidence_ids if eid in evidence_map]
        else:
            candidate_ids = list(evidence_map.keys())
        return [evidence_map[eid] for eid in candidate_ids]

    def _map_evidence_batches(
        self,
        evidence_batches: List[List[RetrievalResult]],
        section: SectionOutline,
    ) -> List[EvidenceSummary]:
        assert self._evidence_mapper is not None

        if not evidence_batches:
            return []

        if (
            self.config.enable_parallel_map
            and len(evidence_batches) > 1
        ):
            max_workers = min(len(evidence_batches), 4)
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(
                            self._evidence_mapper.map_evidence_batch,
                            batch,
                            section,
                            batch_index=index,
                        ): index
                        for index, batch in enumerate(evidence_batches)
                    }
                    results: List[Optional[EvidenceSummary]] = [None] * len(evidence_batches)
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        try:
                            results[idx] = future.result()
                        except Exception as exc:  # noqa: BLE001
                            _LOGGER.warning("并行证据映射失败，索引%d: %s", idx, exc)
                            results[idx] = None

                fallback_indices = [idx for idx, res in enumerate(results) if res is None]
                for idx in fallback_indices:
                    results[idx] = self._evidence_mapper.map_evidence_batch(
                        evidence_batches[idx],
                        section,
                        batch_index=idx,
                    )
                return [res for res in results if res is not None]
            except Exception as exc:  # noqa: BLE001
                _LOGGER.debug("并行Map阶段异常，改为串行执行: %s", exc)

        return [
            self._evidence_mapper.map_evidence_batch(
                batch,
                section,
                batch_index=index,
            )
            for index, batch in enumerate(evidence_batches)
        ]

    def _assemble_report_mapreduce(
        self,
        outline: ReportOutline,
        section_contents: List[SectionContent],
        *,
        query: str,
        evidence_count: int,
    ) -> str:
        self._ensure_mapreduce_components()
        assert self._report_assembler is not None
        section_payload = {
            section.section_id: section.content
            for section in section_contents
        }
        return self._report_assembler.assemble(
            outline,
            section_payload,
            global_context={
                "query": query,
                "evidence_count": evidence_count,
            },
        )

    def _ensure_mapreduce_components(self) -> None:
        if self._evidence_mapper is None:
            self._evidence_mapper = EvidenceMapper(
                max_evidence_per_batch=self.config.section_writer.max_evidence_per_call,
            )
        if self._section_reducer is None:
            self._section_reducer = SectionReducer(
                strategy=self.config.reduce_strategy,
            )
        if self._report_assembler is None:
            self._report_assembler = ReportAssembler()

    def _assemble_report(self, outline: ReportOutline, sections: List[SectionContent]) -> str:
        """
        将标题、摘要、章节内容组装成最终Markdown
        """
        parts: List[str] = [f"# {outline.title}"]
        if outline.report_type == "long_document" and outline.abstract:
            parts.append("## 摘要")
            parts.append(outline.abstract.strip())

        for section in sections:
            parts.append(f"## {section.title}")
            parts.append(section.content.strip())

        return "\n\n".join(parts)

    def _format_evidence_for_check(self, evidence_entries: Iterable[RetrievalResult]) -> str:
        """
        将证据转换为一致性检查所需的文本格式
        """
        lines = []
        for item in evidence_entries:
            snippet = ""
            if isinstance(item.evidence, str):
                snippet = item.evidence.replace("\n", " ")[:200]
            elif isinstance(item.evidence, dict):
                snippet = str(item.evidence)
            lines.append(
                f"{item.result_id} | {item.granularity} | {item.source} | "
                f"{snippet}"
            )
        return "\n".join(lines)

    def _format_references(
        self,
        evidence_map: Dict[str, RetrievalResult],
        used_evidence_ids: List[str],
    ) -> Optional[str]:
        """
        调用引用格式化器生成引用列表
        """
        if not evidence_map or not used_evidence_ids:
            return None
        unique_ids = []
        for eid in used_evidence_ids:
            if eid in evidence_map and eid not in unique_ids:
                unique_ids.append(eid)
        results = [evidence_map[eid] for eid in unique_ids if eid in evidence_map]
        if not results:
            return None
        return self._citation_formatter.format_references(results, self.config.citation_style)

    def _resolve_report_id(self, plan: PlanSpec, report_type: str) -> str:
        """生成用于缓存的report_id"""
        return f"{plan.plan_id}:{plan.version}:{report_type}"

    def _build_evidence_fingerprint(
        self,
        evidence_map: Dict[str, RetrievalResult],
    ) -> Dict[str, str]:
        fingerprint: Dict[str, str] = {}
        for rid, result in evidence_map.items():
            payload = {
                "granularity": result.granularity,
                "source": result.source,
                "score": result.score,
                "metadata": result.metadata.model_dump(mode="json"),
            }
            fingerprint[rid] = hashlib.sha1(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        return fingerprint

    def _load_cached_payload(self, report_id: str) -> Optional[Dict[str, Any]]:
        if self._cache_manager is None:
            return None
        cached = self._cache_manager.get(report_id, skip_validation=True)
        if isinstance(cached, dict):
            return cached
        return None

    def _build_section_cache_index(
        self,
        cached_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not cached_payload:
            return {}
        index: Dict[str, Dict[str, Any]] = {}
        for item in cached_payload.get("sections", []):
            if isinstance(item, dict) and item.get("section_id"):
                index[item["section_id"]] = item
        return index

    def _can_reuse_section(
        self,
        section: SectionOutline,
        cached_section: Dict[str, Any],
        evidence_fingerprint: Dict[str, str],
    ) -> bool:
        if cached_section.get("title") != section.title:
            return False
        if cached_section.get("summary") != section.summary:
            return False
        cached_fp: Dict[str, str] = cached_section.get("evidence_fingerprint", {})  # type: ignore[assignment]
        for evidence_id in cached_section.get("used_evidence_ids", []):
            if evidence_fingerprint.get(evidence_id) != cached_fp.get(evidence_id):
                return False
        return True

    def _save_report_cache(
        self,
        report_id: str,
        evidence_fingerprint: Dict[str, str],
        report_result: ReportResult,
        outline: ReportOutline,
        evidence_map: Dict[str, RetrievalResult],
    ) -> None:
        if self._cache_manager is None:
            return

        outline_lookup = {section.section_id: section for section in outline.sections}
        sections_payload: List[Dict[str, Any]] = []
        for section in report_result.sections:
            outline_section = outline_lookup.get(section.section_id)
            summary = outline_section.summary if outline_section else ""
            sections_payload.append(
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "summary": summary,
                    "content": section.content,
                    "used_evidence_ids": list(section.used_evidence_ids),
                    "evidence_fingerprint": {
                        eid: evidence_fingerprint.get(eid)
                        for eid in section.used_evidence_ids
                        if eid in evidence_fingerprint
                    },
                }
            )

        payload = {
            "outline": report_result.outline.model_dump(mode="json"),
            "sections": sections_payload,
            "final_report": report_result.final_report,
            "references": report_result.references,
            "consistency_check": (
                report_result.consistency_check.model_dump(mode="json")
                if report_result.consistency_check
                else None
            ),
            "evidence_fingerprint": evidence_fingerprint,
            "evidence_ids": list(evidence_map.keys()),
        }

        try:
            self._cache_manager.set(report_id, payload)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("写入报告缓存失败: %s", exc)

    def _deserialize_report_result(self, payload: Dict[str, Any]) -> ReportResult:
        outline = ReportOutline(**payload.get("outline", {}))
        sections = [
            SectionContent(
                section_id=item.get("section_id", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                used_evidence_ids=list(item.get("used_evidence_ids", [])),
            )
            for item in payload.get("sections", [])
            if isinstance(item, dict)
        ]
        consistency = None
        if payload.get("consistency_check"):
            consistency = ConsistencyCheckResult(**payload["consistency_check"])

        return ReportResult(
            outline=outline,
            sections=sections,
            final_report=payload.get("final_report", ""),
            references=payload.get("references"),
            consistency_check=consistency,
        )

    def _update_state_report_context(
        self,
        state: PlanExecuteState,
        report_result: ReportResult,
        report_id: str,
        cache_hit: bool,
    ) -> None:
        """
        将报告结果写回PlanExecuteState的report_context
        """
        context = state.report_context
        if context is None:
            return

        context.report_type = report_result.outline.report_type
        context.outline = report_result.outline.model_dump()
        context.section_drafts = {
            section.section_id: section.content for section in report_result.sections
        }
        context.citations = []
        if report_result.references:
            context.citations.append({"formatted": report_result.references})
        context.consistency_check_results = (
            report_result.consistency_check.model_dump()
            if report_result.consistency_check
            else None
        )
        context.report_id = report_id
        context.cache_hit = cache_hit


def _extract_tracker_from_state(state: PlanExecuteState) -> Optional[EvidenceTracker]:
    if state.execution_context is None:
        return None
    registry = state.execution_context.evidence_registry.get("tracker", {})
    tracker = registry.get("_instance")
    if isinstance(tracker, EvidenceTracker):
        return tracker
    tracker_state = registry.get("state")
    if isinstance(tracker_state, dict):
        return EvidenceTracker(tracker_state)  # type: ignore[arg-type]
    return None
