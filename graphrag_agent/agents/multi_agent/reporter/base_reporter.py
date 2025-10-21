"""
Reporter编排基类

负责串联纲要生成、章节写作、引用整理与一致性校验
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Iterable, Tuple
import logging

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
    ) -> None:
        self.config = config or ReporterConfig()
        self._outline_builder = outline_builder or OutlineBuilder()
        self._section_writer = section_writer or SectionWriter(config=self.config.section_writer)
        self._consistency_checker = consistency_checker or ConsistencyChecker()
        self._citation_formatter = citation_formatter or CitationFormatter()

    def generate_report(
        self,
        state: PlanExecuteState,
        plan: Optional[PlanSpec] = None,
        execution_records: Optional[List[ExecutionRecord]] = None,
        report_type: Optional[str] = None,
    ) -> ReportResult:
        """
        生成报告主流程
        """
        plan = plan or state.plan
        if plan is None:
            raise ValueError("生成报告需要PlanSpec")

        execution_records = execution_records or state.execution_records

        # 聚合证据
        evidence_map = self._collect_evidence(execution_records)
        plan_summary = self._build_plan_summary(plan)
        evidence_summary, limited_ids = self._build_evidence_summary(evidence_map)

        resolved_report_type = report_type or (state.report_context.report_type if state.report_context else None)
        if not resolved_report_type:
            resolved_report_type = self.config.default_report_type

        outline = self._outline_builder.build_outline(
            query=plan.problem_statement.original_query,
            plan_summary=plan_summary,
            evidence_summary=evidence_summary,
            evidence_count=len(evidence_map),
            report_type=resolved_report_type,
        )

        section_contents: List[SectionContent] = []
        used_evidence_ids: List[str] = []
        for section in outline.sections:
            draft = self._section_writer.write_section(
                outline=outline,
                section=section,
                evidence_map=evidence_map,
                fallback_evidence_ids=limited_ids,
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

        final_report = self._assemble_report(outline, section_contents)

        consistency_result: Optional[ConsistencyCheckResult] = None
        if self.config.enable_consistency_check and evidence_map:
            evidence_text = self._format_evidence_for_check(evidence_map.values())
            try:
                consistency_result = self._consistency_checker.check(final_report, evidence_text)
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

        self._update_state_report_context(state, report_result)
        state.response = final_report
        state.update_timestamp()
        return report_result

    def _collect_evidence(self, execution_records: Iterable[ExecutionRecord]) -> Dict[str, RetrievalResult]:
        """
        从执行记录中提取标准化的RetrievalResult
        """
        evidence_map: Dict[str, RetrievalResult] = {}

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

    def _update_state_report_context(self, state: PlanExecuteState, report_result: ReportResult) -> None:
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
