"""
研究执行器

负责处理深度研究类任务，调用DeepResearch/DeeperResearch工具并生成结构化ExecutionRecord。
"""
from typing import Any, Dict, Optional, List
import time
import logging
import json
import re

from graphrag_agent.agents.multi_agent.core.execution_record import (
    ExecutionMetadata,
    ExecutionRecord,
    ToolCall,
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
)
from graphrag_agent.agents.multi_agent.tools.evidence_tracker import get_evidence_tracker
from graphrag_agent.search.tool_registry import TOOL_REGISTRY

_LOGGER = logging.getLogger(__name__)


class ResearchExecutor(BaseExecutor):
    """
    深度研究任务执行器
    """

    worker_type: str = "research_executor"
    SUPPORTED_TASKS = {"deep_research", "deeper_research"}

    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        super().__init__(config)
        self._tool_cache: Dict[str, Any] = {}

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        tool_name = task.task_type
        payload = self.build_default_inputs(task)
        _LOGGER.info("[ResearchExecutor] 执行任务 %s (%s)", task.task_id, tool_name)

        tool = self._get_tool_instance(tool_name)
        start_time = time.perf_counter()
        success = True
        error_message: Optional[str] = None
        result_payload: Any = None

        try:
            result_payload = tool.search(payload)
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)
            _LOGGER.exception("研究任务执行失败 task=%s error=%s", task.task_id, exc)

        latency = time.perf_counter() - start_time

        tool_call = ToolCall(
            tool_name=tool_name,
            args=payload,
            result=result_payload if success else None,
            status="success" if success else "failed",
            error=error_message,
            latency_ms=round(latency * 1000, 3),
        )

        answer_text = ""
        references: List[str] = []
        evidence: List[RetrievalResult] = []
        if success:
            evidence, answer_text, references = self._wrap_research_output(
                state,
                task,
                tool_name,
                result_payload,
            )

        metadata = ExecutionMetadata(
            worker_type=self.worker_type,
            latency_seconds=latency,
            tool_calls_count=1,
            evidence_count=len(evidence),
            environment={
                "execution_mode": signal.execution_mode,
                "references": references,
            },
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            session_id=state.session_id,
            worker_type=self.worker_type,
            inputs={
                "payload": payload,
                "task": task.model_dump(),
            },
            tool_calls=[tool_call],
            evidence=evidence,
            metadata=metadata,
        )

        self._update_state(
            state,
            task,
            record,
            success,
            error_message,
            result_payload,
            answer_text,
            references,
        )

        return TaskExecutionResult(record=record, success=success, error=error_message)

    def _get_tool_instance(self, task_type: str) -> Any:
        if task_type not in self._tool_cache:
            if task_type not in TOOL_REGISTRY:
                raise KeyError(f"未注册的研究工具: {task_type}")
            self._tool_cache[task_type] = TOOL_REGISTRY[task_type]()
        return self._tool_cache[task_type]

    def _wrap_research_output(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        tool_name: str,
        result_payload: Any,
    ) -> tuple[List[RetrievalResult], str, List[str]]:
        """
        将研究结果包装成RetrievalResult并提取核心答案与引用，便于Reporter引用。
        """
        answer_text = self._extract_answer_text(result_payload)
        reference_ids = self._extract_reference_ids(result_payload, answer_text)

        metadata = RetrievalMetadata(
            source_id=f"{task.task_id}:{tool_name}",
            source_type="document",
            confidence=0.65,
            extra={"references": reference_ids} if reference_ids else {},
        )
        result = RetrievalResult(
            granularity="DO",
            evidence=answer_text or "（深度研究未返回结构化文本，仅提供推理概要）",
            metadata=metadata,
            source=tool_name,  # 与RetrievalResult枚举保持一致
            score=0.65,
        )
        evidence: List[RetrievalResult] = [result]

        # 尝试将引用的证据ID解析为额外的RetrievalResult
        reference_results = self._resolve_reference_evidence(state, reference_ids)
        if reference_results:
            evidence.extend(reference_results)
        try:
            tracker = get_evidence_tracker(state)
            evidence = tracker.register(evidence)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("研究结果登记证据失败，使用原始结果: %s", exc)
            # 若登记失败，保持原有顺序，去重
            seen: set[str] = set()
            deduped: List[RetrievalResult] = []
            for item in evidence:
                if item.result_id in seen:
                    continue
                seen.add(item.result_id)
                deduped.append(item)
            evidence = deduped

        if not evidence:
            evidence = [result]
        return evidence, answer_text, reference_ids

    def _update_state(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        record: ExecutionRecord,
        success: bool,
        error: Optional[str],
        result_payload: Any,
        answer_text: str,
        references: List[str],
    ) -> None:
        exec_context = state.execution_context
        if exec_context is None:
            return

        exec_context.current_task_id = task.task_id
        exec_context.tool_call_history.append(
            {
                "task_id": task.task_id,
                "tool_name": record.tool_calls[0].tool_name if record.tool_calls else "",
                "status": record.tool_calls[0].status if record.tool_calls else "unknown",
                "latency_ms": record.metadata.latency_seconds * 1000,
            }
        )

        if success:
            if task.task_id not in exec_context.completed_task_ids:
                exec_context.completed_task_ids.append(task.task_id)
            exec_context.intermediate_results[task.task_id] = {
                "answer": answer_text,
                "references": references,
                "raw_result": result_payload,
            }
        else:
            exec_context.errors.append(
                {
                    "task_id": task.task_id,
                    "error": error,
                    "worker_type": self.worker_type,
                }
            )

        state.execution_records.append(record)

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "completed" if success else "failed")

    @staticmethod
    def _extract_answer_text(result_payload: Any) -> str:
        if isinstance(result_payload, dict):
            for key in ("answer", "summary", "final_report", "response"):
                value = result_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(result_payload, ensure_ascii=False)
        if isinstance(result_payload, str):
            return result_payload.strip()
        return str(result_payload or "").strip()

    @staticmethod
    def _extract_reference_ids(result_payload: Any, answer_text: str) -> List[str]:
        references: List[str] = []

        def _push(candidate: Optional[str]) -> None:
            if not candidate:
                return
            candidate = candidate.strip().strip("'\"")
            if not candidate:
                return
            if candidate not in references:
                references.append(candidate)

        if isinstance(result_payload, dict):
            reference_payload = result_payload.get("reference") or result_payload.get("references")
            if isinstance(reference_payload, dict):
                for key in ("doc_aggs", "chunks", "Chunks"):
                    values = reference_payload.get(key)
                    if isinstance(values, list):
                        for item in values:
                            if isinstance(item, dict):
                                _push(item.get("doc_id") or item.get("chunk_id") or item.get("id"))
                            else:
                                _push(str(item))
            elif isinstance(reference_payload, list):
                for item in reference_payload:
                    if isinstance(item, dict):
                        _push(item.get("doc_id") or item.get("chunk_id") or item.get("id"))
                    else:
                        _push(str(item))

        # 从答案文本中解析 {"Chunks": [...]} 结构或证据ID标签
        if answer_text:
            chunk_matches = re.findall(r"Chunks'\s*:\s*\[([^\]]+)\]", answer_text)
            for block in chunk_matches:
                for part in block.split(","):
                    _push(part)

            id_matches = re.findall(r"\[证据ID[:：]\s*([A-Za-z0-9\-]+)\]", answer_text)
            for eid in id_matches:
                _push(eid)

        return references

    def _resolve_reference_evidence(
        self,
        state: PlanExecuteState,
        reference_ids: List[str],
    ) -> List[RetrievalResult]:
        if not reference_ids:
            return []

        existing_results: Dict[str, RetrievalResult] = {}

        for record in state.execution_records:
            for evidence in record.evidence:
                if not isinstance(evidence, RetrievalResult):
                    continue
                key_candidates = {
                    evidence.result_id,
                    evidence.metadata.source_id,
                    evidence.metadata.extra.get("doc_id") if isinstance(evidence.metadata.extra, dict) else None,
                }
                for candidate in key_candidates:
                    if candidate:
                        existing_results[str(candidate)] = evidence

        resolved: List[RetrievalResult] = []
        for ref_id in reference_ids:
            matched = existing_results.get(ref_id)
            if matched is not None:
                resolved.append(matched)
                continue

            metadata = RetrievalMetadata(
                source_id=ref_id,
                source_type="document",
                confidence=0.5,
                extra={"from_reference": True},
            )
            resolved.append(
                RetrievalResult(
                    granularity="DO",
                    evidence=f"引用原文片段未解析，来源ID: {ref_id}",
                    metadata=metadata,
                    source="deep_research_reference",
                    score=0.5,
                )
            )
        return resolved
