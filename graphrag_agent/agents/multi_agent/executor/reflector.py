"""
反思执行器

基于 AnswerValidationTool 对既有任务输出进行质量校验，并产出反思结果。
"""
from typing import Any, Dict, Optional, Tuple, List
import time
import re
from collections import Counter

from graphrag_agent.agents.multi_agent.core.execution_record import (
    ExecutionMetadata,
    ExecutionRecord,
    ReflectionResult,
    ToolCall,
)
from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
)
from graphrag_agent.search.tool.validation_tool import AnswerValidationTool


class ReflectionExecutor(BaseExecutor):
    """
    反思任务执行器
    """

    worker_type: str = "reflection_executor"

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        *,
        validation_tool: Optional[AnswerValidationTool] = None,
    ) -> None:
        super().__init__(config)
        self._validation_tool = validation_tool or AnswerValidationTool()

    def can_handle(self, task_type: str) -> bool:
        return task_type == "reflection"

    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        """
        对指定任务结果进行反思与验证。
        """
        payload = self.build_default_inputs(task)
        query, answer, target_task_id = self._resolve_query_answer(
            state, payload, current_task_id=task.task_id
        )
        reference_keywords = self._build_reference_keywords(
            state,
            target_task_id,
            query,
            reflection_task_id=task.task_id,
        )

        evaluation_text = self._resolve_evaluation_text(
            answer,
            state,
            target_task_id,
        )
        if not evaluation_text:
            evaluation_text = query

        start_time = time.perf_counter()
        validation_payload: Optional[Dict[str, Any]] = None
        validation_passed = False
        error: Optional[str] = None
        suggestions: List[str] = []

        if not query:
            error = "缺少用于验证的查询内容"
        elif answer is None:
            error = "未找到可验证的答案"
        else:
            try:
                validation_payload = self._validation_tool.validate(
                    query,
                    evaluation_text,
                    reference_keywords=reference_keywords,
                )
                validation_result = validation_payload.get("validation", {})
                validation_passed = bool(validation_result.get("passed", False))
                for key, passed in validation_result.items():
                    if key == "passed":
                        continue
                    if not passed:
                        suggestions.append(f"验证项 {key} 未通过")
            except Exception as exc:  # noqa: BLE001
                error = f"答案验证失败: {exc}"

        if (
            not validation_passed
            and error is None
            and isinstance(evaluation_text, str)
            and evaluation_text.strip()
        ):
            suggestions.extend(
                self._derive_keyword_suggestions(
                    query,
                    evaluation_text,
                    reference_keywords=reference_keywords,
                )
            )

        if suggestions:
            # 去重保持顺序
            suggestions = list(dict.fromkeys(suggestions))

        latency = time.perf_counter() - start_time

        tool_calls = []
        tool_status = "failed" if error else "success"
        if validation_payload is not None:
            tool_calls.append(
                ToolCall(
                    tool_name="answer_validator",
                    args={
                        "query": query,
                        "answer": evaluation_text,
                        "reference_keywords": reference_keywords or {},
                    },
                    result=validation_payload,
                    status=tool_status,
                    latency_ms=round(latency * 1000, 3),
                    error=(
                        error
                        if error
                        else (None if validation_passed else "验证未通过")
                    ),
                )
            )

        metadata = ExecutionMetadata(
            worker_type=self.worker_type,
            latency_seconds=latency,
            tool_calls_count=len(tool_calls),
            evidence_count=0,
            environment={
                "execution_mode": signal.execution_mode,
                "target_task_id": target_task_id,
                "validation_passed": validation_passed,
                "reference_keywords": reference_keywords or {},
                "suggestions": suggestions,
                "evaluation_preview": evaluation_text[:160] if evaluation_text else "",
            },
        )

        if validation_passed:
            reflection_reason = "验证通过，未发现明显问题"
        elif error:
            reflection_reason = error
        else:
            reflection_reason = "验证未通过，建议回滚或追加检索"
        if suggestions and not validation_passed:
            # 使用中文提示串联建议列表，方便在日志中观察反思结论
            reflection_reason = f"{reflection_reason}；建议：{'；'.join(suggestions[:3])}"

        reflection = ReflectionResult(
            success=validation_passed,
            confidence=0.85 if validation_passed else 0.4,
            suggestions=suggestions if not validation_passed else [],
            needs_retry=not validation_passed,
            reasoning=reflection_reason,
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            session_id=state.session_id,
            worker_type=self.worker_type,
            inputs={
                "payload": payload,
                "query": query,
                "answer": answer,
                "evaluation_text": evaluation_text,
            },
            tool_calls=tool_calls,
            reflection=reflection,
            metadata=metadata,
        )

        self._update_state(
            state,
            task,
            record,
            success=error is None,
            error=error,
            target_task_id=target_task_id,
            needs_retry=not validation_passed,
        )

        return TaskExecutionResult(
            record=record,
            success=error is None,
            error=error,
        )

    def _resolve_query_answer(
        self,
        state: PlanExecuteState,
        payload: Dict[str, Any],
        *,
        current_task_id: str,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        确定用于校验的 query 与 answer。
        """
        query = str(payload.get("query") or state.plan_context.refined_query or state.plan_context.original_query or state.input or "")
        answer: Optional[str] = payload.get("answer")

        target_task_id = payload.get("target_task_id")
        if answer is None and target_task_id:
            answer = self._lookup_answer_from_state(state, target_task_id)

        if answer is None:
            answer, target_task_id = self._fallback_from_intermediate(
                state, target_task_id, current_task_id
            )

        if answer is None and state.execution_records:
            for record in reversed(state.execution_records):
                if record.worker_type == self.worker_type:
                    continue
                candidate = self._extract_answer_from_record(record)
                if candidate:
                    answer = candidate
                    if target_task_id is None:
                        target_task_id = record.task_id
                    break

        if answer is None and isinstance(state.response, str) and state.response.strip():
            answer = state.response.strip()

        return query, answer, target_task_id

    def _lookup_answer_from_state(
        self,
        state: PlanExecuteState,
        target_task_id: str,
    ) -> Optional[str]:
        """
        从ExecutionContext中查找目标任务的答案。
        """
        context = state.execution_context
        if context is None:
            return None

        intermediate = context.intermediate_results.get(target_task_id)
        if not intermediate:
            return None

        if isinstance(intermediate, dict):
            answer = intermediate.get("answer") or intermediate.get("research_result")
            if isinstance(answer, str):
                return answer.strip()
            if isinstance(answer, dict):
                candidate = self._extract_from_dict(answer)
                if candidate:
                    return candidate
                nested = answer.get("raw_result")
                if isinstance(nested, dict):
                    nested_candidate = self._extract_from_dict(nested)
                    if nested_candidate:
                        return nested_candidate
        return None

    def _extract_answer_from_record(
        self,
        record: ExecutionRecord,
    ) -> Optional[str]:
        """
        从已有的执行记录中提取可能的答案。
        """
        for call in record.tool_calls:
            result = call.result
            if isinstance(result, dict):
                candidate = self._extract_from_dict(result)
                if candidate:
                    return candidate
            elif isinstance(result, str) and result.strip():
                return result.strip()
        return None

    def _fallback_from_intermediate(
        self,
        state: PlanExecuteState,
        target_task_id: Optional[str],
        current_task_id: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        context = state.execution_context
        if context is None or not context.intermediate_results:
            return None, target_task_id

        if target_task_id:
            candidate = self._lookup_answer_from_state(state, target_task_id)
            if candidate:
                return candidate, target_task_id

        for task_id, payload in reversed(list(context.intermediate_results.items())):
            if task_id in {current_task_id, "__reflection_retry__"}:
                continue
            if not isinstance(payload, dict):
                continue
            candidate = payload.get("answer")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip(), task_id if target_task_id is None else target_task_id
            raw_result = payload.get("raw_result")
            if isinstance(raw_result, dict):
                extracted = self._extract_from_dict(raw_result)
                if extracted:
                    return extracted, task_id if target_task_id is None else target_task_id
        return None, target_task_id

    @staticmethod
    def _extract_from_dict(payload: Dict[str, Any]) -> Optional[str]:
        candidate_keys = (
            "answer",
            "final_answer",
            "final_report",
            "response",
            "summary",
            "output",
        )
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        report_block = payload.get("report")
        if isinstance(report_block, dict):
            report_answer = report_block.get("final_report") or report_block.get("summary")
            if isinstance(report_answer, str) and report_answer.strip():
                return report_answer.strip()
        return None

    def _derive_keyword_suggestions(
        self,
        query: str,
        answer: str,
        *,
        reference_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> List[str]:
        keywords: Optional[Dict[str, List[str]]] = reference_keywords
        if keywords is None:
            extractor = getattr(self._validation_tool.validator, "keyword_extractor", None)
            if extractor is None:
                return []
            try:
                keywords = extractor(query) or {}
            except Exception:
                return []

        if not keywords:
            return []

        suggestions: List[str] = []

        high_level = keywords.get("high_level") or []
        missing_high = [
            kw
            for kw in high_level
            if isinstance(kw, str)
            and (kw if self._contains_chinese(kw) else kw.lower())
            not in (answer if self._contains_chinese(kw) else answer.lower())
        ]
        if missing_high:
            suggestions.append(f"补充高层关键词: {', '.join(missing_high[:5])}")

        low_level = keywords.get("low_level") or []
        missing_low = [
            kw
            for kw in low_level
            if isinstance(kw, str)
            and (kw if self._contains_chinese(kw) else kw.lower())
            not in (answer if self._contains_chinese(kw) else answer.lower())
        ]
        if missing_low:
            suggestions.append(f"加强细节描述，覆盖关键词: {', '.join(missing_low[:6])}")

        return suggestions

    def _resolve_evaluation_text(
        self,
        answer: Optional[Any],
        state: PlanExecuteState,
        target_task_id: Optional[str],
    ) -> str:
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        exec_context = state.execution_context
        if exec_context is not None and target_task_id is not None:
            intermediate = exec_context.intermediate_results.get(target_task_id)
            if isinstance(intermediate, dict):
                candidate = intermediate.get("answer") or intermediate.get("research_result")
                extracted = self._extract_possible_text(candidate)
                if extracted:
                    return extracted
                raw_payload = intermediate.get("raw_result")
                extracted = self._extract_possible_text(raw_payload)
                if extracted:
                    return extracted

        if isinstance(state.response, str) and state.response.strip():
            return state.response.strip()

        evidence_texts = self._collect_evidence_from_records(
            state,
            exclude_tasks=None,
            primary_task_id=target_task_id,
        )
        if evidence_texts:
            combined = "\n".join(evidence_texts[:5]).strip()
            if combined:
                return combined

        return ""

    def _build_reference_keywords(
        self,
        state: PlanExecuteState,
        target_task_id: Optional[str],
        query: str,
        *,
        reflection_task_id: str,
    ) -> Optional[Dict[str, List[str]]]:
        exec_context = state.execution_context
        if exec_context is None:
            return None

        high_counter: Counter[str] = Counter()
        low_counter: Counter[str] = Counter()

        # 1) 目标任务的缓存证据
        if target_task_id is not None:
            cache_entries = exec_context.retrieval_cache.get(target_task_id, [])
            self._accumulate_keywords_from_entries(cache_entries, high_counter, low_counter)
        else:
            cache_entries = []

        # 2) 若仍为空，回退到执行记录中的证据
        if not high_counter and not low_counter:
            evidence_texts = self._collect_evidence_from_records(
                state,
                exclude_tasks={reflection_task_id},
                primary_task_id=target_task_id,
            )
            self._accumulate_keywords_from_strings(evidence_texts, high_counter, low_counter)

        # 3) 若仍没有有效证据，最后使用原始查询兜底
        if not high_counter and not low_counter:
            fallback_texts: List[str] = [query]
            if isinstance(state.response, str) and state.response.strip():
                fallback_texts.append(state.response)
            self._accumulate_keywords_from_strings(fallback_texts, high_counter, low_counter)

        high_keywords = self._counter_to_list(high_counter, limit=8)
        low_keywords = self._counter_to_list(low_counter, limit=10)

        if not high_keywords and not low_keywords:
            return None

        return {
            "high_level": high_keywords,
            "low_level": low_keywords,
        }

    def _tokenize_reference_text(self, text: str) -> List[str]:
        if not text:
            return []
        raw_tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text)
        tokens: List[str] = []
        for token in raw_tokens:
            if self._contains_chinese(token):
                clean_token = token.strip()
                length = len(clean_token)
                if length < 2:
                    continue
                max_window = min(4, length)
                for window in range(2, max_window + 1):
                    for idx in range(0, length - window + 1):
                        substring = clean_token[idx : idx + window]
                        if substring:
                            tokens.append(substring)
            else:
                normalized = self._normalize_keyword_token(token)
                if not normalized:
                    continue
                if len(normalized) > 20:
                    continue
                tokens.append(normalized)
        return tokens

    def _accumulate_keywords_from_entries(
        self,
        entries: List[Any],
        high_counter: Counter[str],
        low_counter: Counter[str],
    ) -> None:
        if not entries:
            return
        for entry in entries:
            text = self._extract_evidence_text(entry)
            if not text:
                continue
            tokens = self._tokenize_reference_text(text)
            for token in tokens:
                if len(token) >= 4:
                    high_counter[token] += 1
                else:
                    low_counter[token] += 1

    def _accumulate_keywords_from_strings(
        self,
        texts: List[str],
        high_counter: Counter[str],
        low_counter: Counter[str],
    ) -> None:
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            for token in self._tokenize_reference_text(text):
                if len(token) >= 4:
                    high_counter[token] += 1
                else:
                    low_counter[token] += 1

    def _collect_evidence_from_records(
        self,
        state: PlanExecuteState,
        *,
        exclude_tasks: Optional[set[str]] = None,
        primary_task_id: Optional[str] = None,
    ) -> List[str]:
        texts: List[str] = []
        for record in state.execution_records:
            if exclude_tasks and record.task_id in exclude_tasks:
                continue
            if primary_task_id and record.task_id != primary_task_id:
                # 允许聚合所有任务的证据，但保持与主任务相关的优先度
                pass
            for evidence in record.evidence:
                text = self._extract_evidence_text(evidence)
                if text:
                    texts.append(text)
            for call in record.tool_calls:
                result = call.result
                if isinstance(result, dict):
                    text = self._extract_answer_content(result)
                    if text:
                        texts.append(text)
        return texts

    def _extract_evidence_text(self, evidence: Any) -> Optional[str]:
        if evidence is None:
            return None
        if isinstance(evidence, str):
            return evidence
        if hasattr(evidence, "evidence"):
            inner = getattr(evidence, "evidence")
            if isinstance(inner, (str, bytes)):
                return inner if isinstance(inner, str) else inner.decode("utf-8", "ignore")
            if isinstance(inner, dict):
                return inner.get("text") or inner.get("content") or str(inner)
        if isinstance(evidence, dict):
            text = evidence.get("evidence") or evidence.get("text") or evidence.get("content")
            if isinstance(text, str):
                return text
            if isinstance(text, dict):
                return text.get("text") or text.get("content") or str(text)
        return str(evidence) if isinstance(evidence, (int, float)) else None

    def _extract_answer_content(self, payload: Dict[str, Any]) -> Optional[str]:
        candidate_keys = (
            "final_report",
            "final_answer",
            "summary",
            "answer",
            "response",
            "content",
        )
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = payload.get("report")
        if isinstance(nested, dict):
            text = nested.get("final_report") or nested.get("summary")
            if isinstance(text, str) and text.strip():
                return text
        return None

    def _extract_possible_text(self, payload: Any) -> Optional[str]:
        if payload is None:
            return None
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        if isinstance(payload, dict):
            direct = payload.get("answer") or payload.get("text") or payload.get("content")
            if isinstance(direct, str) and direct.strip():
                return direct.strip()
            return self._extract_answer_content(payload)
        if hasattr(payload, "final_report") and isinstance(payload.final_report, str):
            text = payload.final_report.strip()
            if text:
                return text
        return None

    def _normalize_keyword_token(self, token: str) -> str:
        token = token.strip()
        if not token:
            return ""
        if self._contains_chinese(token):
            return token
        return token.lower()

    @staticmethod
    def _counter_to_list(counter: Counter[str], limit: int) -> List[str]:
        if not counter:
            return []
        ordered = [token for token, _ in counter.most_common(limit * 3)]
        dedup = []
        seen = set()
        for token in ordered:
            if token not in seen:
                seen.add(token)
                dedup.append(token)
            if len(dedup) >= limit:
                break
        return dedup

    @staticmethod
    def _contains_chinese(token: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in token)

    def _update_state(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        record: ExecutionRecord,
        success: bool,
        error: Optional[str],
        target_task_id: Optional[str],
        needs_retry: bool,
    ) -> None:
        """
        将执行结果写回状态，维护任务完成情况。
        """
        state.execution_records.append(record)

        exec_context = state.execution_context
        if exec_context is not None:
            action = "retry" if needs_retry else "confirm"
            exec_context.current_task_id = task.task_id
            exec_context.tool_call_history.append(
                {
                    "task_id": task.task_id,
                    "tool_name": "answer_validator",
                    "status": "success" if success else "failed",
                    "latency_ms": record.metadata.latency_seconds * 1000,
                }
            )
            exec_context.intermediate_results[task.task_id] = {
                "reflection": record.reflection.model_dump(),
                "target_task_id": target_task_id,
                "action": action,
            }

            if success:
                if task.task_id not in exec_context.completed_task_ids:
                    exec_context.completed_task_ids.append(task.task_id)
            if needs_retry:
                reason = error or "validation_failed"
                if target_task_id and target_task_id in exec_context.completed_task_ids:
                    exec_context.completed_task_ids.remove(target_task_id)
                retry_bucket = exec_context.intermediate_results.setdefault(
                    "__reflection_retry__", []
                )
                if isinstance(retry_bucket, list):
                    retry_bucket.append(
                        {
                            "target_task_id": target_task_id,
                            "reason": reason,
                            "reflection_task_id": task.task_id,
                        }
                    )
                exec_context.errors.append(
                    {
                        "task_id": task.task_id,
                        "error": reason,
                        "worker_type": self.worker_type,
                        "target_task_id": target_task_id,
                    }
                )
            elif not success:
                if target_task_id and target_task_id in exec_context.completed_task_ids:
                    exec_context.completed_task_ids.remove(target_task_id)
                exec_context.errors.append(
                    {
                        "task_id": task.task_id,
                        "error": error or "unknown",
                        "worker_type": self.worker_type,
                        "target_task_id": target_task_id,
                    }
                )

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "completed" if success else "failed")
            if needs_retry and target_task_id:
                state.plan.update_task_status(target_task_id, "pending")
            elif not success and target_task_id:
                state.plan.update_task_status(target_task_id, "pending")
