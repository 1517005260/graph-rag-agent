"""执行层与报告层共享的证据追踪工具。"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState


EvidenceTrackingState = Dict[str, Dict[str, Dict[str, object]]]


@dataclass
class EvidenceTracker:
    """维护标准化的检索证据，避免重复并统计引用情况。"""

    registry: EvidenceTrackingState = field(init=False)

    def __init__(self, registry: Optional[EvidenceTrackingState] = None) -> None:
        self.registry = registry or {"by_key": {}, "by_id": {}}

    def register(self, entries: Iterable[RetrievalResult]) -> List[RetrievalResult]:
        """登记检索结果，按来源与粒度去重并优先保留高分证据。"""
        canonical: List[RetrievalResult] = []
        for item in entries:
            key = self._make_key(item)
            stored = self.registry["by_key"].get(key)
            if stored is None:
                self.registry["by_key"][key] = {
                    "result": item,
                    "occurrences": 1,
                }
                self.registry["by_id"][item.result_id] = {
                    "key": key,
                    "result": item,
                }
                canonical.append(item)
                continue

            stored_result: RetrievalResult = stored["result"]  # type: ignore[assignment]
            stored["occurrences"] = int(stored.get("occurrences", 0)) + 1

            if item.score > stored_result.score:
                stored["result"] = item
                self.registry["by_id"][item.result_id] = {"key": key, "result": item}
                # 更新所有指向同一来源的映射，确保引用一致
                for rid, meta in list(self.registry["by_id"].items()):
                    if meta.get("key") == key:
                        meta["result"] = item
                canonical.append(item)
            else:
                canonical.append(stored_result)
        return canonical

    def all_results(self) -> List[RetrievalResult]:
        """返回当前已收集的标准化证据列表。"""
        return [entry["result"] for entry in self.registry["by_key"].values()]  # type: ignore[return-value, assignment]

    def lookup(self, result_id: str) -> Optional[RetrievalResult]:
        """根据 ``result_id`` 查找对应的标准化证据。"""
        data = self.registry["by_id"].get(result_id)
        if not data:
            return None
        return data.get("result")  # type: ignore[return-value]

    def resolve(self, result_id: str) -> Optional[Dict[str, object]]:
        """
        返回便于溯源的证据信息，包括原文、来源标识等。
        """
        result = self.lookup(result_id)
        if result is None:
            return None
        return {
            "result_id": result.result_id,
            "source_id": result.metadata.source_id,
            "source_type": result.metadata.source_type,
            "evidence": result.evidence,
            "metadata": result.metadata.model_dump(),
        }

    @staticmethod
    def _make_key(result: RetrievalResult) -> str:
        return f"{result.metadata.source_id}:{result.granularity}"


def get_evidence_tracker(state: PlanExecuteState) -> EvidenceTracker:
    """从会话状态中获取（或惰性创建）证据追踪器实例。"""
    if state.execution_context is None:
        raise ValueError("ExecutionContext is required for evidence tracking")
    registry = state.execution_context.evidence_registry.setdefault("tracker", {})
    tracker = registry.get("_instance")
    if not isinstance(tracker, EvidenceTracker):
        tracker_state: EvidenceTrackingState = registry.get("state")  # type: ignore[assignment]
        tracker = EvidenceTracker(tracker_state)
        registry["_instance"] = tracker
        registry["state"] = tracker.registry
    return tracker


__all__ = ["EvidenceTracker", "EvidenceTrackingState", "get_evidence_tracker"]
