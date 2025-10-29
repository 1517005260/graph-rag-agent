"""
检索结果适配器

将不同搜索工具的原始输出统一转换为RetrievalResult数据模型，便于多Agent管线消费。
"""
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
import uuid

from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)


def create_retrieval_metadata(
    *,
    source_id: str,
    source_type: str,
    confidence: float = 0.5,
    timestamp: Optional[datetime] = None,
    do_level: Optional[str] = None,
    community_id: Optional[str] = None,
    hop_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RetrievalMetadata:
    """构建统一的RetrievalMetadata对象。"""
    return RetrievalMetadata(
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        confidence=confidence,
        timestamp=timestamp or datetime.now(),
        do_level=do_level,  # type: ignore[arg-type]
        community_id=community_id,
        hop_count=hop_count,
        extra=extra or {},
    )


def create_retrieval_result(
    *,
    evidence: Any,
    source: str,
    granularity: str,
    metadata: RetrievalMetadata,
    score: float = 0.5,
    result_id: Optional[str] = None,
) -> RetrievalResult:
    """构建RetrievalResult对象。"""
    return RetrievalResult(
        result_id=result_id or str(uuid.uuid4()),
        granularity=granularity,  # type: ignore[arg-type]
        evidence=evidence,
        metadata=metadata,
        source=source,  # type: ignore[arg-type]
        score=score,
    )


def results_to_payload(results: Sequence[RetrievalResult]) -> List[Dict[str, Any]]:
    """将RetrievalResult列表转换为可序列化的字典列表。"""
    return [result.to_dict() for result in results]


def results_from_documents(
    docs: Iterable[Any],
    *,
    source: str,
    default_confidence: float = 0.6,
    granularity: str = "Chunk",
) -> List[RetrievalResult]:
    """
    从LangChain Document或类似对象生成RetrievalResult列表。

    参数:
        docs: LangChain Document或兼容对象迭代器
        source: 检索来源（local_search / global_search 等）
        default_confidence: 默认置信度
        granularity: 结果粒度
    """
    results: List[RetrievalResult] = []
    for doc in docs:
        if isinstance(doc, dict):
            metadata_dict = doc.get("metadata", {}) or {}
            score_value = doc.get("score", metadata_dict.get("score", default_confidence))
            page_content = doc.get("page_content") or doc.get("text") or metadata_dict.get("text") or ""
        else:
            metadata_dict = getattr(doc, "metadata", {}) or {}
            score_value = getattr(doc, "score", metadata_dict.get("score", default_confidence))
            page_content = getattr(doc, "page_content", None) or metadata_dict.get("text") or ""

        source_id = str(
            metadata_dict.get("id")
            or metadata_dict.get("source_id")
            or metadata_dict.get("chunk_id")
            or uuid.uuid4()
        )
        community_id = metadata_dict.get("community_id") or metadata_dict.get("community")
        score = float(score_value or default_confidence)
        extra: Dict[str, Any] = {
            "source": metadata_dict.get("source"),
            "document_id": metadata_dict.get("document_id"),
            "raw_metadata": metadata_dict,
        }
        modal_segments = metadata_dict.get("modal_segments") or []
        if modal_segments:
            extra["modal_segments"] = modal_segments
            extra["modal_asset_urls"] = [
                segment.get("image_url")
                for segment in modal_segments
                if isinstance(segment, dict) and segment.get("image_url")
            ]
            extra["modal_segment_ids"] = [
                segment.get("segment_id")
                for segment in modal_segments
                if isinstance(segment, dict) and segment.get("segment_id")
            ]
        modal_context = metadata_dict.get("modal_context")
        if modal_context:
            extra["modal_context"] = modal_context
            if modal_context not in page_content:
                page_content = f"{page_content}\n\n{modal_context}".strip()

        metadata = create_retrieval_metadata(
            source_id=source_id,
            source_type="chunk",
            confidence=metadata_dict.get("confidence", score),
            community_id=community_id,
            extra=extra,
        )
        evidence = page_content
        results.append(
            create_retrieval_result(
                evidence=evidence,
                source=source,
                granularity=granularity,
                metadata=metadata,
                score=score,
            )
        )
    return results


def results_from_entities(
    entities: Iterable[Dict[str, Any]],
    *,
    source: str,
    confidence: float = 0.55,
) -> List[RetrievalResult]:
    """从实体列表构造RetrievalResult。"""
    results: List[RetrievalResult] = []
    for entity in entities:
        entity_id = str(entity.get("id") or uuid.uuid4())
        description = entity.get("description") or entity.get("text") or ""
        metadata = create_retrieval_metadata(
            source_id=entity_id,
            source_type="entity",
            confidence=entity.get("confidence", confidence),
            extra={"raw_entity": entity},
        )
        results.append(
            create_retrieval_result(
                evidence=description,
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=entity.get("weight", confidence),
            )
        )
    return results


def results_from_relationships(
    relationships: Iterable[Dict[str, Any]],
    *,
    source: str,
    confidence: float = 0.5,
) -> List[RetrievalResult]:
    """从关系数据生成RetrievalResult。"""
    results: List[RetrievalResult] = []
    for relation in relationships:
        relation_id = str(
            relation.get("id")
            or f"{relation.get('start')}->{relation.get('end')}:{relation.get('type')}"
            or uuid.uuid4()
        )
        description = relation.get("description") or ""
        metadata = create_retrieval_metadata(
            source_id=relation_id,
            source_type="relationship",
            confidence=relation.get("confidence", confidence),
            extra={"raw_relationship": relation},
        )
        evidence = description or f"{relation.get('start')} -{relation.get('type')}-> {relation.get('end')}"
        # 将 weight 归一化到 [0, 1] 范围，避免超出 score 的验证范围
        raw_weight = relation.get("weight", confidence)
        normalized_score = min(1.0, max(0.0, raw_weight / 10.0 if raw_weight > 1.0 else raw_weight))
        results.append(
            create_retrieval_result(
                evidence=evidence,
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=normalized_score,
            )
        )
    return results


def merge_retrieval_results(*result_groups: Iterable[RetrievalResult]) -> List[RetrievalResult]:
    """合并多个RetrievalResult序列并去重（基于source_id与granularity）。"""
    merged: Dict[tuple[str, str], RetrievalResult] = {}
    for group in result_groups:
        for result in group:
            key = (result.metadata.source_id, result.granularity)
            existing = merged.get(key)
            if existing is None or result.score > existing.score:
                merged[key] = result
    return list(merged.values())
