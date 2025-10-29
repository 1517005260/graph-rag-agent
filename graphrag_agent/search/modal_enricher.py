from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd
from langchain_core.documents import Document
from neo4j import Result

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import MINERU_ASSET_BASE_URL


@dataclass
class ModalSummary:
    """多模态聚合结果。"""

    segments: List[Dict[str, Any]]
    asset_urls: List[str]
    contexts: List[str]


class ModalEnricher:
    """提供多模态段落的批量查询、文本补充与聚合能力。"""

    def __init__(self, driver=None, asset_base_url: Optional[str] = None):
        db_manager = get_db_manager()
        self.driver = driver or db_manager.get_driver()
        self.asset_base_url = (asset_base_url or MINERU_ASSET_BASE_URL or "").rstrip("/")
        self._chunk_modal_cache: Dict[str, Dict[str, Any]] = {}
        self._last_modal_map: Dict[str, Dict[str, Any]] = {}

    def fetch_modal_map(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """批量查询 Chunk 关联的多模态段落。"""
        normalized_ids = [cid for cid in chunk_ids if cid]
        if not normalized_ids:
            self._last_modal_map = {}
            return {}

        missing_ids = [cid for cid in normalized_ids if cid not in self._chunk_modal_cache]
        if missing_ids:
            query = """
            MATCH (c:`__Chunk__`)
            WHERE c.id IN $chunk_ids
            OPTIONAL MATCH (c)-[r:HAS_MODAL]->(s:`__ModalSegment__`)
            WITH c, r, s, properties(s) AS s_props
            WITH
                c,
                collect(
                    CASE
                        WHEN s IS NULL THEN NULL
                        ELSE {
                            segment_id: s.id,
                            type: coalesce(s_props["modalType"], "text"),
                            text: coalesce(s_props["text"], ""),
                            source: s_props["source"],
                            page_index: s_props["pageIndex"],
                            bbox: s_props["bbox"],
                            order: coalesce(r.sequence, s_props["order"], 0),
                            image_relative_path: s_props["imageRelativePath"],
                            image_caption: coalesce(s_props["imageCaption"], []),
                            image_footnote: coalesce(s_props["imageFootnote"], []),
                            table_html: s_props["tableHtml"],
                            table_caption: s_props["tableCaption"],
                            table_footnote: coalesce(s_props["tableFootnote"], []),
                            latex: s_props["latex"]
                        }
                    END
                ) AS raw_segments
            OPTIONAL MATCH (c)-[:PART_OF]->(d:`__Document__`)
            WITH c, raw_segments, d, properties(d) AS d_props
            RETURN
                c.id AS chunk_id,
                [seg IN raw_segments WHERE seg IS NOT NULL] AS segments,
                c.modal_segment_ids AS chunk_modal_ids,
                c.modal_segment_types AS chunk_modal_types,
                c.modal_segment_sources AS chunk_modal_sources,
                d.fileName AS document_name,
                d_props["mineruTaskId"] AS mineru_task_id
            """
            result_df = self.driver.execute_query(
                query,
                parameters_={"chunk_ids": list(set(missing_ids))},
                result_transformer_=Result.to_df,
            )

            if isinstance(result_df, pd.DataFrame):
                for _, row in result_df.iterrows():
                    chunk_id = row.get("chunk_id")
                    if not chunk_id:
                        continue
                    mineru_task_id = row.get("mineru_task_id")
                    if pd.isna(mineru_task_id):
                        mineru_task_id = None
                    document_name = row.get("document_name")
                    if pd.isna(document_name):
                        document_name = None
                    prepared_segments = self._prepare_segment_payloads(
                        row.get("segments") or [],
                        mineru_task_id,
                    )
                    modal_context = self._compose_modal_text(prepared_segments)
                    asset_urls = [
                        segment.get("image_url")
                        for segment in prepared_segments
                        if segment.get("image_url")
                    ]

                    self._chunk_modal_cache[chunk_id] = {
                        "modal_segments": prepared_segments,
                        "chunk_modal_ids": row.get("chunk_modal_ids") or [],
                        "chunk_modal_types": row.get("chunk_modal_types") or [],
                        "chunk_modal_sources": row.get("chunk_modal_sources") or [],
                        "modal_context": modal_context,
                        "modal_asset_urls": asset_urls,
                        "mineru_task_id": mineru_task_id,
                        "document_name": document_name,
                    }

            for cid in missing_ids:
                if cid not in self._chunk_modal_cache:
                    self._chunk_modal_cache[cid] = {
                        "modal_segments": [],
                        "chunk_modal_ids": [],
                        "chunk_modal_types": [],
                        "chunk_modal_sources": [],
                        "modal_context": "",
                        "modal_asset_urls": [],
                        "mineru_task_id": None,
                        "document_name": None,
                    }

        modal_map = {
            cid: self._chunk_modal_cache.get(
                cid,
                {
                    "modal_segments": [],
                    "chunk_modal_ids": [],
                    "chunk_modal_types": [],
                    "chunk_modal_sources": [],
                    "modal_context": "",
                    "modal_asset_urls": [],
                    "mineru_task_id": None,
                    "document_name": None,
                },
            )
            for cid in normalized_ids
        }
        self._last_modal_map = modal_map
        return modal_map

    def enrich_documents(self, documents: Sequence[Document]) -> List[Document]:
        """为 LangChain Document 补充多模态信息。"""
        if not documents:
            self._last_modal_map = {}
            return []

        chunk_ids: List[str] = []
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            chunk_id = (
                metadata.get("id")
                or metadata.get("chunk_id")
                or metadata.get("source_id")
            )
            if chunk_id:
                chunk_ids.append(chunk_id)

        modal_map = self.fetch_modal_map(chunk_ids)
        for doc in documents:
            metadata = getattr(doc, "metadata", {})
            chunk_id = (
                metadata.get("id")
                or metadata.get("chunk_id")
                or metadata.get("source_id")
            )
            if not chunk_id:
                continue

            modal_data = modal_map.get(chunk_id) or {}
            modal_segments = modal_data.get("modal_segments") or []
            metadata["modal_segments"] = modal_segments
            metadata["modal_segment_ids"] = modal_data.get("chunk_modal_ids") or []
            metadata["modal_segment_types"] = modal_data.get("chunk_modal_types") or []
            metadata["modal_segment_sources"] = modal_data.get("chunk_modal_sources") or []
            metadata["modal_context"] = modal_data.get("modal_context") or ""
            metadata["mineru_task_id"] = modal_data.get("mineru_task_id")
            metadata["document_name"] = modal_data.get("document_name")
            metadata["modal_asset_urls"] = modal_data.get("modal_asset_urls") or []

            modal_text = metadata.get("modal_context")
            page_content = getattr(doc, "page_content", "") or ""
            if modal_text and modal_text not in page_content:
                doc.page_content = f"{page_content}\n\n{modal_text}".strip()

        return list(documents)

    def aggregate_modal_summary(
        self,
        documents: Optional[Sequence[Document]] = None,
        modal_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ModalSummary:
        """聚合多模态段落与图片 URL，供回答或前端直接使用。"""
        if modal_map is None:
            if documents is not None:
                chunk_ids = []
                for doc in documents:
                    metadata = getattr(doc, "metadata", {}) or {}
                    chunk_id = (
                        metadata.get("id")
                        or metadata.get("chunk_id")
                        or metadata.get("source_id")
                    )
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                modal_map = self.fetch_modal_map(chunk_ids)
            else:
                modal_map = self._last_modal_map

        segments: List[Dict[str, Any]] = []
        asset_urls: List[str] = []
        contexts: List[str] = []
        seen_segments = set()

        for info in modal_map.values():
            modal_segments = info.get("modal_segments") or []
            for segment in modal_segments:
                segment_id = segment.get("segment_id")
                if segment_id and segment_id in seen_segments:
                    continue
                if segment_id:
                    seen_segments.add(segment_id)
                segments.append(segment)
                image_url = segment.get("image_url")
                if image_url and image_url not in asset_urls:
                    asset_urls.append(image_url)

            modal_context = info.get("modal_context")
            if modal_context:
                contexts.append(modal_context)

        return ModalSummary(
            segments=segments,
            asset_urls=asset_urls,
            contexts=contexts,
        )

    def _prepare_segment_payloads(
        self,
        segments: Sequence[Dict[str, Any]],
        mineru_task_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """格式化Cypher返回的段落结构，补全URL等字段。"""
        prepared: List[Dict[str, Any]] = []
        for segment in segments:
            if not segment:
                continue
            payload = {
                "segment_id": segment.get("segment_id"),
                "type": (segment.get("type") or "text").lower(),
                "text": segment.get("text") or "",
                "source": segment.get("source"),
                "page_index": segment.get("page_index"),
                "bbox": segment.get("bbox"),
                "order": segment.get("order"),
                "image_relative_path": segment.get("image_relative_path"),
                "image_caption": segment.get("image_caption") or [],
                "image_footnote": segment.get("image_footnote") or [],
                "table_html": segment.get("table_html"),
                "table_caption": segment.get("table_caption"),
                "table_footnote": segment.get("table_footnote") or [],
                "latex": segment.get("latex"),
            }
            image_url = self._build_image_url(
                payload.get("image_relative_path"),
                mineru_task_id or payload.get("source"),
            )
            if image_url:
                payload["image_url"] = image_url
            prepared.append(payload)

        prepared.sort(
            key=lambda item: (
                item.get("order") if item.get("order") is not None else 0,
                item.get("segment_id") or "",
            )
        )
        return prepared

    def _compose_modal_text(self, segments: Sequence[Dict[str, Any]]) -> str:
        """将多模态段落转换为易于LLM理解的文本描述。"""
        if not segments:
            return ""

        lines: List[str] = ["[多模态补充信息]"]
        for index, segment in enumerate(segments, start=1):
            seg_type = segment.get("type", "text")
            lines.append(f"- 段落{index} 类型: {seg_type}")

            if seg_type == "image":
                if segment.get("image_url"):
                    lines.append(f"  图片URL: {segment['image_url']}")
                elif segment.get("image_relative_path"):
                    lines.append(f"  图片路径: {segment['image_relative_path']}")
                if segment.get("text"):
                    lines.append(f"  图片描述: {segment['text']}")
                if segment.get("image_caption"):
                    lines.append(f"  图注: {'；'.join(segment['image_caption'])}")
                if segment.get("image_footnote"):
                    lines.append(f"  图脚注: {'；'.join(segment['image_footnote'])}")
            elif seg_type == "table":
                if segment.get("table_caption"):
                    lines.append(f"  表注: {segment['table_caption']}")
                if segment.get("text"):
                    lines.append(f"  表格摘要: {segment['text']}")
                if segment.get("table_html"):
                    lines.append("  表格HTML如下：")
                    lines.append(segment["table_html"])
                if segment.get("table_footnote"):
                    lines.append(f"  表脚注: {'；'.join(segment['table_footnote'])}")
            elif seg_type == "equation":
                if segment.get("latex"):
                    lines.append(f"  公式LaTeX: {segment['latex']}")
                if segment.get("text"):
                    lines.append(f"  公式说明: {segment['text']}")
            else:
                if segment.get("text"):
                    lines.append(f"  文本: {segment['text']}")

            if segment.get("source"):
                lines.append(f"  来源: {segment['source']}")

        return "\n".join(lines)

    def _build_image_url(self, rel_path: Optional[str], mineru_task_id: Optional[str]) -> Optional[str]:
        """根据相对路径推导图片下载URL。"""
        if not rel_path or not self.asset_base_url:
            return None
        cleaned = rel_path.strip().strip("/")
        if not cleaned:
            return None
        parts = cleaned.split("/")
        filename = parts[-1]
        task_id = mineru_task_id or (parts[0] if len(parts) > 1 else None)
        if not task_id:
            return None
        return f"{self.asset_base_url}/{quote(str(task_id))}/{quote(str(filename))}"
