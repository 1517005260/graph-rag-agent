from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
from graphrag_agent.graph.core import connection_manager
from graphrag_agent.search.modal_enricher import ModalEnricher
from graphrag_agent.search.retrieval_adapter import (
    merge_retrieval_results,
    results_from_documents,
    results_from_entities,
    results_from_relationships,
    results_to_payload,
)
from graphrag_agent.search.tool.reasoning.chain_of_exploration import ChainOfExplorationSearcher


class ChainOfExplorationTool:
    """将ChainOfExplorationSearcher封装为LangChain Tool。"""

    def __init__(self, max_steps: int = 5, exploration_width: int = 3):
        self.max_steps = max_steps
        self.exploration_width = exploration_width
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        self.graph = connection_manager.get_connection()
        self.searcher = ChainOfExplorationSearcher(self.graph, self.llm, self.embeddings)
        self.modal_enricher = ModalEnricher()

    def explore(
        self,
        query: str,
        *,
        start_entities: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        exploration_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行图谱探索并返回统一结构。"""
        if not query:
            raise ValueError("query不能为空")

        start_entities = start_entities or []
        results = self.searcher.explore(
            query=query,
            starting_entities=start_entities,
            max_steps=max_steps or self.max_steps,
            exploration_width=exploration_width or self.exploration_width,
        )

        content_items = results.get("content", []) or []
        chunk_ids: List[str] = []
        for item in content_items:
            metadata = item.get("metadata") or {}
            chunk_id = item.get("id") or item.get("chunk_id") or metadata.get("id")
            if chunk_id:
                chunk_ids.append(str(chunk_id))

        modal_summary = None
        if chunk_ids:
            modal_map = self.modal_enricher.fetch_modal_map(chunk_ids)
            modal_summary = self.modal_enricher.aggregate_modal_summary(modal_map=modal_map)
            for item in content_items:
                metadata = item.setdefault("metadata", {})
                chunk_id = item.get("id") or item.get("chunk_id") or metadata.get("id")
                if chunk_id:
                    metadata.setdefault("id", chunk_id)
                modal_data = modal_map.get(str(chunk_id)) if chunk_id else None
                if not modal_data:
                    continue
                metadata["modal_segments"] = modal_data.get("modal_segments", [])
                metadata["modal_asset_urls"] = modal_data.get("modal_asset_urls", [])
                metadata["modal_context"] = modal_data.get("modal_context", "")
                if modal_data.get("modal_context"):
                    current_text = item.get("text") or ""
                    if modal_data["modal_context"] not in current_text:
                        merged_text = f"{current_text}\n\n{modal_data['modal_context']}" if current_text else modal_data["modal_context"]
                        item["text"] = merged_text.strip()

        entity_results = results_from_entities(
            results.get("entities", []), source="chain_exploration"
        )
        relation_results = results_from_relationships(
            results.get("relationships", []), source="chain_exploration"
        )
        content_results = results_from_documents(
            results.get("content", []),
            source="chain_exploration",
            granularity="Chunk",
        )

        merged_results = merge_retrieval_results(
            entity_results, relation_results, content_results
        )

        if modal_summary:
            results["modal_segments"] = modal_summary.segments
            results["modal_asset_urls"] = modal_summary.asset_urls
            results["modal_context"] = "\n\n".join(modal_summary.contexts)

        summary = {
            "exploration_path": results.get("exploration_path", []),
            "statistics": results.get("statistics", {}),
            "communities": results.get("communities", []),
        }
        if modal_summary:
            summary["modal_asset_urls"] = modal_summary.asset_urls
            summary["modal_context"] = "\n\n".join(modal_summary.contexts)

        return {
            "query": query,
            "start_entities": start_entities,
            "summary": summary,
            "raw_result": results,
            "retrieval_results": results_to_payload(merged_results),
            "modal_segments": modal_summary.segments if modal_summary else [],
            "modal_asset_urls": modal_summary.asset_urls if modal_summary else [],
            "modal_context": "\n\n".join(modal_summary.contexts) if modal_summary else "",
        }

    def get_tool(self) -> BaseTool:
        """返回LangChain兼容的探索工具。"""

        chain_tool = self

        class ChainExplorationLC(BaseTool):
            name: str = "chain_of_exploration"
            description: str = (
                "图谱路径探索工具：给定查询与起始实体，沿关系链探索相关实体与证据。"
                "返回结构化的探索摘要与标准化检索结果。"
            )

            def _run(
                self_tool, query: Any, start_entities: Optional[List[str]] = None, **kwargs: Any
            ) -> Dict[str, Any]:
                if isinstance(query, dict):
                    # 支持传入完整payload
                    payload = query
                    query_text = payload.get("query", "")
                    start = payload.get("start_entities") or payload.get("entities") or []
                    return chain_tool.explore(
                        query_text,
                        start_entities=start,
                        max_steps=payload.get("max_steps"),
                        exploration_width=payload.get("exploration_width"),
                    )
                return chain_tool.explore(
                    str(query),
                    start_entities=start_entities,
                    max_steps=kwargs.get("max_steps"),
                    exploration_width=kwargs.get("exploration_width"),
                )

            def _arun(self_tool, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("异步执行未实现")

        return ChainExplorationLC()
