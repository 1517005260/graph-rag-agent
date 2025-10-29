import time
import json
from typing import List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import Result

from graphrag_agent.config.prompts import (
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    GLOBAL_SEARCH_MAP_PROMPT,
    GLOBAL_SEARCH_REDUCE_PROMPT,
    GLOBAL_SEARCH_KEYWORD_PROMPT,
)
from graphrag_agent.config.settings import gl_description, GLOBAL_SEARCH_SETTINGS
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.modal_enricher import ModalSummary
from graphrag_agent.search.retrieval_adapter import (
    create_retrieval_metadata,
    create_retrieval_result,
    results_to_payload,
)


class GlobalSearchTool(BaseSearchTool):
    """全局搜索工具，基于知识图谱和Map-Reduce模式实现跨社区的广泛查询"""

    def __init__(self, level: int = None):
        """
        初始化全局搜索工具
        
        参数:
            level: 社区层级，默认为0
        """
        # 设置社区层级
        self.level = (
            level if level is not None else GLOBAL_SEARCH_SETTINGS["default_level"]
        )
        
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/global_search")
        self.community_chunk_limit = GLOBAL_SEARCH_SETTINGS.get("community_chunk_limit", 5)

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 设置Map阶段的处理链
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", GLOBAL_SEARCH_MAP_PROMPT),
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()
        
        # 设置Reduce阶段的处理链
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", GLOBAL_SEARCH_REDUCE_PROMPT),
        ])
        self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", GLOBAL_SEARCH_KEYWORD_PROMPT),
            ("human", "{query}"),
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典
        """
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            llm_start = time.time()
            
            # 调用LLM提取关键词
            result = self.keyword_chain.invoke({"query": query})
            
            # 解析JSON结果
            keywords = json.loads(result)
            
            # 记录LLM处理时间
            self.performance_metrics["llm_time"] = time.time() - llm_start
            
            # 将关键词数组转换为标准格式
            if isinstance(keywords, list):
                formatted_keywords = {
                    "keywords": keywords,
                    "low_level": [],
                    "high_level": keywords  # 全局搜索主要关注高级概念
                }
            else:
                # 默认空结构
                formatted_keywords = {
                    "keywords": [],
                    "low_level": [],
                    "high_level": []
                }
                
            # 缓存结果
            self.cache_manager.set(f"keywords:{query}", formatted_keywords)
            
            return formatted_keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 返回空字典作为默认值
            return {"keywords": [], "low_level": [], "high_level": []}
    
    def _get_community_data(self, keywords: List[str] = None) -> List[dict]:
        """
        使用关键词检索社区数据
        
        参数:
            keywords: 关键词列表，用于过滤社区
            
        返回:
            List[dict]: 社区数据列表
        """
        # 构建基础查询
        cypher_query = """
        MATCH (c:__Community__)
        WHERE c.level = $level
        """
        
        params = {"level": self.level}
        
        # 如果提供了关键词，使用它们过滤社区
        if keywords and len(keywords) > 0:
            keywords_condition = []
            for i, keyword in enumerate(keywords):
                keyword_param = f"keyword{i}"
                keywords_condition.append(f"c.full_content CONTAINS ${keyword_param}")
                params[keyword_param] = keyword
            
            if keywords_condition:
                cypher_query += " AND (" + " OR ".join(keywords_condition) + ")"
        
        # 添加排序和返回语句
        cypher_query += """
        WITH c
        ORDER BY c.community_rank DESC, c.weight DESC
        LIMIT 20
        RETURN {communityId: c.id, full_content: c.full_content} AS output
        """
        
        # 执行查询
        return self.graph.query(cypher_query, params=params)
    
    def _attach_modal_to_communities(self, communities: List[dict]):
        """为社区数据补充多模态段落，并返回聚合摘要。"""
        community_ids: List[str] = []
        for item in communities:
            info = item.get("output", item)
            community_id = info.get("communityId") or info.get("id")
            if community_id:
                community_ids.append(str(community_id))

        if not community_ids:
            for item in communities:
                info = item.get("output", item)
                if info:
                    base_text = info.get("full_content") or info.get("summary") or ""
                    info.setdefault("modal_segments", [])
                    info.setdefault("modal_asset_urls", [])
                    info.setdefault("modal_context", "")
                    info.setdefault("context_for_llm", base_text)
            return self.modal_enricher.aggregate_modal_summary(modal_map={})

        query = """
        MATCH (c:__Community__)
        WHERE c.id IN $community_ids
        MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        MATCH (chunk:__Chunk__)-[:MENTIONS]->(e)
        WITH c, chunk
        ORDER BY chunk.position ASC, chunk.id ASC
        WITH c, collect(DISTINCT chunk.id)[0..$limit] AS chunk_ids
        RETURN c.id AS community_id, chunk_ids
        """

        result_df = self.driver.execute_query(
            query,
            parameters_={
                "community_ids": community_ids,
                "limit": self.community_chunk_limit,
            },
            result_transformer_=Result.to_df,
        )

        aggregated_modal_map: Dict[str, Dict[str, Any]] = {}
        community_modal_map: Dict[str, Dict[str, Any]] = {}

        if isinstance(result_df, list):
            iterable = result_df
        elif result_df is None:
            iterable = []
        elif hasattr(result_df, "to_dict"):
            iterable = result_df.to_dict("records")
        else:
            iterable = []

        for row in iterable:
            community_id = str(row.get("community_id") or "")
            chunk_ids = row.get("chunk_ids") or []
            if not community_id:
                continue
            modal_map = self.modal_enricher.fetch_modal_map(chunk_ids)
            aggregated_modal_map.update(modal_map)
            summary = self.modal_enricher.aggregate_modal_summary(modal_map=modal_map)
            community_modal_map[community_id] = {
                "summary": summary,
                "chunk_ids": chunk_ids,
            }

        for item in communities:
            info = item.get("output", item)
            if not info:
                continue
            community_id = str(info.get("communityId") or info.get("id") or "")
            base_text = info.get("full_content") or info.get("summary") or ""
            modal_entry = community_modal_map.get(community_id)
            if modal_entry:
                summary = modal_entry["summary"]
                modal_context = "\n\n".join(summary.contexts)
                info["modal_segments"] = summary.segments
                info["modal_asset_urls"] = summary.asset_urls
                info["modal_context"] = modal_context
                info["modal_chunk_ids"] = modal_entry.get("chunk_ids", [])
                if modal_context:
                    info["context_for_llm"] = (
                        f"{base_text}\n\n[多模态补充]\n{modal_context}"
                        if base_text
                        else f"[多模态补充]\n{modal_context}"
                    )
                else:
                    info["context_for_llm"] = base_text
            else:
                info.setdefault("modal_segments", [])
                info.setdefault("modal_asset_urls", [])
                info.setdefault("modal_context", "")
                info.setdefault("modal_chunk_ids", [])
                info["context_for_llm"] = base_text

        return self.modal_enricher.aggregate_modal_summary(modal_map=aggregated_modal_map)
    
    def _process_community_batch(self, query: str, batch: List[dict]) -> str:
        """
        处理社区批次，提高效率
        
        参数:
            query: 查询字符串
            batch: 社区数据批次
            
        返回:
            str: 批次处理结果
        """
        # 合并批次内的社区数据
        combined_data = []
        for item in batch:
            info = item.get("output", item)
            community_id = info.get("communityId") or info.get("id") or "unknown"
            context_text = info.get("context_for_llm") or info.get("full_content") or ""
            combined_data.append(f"社区ID: {community_id}\n内容: {context_text}")
        
        batch_context = "\n---\n".join(combined_data)
        
        # 一次性处理整个批次
        return self.map_chain.invoke({
            "question": query, 
            "context_data": batch_context
        })
    
    def _process_communities(self, query: str, communities: List[dict]) -> List[str]:
        """
        处理社区数据生成中间结果（Map阶段）
        
        参数:
            query: 搜索查询字符串
            communities: 社区数据列表
            
        返回:
            List[str]: 中间结果列表
        """
        batch_size = GLOBAL_SEARCH_SETTINGS["community_batch_size"]  # 每批处理若干社区，提高效率
        
        results = []
        
        # 使用批处理提高效率
        for i in range(0, len(communities), batch_size):
            batch = communities[i:i+batch_size]
            try:
                batch_result = self._process_community_batch(query, batch)
                if batch_result and len(batch_result.strip()) > 0:
                    results.append(batch_result)
            except Exception as e:
                print(f"批处理失败: {e}")
        
        return results
    
    def _reduce_results(self, query: str, intermediate_results: List[str]) -> str:
        """
        整合中间结果生成最终答案（Reduce阶段）
        
        参数:
            query: 搜索查询字符串
            intermediate_results: 中间结果列表
            
        返回:
            str: 最终生成的答案
        """
        # 调用Reduce链生成最终答案
        return self.reduce_chain.invoke({
            "report_data": intermediate_results,
            "question": query,
            "response_type": "多个段落",
        })
    
    def _normalize_input(self, query_input: Any) -> Dict[str, Any]:
        """标准化输入格式。"""
        if isinstance(query_input, dict):
            query = query_input.get("query") or query_input.get("input") or ""
            keywords = query_input.get("keywords")
        else:
            query = str(query_input)
            keywords = None
        if not keywords:
            extracted = self.extract_keywords(query)
            keywords = extracted.get("keywords", [])
        return {"query": query, "keywords": keywords}

    def _community_results_to_retrieval(self, communities: List[dict]) -> List[Dict[str, Any]]:
        """将社区数据转换为RetrievalResult payload。"""
        retrieval_results = []
        for item in communities:
            info = item.get("output", item)
            community_id = str(info.get("communityId") or info.get("id") or "")
            summary = info.get("full_content") or info.get("summary") or ""
            if not community_id:
                community_id = str(hash(summary))  # fallback
            metadata = create_retrieval_metadata(
                source_id=community_id,
                source_type="community",
                confidence=info.get("confidence", 0.6),
                community_id=community_id,
                extra={
                    "raw": info,
                    "modal_segments": info.get("modal_segments"),
                    "modal_asset_urls": info.get("modal_asset_urls"),
                    "modal_context": info.get("modal_context"),
                },
            )
            result = create_retrieval_result(
                evidence=summary,
                source="global_search",
                granularity="DO",
                metadata=metadata,
                score=info.get("score", 0.6),
            )
            retrieval_results.append(result)
        return results_to_payload(retrieval_results)

    def search(self, query_input: Any) -> List[str]:
        """兼容旧接口，返回中间结果列表。"""
        structured = self.structured_search(query_input)
        return structured.get("intermediate_results", [])

    def structured_search(self, query_input: Any) -> Dict[str, Any]:
        """执行全局搜索并返回结构化数据。"""
        overall_start = time.time()
        parsed = self._normalize_input(query_input)
        query = parsed["query"]
        keywords = parsed["keywords"]

        if not query:
            raise ValueError("query不能为空")

        cache_key = query if not keywords else f"{query}||{','.join(sorted(keywords))}"
        structured_cache_key = f"{cache_key}::structured"

        cached_structured = self.cache_manager.get(structured_cache_key)
        if isinstance(cached_structured, dict):
            return cached_structured

        cached_intermediate = self.cache_manager.get(cache_key)

        try:
            community_data = self._get_community_data(keywords)
            if not community_data:
                return {
                    "query": query,
                    "keywords": keywords,
                    "intermediate_results": [],
                    "final_answer": "",
                    "retrieval_results": [],
                    "communities": [],
                    "modal_segments": [],
                    "modal_asset_urls": [],
                    "modal_context": "",
                }

            overall_modal_summary: ModalSummary = self._attach_modal_to_communities(community_data)
            intermediate_results = self._process_communities(query, community_data)
            final_answer = self._reduce_results(query, intermediate_results) if intermediate_results else ""
            retrieval_payload = self._community_results_to_retrieval(community_data)
            modal_segments = overall_modal_summary.segments if overall_modal_summary else []
            modal_asset_urls = overall_modal_summary.asset_urls if overall_modal_summary else []
            modal_context_text = "\n\n".join(overall_modal_summary.contexts) if overall_modal_summary else ""

            structured_result = {
                "query": query,
                "keywords": keywords,
                "intermediate_results": intermediate_results,
                "final_answer": final_answer,
                "retrieval_results": retrieval_payload,
                "communities": community_data,
                "modal_segments": modal_segments,
                "modal_asset_urls": modal_asset_urls,
                "modal_context": modal_context_text,
            }

            self.cache_manager.set(structured_cache_key, structured_result)
            if cached_intermediate is None:
                self.cache_manager.set(cache_key, intermediate_results)

            self.performance_metrics["total_time"] = time.time() - overall_start
            return structured_result

        except Exception as e:
            print(f"全局搜索失败: {e}")
            return {
                "query": query,
                "keywords": keywords,
                "intermediate_results": [],
                "final_answer": f"搜索过程中出现错误: {str(e)}",
                "retrieval_results": [],
                "communities": [],
                "error": str(e),
                "modal_segments": [],
                "modal_asset_urls": [],
                "modal_context": "",
            }
    
    def get_tool(self) -> BaseTool:
        """兼容旧流程的工具。"""
        class GlobalRetrievalTool(BaseTool):
            name : str= "global_retriever"
            description : str = gl_description
            
            def _run(self_tool, query: Any) -> List[str]:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> List[str]:
                raise NotImplementedError("异步执行未实现")
        
        return GlobalRetrievalTool()

    def get_structured_tool(self) -> BaseTool:
        """可返回结构化结果的工具。"""
        outer = self

        class GlobalStructuredTool(BaseTool):
            name: str = "global_search_structured"
            description: str = "结构化全局搜索工具：返回Map/Reduce结果及RetrievalResult列表。"

            def _run(self_tool, query: Any, **kwargs: Any) -> Dict[str, Any]:
                payload = query
                if not isinstance(query, dict):
                    payload = {"query": query}
                payload.update(kwargs)
                return outer.structured_search(payload)

            def _arun(self_tool, *args: Any, **kwargs: Any):
                raise NotImplementedError("异步执行未实现")

        return GlobalStructuredTool()
    
    def close(self):
        """关闭资源"""
        # 调用父类方法关闭资源
        super().close()
