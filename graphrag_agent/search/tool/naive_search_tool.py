from typing import List, Dict, Any
import time
import numpy as np
import json

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompts import NAIVE_PROMPT, NAIVE_SEARCH_QUERY_PROMPT
from graphrag_agent.config.settings import (
    response_type,
    naive_description,
    NAIVE_SEARCH_TOP_K,
    NAIVE_SEARCH_CANDIDATE_LIMIT,
)
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.utils import VectorUtils


class NaiveSearchTool(BaseSearchTool):
    """简单的Naive RAG搜索工具，只使用embedding进行向量搜索"""
    
    def __init__(self):
        """初始化Naive搜索工具"""
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/naive_search")
        
        # 搜索参数设置
        self.top_k = NAIVE_SEARCH_TOP_K  # 检索的最大文档数量
        self.candidate_limit = NAIVE_SEARCH_CANDIDATE_LIMIT  # 参与相似度计算的候选上限
        
        # 设置处理链
        self._setup_chains()
        
    def _setup_chains(self):
        """设置处理链"""
        # 创建查询处理链
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", NAIVE_SEARCH_QUERY_PROMPT),
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词（naive rag不需要复杂的关键词提取）
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 空的关键词字典
        """
        return {"low_level": [], "high_level": []}
    
    def _cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 相似度值
        """
        # 确保向量是numpy数组
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # 避免被零除
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def search(self, query_input: Any) -> str:
        """
        执行Naive RAG搜索 - 纯向量搜索
        
        参数:
            query_input: 用户查询或包含查询的字典
            
        返回:
            str: 基于检索结果生成的回答
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        # 检查缓存
        cache_key = f"naive:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            print(f"缓存命中: {query[:30]}...")
            return cached_result
        
        try:
            # 生成查询的嵌入向量
            search_start = time.time()
            query_embedding = self.embeddings.embed_query(query)
            
            # 获取带embedding的Chunk节点
            candidate_query = """
            MATCH (c:__Chunk__)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS id, c.text AS text, c.embedding AS embedding
            """
            query_params: Dict[str, Any] = {}
            if self.candidate_limit:
                candidate_query += "\n            LIMIT $candidate_limit"
                query_params["candidate_limit"] = int(self.candidate_limit)
            if query_params:
                chunks_with_embedding = self.graph.query(candidate_query, query_params)
            else:
                chunks_with_embedding = self.graph.query(candidate_query)
            
            # 使用工具类对候选集进行排序
            scored_chunks = VectorUtils.rank_by_similarity(
                query_embedding,
                chunks_with_embedding,
                "embedding",
                self.top_k
            )
            
            # 取top_k个结果
            results = scored_chunks[:self.top_k]
            chunk_ids = [item.get("id") for item in results if item.get("id")]
            modal_map = self.modal_enricher.fetch_modal_map(chunk_ids)
            modal_summary = self.modal_enricher.aggregate_modal_summary(modal_map=modal_map)
            
            search_time = time.time() - search_start
            self.performance_metrics["query_time"] = search_time
            
            if not results:
                return f"没有找到与'{query}'相关的信息。\n\n{{'data': {{'Chunks':[] }} }}"
            
            # 格式化检索到的文档片段
            chunks_content = []
            for item in results:
                chunk_id = item.get("id", "unknown")
                text = item.get("text", "")
                modal_data = modal_map.get(item.get("id")) if item.get("id") else {}
                chunk_lines = []
                if chunk_id:
                    chunk_lines.append(f"Chunk ID: {chunk_id}")
                if text:
                    chunk_lines.append(text)
                modal_context = modal_data.get("modal_context") if modal_data else ""
                if modal_context:
                    chunk_lines.append(f"多模态补充:\n{modal_context}")
                if chunk_lines:
                    chunks_content.append("\n".join(chunk_lines))
            
            context = "\n\n---\n\n".join(chunks_content)
            
            # 生成回答
            llm_start = time.time()
            
            answer = self.query_chain.invoke({
                "query": query,
                "context": context,
                "response_type": response_type
            })
            modal_context_text = "\n\n".join(modal_summary.contexts)
            modal_enhancement = self.modal_asset_processor.enhance_answer(
                question=query,
                answer=answer,
                modal_summary=modal_summary,
                context=modal_context_text,
            )
            enhanced_answer = modal_enhancement.apply_to_answer(answer)
            
            llm_time = time.time() - llm_start
            self.performance_metrics["llm_time"] = llm_time
            
            # 附带引用信息与多模态素材
            reference_payload = {
                "data": {
                    "Chunks": list(dict.fromkeys(chunk_ids))[:5],
                    "modalAssets": modal_summary.asset_urls,
                    "modalContexts": modal_summary.contexts,
                    "imageMarkdown": modal_enhancement.markdown,
                    "visionAnalysis": modal_enhancement.vision_analysis,
                    "modalImageDetails": [
                        detail.to_dict() for detail in modal_enhancement.image_details
                    ],
                }
            }
            enhanced_answer += f"\n\n{json.dumps(reference_payload, ensure_ascii=False)}"
            
            # 缓存结果
            self.cache_manager.set(cache_key, enhanced_answer)
            
            # 记录总耗时
            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time
            
            return enhanced_answer
            
        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            return f"搜索过程中出错: {str(e)}\n\n{{'data': {{'Chunks':[] }} }}"
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 搜索工具实例
        """
        class NaiveRetrievalTool(BaseTool):
            name : str= "naive_retriever"
            description : str = naive_description
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return NaiveRetrievalTool()
    
    def close(self):
        """关闭资源"""
        super().close()
