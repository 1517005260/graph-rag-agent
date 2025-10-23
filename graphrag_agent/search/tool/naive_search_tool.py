from typing import List, Dict, Any
import time
import numpy as np

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompt import NAIVE_PROMPT
from graphrag_agent.config.settings import response_type, naive_description, NAIVE_SEARCH_TOP_K
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
        
        # 设置处理链
        self._setup_chains()
        
    def _setup_chains(self):
        """设置处理链"""
        # 创建查询处理链
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", """
                ---文档片段--- 
                {context}
                
                问题：
                {query}
                """),
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
            chunks_with_embedding = self.graph.query("""
            MATCH (c:__Chunk__)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS id, c.text AS text, c.embedding AS embedding
            LIMIT 100  // 获取候选集
            """)
            
            # 使用工具类对候选集进行排序
            scored_chunks = VectorUtils.rank_by_similarity(
                query_embedding,
                chunks_with_embedding,
                "embedding",
                self.top_k
            )
            
            # 取top_k个结果
            results = scored_chunks[:self.top_k]
            
            search_time = time.time() - search_start
            self.performance_metrics["query_time"] = search_time
            
            if not results:
                return f"没有找到与'{query}'相关的信息。\n\n{{'data': {{'Chunks':[] }} }}"
            
            # 格式化检索到的文档片段
            chunks_content = []
            chunk_ids = []
            
            for item in results:
                chunk_id = item.get("id", "unknown")
                text = item.get("text", "")
                
                if text:
                    chunks_content.append(f"Chunk ID: {chunk_id}\n{text}")
                    chunk_ids.append(chunk_id)
            
            context = "\n\n---\n\n".join(chunks_content)
            
            # 生成回答
            llm_start = time.time()
            
            answer = self.query_chain.invoke({
                "query": query,
                "context": context,
                "response_type": response_type
            })
            
            llm_time = time.time() - llm_start
            self.performance_metrics["llm_time"] = llm_time
            
            # 确保回答中包含Chunk ID
            if "{'data': {'Chunks':" not in answer:
                # 添加引用信息
                chunk_references = ", ".join([f"'{id}'" for id in chunk_ids[:5]])
                answer += f"\n\n{{'data': {{'Chunks':[{chunk_references}] }} }}"
            
            # 缓存结果
            self.cache_manager.set(cache_key, answer)
            
            # 记录总耗时
            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time
            
            return answer
            
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
