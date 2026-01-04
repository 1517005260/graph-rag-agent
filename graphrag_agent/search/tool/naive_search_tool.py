from typing import List, Dict, Any
import time

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompts import NAIVE_PROMPT, NAIVE_SEARCH_QUERY_PROMPT
from graphrag_agent.config.settings import response_type, naive_description, NAIVE_SEARCH_TOP_K
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.neo4j_vector_search import Neo4jVectorSearch
from graphrag_agent.search.response_models import ResponseBuilder, create_error_response


class NaiveSearchTool(BaseSearchTool):
    """简单的Naive RAG搜索工具，只使用embedding进行向量搜索"""
    
    def __init__(self):
        """初始化Naive搜索工具"""
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/naive_search")

        # 搜索参数设置
        self.top_k = NAIVE_SEARCH_TOP_K  # 检索的最大文档数量

        # 初始化 Neo4j 原生向量搜索（工程级实践）
        self.vector_search = Neo4jVectorSearch()

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
    
    def search(self, query_input: Any) -> Dict[str, Any]:
        """
        执行Naive RAG搜索 - 纯向量搜索（工程级实践）

        参数:
            query_input: 用户查询或包含查询的字典

        返回:
            Dict[str, Any]: 标准化的响应结构
                - answer: 生成的回答
                - references: 引用的资源 (chunks, entities等)
                - meta: 元数据 (retriever类型、性能指标等)
        """
        overall_start = time.time()

        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        # 初始化响应构建器
        builder = ResponseBuilder(retriever_name="naive")

        # 检查缓存
        cache_key = f"naive:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            print(f"缓存命中: {query[:30]}...")
            # 缓存的结果应该已经是标准格式
            if isinstance(cached_result, dict) and "answer" in cached_result:
                return cached_result
            # 兼容旧格式缓存（纯文本）
            builder.set_cache_hit(True)
            return builder.build_dict(answer=str(cached_result))

        try:
            # 生成查询的嵌入向量
            search_start = time.time()
            query_embedding = self.embeddings.embed_query(query)

            # ✅ 使用 Neo4j 原生 vector search（工程级实践）
            # 直接在服务端完成向量搜索和排序，避免客户端计算
            results = self.vector_search.search_chunks(
                query_embedding=query_embedding,
                top_k=self.top_k,
                return_properties=["id", "text", "fileName"]
            )

            search_time = time.time() - search_start
            self.performance_metrics["query_time"] = search_time

            if not results:
                no_result_response = builder.set_timing(
                    search_time=search_time,
                    total_time=time.time() - overall_start
                ).build_dict(answer=f"没有找到与'{query}'相关的信息。")

                # 缓存空结果
                self.cache_manager.set(cache_key, no_result_response)
                return no_result_response

            # 提取结果数据
            chunk_ids = []
            chunks_content = []
            scores = []

            for item in results:
                chunk_id = item.get("id", "unknown")
                text = item.get("text", "")
                score = item.get("score", 0.0)

                if text:
                    chunks_content.append(f"Chunk ID: {chunk_id}\n{text}")
                    chunk_ids.append(chunk_id)
                    scores.append(score)

            context = "\n\n---\n\n".join(chunks_content)

            # 添加引用信息到构建器
            builder.add_chunks(chunk_ids)
            builder.add_scores(scores)
            builder.set_top_k(self.top_k)

            # 生成回答
            llm_start = time.time()

            answer = self.query_chain.invoke({
                "query": query,
                "context": context,
                "response_type": response_type
            })

            llm_time = time.time() - llm_start
            self.performance_metrics["llm_time"] = llm_time

            # 记录性能指标
            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time

            builder.set_timing(
                search_time=search_time,
                llm_time=llm_time,
                total_time=total_time
            )

            # 构建标准响应
            response = builder.build_dict(answer=answer)

            # 缓存结果
            self.cache_manager.set(cache_key, response)

            return response

        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            return create_error_response(
                retriever_name="naive",
                error_message=str(e),
                error_type="search_error"
            )
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具（工程级实践）

        返回:
            BaseTool: 搜索工具实例
                - _run() 返回纯文本 answer（Agent 可直接使用）
                - 内部使用结构化 dict，但对外只返回 answer 字段
        """
        class NaiveRetrievalTool(BaseTool):
            name : str= "naive_retriever"
            description : str = naive_description

            def _run(self_tool, query: Any) -> str:
                """
                执行搜索并返回纯文本 answer

                工程级实践：
                1. 内部 search() 返回结构化 dict（类型安全、便于扩展）
                2. 对外 _run() 返回纯文本 answer（Agent 直接使用，无需解析）

                这样避免了：
                - Agent 层解析 JSON 失败
                - 类型错误（expected str instance, dict found）
                - LLM 生成的 JSON 格式不正确
                """
                response_dict = self.search(query)

                # ✅ 只返回 answer 纯文本（Agent 可直接使用）
                # 结构化数据（references, meta）保留在内部，供后续扩展使用
                return response_dict.get("answer", "搜索失败：未返回有效答案")

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")

        return NaiveRetrievalTool()
    
    def close(self):
        """关闭资源"""
        super().close()
