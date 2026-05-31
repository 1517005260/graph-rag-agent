from typing import List, Dict, Any
import time
import json
from langsmith import traceable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from graphrag_agent.config.prompts import (
    LC_SYSTEM_PROMPT,
    contextualize_q_system_prompt,
    LOCAL_SEARCH_CONTEXT_PROMPT,
    LOCAL_SEARCH_KEYWORD_PROMPT,
)
from graphrag_agent.config.settings import lc_description
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.local_search import LocalSearch
from graphrag_agent.search.retrieval_adapter import results_from_documents, results_to_payload


class LocalSearchTool(BaseSearchTool):
    """本地搜索工具，基于向量检索实现社区内部的精确查询"""
    
    def __init__(self):
        """初始化本地搜索工具"""
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/local_search")
        
        # 设置聊天历史，用于连续对话
        self.chat_history = []
                
        # 创建本地搜索器和检索器
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        self.retriever = self.local_searcher.as_retriever()

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 创建上下文理解提示模板
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 创建历史感知检索器
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt,
        )

        # 创建带历史的本地查询提示模板
        lc_prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", LOCAL_SEARCH_CONTEXT_PROMPT),
        ])

        # 创建问答链
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            lc_prompt_with_history,
        )

        # 创建完整的RAG链
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain,
        )
        
        # 创建关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", LOCAL_SEARCH_KEYWORD_PROMPT),
            ("human", "{query}"),
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
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
            
            # 确保包含必要的键
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []
                
            # 缓存结果
            self.cache_manager.set(f"keywords:{query}", keywords)
            
            return keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 返回空字典作为默认值
            return {"low_level": [], "high_level": []}
    
    def _filter_documents_by_relevance(self, docs, query: str) -> List:
        """
        根据相关性过滤文档
        
        参数:
            docs: 文档列表
            query: 查询字符串
            
        返回:
            List: 按相关性排序的文档列表
        """
        # 使用基类的标准方法
        return self.filter_by_relevance(query, docs, top_k=5)

    def _normalize_input(self, query_input: Any) -> Dict[str, Any]:
        """规范化输入，返回包含query与keywords的字典。"""
        if isinstance(query_input, dict):
            query = query_input.get("query") or query_input.get("input") or ""
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            keywords = []
        return {"query": query, "keywords": keywords}

    @traceable
    def search(self, query_input: Any) -> str:
        """兼容旧接口，返回纯文本答案。"""
        structured = self.structured_search(query_input)
        return structured.get("answer", "未找到相关信息")

    def structured_search(self, query_input: Any) -> Dict[str, Any]:
        """
        执行本地搜索并返回结构化结果，包含标准化的RetrievalResult。
        """
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

        cached_answer = self.cache_manager.get(cache_key)

        try:
            chain_output = self.rag_chain.invoke(
                {
                    "input": query,
                    "response_type": "多个段落",
                    "chat_history": self.chat_history,
                }
            )

            answer = chain_output.get("answer") or "抱歉，我无法回答这个问题。"
            documents = chain_output.get("context") or []
            retrieval_results = results_to_payload(
                results_from_documents(documents, source="local_search")
            )

            structured_result = {
                "query": query,
                "keywords": keywords,
                "answer": answer,
                "retrieval_results": retrieval_results,
                "raw_context": [
                    {"page_content": getattr(doc, "page_content", ""), "metadata": getattr(doc, "metadata", {})}
                    for doc in documents
                ],
            }

            # 缓存结构化结果与纯文本答案
            self.cache_manager.set(structured_cache_key, structured_result)
            if cached_answer is None:
                self.cache_manager.set(cache_key, answer)

            self.performance_metrics["total_time"] = time.time() - overall_start
            return structured_result

        except Exception as e:
            print(f"本地搜索失败: {e}")
            error_msg = f"搜索过程中出现问题: {str(e)}"
            self.performance_metrics["total_time"] = time.time() - overall_start
            return {
                "query": query,
                "keywords": keywords,
                "answer": error_msg,
                "retrieval_results": [],
                "raw_context": [],
                "error": str(e),
            }
    
    def get_tool(self):
        """返回兼容旧流程的Retriever工具。"""
        return create_retriever_tool(
            self.retriever,
            "lc_search_tool",
            lc_description,
        )

    def get_structured_tool(self) -> BaseTool:
        """返回可直接输出结构化结果的工具版本。"""
        outer = self

        class LocalSearchStructuredTool(BaseTool):
            name: str = "local_search_structured"
            description: str = (
                "结构化本地搜索工具：返回回答、检索上下文以及标准化的RetrievalResult列表。"
            )

            def _run(self_tool, query: Any, **kwargs: Any) -> Dict[str, Any]:
                payload = query
                if not isinstance(query, dict):
                    payload = {"query": query}
                payload.update(kwargs)
                return outer.structured_search(payload)

            def _arun(self_tool, *args: Any, **kwargs: Any):
                raise NotImplementedError("异步执行未实现")

        return LocalSearchStructuredTool()
    
    def close(self):
        """关闭资源"""
        # 先调用父类方法关闭基础资源
        super().close()
        
        # 关闭本地搜索器
        if hasattr(self, 'local_searcher'):
            self.local_searcher.close()
