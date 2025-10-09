import time
import json
from typing import List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from graphrag_agent.config.settings import gl_description
from graphrag_agent.search.tool.base import BaseSearchTool


class GlobalSearchTool(BaseSearchTool):
    """全局搜索工具，基于知识图谱和Map-Reduce模式实现跨社区的广泛查询"""

    def __init__(self, level: int = 0):
        """
        初始化全局搜索工具
        
        参数:
            level: 社区层级，默认为0
        """
        # 设置社区层级
        self.level = level
        
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/global_search")

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 设置Map阶段的处理链
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """),
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()
        
        # 设置Reduce阶段的处理链
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专门从用户查询中提取搜索关键词的助手。提取最相关的关键词，这些关键词将用于在知识库中查找信息。
                
                请返回一个关键词列表，格式为JSON数组：
                ["关键词1", "关键词2", ...]
                
                注意：
                - 提取5-8个关键词即可
                - 不要添加任何解释或其他文本，只返回JSON数组
                - 关键词应该是名词短语、概念或专有名词
                """),
            ("human", "{query}")
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
            combined_data.append(f"社区ID: {item['output']['communityId']}\n内容: {item['output']['full_content']}")
        
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
        batch_size = 5  # 每批处理5个社区，提高效率
        
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
    
    def search(self, query_input: Any) -> List[str]:
        """
        执行全局搜索，实现Map-Reduce模式
        
        参数:
            query_input: 查询输入，可以是字符串或包含查询和关键词的字典
            
        返回:
            List[str]: 中间结果列表（用于GraphAgent的reduce阶段）
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            extracted_keywords = self.extract_keywords(query)
            keywords = extracted_keywords.get("keywords", [])
        
        # 检查缓存
        cache_key = query
        if keywords:
            cache_key = f"{query}||{','.join(sorted(keywords))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 获取社区数据
            community_data = self._get_community_data(keywords)
            
            # 如果没有找到相关社区，返回空结果
            if not community_data:
                return []
            
            # 处理社区数据，生成中间结果
            intermediate_results = self._process_communities(query, community_data)
            
            # 缓存结果
            self.cache_manager.set(cache_key, intermediate_results)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return intermediate_results
        
        except Exception as e:
            print(f"全局搜索失败: {e}")
            return [f"搜索过程中出现错误: {str(e)}"]
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 搜索工具实例
        """
        class GlobalRetrievalTool(BaseTool):
            name : str= "global_retriever"
            description : str = gl_description
            
            def _run(self_tool, query: Any) -> List[str]:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> List[str]:
                raise NotImplementedError("异步执行未实现")
        
        return GlobalRetrievalTool()
    
    def close(self):
        """关闭资源"""
        # 调用父类方法关闭资源
        super().close()