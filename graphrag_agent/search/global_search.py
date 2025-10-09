from typing import List
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from graphrag_agent.config.neo4jdb import get_db_manager

class GlobalSearch:
    """
    全局搜索类：使用Neo4j和LangChain实现基于Map-Reduce模式的全局搜索功能
    
    该类主要用于在整个知识图谱范围内进行搜索，采用以下步骤：
    1. 获取指定层级的所有社区数据
    2. Map阶段：为每个社区生成中间结果
    3. Reduce阶段：整合所有中间结果生成最终答案
    """
    
    def __init__(self, llm, response_type: str = "多个段落"):
        """
        初始化全局搜索类
        
        参数:
            llm: 大语言模型实例
            response_type: 响应类型，默认为"多个段落"
        """
        # 保存模型实例和配置
        self.llm = llm
        self.response_type = response_type
        
        # 使用数据库连接管理
        db_manager = get_db_manager()
        
        # 初始化Neo4j图实例
        self.graph = db_manager.get_graph()
        
    def _get_community_data(self, level: int) -> List[dict]:
        """
        获取指定层级的社区数据
        
        参数:
            level: 社区层级
            
        返回:
            List[dict]: 社区数据字典列表
        """
        return self.graph.query(
            """
            MATCH (c:__Community__)
            WHERE c.level = $level
            RETURN {communityId:c.id, full_content:c.full_content} AS output
            """,
            params={"level": level},
        )
    
    def _process_communities(self, query: str, communities: List[dict]) -> List[str]:
        """
        处理社区数据生成中间结果（Map阶段）
        
        参数:
            query: 搜索查询字符串
            communities: 社区数据列表
            
        返回:
            List[str]: 中间结果列表
        """
        # 设置Map阶段的提示模板
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """),
        ])
        
        # 创建Map阶段的处理链
        map_chain = map_prompt | self.llm | StrOutputParser()
        
        # 处理每个社区
        results = []
        for community in tqdm(communities, desc="正在处理社区数据"):
            response = map_chain.invoke({
                "question": query,
                "context_data": community["output"]
            })
            results.append(response)
            print(response)  # 输出处理进度
            
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
        # 设置Reduce阶段的提示模板
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        
        # 创建Reduce阶段的处理链
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        
        # 生成最终答案
        return reduce_chain.invoke({
            "report_data": intermediate_results,
            "question": query,
            "response_type": self.response_type,
        })
    
    def search(self, query: str, level: int) -> str:
        """
        执行全局搜索
        
        参数:
            query: 搜索查询字符串
            level: 要搜索的社区层级
            
        返回:
            str: 生成的最终答案
        """
        # 获取社区数据
        communities = self._get_community_data(level)
        
        # 处理社区数据（Map阶段）
        intermediate_results = self._process_communities(query, communities)
        
        # 生成最终答案（Reduce阶段）
        return self._reduce_results(query, intermediate_results)
        
    def close(self):
        """关闭资源连接"""
        pass
            
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()