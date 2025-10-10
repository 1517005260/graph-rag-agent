from typing import List, Dict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import asyncio

from graphrag_agent.config.prompt import NAIVE_PROMPT
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.naive_search_tool import NaiveSearchTool
from graphrag_agent.agents.base import BaseAgent


class NaiveRagAgent(BaseAgent):
    """使用简单向量检索的Naive RAG Agent实现"""
    
    def __init__(self):
        # 初始化Naive搜索工具
        self.search_tool = NaiveSearchTool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/naive_agent"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.search_tool.get_tool(),
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成，无需复杂路由
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        提取查询关键词 - 简化版本，不做实际的关键词提取
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词（空列表）
        """
        # Naive实现不需要关键词提取
        return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全地获取问题和检索结果
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 首先尝试全局缓存
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 然后检查会话缓存
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
        ("system", NAIVE_PROMPT),
        ("human", """
            ---检索结果--- 
            {context}
            
            问题：
            {question}
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            response = rag_chain.invoke({
                "context": docs, 
                "question": question, 
                "response_type": response_type
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            if response and len(response) > 10:
                # 更新会话缓存
                self.cache_manager.set(question, response, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(question, response)
            
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            response)
            
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error", 
                            {"question": question, "docs_length": len(docs)}, 
                            error_msg)
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
    
    async def _stream_process(self, inputs, config):
        """实现流式处理过程"""
        # 获取会话信息
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = inputs["messages"][-1].content
        
        # 开始处理提示
        yield "**开始处理查询**...\n\n"
        
        try:
            # 执行Naive搜索
            search_result = self.search_tool.search(query)
            
            # 分块返回结果
            if search_result:
                yield "**已找到相关信息，正在生成回答**...\n\n"
                
                # 分块返回
                sentences = re.split(r'([.!?。！？]\s*)', search_result)
                buffer = ""
                
                for i in range(0, len(sentences)):
                    buffer += sentences[i]
                    
                    # 当缓冲区包含完整句子或达到合理大小时输出
                    if (i % 2 == 1) or len(buffer) >= 40:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.01)
                
                # 输出任何剩余内容
                if buffer:
                    yield buffer
            else:
                yield "未找到与您问题相关的信息。请尝试更换关键词或提供更多细节。"
        
        except Exception as e:
            yield f"**处理查询时出错**: {str(e)}"
    
    def close(self):
        """关闭资源"""
        # 先关闭父类资源
        super().close()
        
        # 再关闭搜索工具资源
        if self.search_tool:
            self.search_tool.close()