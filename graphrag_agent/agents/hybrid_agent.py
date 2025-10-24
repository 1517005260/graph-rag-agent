from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import tools_condition
import asyncio
import re

from graphrag_agent.config.prompt import LC_SYSTEM_PROMPT
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool

from graphrag_agent.agents.base import BaseAgent


class HybridAgent(BaseAgent):
    """使用混合搜索的Agent实现"""
    
    def __init__(self):
        # 初始化混合搜索工具
        self.search_tool = HybridSearchTool()
        
        # 首先初始化基础属性
        self.cache_dir = "./cache/hybrid_agent"
        
        # 调用父类构造函数 - 使用默认的ContextAwareCacheKeyStrategy
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.search_tool.get_tool(),
            self.search_tool.get_global_tool(),
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            # 使用增强型搜索工具的关键词提取功能
            keywords = self.search_tool.extract_keywords(query)
            
            # 确保返回有效的关键词格式
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
            # 出错时返回默认空关键词
            return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全地获取问题内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        # 安全地获取文档内容
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

        # 获取当前会话ID，用于上下文感知缓存
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

        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            以下是检索到的相关信息，按重要性排序：
            
            {context}
            
            用户的问题是：
            {question}
            
            请以清晰、全面的方式回答问题，确保：
            1. 回答结合了检索到的低级（实体细节）和高级（主题概念）信息
            2. 使用三级标题(###)组织内容，增强可读性
            3. 结尾处用"#### 引用数据"标记引用来源
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
    
    async def _generate_node_stream(self, state):
        """生成回答节点逻辑的流式版本"""
        messages = state["messages"]
        
        # 安全地获取问题内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        # 安全地获取文档内容
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            # 按句子分块输出
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return

        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            以下是检索到的相关信息，按重要性排序：
            
            {context}
            
            用户的问题是：
            {question}
            
            请以清晰、全面的方式回答问题，确保：
            1. 回答结合了检索到的低级（实体细节）和高级（主题概念）信息
            2. 使用三级标题(###)组织内容，增强可读性
            3. 结尾处用"#### 引用数据"标记引用来源
            """),
        ])

        # 使用流式模型
        # 用同步模型直接生成完整结果
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs, 
            "question": question, 
            "response_type": response_type
        })

        # 分块输出结果
        sentences = re.split(r'([.!?。！？]\s*)', response)
        buffer = ""

        for i in range(len(sentences)):
            buffer += sentences[i]
        if i % 2 == 1 or len(buffer) >= self.stream_flush_threshold:
                yield buffer
                buffer = ""
                await asyncio.sleep(0.01)

        if buffer:
            yield buffer

    async def _stream_process(self, inputs, config):
        """实现流式处理过程"""
        # 实现与 GraphAgent 类似，但针对 HybridAgent 的特性
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        # 安全地获取查询内容
        query = ""
        if "messages" in inputs and inputs["messages"] and len(inputs["messages"]) > 0:
            last_message = inputs["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                query = last_message.content
        
        if not query:
            yield "无法获取查询内容，请重试。"
            return
        
        # 缓存检查与处理同GraphAgent相同
        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response:
            # 对于缓存的响应，按自然语言单位分块返回
            chunks = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        # 工作流处理与GraphAgent相同，但添加进度提示
        workflow_state = {"messages": [HumanMessage(content=query)]}
        
        # 输出一个处理开始的提示
        yield "**正在分析问题**...\n\n"
        
        # 执行 agent 节点
        agent_output = await self._agent_node_async(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}
        
        # 检查是否需要使用工具
        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            # 告知用户正在检索
            yield "**正在检索相关信息**...\n\n"
            
            # 执行检索节点
            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}
            
            # 告知用户正在生成回答
            yield "**正在生成回答**...\n\n"
            
            # 流式生成节点输出
            async for token in self._generate_node_stream(workflow_state):
                yield token
        else:
            # 不需要工具，直接返回Agent的响应
            final_msg = workflow_state["messages"][-1]
            content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            
            # 按自然语言单位分块
            chunks = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
    
    async def _retrieve_node_async(self, state):
        """检索节点的异步版本"""
        try:
            # 获取最后一条消息
            last_message = state["messages"][-1]
            
            # 安全获取工具调用信息
            tool_calls = []
            
            # 检查additional_kwargs中的tool_calls
            if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs.get('tool_calls', [])
            
            # 检查直接的tool_calls属性
            if not tool_calls and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_calls = last_message.tool_calls
                
            # 如果没有找到工具调用
            if not tool_calls:
                return {
                    "messages": [
                        AIMessage(content="无法获取查询信息，请重试。")
                    ]
                }
            
            # 获取第一个工具调用
            tool_call = tool_calls[0]
            
            # 安全获取查询
            query = ""
            tool_id = "tool_call_0"
            tool_name = "search_tool"
            
            # 根据工具调用格式提取参数
            if isinstance(tool_call, dict):
                # 提取ID
                tool_id = tool_call.get("id", tool_id)
                
                # 提取函数名称
                if "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", tool_name)
                    
                    # 提取参数
                    args = tool_call["function"].get("arguments", {})
                    if isinstance(args, str):
                        # 尝试解析JSON
                        try:
                            import json
                            args_dict = json.loads(args)
                            query = args_dict.get("query", "")
                        except:
                            query = args  # 如果解析失败，使用整个字符串作为查询
                    elif isinstance(args, dict):
                        query = args.get("query", "")
                # 直接在root级别检查
                elif "name" in tool_call:
                    tool_name = tool_call.get("name", tool_name)
                
                # 检查args字段
                if not query and "args" in tool_call:
                    args = tool_call["args"]
                    if isinstance(args, dict):
                        query = args.get("query", "")
                    elif isinstance(args, str):
                        query = args
            
            # 如果仍然没有查询，尝试使用最简单的提取
            if not query and hasattr(last_message, 'content'):
                query = last_message.content
                
            # 执行搜索
            tool_result = self.search_tool.search(query)
            
            # 返回正确格式的工具消息
            return {
                "messages": [
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                ]
            }
        except Exception as e:
            # 处理错误
            error_msg = f"处理工具调用时出错: {str(e)}"
            print(error_msg)
            return {
                "messages": [
                    AIMessage(content=error_msg)
                ]
            }
    
    async def _agent_node_async(self, state):
        """Agent 节点的异步版本"""
        def sync_agent():
            return self._agent_node(state)
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        return await asyncio.get_event_loop().run_in_executor(None, sync_agent)
    
    def _get_tool_call_info(self, message):
        """
        从消息中提取工具调用信息
        
        参数:
            message: 包含工具调用的消息
            
        返回:
            Dict: 工具调用信息，包括id、name和args
        """
        # 检查additional_kwargs中的tool_calls
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                return {
                    "id": tool_call.get("id", "tool_call_0"),
                    "name": tool_call.get("function", {}).get("name", "search_tool"),
                    "args": tool_call.get("function", {}).get("arguments", {})
                }
        
        # 检查直接的tool_calls属性
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "id": tool_call.get("id", "tool_call_0"),
                "name": tool_call.get("name", "search_tool"),
                "args": tool_call.get("args", {})
            }
        
        # 默认返回
        return {
            "id": "tool_call_0",
            "name": "search_tool",
            "args": {"query": ""}
        }
    
    def close(self):
        """关闭资源"""
        # 先关闭父类资源
        super().close()
        
        # 再关闭搜索工具资源
        if self.search_tool:
            self.search_tool.close()
