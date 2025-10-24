from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
import asyncio

import json
import re

from graphrag_agent.config.prompt import LC_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.local_search_tool import LocalSearchTool
from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool

from graphrag_agent.agents.base import BaseAgent


class GraphAgent(BaseAgent):
    """使用图结构的Agent实现"""
    
    def __init__(self):
        # 初始化本地和全局搜索工具
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/graph_agent"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.local_tool.get_tool(),
            self.global_tool.search,
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 添加 reduce 节点
        workflow.add_node("reduce", self._reduce_node)
        
        # 添加条件边，根据文档评分决定路由
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
            {
                "generate": "generate", 
                "reduce": "reduce"
            }
        )
        
        # 添加从 reduce 到结束的边
        workflow.add_edge("reduce", END)

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 检查查询是否为空
        if not query or not isinstance(query, str):
            return {"low_level": [], "high_level": []}
        
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        # 使用LLM提取关键词
        try:
            # 使用简单的prompt模板，避免复杂格式
            prompt = f"""提取以下查询的关键词:
            查询: {query}
            
            请提取两类关键词:
            1. 低级关键词: 具体实体、名称、术语
            2. 高级关键词: 主题、概念、领域
            
            以JSON格式返回。
            """
            
            result = self.llm.invoke(prompt)
            
            # 解析LLM返回的内容
            content = result.content if hasattr(result, 'content') else result
            
            # 尝试提取JSON部分
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    keywords = json.loads(json_str)
                    # 确保结果有正确的格式
                    if not isinstance(keywords, dict):
                        keywords = {}
                    if "low_level" not in keywords:
                        keywords["low_level"] = []
                    if "high_level" not in keywords:
                        keywords["high_level"] = []
                        
                    # 缓存结果
                    self.cache_manager.set(f"keywords:{query}", keywords)
                    return keywords
                except:
                    pass
        except Exception as e:
            print(f"关键词提取失败: {e}")
            
        # 如果提取失败，返回默认值
        default_keywords = {"low_level": [], "high_level": []}
        return default_keywords

    def _grade_documents(self, state) -> str:
        """评估文档相关性 - 返回 'generate' 或 'reduce'"""
        messages = state["messages"]
        retrieve_message = messages[-2]
        
        # 检查是否为全局检索工具调用
        tool_calls = retrieve_message.additional_kwargs.get("tool_calls", [])
        if tool_calls and tool_calls[0].get("function", {}).get("name") == "global_retriever":
            self._log_execution("grade_documents", messages, "reduce")
            return "reduce"

        # 获取问题和文档内容
        try:
            question = messages[-3].content
            docs = messages[-1].content
        except Exception as e:
            # 如果出错，默认为 generate 模式
            print(f"文档评分出错: {e}")
            return "generate"
        
        # 检查文档内容是否足够
        if not docs or len(docs) < 100:
            print("文档内容不足，尝试使用本地搜索")
            # 尝试使用local_tool进行更精确搜索
            try:
                local_result = self.local_tool.search(question)
                if local_result and len(local_result) > 100:
                    # 替换原来的结果
                    messages[-1].content = local_result
            except Exception as e:
                print(f"本地搜索失败: {e}")
        
        # 从问题中提取关键词
        keywords = []
        if hasattr(messages[-3], 'additional_kwargs') and messages[-3].additional_kwargs:
            kw_data = messages[-3].additional_kwargs.get("keywords", {})
            if isinstance(kw_data, dict):
                keywords = kw_data.get("low_level", []) + kw_data.get("high_level", [])
        
        if not keywords:
            # 如果没有提取到关键词，使用简单的关键词提取
            keywords = [word for word in question.lower().split() if len(word) > 2]
        
        # 计算关键词匹配率
        docs_text = docs.lower() if docs else ""
        matches = sum(1 for keyword in keywords if keyword.lower() in docs_text)
        match_rate = matches / len(keywords) if keywords else 0
        
        # 记录匹配情况
        self._log_execution("grade_documents", {
            "question": question,
            "keywords": keywords,
            "match_rate": match_rate,
            "docs_length": len(docs_text)
        }, f"匹配率: {match_rate}")
        
        # 总是返回 "generate" 而不是 "rewrite"，避免路由错误
        return "generate"

    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 首先尝试全局缓存
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 然后检查会话缓存
        thread_id = state.get("configurable", {}).get("thread_id", "default")
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
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {question}
            
            请严格按照以下格式输出回答：
            1. 使用三级标题(###)标记主题
            2. 主要内容用清晰的段落展示
            3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
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

    def _reduce_node(self, state):
        """处理全局搜索的Reduce节点逻辑"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 检查缓存
        cached_result = self.cache_manager.get(f"reduce:{question}")
        if cached_result:
            self._log_execution("reduce", 
                               {"question": question, "docs_length": len(docs)}, 
                               cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        response = reduce_chain.invoke({
            "report_data": docs,
            "question": question,
            "response_type": response_type,
        })
        
        # 缓存结果
        self.cache_manager.set(f"reduce:{question}", response)
        
        self._log_execution("reduce", 
                           {"question": question, "docs_length": len(docs)}, 
                           response)
        
        return {"messages": [AIMessage(content=response)]}
    
    async def _generate_node_stream(self, state):
        """生成回答节点逻辑的流式版本"""        
        messages = state["messages"]
        
        # 安全获取问题和文档内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception as e:
            yield f"**获取问题或文档时出错**: {str(e)}"
            return

        # 获取线程ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
        
        # 检查缓存
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            # 分块输出缓存内容
            sentences = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return

        # 构建提示模板
        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {question}
            
            请严格按照以下格式输出回答：
            1. 使用三级标题(###)标记主题
            2. 主要内容用清晰的段落展示
            3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
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
        # 获取会话信息
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = inputs["messages"][-1].content
        
        # 首先检查缓存
        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response:
            # 分块返回缓存结果
            sentences = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        # 处理工作流
        workflow_state = {"messages": [HumanMessage(content=query)]}
        
        # 执行agent节点 - 提供状态更新
        yield "**正在分析问题**...\n\n"
        agent_output = self._agent_node(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}
        
        # 检查是否需要使用工具
        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            # 执行检索节点
            yield "**正在检索相关信息**...\n\n"
            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}
            
            # 确保检索到内容
            last_message = workflow_state["messages"][-1]
            content = last_message.content if hasattr(last_message, 'content') else ""
            
            if not content or len(content) < 100:
                # 如果检索结果不足，尝试使用本地搜索
                try:
                    yield "**检索内容不足，正在尝试更深入的搜索**...\n\n"
                    local_result = self.local_tool.search(query)
                    if local_result and len(local_result) > 100:
                        # 使用本地搜索结果替换
                        workflow_state["messages"][-1] = ToolMessage(
                            content=local_result,
                            tool_call_id="local_search",
                            name="local_search_tool"
                        )
                        yield "**找到更多相关信息，继续生成回答**...\n\n"
                except Exception as e:
                    yield f"**尝试深入搜索时出错**: {str(e)}"
            
            # 流式生成节点输出
            yield "**正在生成回答**...\n\n"
            async for token in self._generate_node_stream(workflow_state):
                yield token
        else:
            # 不需要工具，直接返回Agent的响应
            final_msg = workflow_state["messages"][-1]
            content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            
            # 分块返回
            sentences = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
    
    # 异步检索节点辅助方法
    async def _retrieve_node_async(self, state):
        """检索节点的异步版本，用于流式处理"""
        try:
            # 获取工具调用信息
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
                print("无法获取工具调用信息")
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
            if tool_name == "global_retriever":
                # 使用全局搜索
                tool_result = self.global_tool.search(query)
            else:
                # 使用本地搜索
                tool_result = self.local_tool.search(query)
            
            # 检查搜索结果
            if not tool_result or (isinstance(tool_result, str) and len(tool_result.strip()) < 50):
                print("搜索结果内容不足，使用备用方法")
                # 尝试使用另一种搜索方法
                backup_result = self.local_tool.search(query)
                if backup_result and len(backup_result.strip()) > 50:
                    tool_result = backup_result
            
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
            import traceback
            error_msg = f"处理工具调用时出错: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return {
                "messages": [
                    AIMessage(content=f"搜索过程中出现错误: {str(e)}")
                ]
            }
    
    async def _agent_node_async(self, state):
        """Agent 节点的异步版本"""
        def sync_agent():
            return self._agent_node(state)
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        return await asyncio.get_event_loop().run_in_executor(None, sync_agent)
        
