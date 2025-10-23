from typing import List, Dict, AsyncGenerator, Optional
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio
import re

from graphrag_agent.config.prompt import LC_SYSTEM_PROMPT
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.deeper_research_tool import DeeperResearchTool
from graphrag_agent.search.tool.deep_research_tool import DeepResearchTool 

from graphrag_agent.agents.base import BaseAgent


class DeepResearchAgent(BaseAgent):
    """
    深度研究Agent
    
    该Agent扩展了基础Agent架构，使用多回合的思考、搜索和推理来解决复杂问题。
    主要特点：
    1. 显式推理过程
    2. 迭代式搜索
    3. 高质量知识整合
    4. 支持流式输出
    5. 社区感知和知识图谱增强
    6. 多分支推理和矛盾检测
    7. 知识图谱探索能力
    8. 推理链分析
    """
    
    def __init__(self, use_deeper_tool=True):
        """
        初始化增强版深度研究Agent
        
        Args:
            use_deeper_tool: 是否使用增强版研究工具
        """
        # 初始化研究工具
        self.use_deeper_tool = use_deeper_tool
        
        if use_deeper_tool:
            # 使用增强版研究工具
            try:
                self.research_tool = DeeperResearchTool()
                print("已加载增强版深度研究工具")
                
                # 加载额外工具
                self.exploration_tool = self.research_tool.get_exploration_tool()
                self.reasoning_analysis_tool = self.research_tool.get_reasoning_analysis_tool()
                self.stream_tool = self.research_tool.get_stream_tool()
            except Exception as e:
                print(f"加载增强版研究工具失败: {e}，将使用标准版")
                self.research_tool = DeepResearchTool()
                self.use_deeper_tool = False
                
                # 标准版工具
                self.stream_tool = self.research_tool.get_thinking_stream_tool()
        else:
            # 使用标准版研究工具
            self.research_tool = DeepResearchTool()
            self.stream_tool = self.research_tool.get_thinking_stream_tool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/enhanced_research_agent"
        
        # 设置查看推理过程的模式
        self.show_thinking = False
        
        # 添加会话上下文
        self.session_context = {}
        
        # 加载社区知识增强器
        if use_deeper_tool and hasattr(self.research_tool, 'community_search'):
            self.community_enhancer = self.research_tool.community_search
        else:
            self.community_enhancer = None

        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)
    
    def _setup_chains(self):
        """设置处理链 - 由于我们直接使用工具，不需要特别设置"""
        pass
    
    def _setup_tools(self) -> List:
        """设置工具，根据模式选择不同的工具组合"""
        tools = []
        
        # 基础研究工具 - 根据显示思考过程的模式选择
        if self.show_thinking:
            # 思考过程可见模式
            tools.append(self.research_tool.get_thinking_tool())
        else:
            # 标准模式
            tools.append(self.research_tool.get_tool())
        
        # 添加增强工具 - 只有使用增强版深度研究工具时才有
        if self.use_deeper_tool:
            # 添加知识图谱探索工具
            tools.append(self.exploration_tool)
            
            # 添加推理链分析工具
            tools.append(self.reasoning_analysis_tool)
        
        # 流式工具总是添加
        tools.append(self.stream_tool)
            
        return tools
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成
        workflow.add_edge("retrieve", "generate")
    
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词"""
        # 使用研究工具的关键词提取功能
        return self.research_tool.extract_keywords(query)
    
    def _generate_node(self, state):
        """生成回答节点逻辑 - 处理各种返回格式"""
        messages = state["messages"]
        
        # 安全地获取问题和检索结果
        try:
            # 原始问题在倒数第三个消息
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
            # 检索结果在最后一个消息
            retrieval_result = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception as e:
            return {"messages": [AIMessage(content=f"生成回答时出错: {str(e)}")]}

        # 首先尝试全局缓存
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "source": "全局缓存"}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 然后检查会话缓存
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "source": "会话缓存"}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 处理流式输出的情况 - 生成器或字典结果
        if isinstance(retrieval_result, (AsyncGenerator, Dict)) and not isinstance(retrieval_result, str):
            # 如果结果是字典且包含'answer'字段，提取答案
            if isinstance(retrieval_result, dict) and 'answer' in retrieval_result:
                answer = retrieval_result['answer']
                # 根据结果结构处理
                if '<think>' in answer and '</think>' in answer:
                    # 包含思考过程，提取干净的答案
                    clean_answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL)
                    # 缓存清理后的答案
                    if clean_answer and len(clean_answer) > 10:
                        self.cache_manager.set(question, clean_answer, thread_id=thread_id)
                        self.global_cache_manager.set(question, clean_answer)
                    return {"messages": [AIMessage(content=clean_answer)]}
                else:
                    # 没有特殊标记，直接使用
                    if answer and len(answer) > 10:
                        self.cache_manager.set(question, answer, thread_id=thread_id)
                        self.global_cache_manager.set(question, answer)
                    return {"messages": [AIMessage(content=answer)]}
            # 生成器或其他复杂结构，直接返回
            return {"messages": [AIMessage(content=retrieval_result)]}

        # 如果检索结果不是包含思考过程的字符串，直接返回
        if not isinstance(retrieval_result, str) or not retrieval_result.startswith("<think>"):
            # 直接返回检索结果
            if self.cache_manager.validate_answer(question, retrieval_result):
                # 更新会话缓存
                self.cache_manager.set(question, retrieval_result, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(question, retrieval_result)
            return {"messages": [AIMessage(content=retrieval_result)]}
        
        # 处理思考过程（当使用思考工具时）
        try:
            # 提取思考过程
            thinking = retrieval_result
            
            # 创建总结提示
            prompt = ChatPromptTemplate.from_messages([
                ("system", LC_SYSTEM_PROMPT),
                ("human", """
                    以下是对问题的思考过程:
                    
                    {thinking}
                    
                    原始问题是:
                    {question}
                    
                    请生成一个全面、有深度的回答，不要重复思考过程，直接给出最终综合结论。
                    结论应该清晰、直接地回答问题，包含相关的事实和见解。
                    如果有不同的观点或矛盾的信息，请指出并提供平衡的视角。
                    """),
            ])
            
            # 创建处理链
            chain = prompt | self.llm | StrOutputParser()
            
            # 生成回答
            response = chain.invoke({
                "thinking": thinking,
                "question": question,
                "response_type": response_type
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            if self.cache_manager.validate_answer(question, response):
                # 更新会话缓存
                self.cache_manager.set(question, response, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(question, response)
            
            return {"messages": [AIMessage(content=response)]}
            
        except Exception as e:
            error_msg = f"处理思考过程时出错: {str(e)}"
            return {"messages": [AIMessage(content=error_msg)]}
    
    def ask(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None, 
            show_thinking: bool = False, exploration_mode: bool = False):
        """
        向Agent提问，可选显示思考过程
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            recursion_limit: 递归限制
            show_thinking: 是否显示思考过程
            exploration_mode: 是否使用知识图谱探索模式
                
        返回:
            str: 生成的回答或包含思考过程的字典
        """
        # 设置是否显示思考过程
        old_thinking = self.show_thinking
        self.show_thinking = show_thinking
        
        try:
            # 检查是否使用知识图谱探索模式
            if exploration_mode and self.use_deeper_tool:
                # 知识图谱探索模式
                return self.explore_knowledge(query, thread_id)
            
            # 正常模式 - 调用父类方法
            result = super().ask(query, thread_id, recursion_limit)
            return result
        finally:
            # 重置状态
            self.show_thinking = old_thinking
    
    def ask_with_thinking(self, query: str, thread_id: str = "default", community_aware: bool = True):
        """
        提问并返回带思考过程的答案
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            community_aware: 是否启用社区感知
            
        返回:
            dict: 包含思考过程和答案的字典
        """
        # 如果启用社区感知且工具支持
        if community_aware and self.community_enhancer:
            # 提取关键词
            keywords = self.research_tool.extract_keywords(query)
            
            # 使用社区感知增强搜索
            enhanced_context = self.community_enhancer.enhance_search(query, keywords)
            
            # 传递增强上下文到思考过程
            result = self.research_tool.thinking(query)
            
            # 添加社区上下文到结果
            if "community_info" in enhanced_context:
                result["community_context"] = enhanced_context["community_info"]
            
            return result
        else:
            # 直接调用研究工具的thinking方法
            result = self.research_tool.thinking(query)
            
            # 确保结果包含执行日志
            if "execution_logs" not in result:
                result["execution_logs"] = []
                
            return result
    
    async def ask_stream(self, query: str, thread_id: str = "default", 
                         recursion_limit: Optional[int] = None, show_thinking: bool = False) -> AsyncGenerator[str, None]:
        """
        向Agent提问，返回流式响应
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            recursion_limit: 递归限制
            show_thinking: 是否显示思考过程
            
        返回:
            AsyncGenerator: 流式响应内容
        """
        # 使用父类的缓存检查逻辑
        fast_result = self.check_fast_cache(query, thread_id)
        if fast_result:
            # 缓存命中，按句子分割返回
            chunks = re.split(r'([.!?。！？]\s*)', fast_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= self.deep_stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        # 无缓存，根据是否显示思考过程和工具类型选择流式方法
        if show_thinking:
            # 使用工具的流式思考接口
            if self.use_deeper_tool:
                async for chunk in self.research_tool.thinking_stream(query):
                    if isinstance(chunk, dict) and "answer" in chunk:
                        # 这是最终答案，处理后返回
                        final_answer = chunk["answer"]
                        
                        # 如果包含思考标记，提取干净部分
                        if "<think>" in final_answer and "</think>" in final_answer:
                            clean_answer = re.sub(r'<think>.*?</think>\s*', '', final_answer, flags=re.DOTALL)
                            
                            # 缓存清理后的答案
                            if clean_answer and len(clean_answer) > 10:
                                self.cache_manager.set(f"deep:{query}", clean_answer, thread_id=thread_id)
                            
                            yield clean_answer
                        else:
                            # 没有思考标记，直接使用
                            if final_answer and len(final_answer) > 10:
                                self.cache_manager.set(f"deep:{query}", final_answer, thread_id=thread_id)
                            
                            yield final_answer
                    else:
                        # 思考过程，直接返回
                        yield chunk
            else:
                # 标准版深度研究工具
                async for chunk in self.research_tool.thinking_stream(query):
                    if isinstance(chunk, dict) and "answer" in chunk:
                        # 最终答案，提取干净部分
                        final_answer = chunk["answer"]
                        if "<think>" in final_answer and "</think>" in final_answer:
                            clean_answer = re.sub(r'<think>.*?</think>\s*', '', final_answer, flags=re.DOTALL)
                            # 缓存清理后的答案
                            if clean_answer and len(clean_answer) > 10:
                                self.cache_manager.set(f"deep:{query}", clean_answer, thread_id=thread_id)
                            yield clean_answer
                        else:
                            # 没有思考标记，直接使用
                            if final_answer and len(final_answer) > 10:
                                self.cache_manager.set(f"deep:{query}", final_answer, thread_id=thread_id)
                            yield final_answer
                    else:
                        # 思考过程，直接返回
                        yield chunk
        else:
            # 普通搜索，仅返回最终答案
            if self.use_deeper_tool:
                # 增强版深度研究工具的流式搜索
                async for chunk in self.research_tool.search_stream(query):
                    yield chunk
            else:
                # 标准版深度研究工具的流式搜索
                async for chunk in self.research_tool.search_stream(query):
                    yield chunk
            
    def explore_knowledge(self, query: str, thread_id: str = "default"):
        """
        使用知识图谱探索模式探索主题
        
        参数:
            query: 用户查询或探索主题
            thread_id: 会话ID
            
        返回:
            dict: 包含探索结果的字典
        """
        # 需要使用增强版工具且有探索工具
        if not self.use_deeper_tool or not hasattr(self, 'exploration_tool'):
            return {"error": "知识图谱探索功能需要使用增强版研究工具"}
        
        # 提取关键词作为起始实体
        keywords = self.research_tool.extract_keywords(query)
        
        # 构建查询参数
        entities = keywords.get("high_level", []) + keywords.get("low_level", [])
        entities = entities[:3]  # 最多使用3个实体
        
        # 准备探索参数
        explore_query = {
            "query": query,
            "entities": entities
        }
        
        # 执行探索
        result = self.exploration_tool._run(explore_query)
        
        # 美化结果
        if isinstance(result, dict) and "exploration_path" in result:
            # 记录当前探索会话ID
            if "session_id" in result:
                self.session_context[thread_id] = {
                    "exploration_session_id": result["session_id"]
                }
                
            # 使用LLM生成探索摘要
            path_desc = []
            for step in result.get("exploration_path", []):
                step_num = step.get("step", 0)
                node_id = step.get("node_id", "")
                reasoning = step.get("reasoning", "")
                if step_num > 0:  # 跳过起始实体
                    path_desc.append(f"步骤{step_num}: 从'{node_id}'发现 - {reasoning}")
            
            path_summary = "\n".join(path_desc)
            
            # 提取关键内容
            key_content = []
            for item in result.get("content", [])[:5]:
                if "text" in item:
                    key_content.append(item["text"])
            
            content_summary = "\n\n".join(key_content)
            
            # 构建摘要提示
            summary_prompt = f"""
            基于以下知识图谱探索路径和发现的内容，生成一个关于"{query}"的综合性摘要:
            
            探索路径:
            {path_summary}
            
            关键内容:
            {content_summary}
            
            请提供一个全面、有深度的分析，包括关键发现、关联关系和见解。
            """
            
            # 生成探索摘要
            try:
                response = self.llm.invoke(summary_prompt)
                exploration_summary = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                exploration_summary = f"生成探索摘要时出错: {str(e)}"
            
            # 美化返回结果
            enhanced_result = {
                "exploration_path": result.get("exploration_path", []),
                "summary": exploration_summary,
                "key_entities": entities,
                "discovered_entities": [step.get("node_id") for step in result.get("exploration_path", []) if step.get("step", 0) > 0],
                "content_samples": key_content[:3],
                "path_visualization": path_desc,
            }
            
            return enhanced_result
        else:
            # 返回原始结果
            return result
    
    def analyze_reasoning_chain(self, query_id: str = None, thread_id: str = "default"):
        """
        分析推理链和证据
        
        参数:
            query_id: 查询ID，如果为None则使用当前会话的查询ID
            thread_id: 会话ID
            
        返回:
            dict: 包含推理链分析结果的字典
        """
        # 需要使用增强版工具
        if not self.use_deeper_tool or not hasattr(self, 'reasoning_analysis_tool'):
            return {"error": "推理链分析功能需要使用增强版研究工具"}
        
        # 如果未提供查询ID，尝试从会话上下文获取
        if not query_id:
            if thread_id in self.session_context and "latest_query_id" in self.session_context[thread_id]:
                query_id = self.session_context[thread_id]["latest_query_id"]
            elif hasattr(self.research_tool, 'current_query_context') and self.research_tool.current_query_context.get("query_id"):
                query_id = self.research_tool.current_query_context.get("query_id")
            else:
                return {"error": "未找到有效的查询ID，请先执行一次深度研究查询"}
        
        # 执行推理链分析
        return self.reasoning_analysis_tool._run(query_id)
    
    def detect_contradictions(self, query: str, thread_id: str = "default"):
        """
        检测和分析信息矛盾
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            
        返回:
            dict: 包含矛盾分析的字典
        """
        # 需要使用增强版工具
        if not self.use_deeper_tool:
            return {"error": "矛盾检测功能需要使用增强版研究工具"}
            
        # 执行深度思考
        result = self.ask_with_thinking(query, thread_id)
        
        # 提取矛盾信息
        contradictions = result.get("contradictions", [])
        
        # 如果结果中没有矛盾信息但有查询ID，尝试使用矛盾检测方法
        if not contradictions and hasattr(self.research_tool, '_detect_and_resolve_contradictions'):
            query_id = None
            if hasattr(self.research_tool, 'current_query_context'):
                query_id = self.research_tool.current_query_context.get("query_id")
                
            if query_id:
                contradiction_result = self.research_tool._detect_and_resolve_contradictions(query_id)
                contradictions = contradiction_result.get("contradictions", [])
        
        # 结构化返回矛盾信息
        if contradictions:
            result = {
                "has_contradictions": True,
                "count": len(contradictions),
                "contradictions": contradictions,
                "analysis": "发现信息来源中存在矛盾，请谨慎对待最终结论。"
            }
            
            # 使用LLM分析矛盾影响
            if len(contradictions) > 0:
                # 准备提示
                contradiction_texts = []
                for i, contradiction in enumerate(contradictions):
                    if contradiction.get("type") == "numerical":
                        contradiction_texts.append(
                            f"{i+1}. 数值矛盾: 在'{contradiction.get('context', '')}'中，" +
                            f"发现值 {contradiction.get('value1')} 和 {contradiction.get('value2')}"
                        )
                    else:
                        contradiction_texts.append(f"{i+1}. 语义矛盾: {contradiction.get('analysis', '')}")
                
                contradictions_text = "\n".join(contradiction_texts)
                
                impact_prompt = f"""
                在回答关于"{query}"的问题时，发现以下信息矛盾:
                
                {contradictions_text}
                
                请分析这些矛盾对最终答案可能产生的影响，以及如何在存在这些矛盾的情况下给出最准确的回答。
                """
                
                try:
                    response = self.llm.invoke(impact_prompt)
                    impact_analysis = response.content if hasattr(response, 'content') else str(response)
                    result["impact_analysis"] = impact_analysis
                except Exception as e:
                    result["impact_analysis"] = f"分析矛盾影响时出错: {str(e)}"
            
            return result
        else:
            return {
                "has_contradictions": False,
                "count": 0,
                "contradictions": [],
                "analysis": "未在信息来源中检测到明显矛盾。"
            }
        
    def is_deeper_tool(self, use_deeper=True):
        """
        切换是否使用增强版研究工具
        
        参数:
            use_deeper: 是否使用增强版
            
        返回:
            str: 状态消息
        """
        # 切换工具
        self.use_deeper_tool = use_deeper
        
        if use_deeper:
            # 切换到增强版
            try:
                self.research_tool = DeeperResearchTool()
                
                # 加载额外工具
                self.exploration_tool = self.research_tool.get_exploration_tool()
                self.reasoning_analysis_tool = self.research_tool.get_reasoning_analysis_tool()
                self.stream_tool = self.research_tool.get_stream_tool()
                
                # 重新设置工具
                self._tools = self._setup_tools()
                return "已切换到增强版研究工具，启用知识图谱探索和推理链分析功能"
            except Exception as e:
                self.use_deeper_tool = False
                self.stream_tool = self.research_tool.get_thinking_stream_tool() if hasattr(self.research_tool, 'get_thinking_stream_tool') else None
                return f"切换到增强版失败: {e}"
        else:
            # 切换回标准版
            self.research_tool = DeepResearchTool()
            self.stream_tool = self.research_tool.get_thinking_stream_tool() if hasattr(self.research_tool, 'get_thinking_stream_tool') else None
            # 清除增强工具
            self.exploration_tool = None
            self.reasoning_analysis_tool = None
            # 重新设置工具
            self._tools = self._setup_tools()
            return "已切换到标准版研究工具，部分高级功能将不可用"
            
    def close(self):
        """关闭资源"""
        # 调用父类方法
        super().close()
        
        # 关闭研究工具资源
        if hasattr(self, 'research_tool') and hasattr(self.research_tool, 'close'):
            self.research_tool.close()
