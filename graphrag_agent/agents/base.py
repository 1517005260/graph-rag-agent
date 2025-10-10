from typing import Annotated, Sequence, TypedDict, List, Dict, Any, AsyncGenerator
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import pprint
import time
import asyncio

from graphrag_agent.models.get_models import get_llm_model, get_stream_llm_model, get_embeddings_model
from graphrag_agent.cache_manager.manager import (
    CacheManager, 
    ContextAwareCacheKeyStrategy, 
    HybridCacheBackend
)
from graphrag_agent.cache_manager.strategies.global_strategy import GlobalCacheKeyStrategy

class BaseAgent(ABC):
    """Agent 基类，定义通用功能和接口"""
    
    def __init__(self, cache_dir="./cache", memory_only=False):
        """
        初始化搜索工具
        
        参数:
            cache_dir: 缓存目录，用于存储搜索结果
        """
        # 初始化普通 LLM 和流式 LLM
        self.llm = get_llm_model()
        self.stream_llm = get_stream_llm_model()
        self.embeddings = get_embeddings_model()
        
        self.memory = MemorySaver()
        self.execution_log = []
    
        # 常规上下文感知缓存（会话内）
        self.cache_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=200,
                disk_max_size=2000
            ) if not memory_only else None,
            cache_dir=cache_dir,
            memory_only=memory_only
        )
        
        # 全局缓存（跨会话）
        self.global_cache_manager = CacheManager(
            key_strategy=GlobalCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=f"{cache_dir}/global",
                memory_max_size=500,
                disk_max_size=5000
            ) if not memory_only else None,
            cache_dir=f"{cache_dir}/global",
            memory_only=memory_only
        )
        
        self.performance_metrics = {}  # 性能指标收集
        
        # 初始化工具
        self.tools = self._setup_tools()
        
        # 设置工作流图
        self._setup_graph()
    
    @abstractmethod
    def _setup_tools(self) -> List:
        """设置工具，子类必须实现"""
        pass
    
    def _setup_graph(self):
        """设置工作流图 - 基础结构，子类可以通过_add_retrieval_edges自定义"""
        # 定义状态类型
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # 创建工作流图
        workflow = StateGraph(AgentState)
        
        # 添加节点 - 节点与原始代码保持一致
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)
        
        # 添加从开始到Agent的边
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # 添加从检索到生成的边 - 这个逻辑由子类实现
        self._add_retrieval_edges(workflow)
        
        # 从生成到结束
        workflow.add_edge("generate", END)
        
        # 编译图
        self.graph = workflow.compile(checkpointer=self.memory)
    
    async def _stream_process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        执行流式处理的默认实现
        
        子类应该覆盖此方法以实现特定的流式处理逻辑
        
        参数:
            inputs: 输入消息
            config: 配置
            
        返回:
            AsyncGenerator[str, None]: 流式响应生成器
        """
        # 获取消息
        messages = inputs.get("messages", [])
        query = messages[-1].content if messages else ""
        
        # 构建状态字典
        state = {
            "messages": messages,
            "configurable": config.get("configurable", {})
        }
        
        # 获取生成结果
        result = await self._generate_node_async(state)
        
        if "messages" in result and result["messages"]:
            message = result["messages"][0]
            content = message.content if hasattr(message, "content") else str(message)
            
            # 按句子或段落分块，更自然
            import re
            chunks = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""
            
            for i in range(0, len(chunks)):
                if i < len(chunks):
                    buffer += chunks[i]
                    
                    # 当缓冲区包含完整句子或达到合理大小时输出
                    if (i % 2 == 1) or len(buffer) >= 40:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.01)  # 微小延迟确保流畅显示
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
        else:
            yield "无法生成响应。"

    
    @abstractmethod
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边，子类必须实现"""
        pass
    
    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """记录节点执行"""
        self.execution_log.append({
            "node": node_name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        })
    
    def _log_performance(self, operation, metrics):
        """记录性能指标"""
        self.performance_metrics[operation] = {
            "timestamp": time.time(),
            **metrics
        }
        
        # 输出关键性能指标
        if "duration" in metrics:
            print(f"性能指标 - {operation}: {metrics['duration']:.4f}s")
    
    def _agent_node(self, state):
        """Agent 节点逻辑"""
        messages = state["messages"]
        
        # 提取关键词优化查询
        if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
            query = messages[-1].content
            keywords = self._extract_keywords(query)
            
            # 记录关键词
            self._log_execution("extract_keywords", query, keywords)
            
            # 增强消息，添加关键词信息
            if keywords:
                # 创建一个新的消息，带有关键词元数据
                enhanced_message = HumanMessage(
                    content=query,
                    additional_kwargs={"keywords": keywords}
                )
                # 替换原始消息
                messages = messages[:-1] + [enhanced_message]
        
        # 使用工具处理请求
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        
        self._log_execution("agent", messages, response)
        return {"messages": [response]}
    
    @abstractmethod
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词，子类必须实现"""
        pass
    
    @abstractmethod
    def _generate_node(self, state):
        """生成回答节点逻辑，子类必须实现"""
        pass

    async def _generate_node_stream(self, state):
        """
        生成回答节点逻辑的流式版本
        
        参数:
            state: 当前状态
            
        返回:
            AsyncGenerator[str, None]: 流式响应生成器
        """
        # 默认实现 - 应由子类覆盖
        result = self._generate_node(state)
        if "messages" in result and result["messages"]:
            message = result["messages"][0]
            content = message.content if hasattr(message, "content") else str(message)
            
            # 模拟流式输出
            chunk_size = 4  # 每个分块的字符数
            for i in range(0, len(content), chunk_size):
                yield content[i:i+chunk_size]
                await asyncio.sleep(0.01)
    
    async def _generate_node_async(self, state):
        """
        生成回答节点逻辑的异步版本
        
        参数:
            state: 当前状态
            
        返回:
            Dict: 包含消息的结果字典
        """
        # 这个默认实现只是调用同步版本
        # 子类应该提供真正的异步实现
        def sync_generate():
            return self._generate_node(state)
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        return await asyncio.get_event_loop().run_in_executor(None, sync_generate)
    
    def check_fast_cache(self, query: str, thread_id: str = "default") -> str:
        """专用的快速缓存检查方法，用于高性能路径"""
        start_time = time.time()
        
        # 提取关键词，确保在缓存键中使用
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }
        
        # 使用缓存管理器的快速获取方法，传递相关参数
        result = self.cache_manager.get_fast(query, **cache_params)
        duration = time.time() - start_time
        self._log_performance("fast_cache_check", {
            "duration": duration,
            "hit": result is not None
        })
        
        return result
    
    def _check_all_caches(self, query: str, thread_id: str = "default"):
        """整合的缓存检查方法"""
        cache_check_start = time.time()
        
        # 1. 首先尝试全局缓存（跨会话缓存）
        global_result = self.global_cache_manager.get(query)
        if global_result:
            print(f"全局缓存命中: {query[:30]}...")
            
            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "global"
            })
            
            return global_result
        
        # 2. 尝试快速路径 - 跳过验证的高质量缓存
        fast_result = self.check_fast_cache(query, thread_id)
        if fast_result:
            print(f"快速路径缓存命中: {query[:30]}...")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(query, fast_result)
            
            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "fast"
            })
            
            return fast_result
        
        # 3. 尝试常规缓存路径，但优化验证
        cached_response = self.cache_manager.get(query, skip_validation=True, thread_id=thread_id)
        if cached_response:
            print(f"常规缓存命中，跳过验证: {query[:30]}...")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(query, cached_response)
            
            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "standard"
            })
            
            return cached_response
        
        # 没有命中任何缓存
        cache_time = time.time() - cache_check_start
        self._log_performance("cache_check", {
            "duration": cache_time,
            "type": "miss"
        })
        
        return None
    
    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> Dict:
        """执行查询并获取带执行轨迹的回答"""
        overall_start = time.time()
        self.execution_log = []  # 重置执行日志
        
        # 确保查询字符串是干净的
        safe_query = query.strip()
        
        # 首先尝试全局缓存（跨会话缓存）
        global_cache_start = time.time()
        global_result = self.global_cache_manager.get(safe_query)
        global_cache_time = time.time() - global_cache_start
        
        if global_result:
            print(f"全局缓存命中: {safe_query[:30]}... ({global_cache_time:.4f}s)")
            
            return {
                "answer": global_result,
                "execution_log": [{"node": "global_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "全局缓存命中"}]
            }
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_cache_start = time.time()
        fast_result = self.check_fast_cache(safe_query, thread_id)
        fast_cache_time = time.time() - fast_cache_start
        
        if fast_result:
            print(f"快速路径缓存命中: {safe_query[:30]}... ({fast_cache_time:.4f}s)")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, fast_result)
            
            return {
                "answer": fast_result,
                "execution_log": [{"node": "fast_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "高质量缓存命中"}]
            }
        
        # 尝试常规缓存路径
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            print(f"完整问答缓存命中: {safe_query[:30]}... ({cache_time:.4f}s)")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, cached_response)
            
            return {
                "answer": cached_response,
                "execution_log": [{"node": "cache_hit", "timestamp": time.time(), "input": safe_query, "output": "常规缓存命中"}]
            }
        
        # 未命中缓存，执行标准流程
        process_start = time.time()
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # 执行完整的处理流程
            for output in self.graph.stream(inputs, config=config):
                pprint.pprint(f"Output from node '{list(output.keys())[0]}':")
                pprint.pprint("---")
                pprint.pprint(output, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")
                
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content
            
            # 缓存处理结果 - 同时更新会话缓存和全局缓存
            if answer and len(answer) > 10:
                # 更新会话缓存
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(safe_query, answer)
            
            process_time = time.time() - process_start
            print(f"完整处理耗时: {process_time:.4f}s")
            
            overall_time = time.time() - overall_start
            self._log_performance("ask_with_trace", {
                "total_duration": overall_time,
                "cache_check": cache_time,
                "processing": process_time
            })
            
            return {
                "answer": answer,
                "execution_log": self.execution_log
            }
        except Exception as e:
            error_time = time.time() - process_start
            print(f"处理查询时出错: {e} ({error_time:.4f}s)")
            return {
                "answer": f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}",
                "execution_log": self.execution_log + [{"node": "error", "timestamp": time.time(), "input": query, "output": str(e)}]
            }
        
    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        """向Agent提问"""
        overall_start = time.time()
        
        # 确保查询字符串是干净的
        safe_query = query.strip()
        
        cached_result = self._check_all_caches(safe_query, thread_id)
        if cached_result:
            return cached_result
        
        # 未命中缓存，执行标准流程
        process_start = time.time()
        
        # 正常处理请求
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            for output in self.graph.stream(inputs, config=config):
                pass
                    
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content
            
            # 缓存处理结果 - 同时更新会话缓存和全局缓存
            if answer and len(answer) > 10:
                # 更新会话缓存
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(safe_query, answer)
            
            process_time = time.time() - process_start
            overall_time = time.time() - overall_start
            
            self._log_performance("ask", {
                "total_duration": overall_time,
                "cache_check": 0,  # 由_check_all_caches记录
                "processing": process_time
            })
            
            return answer
        except Exception as e:
            error_time = time.time() - process_start
            print(f"处理查询时出错: {e} ({error_time:.4f}s)")
            return f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}"
    
    async def ask_stream(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> AsyncGenerator[str, None]:
        """
        向Agent提问，返回流式响应
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            recursion_limit: 递归限制
                
        返回:
            AsyncGenerator[str, None]: 流式响应生成器
        """
        overall_start = time.time()
        
        # 确保查询字符串是干净的
        safe_query = query.strip()
        
        # 首先尝试全局缓存（跨会话缓存）
        global_result = self.global_cache_manager.get(safe_query)
        if global_result:
            # 对于缓存响应，按自然语言单位分块返回
            import re
            chunks = re.split(r'([.!?。！？]\s*)', global_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_result = self.check_fast_cache(safe_query, thread_id)
        if fast_result:
            # 对于缓存响应，按自然语言单位分块返回
            import re
            chunks = re.split(r'([.!?。！？]\s*)', fast_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
                
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, fast_result)
            return
        
        # 尝试常规缓存路径
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            # 同样按自然语言单位分块
            import re
            chunks = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
                
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, cached_response)
            return
        
        # 未命中缓存，执行标准流程
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit,
                "stream_mode": True  # 指示流式输出模式
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        answer = ""
        
        try:
            # 执行流式处理
            async for chunk in self._stream_process(inputs, config):
                yield chunk
                answer += chunk
            
            # 缓存完整回答 - 同时更新会话缓存和全局缓存
            if answer and len(answer) > 10:
                # 更新会话缓存
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(safe_query, answer)
            
            process_time = time.time() - overall_start
            self._log_performance("ask_stream", {
                "total_duration": process_time,
                "processing": process_time
            })
            
        except Exception as e:
            error_time = time.time() - overall_start
            error_msg = f"处理查询时出错: {str(e)} ({error_time:.4f}s)"
            print(error_msg)
            yield error_msg
    
    def mark_answer_quality(self, query: str, is_positive: bool, thread_id: str = "default"):
        """标记回答质量，用于缓存质量控制"""
        start_time = time.time()
        
        # 提取关键词
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }
        
        # 调用缓存管理器的质量标记方法，传递相关参数
        marked = self.cache_manager.mark_quality(query.strip(), is_positive, **cache_params)
        
        mark_time = time.time() - start_time
        self._log_performance("mark_quality", {
            "duration": mark_time,
            "is_positive": is_positive
        })
    
    def clear_cache_for_query(self, query: str, thread_id: str = "default"):
        """
        清除特定查询的缓存（会话缓存和全局缓存）
        
        参数:
            query: 查询字符串
            thread_id: 会话ID
        
        返回:
            bool: 是否成功删除
        """
        # 清除会话缓存
        success = False
        
        try:
            # 尝试移除可能存在的前缀
            clean_query = query.strip()
            if ":" in clean_query:
                parts = clean_query.split(":", 1)
                if len(parts) > 1:
                    clean_query = parts[1].strip()
            
            # 清除原始查询的会话缓存
            session_cache_deleted = self.cache_manager.delete(query.strip(), thread_id=thread_id)
            success = session_cache_deleted
            
            # 清除没有前缀的查询缓存
            if clean_query != query.strip():
                self.cache_manager.delete(clean_query, thread_id=thread_id)
            
            # 清除带前缀的查询缓存变体
            prefixes = ["generate:", "deep:", "query:"]
            for prefix in prefixes:
                self.cache_manager.delete(f"{prefix}{clean_query}", thread_id=thread_id)
            
            # 清除全局缓存 - 使用所有可能的变体
            if hasattr(self, 'global_cache_manager'):
                # 删除原始查询
                global_cache_deleted = self.global_cache_manager.delete(query.strip())
                success = success or global_cache_deleted
                
                # 删除清理后的查询
                if clean_query != query.strip():
                    self.global_cache_manager.delete(clean_query)
                
                # 删除带前缀的查询变体
                for prefix in prefixes:
                    self.global_cache_manager.delete(f"{prefix}{clean_query}")
            
            # 强制刷新缓存写入
            if hasattr(self.cache_manager.storage, '_flush_write_queue'):
                self.cache_manager.storage._flush_write_queue()
            
            if hasattr(self, 'global_cache_manager') and hasattr(self.global_cache_manager.storage, '_flush_write_queue'):
                self.global_cache_manager.storage._flush_write_queue()
                
            # 记录日志
            print(f"已清除查询缓存: {query.strip()}")
            
            return success
        except Exception as e:
            print(f"清除缓存时出错: {e}")
            return False
    
    def _validate_answer(self, query: str, answer: str, thread_id: str = "default") -> bool:
        """验证答案质量"""
        
        # 使用缓存管理器的验证方法
        def validator(query, answer):
            # 基本检查 - 长度
            if len(answer) < 20:
                return False
                
            # 检查是否包含错误消息
            error_patterns = [
                "抱歉，处理您的问题时遇到了错误",
                "技术原因:",
                "无法获取",
                "无法回答这个问题"
            ]
            
            for pattern in error_patterns:
                if pattern in answer:
                    return False
                    
            # 相关性检查 - 检查问题关键词是否在答案中出现
            keywords = self._extract_keywords(query)
            if keywords:
                low_level_keywords = keywords.get("low_level", [])
                if low_level_keywords:
                    # 至少有一个低级关键词应该在答案中出现
                    keyword_found = any(keyword.lower() in answer.lower() for keyword in low_level_keywords)
                    if not keyword_found:
                        return False
            
            # 通过所有检查
            return True
        
        return self.cache_manager.validate_answer(query, answer, validator, thread_id=thread_id)
    
    def close(self):
        """关闭资源"""
        # 确保所有延迟写入的缓存项都被保存
        if hasattr(self.cache_manager.storage, '_flush_write_queue'):
            self.cache_manager.storage._flush_write_queue()
            
        # 同样确保全局缓存的写入被保存
        if hasattr(self.global_cache_manager.storage, '_flush_write_queue'):
            self.global_cache_manager.storage._flush_write_queue()