from typing import List, Dict
import time
import re
import asyncio
import traceback
import json

from graphrag_agent.agents.base import BaseAgent
from graphrag_agent.agents.agent_coordinator import GraphRAGAgentCoordinator
from graphrag_agent.search.tool.local_search_tool import LocalSearchTool
from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool
from graphrag_agent.search.tool.deeper_research_tool import DeeperResearchTool
from graphrag_agent.search.tool.deep_research_tool import DeepResearchTool
from graphrag_agent.search.tool.reasoning.chain_of_exploration import ChainOfExplorationSearcher
from graphrag_agent.search.tool.reasoning.validator import complexity_estimate
from graphrag_agent.search.tool.reasoning.evidence import EvidenceChainTracker
from graphrag_agent.models.get_models import get_embeddings_model

class FusionGraphRAGAgent(BaseAgent):
    """
    Fusion GraphRAG Agent
    
    基于多Agent协作架构的增强型GraphRAGAgent，集成了多种搜索策略和知识融合方法。
    提供图谱感知、社区结构、Chain of Exploration等高级功能，实现更深度的知识检索和推理。
    """
    
    def __init__(self):
        """初始化Fusion GraphRAG Agent"""
        # 设置缓存目录
        self.cache_dir = "./cache/fusion_graphrag"

        self.use_deeper_tool = True
        self.has_all_tools = False
        self.research_tool = DeepResearchTool()
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)
        
        # 创建协调器
        self.coordinator = GraphRAGAgentCoordinator(self.llm)
        
        # 初始化基础搜索工具 - 用于关键词提取
        self.search_tool = LocalSearchTool()
        
        # 初始化深度研究工具 - 使用增强版本
        try:
            # 尝试加载增强版深度研究工具
            self.research_tool = DeeperResearchTool()
            print("已加载增强版深度研究工具")

            # 获取各种专用工具
            self.chain_explorer = self.research_tool.get_exploration_tool()  # 知识图谱探索工具
            self.reasoning_analysis_tool = self.research_tool.get_reasoning_analysis_tool()  # 推理链分析工具
            self.stream_tool = self.research_tool.get_stream_tool()  # 流式处理工具
            self.evidence_tracker = EvidenceChainTracker()  # 证据链跟踪器
            
            # 设置为True表示成功加载了所有增强工具
            self.has_all_tools = True
        except Exception as e:
            print(f"加载增强版研究工具失败: {e}，将使用标准版")
            self.research_tool = DeepResearchTool()
            self.use_deeper_tool = False
            self.has_all_tools = False
            
            # 使用标准版的chain explorer
            self.chain_explorer = ChainOfExplorationSearcher(
                self.graph, self.llm, get_embeddings_model()
            )
            
            # 尝试获取流式工具
            try:
                self.stream_tool = self.research_tool.get_thinking_stream_tool()
            except:
                self.stream_tool = None
        
        # 设置思考可见性
        self.show_thinking = False
        
        # 添加查询历史记录，用于跟踪会话上下文
        self.query_context = {}
        
        # 跟踪已探索的查询分支
        self.explored_branches = {}
        
        # 矛盾检测结果缓存
        self.contradiction_cache = {}
    
    def _setup_tools(self) -> List:
        """设置工具"""
        # 创建工具实例
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        
        tools = [
            self.local_tool.get_tool(),
            self.global_tool.get_tool(),
        ]
        
        # 如果使用深度研究版本，则添加所有增强工具
        if self.use_deeper_tool and self.has_all_tools:
            tools.extend([
                self.research_tool.get_tool(),  # 基础深度研究工具
                self.chain_explorer,  # 知识图谱探索工具
                self.reasoning_analysis_tool,  # 推理链分析工具
                self.stream_tool,  # 流式处理工具
            ])
        elif self.use_deeper_tool:
            # 只使用基础深度研究工具
            tools.append(self.research_tool.get_tool())
        else:
            # 使用标准研究工具
            self.deep_research = DeepResearchTool()
            tools.append(self.deep_research.get_tool())
            
            # 如果有流式工具也添加
            if hasattr(self, 'stream_tool') and self.stream_tool:
                tools.append(self.stream_tool)
        
        return tools
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成，具体逻辑由协调器处理
        workflow.add_edge("retrieve", "generate")
    
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 使用搜索工具提取关键词
        return self.search_tool.extract_keywords(query)
    
    def _estimate_complexity(self, query: str) -> float:
        """
        估计查询复杂度
        
        参数:
            query: 查询字符串
            
        返回:
            float: 复杂度评分 (0.0-1.0)
        """
        # 使用复杂度估计器
        return complexity_estimate(query)
    
    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全获取问题内容
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
            return {"messages": [{"role": "assistant", "content": global_result}]}
            
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
            return {"messages": [{"role": "assistant", "content": cached_result}]}
        
        # 使用协调器处理
        try:
            start_time = time.time()
            
            # 检查查询复杂度
            complexity = self._estimate_complexity(question)
            
            if complexity > 0.7 and self.use_deeper_tool:
                # 使用增强版深度研究工具处理复杂查询
                self._log_execution("deep_research", 
                              {"question": question, "complexity": complexity}, 
                              "使用增强版深度研究工具")
                
                # 创建新的查询ID，用于跟踪
                query_id = f"query_{int(time.time())}"
                self.query_context[thread_id] = {"current_query_id": query_id}
                
                # 执行深度研究
                result = self.research_tool.thinking(question)
                answer = result.get("answer", "未能生成回答")
                
                # 如果有矛盾检测功能，进行矛盾检测
                if hasattr(self.research_tool, '_detect_and_resolve_contradictions'):
                    try:
                        contradiction_result = self.research_tool._detect_and_resolve_contradictions(query_id)
                        if contradiction_result and contradiction_result.get("contradictions"):
                            # 缓存矛盾检测结果
                            self.contradiction_cache[question] = contradiction_result
                            # 如果发现矛盾，在答案中添加提示
                            if "<think>" not in answer:  # 确保不重复添加
                                answer += "\n\n**注意**: 在分析过程中发现了信息来源中存在一些不一致之处。以上答案已尝试综合各方观点，提供最准确的信息。"
                    except Exception as e:
                        print(f"矛盾检测失败: {e}")
                
                # 清理答案 - 删除思考过程部分
                if "<think>" in answer and "</think>" in answer:
                    clean_answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL)
                else:
                    clean_answer = answer
            else:
                # 使用协调器处理标准复杂度的查询
                self._log_execution("coordinator", 
                              {"question": question, "complexity": complexity}, 
                              "使用标准协调器")
                result = self.coordinator.process_query(question)
                answer = result.get("answer", "未能生成回答")
                clean_answer = answer
            
            # 记录性能
            self._log_performance("process", {
                "duration": time.time() - start_time,
                "complexity": complexity
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            # 更新会话缓存
            self.cache_manager.set(question, clean_answer, thread_id=thread_id)
            # 更新全局缓存
            self.global_cache_manager.set(question, clean_answer)
            
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            clean_answer)
            
            return {"messages": [{"role": "assistant", "content": clean_answer}]}
            
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}\n{traceback.format_exc()}"
            self._log_execution("generate_error", 
                            {"question": question, "docs_length": len(docs)}, 
                            error_msg)
            return {"messages": [{"role": "assistant", "content": f"抱歉，我无法回答这个问题。技术原因: {str(e)}"}]}
    
    async def _async_enhance_search(self, query, thread_id):
        """
        异步执行社区感知搜索增强
        
        参数:
            query: 用户查询
            thread_id: 会话ID
            
        返回:
            Dict: 增强搜索结果
        """
        # 提取查询的关键词
        keywords = self._extract_keywords(query)
        
        # 使用Chain of Exploration增强搜索
        async def enhance_wrapper():
            # 防止报错的兜底逻辑
            try:
                if hasattr(self.research_tool, '_enhance_search_with_coe'):
                    return self.research_tool._enhance_search_with_coe(query, keywords)
                else:
                    # 如果方法不存在，使用标准链式探索
                    focus_entities = keywords.get("high_level", []) + keywords.get("low_level", [])
                    if focus_entities:
                        return self.chain_explorer.explore(
                            query, 
                            focus_entities[:3],  # 使用前3个关注实体作为起点
                            max_steps=3
                        )
                    return {}
            except Exception as e:
                print(f"增强搜索失败: {e}")
                return {}
        
        return await asyncio.get_event_loop().run_in_executor(None, enhance_wrapper)
    
    async def _async_detect_contradictions(self, query_id):
        """异步检测矛盾"""
        # 检查缓存
        if hasattr(self, '_contradiction_cache') and query_id in self._contradiction_cache:
            return self._contradiction_cache[query_id]
            
        def detect_wrapper():
            result = self._detect_and_resolve_contradictions(query_id)
            # 更新缓存
            if not hasattr(self, '_contradiction_cache'):
                self._contradiction_cache = {}
            self._contradiction_cache[query_id] = result
            return result
        
        return await asyncio.get_event_loop().run_in_executor(None, detect_wrapper)
    
    async def _stream_process(self, inputs, config):
        """流式处理过程"""
        # 获取查询
        query = inputs["messages"][-1].content if len(inputs["messages"]) > 0 else ""
        if not query:
            yield "无法获取查询内容，请重试。"
            return
            
        # 获取会话ID
        thread_id = config.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存
        cached_result = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_result:
            self._log_execution("stream_cache_hit", {"query": query}, "缓存命中")
            # 分块返回缓存结果
            sentences = re.split(r'([.!?。！？]\s*)', cached_result)
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
            return
        
        # 估计查询复杂度
        complexity = self._estimate_complexity(query)
        
        # 根据复杂度决定使用哪种处理方式
        if complexity > 0.7 and self.use_deeper_tool:
            # 复杂查询使用深度研究工具处理
            yield "**开始进行深度分析**...\n\n"
            
            # 创建新的查询ID，用于跟踪
            query_id = f"query_{int(time.time())}"
            self.query_context[thread_id] = {"current_query_id": query_id}
            
            try:
                # 尝试使用社区感知增强搜索
                yield "**分析相关知识社区与实体关联**...\n"
                enhanced_context = await self._async_enhance_search(query, thread_id)
                
                if enhanced_context:
                    # 如果找到了相关社区或实体
                    if "exploration_results" in enhanced_context:
                        # 提取探索路径
                        exp_results = enhanced_context["exploration_results"]
                        if "exploration_path" in exp_results and len(exp_results["exploration_path"]) > 1:
                            path_msg = "\n**发现相关知识路径**:\n"
                            for i, step in enumerate(exp_results["exploration_path"][:5]):
                                if i > 0:  # 跳过起始实体
                                    path_msg += f"- {step.get('node_id')}: {step.get('reasoning')[:50]}...\n"
                            yield path_msg
            except Exception as e:
                # 打印错误但继续处理
                print(f"增强搜索失败: {e}")
                # 不返回错误消息给用户，继续正常流程
            
            # 使用深度研究工具的流式思考方法
            try:
                last_chunk = None
                async for chunk in self.research_tool.thinking_stream(query):
                    if isinstance(chunk, dict) and "answer" in chunk:
                        # 这是最终结果对象
                        final_answer = chunk["answer"]
                        last_chunk = chunk
                        
                        # 在最终答案之前添加矛盾检测结果
                        try:
                            # 检测矛盾
                            contradiction_result = await self._async_detect_contradictions(query_id)
                            
                            if contradiction_result["contradictions"]:
                                yield "\n**信息一致性分析**：发现信息中存在一些不一致之处。在综合答案时已考虑这些因素。\n\n"
                        except Exception as e:
                            # 矛盾检测失败，不影响正常流程
                            print(f"矛盾检测失败: {e}")
                        
                        # 清理答案，移除思考过程
                        if "<think>" in final_answer and "</think>" in final_answer:
                            clean_answer = re.sub(r'<think>.*?</think>\s*', '', final_answer, flags=re.DOTALL)
                            
                            # 缓存清理后的答案
                            self.cache_manager.set(query, clean_answer, thread_id=thread_id)
                            self.global_cache_manager.set(query, clean_answer)
                            
                            yield clean_answer
                        else:
                            # 没有思考标记，直接使用
                            self.cache_manager.set(query, final_answer, thread_id=thread_id)
                            self.global_cache_manager.set(query, final_answer)
                            
                            yield final_answer
                    else:
                        # 返回思考过程
                        yield chunk
            except Exception as e:
                error_msg = f"深度研究过程中出错: {str(e)}"
                print(error_msg)
                yield f"**处理查询时出错**: {str(e)}"
        else:
            # 使用协调器的标准流式处理
            try:
                self._log_execution("stream_process_start", {"query": query}, "开始流式处理")
                async for chunk in self.coordinator.process_query_stream(query):
                    yield chunk
                
                # 注意：协调器内部会处理缓存
                self._log_execution("stream_process_complete", {"query": query}, "完成流式处理")
                
            except Exception as e:
                error_msg = f"处理查询时出错: {str(e)}"
                self._log_execution("stream_process_error", {"query": query}, error_msg)
                yield f"**处理查询时出错**: {str(e)}"
    
    async def explore_knowledge_stream(self, query, thread_id="default"):
        """
        流式知识图谱探索
        
        参数:
            query: 用户查询
            thread_id: 会话ID
        
        返回:
            AsyncGenerator: 流式探索结果
        """
        # 提取关键词作为起始实体
        keywords = self._extract_keywords(query)
        entities = keywords.get("high_level", []) + keywords.get("low_level", [])
        entities = entities[:3]  # 最多使用3个实体
        
        # 检查是否有实体
        if not entities:
            yield "未能从查询中提取到实体，无法执行知识图谱探索。"
            return
            
        # 通知用户开始探索
        yield f"**开始从实体 {', '.join(entities)} 探索相关知识**...\n"
        
        try:
            # 如果使用增强版研究工具且有专用探索方法
            if self.use_deeper_tool and hasattr(self.research_tool, 'get_exploration_tool'):
                # 使用专用探索工具
                exploration_tool = self.research_tool.get_exploration_tool()
                # 执行探索
                exploration_results = exploration_tool._run({
                    "query": query,
                    "entities": entities
                })
            else:
                # 使用标准链式探索
                exploration_results = self.chain_explorer.explore(query, entities, max_steps=3)
                
            # 处理结果
            if not exploration_results or not isinstance(exploration_results, dict):
                yield "未找到相关的知识路径。"
                return
                
            # 提取探索路径
            if "exploration_path" in exploration_results:
                path = exploration_results["exploration_path"]
                
                # 显示路径
                yield "**发现的知识路径**:\n"
                
                for step in path:
                    step_num = step.get("step", 0)
                    node_id = step.get("node_id", "")
                    reasoning = step.get("reasoning", "")
                    
                    if step_num > 0:  # 跳过起始实体
                        yield f"- 步骤{step_num}: 从 '{node_id}' 发现 - {reasoning}\n"
                        await asyncio.sleep(0.01)
                
                # 提取关键内容
                if "content" in exploration_results:
                    content = exploration_results["content"]
                    
                    yield "\n**发现的相关内容**:\n"
                    
                    for i, item in enumerate(content[:5]):  # 限制显示5条
                        if "text" in item:
                            text = item["text"]
                            if len(text) > 200:
                                text = text[:200] + "..."
                            yield f"内容 {i+1}: {text}\n\n"
                            await asyncio.sleep(0.02)
            else:
                yield "未找到清晰的知识探索路径。"
                
        except Exception as e:
            error_msg = f"知识探索过程中出错: {str(e)}"
            print(error_msg)
            yield f"**探索过程中出错**: {str(e)}"
    
    async def analyze_reasoning_chain(self, query_id=None, thread_id="default"):
        """
        分析推理链和证据，流式返回
        
        参数:
            query_id: 查询ID，如果为None则使用当前会话的查询ID
            thread_id: 会话ID
            
        返回:
            AsyncGenerator: 流式分析结果
        """
        # 检查是否有推理分析工具
        if not self.has_all_tools or not hasattr(self.research_tool, 'get_reasoning_analysis_tool'):
            yield "推理链分析功能需要使用增强版研究工具，当前环境不支持此功能。"
            return
            
        # 如果未提供查询ID，尝试从会话上下文获取
        if not query_id:
            if thread_id in self.query_context and "current_query_id" in self.query_context[thread_id]:
                query_id = self.query_context[thread_id]["current_query_id"]
            else:
                yield "未找到有效的查询ID，请先执行一次深度研究查询。"
                return
        
        # 使用推理分析工具
        analysis_tool = self.research_tool.get_reasoning_analysis_tool()
        
        yield "**开始分析推理链和证据**...\n\n"
        
        try:
            # 执行推理链分析
            results = analysis_tool._run(query_id)
            
            if not results:
                yield "未找到相关的推理链数据。"
                return
                
            # 展示推理摘要
            if "summary" in results:
                summary = results["summary"]
                yield "**推理过程摘要**:\n\n"
                
                for key, value in summary.items():
                    if key == "steps_count":
                        yield f"- 推理步骤总数: {value}\n"
                    elif key == "evidence_count":
                        yield f"- 使用的证据数量: {value}\n"
                    elif key == "duration_seconds":
                        yield f"- 推理总时间: {value:.2f}秒\n"
                    elif key == "confidence":
                        yield f"- 答案置信度: {value:.2f}\n"
                
                yield "\n"
                
            # 展示矛盾分析
            if "contradictions" in results and results["contradictions"]:
                contradictions = results["contradictions"]
                yield f"**发现的矛盾信息** ({len(contradictions)}项):\n\n"
                
                for i, c in enumerate(contradictions[:3]):  # 限制显示3个
                    if c["type"] == "numerical":
                        yield f"{i+1}. 数值矛盾: 在'{c.get('context', '')}'中，"
                        yield f"值 {c.get('value1')} 和 {c.get('value2')} 不一致\n"
                    else:
                        yield f"{i+1}. 语义矛盾: {c.get('analysis', '')}\n"
                
                if len(contradictions) > 3:
                    yield f"(共有 {len(contradictions)} 个矛盾，仅显示前3个)\n\n"
                    
            # 展示证据统计
            if "evidence_stats" in results:
                stats = results["evidence_stats"]
                yield "**证据来源统计**:\n\n"
                
                for source, count in stats.items():
                    yield f"- {source}: {count}项\n"
                
                yield "\n"
                
            # 展示完整推理链
            if "reasoning_chain" in results and "steps" in results["reasoning_chain"]:
                steps = results["reasoning_chain"]["steps"]
                yield f"**推理链详情** ({len(steps)}步):\n\n"
                
                for i, step in enumerate(steps[:5]):  # 限制显示5步
                    step_type = step.get("type", "未知类型")
                    description = step.get("description", "无描述")
                    evidence_count = len(step.get("evidence_ids", []))
                    
                    yield f"{i+1}. [{step_type}] {description} ({evidence_count}个证据)\n"
                
                if len(steps) > 5:
                    yield f"(共有 {len(steps)} 个推理步骤，仅显示前5个)\n"
            
        except Exception as e:
            error_msg = f"分析推理链时出错: {str(e)}"
            print(error_msg)
            yield f"**分析过程中出错**: {str(e)}"
    
    async def detect_contradictions(self, query, thread_id="default"):
        """
        检测和分析信息矛盾，流式返回
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            
        返回:
            AsyncGenerator: 流式矛盾分析结果
        """
        # 检查是否有推理分析工具
        if not self.has_all_tools:
            yield "矛盾检测功能需要使用增强版研究工具，当前环境不支持此功能。"
            return
            
        # 首先检查缓存
        if query in self.contradiction_cache:
            cached_result = self.contradiction_cache[query]
            yield f"**从缓存中获取矛盾分析结果**\n\n"
            
            # 显示缓存的矛盾结果
            contradictions = cached_result.get("contradictions", [])
            if contradictions:
                yield f"**发现 {len(contradictions)} 个矛盾信息**:\n\n"
                
                for i, c in enumerate(contradictions):
                    if c["type"] == "numerical":
                        yield f"{i+1}. 数值矛盾: 在'{c.get('context', '')}'中，"
                        yield f"值 {c.get('value1')} 和 {c.get('value2')} 不一致\n"
                    else:
                        yield f"{i+1}. 语义矛盾: {c.get('analysis', '')}\n"
            else:
                yield "未检测到明显的信息矛盾。"
            return
            
        # 如果没有缓存，执行矛盾检测
        yield "**开始检测信息矛盾**...\n\n"
        
        # 创建新的查询ID用于此次分析
        query_id = f"query_{int(time.time())}"
        self.query_context[thread_id] = {"current_query_id": query_id}
        
        if hasattr(self.research_tool, 'detect_contradictions'):
            # 如果直接有矛盾检测方法
            try:
                result = self.research_tool.detect_contradictions(query, thread_id)
                
                # 缓存结果
                self.contradiction_cache[query] = result
                
                # 显示矛盾分析结果
                if result["has_contradictions"]:
                    yield f"**发现 {result['count']} 个信息矛盾**:\n\n"
                    
                    for i, c in enumerate(result["contradictions"]):
                        if isinstance(c, dict):
                            if c.get("type") == "numerical":
                                yield f"{i+1}. 数值矛盾: 在'{c.get('context', '')}'中，"
                                yield f"值 {c.get('value1')} 和 {c.get('value2')} 不一致\n"
                            else:
                                yield f"{i+1}. 语义矛盾: {c.get('analysis', '')}\n"
                    
                    # 显示矛盾影响分析
                    if "impact_analysis" in result:
                        yield "\n**矛盾对结论的影响分析**:\n\n"
                        yield result["impact_analysis"]
                else:
                    yield "**分析结果**: 未在信息来源中检测到明显矛盾。"
            except Exception as e:
                print(f"矛盾检测失败: {e}")
                yield f"**检测过程中出错**: {str(e)}"
        else:
            # 没有专用方法，执行深度思考后检测矛盾
            try:
                # 执行深度思考
                yield "**执行深度分析以检测潜在矛盾**...\n"
                result = self.research_tool.thinking(query)
                
                # 提取查询ID
                if hasattr(self.research_tool, 'current_query_context'):
                    query_id = self.research_tool.current_query_context.get("query_id", query_id)
                
                # 使用推理分析工具检测矛盾
                if hasattr(self, 'reasoning_analysis_tool'):
                    analysis_result = self.reasoning_analysis_tool._run(query_id)
                    
                    # 提取矛盾信息
                    contradictions = analysis_result.get("contradictions", [])
                    
                    # 缓存矛盾结果
                    self.contradiction_cache[query] = {"contradictions": contradictions}
                    
                    if contradictions:
                        yield f"**发现 {len(contradictions)} 个信息矛盾**:\n\n"
                        
                        for i, c in enumerate(contradictions):
                            if c["type"] == "numerical":
                                yield f"{i+1}. 数值矛盾: 在'{c.get('context', '')}'中，"
                                yield f"值 {c.get('value1')} 和 {c.get('value2')} 不一致\n"
                            else:
                                yield f"{i+1}. 语义矛盾: {c.get('analysis', '')}\n"
                        
                        # 生成矛盾影响分析
                        yield "\n**生成矛盾影响分析**...\n"
                        
                        impact_prompt = f"""
                        在回答关于"{query}"的问题时，发现以下信息矛盾:
                        
                        {json.dumps(contradictions, ensure_ascii=False)}
                        
                        请分析这些矛盾对最终答案可能产生的影响，以及如何在存在这些矛盾的情况下给出最准确的回答。
                        """
                        
                        impact_response = self.llm.invoke(impact_prompt)
                        impact_analysis = impact_response.content if hasattr(impact_response, 'content') else str(impact_response)
                        
                        yield "\n**矛盾对结论的影响分析**:\n\n"
                        yield impact_analysis
                    else:
                        yield "**分析结果**: 未在信息来源中检测到明显矛盾。"
                else:
                    yield "**注意**: 无法执行详细的矛盾分析，因为缺少推理分析工具。"
            except Exception as e:
                print(f"矛盾检测失败: {e}")
                yield f"**检测过程中出错**: {str(e)}"

    def generate_multi_hypothesis(self, query, thread_id="default", max_hypotheses=3):
        """
        为复杂查询生成多个假设
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            max_hypotheses: 最大假设数
            
        返回:
            Dict: 假设结果
        """
        # 检查是否有生成假设的能力
        if not self.has_all_tools:
            return {"error": "多假设生成功能需要使用增强版研究工具"}
        
        try:
            # 检查查询复杂度
            complexity = self._estimate_complexity(query)
            
            if complexity < 0.5:
                return {"error": "查询复杂度较低，不需要多假设分析", "complexity": complexity}
            
            # 使用查询生成器生成假设
            if hasattr(self.research_tool, 'query_generator'):
                hypotheses = self.research_tool.query_generator.generate_multiple_hypotheses(
                    query, self.llm, max_hypotheses=max_hypotheses
                )
            else:
                # 直接使用LLM生成假设
                hypothesis_prompt = f"""
                针对问题: "{query}"
                
                请生成 {max_hypotheses} 个不同的假设或可能的答案路径。这些假设应该:
                1. 相互不同，覆盖问题的不同角度
                2. 都是合理的可能性，而不是随意猜测
                3. 具体、可验证，而不是笼统的表述
                
                请直接列出假设，无需额外解释。
                """
                
                response = self.llm.invoke(hypothesis_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # 解析假设
                hypotheses = []
                lines = response_text.strip().split('\n')
                for line in lines:
                    # 移除数字、破折号等前缀
                    clean_line = re.sub(r'^\d+[\.\)、\-\s]+', '', line).strip()
                    if clean_line and clean_line not in hypotheses:
                        hypotheses.append(clean_line)
                
                # 限制最大数量
                hypotheses = hypotheses[:max_hypotheses]
            
            # 创建新的查询ID
            query_id = f"query_{int(time.time())}"
            self.query_context[thread_id] = {"current_query_id": query_id}
            
            # 为假设创建分支
            branch_results = {}
            
            for i, hypothesis in enumerate(hypotheses):
                branch_name = f"branch_{i+1}"
                
                # 记录分支
                branch_results[branch_name] = {
                    "hypothesis": hypothesis,
                    "status": "created",
                    "branch_id": i+1
                }
            
            return {
                "query_id": query_id,
                "hypotheses": hypotheses,
                "branches": branch_results,
                "complexity": complexity
            }
        except Exception as e:
            print(f"假设生成失败: {e}")
            return {"error": f"假设生成失败: {str(e)}"}

    async def analyze_branch(self, branch_data, thread_id="default"):
        """
        分析特定假设分支
        
        参数:
            branch_data: 分支信息，包含query和hypothesis
            thread_id: 会话ID
            
        返回:
            AsyncGenerator: 流式分析结果
        """
        # 检查输入
        if not isinstance(branch_data, dict) or "query" not in branch_data or "hypothesis" not in branch_data:
            yield "输入格式错误: 需要包含 'query' 和 'hypothesis' 字段"
            return
        
        query = branch_data["query"]
        hypothesis = branch_data["hypothesis"]
        
        yield f"**开始分析假设**: {hypothesis}\n\n"
        
        # 创建反事实思考提示
        counter_prompt = f"""
        请分析以下问题和假设:
        
        问题: {query}
        假设: {hypothesis}
        
        请进行反事实分析，思考如果该假设不成立会怎样。具体来说:
        1. 如果假设不正确，哪些条件必须不成立？
        2. 有哪些证据可能直接反驳这个假设？
        3. 这个假设可能忽略了哪些重要因素？
        
        请提供详细分析。
        """
        
        try:
            # 使用LLM生成反事实分析
            response = self.llm.invoke(counter_prompt)
            counter_analysis = response.content if hasattr(response, 'content') else str(response)
            
            # 构建分析提示
            analysis_prompt = f"""
            请基于以下问题和假设进行深入分析:
            
            问题: {query}
            假设: {hypothesis}
            
            反事实分析: {counter_analysis}
            
            请评估这个假设的可靠性和可能的证据支持:
            1. 支持该假设的可能证据有哪些？
            2. 该假设解释了问题的哪些方面？
            3. 该假设忽略了哪些关键因素？
            4. 总体可信度评分(0-100)
            """
            
            yield "**正在进行反事实分析**...\n\n"
            yield counter_analysis + "\n\n"
            
            # 使用LLM生成假设分析
            response = self.llm.invoke(analysis_prompt)
            hypothesis_analysis = response.content if hasattr(response, 'content') else str(response)
            
            yield "**假设分析**:\n\n"
            yield hypothesis_analysis
            
            # 提取可信度评分
            confidence_match = re.search(r'可信度评分.*?(\d+)', hypothesis_analysis)
            confidence = int(confidence_match.group(1)) if confidence_match else 50
            
            # 更新分支信息
            branch_name = f"branch_{int(time.time())}"
            self.explored_branches[branch_name] = {
                "hypothesis": hypothesis,
                "counter_analysis": counter_analysis,
                "hypothesis_analysis": hypothesis_analysis,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"分支分析失败: {e}")
            yield f"**分析过程中出错**: {str(e)}"

    async def run_advanced_query(self, advanced_query, thread_id="default"):
        """
        运行高级查询，支持各种增强功能
        
        参数:
            advanced_query: 高级查询信息，包含query和mode字段
            thread_id: 会话ID
            
        返回:
            AsyncGenerator: 流式查询结果
        """
        # 检查输入
        if not isinstance(advanced_query, dict) or "query" not in advanced_query:
            yield "输入格式错误: 需要包含 'query' 字段"
            return
        
        query = advanced_query["query"]
        mode = advanced_query.get("mode", "auto")
        
        # 根据模式执行不同的操作
        if mode == "explore_knowledge":
            # 知识图谱探索模式
            async for chunk in self.explore_knowledge_stream(query, thread_id):
                yield chunk
        elif mode == "detect_contradictions":
            # 矛盾检测模式
            async for chunk in self.detect_contradictions(query, thread_id):
                yield chunk
        elif mode == "reasoning_analysis":
            # 推理链分析模式
            query_id = advanced_query.get("query_id")
            async for chunk in self.analyze_reasoning_chain(query_id, thread_id):
                yield chunk
        elif mode == "multi_hypothesis":
            # 多假设生成模式
            max_hyp = advanced_query.get("max_hypotheses", 3)
            result = self.generate_multi_hypothesis(query, thread_id, max_hyp)
            
            if "error" in result:
                yield f"**错误**: {result['error']}"
            else:
                yield f"**为查询生成了 {len(result['hypotheses'])} 个假设**:\n\n"
                
                for i, hyp in enumerate(result["hypotheses"]):
                    yield f"{i+1}. {hyp}\n"
                
                yield f"\n**查询复杂度**: {result['complexity']:.2f}/1.0\n"
                yield f"**查询ID**: {result['query_id']}\n"
        elif mode == "branch_analysis":
            # 分支分析模式
            hypothesis = advanced_query.get("hypothesis")
            if not hypothesis:
                yield "错误: 分支分析需要提供 'hypothesis' 字段"
            else:
                branch_data = {"query": query, "hypothesis": hypothesis}
                async for chunk in self.analyze_branch(branch_data, thread_id):
                    yield chunk
        else:
            # 自动模式 - 根据复杂度选择合适的处理方式
            complexity = self._estimate_complexity(query)
            
            if complexity > 0.7:
                # 复杂查询，先尝试多假设生成
                yield f"**检测到复杂查询 (复杂度: {complexity:.2f}/1.0)**\n\n"
                yield "**生成多角度分析假设**...\n\n"
                
                result = self.generate_multi_hypothesis(query, thread_id)
                
                if "error" not in result and "hypotheses" in result and result["hypotheses"]:
                    yield f"**从以下 {len(result['hypotheses'])} 个角度分析问题**:\n\n"
                    
                    for i, hyp in enumerate(result["hypotheses"][:3]):
                        yield f"{i+1}. {hyp}\n"
                    
                    yield "\n**开始深度分析**...\n\n"
                
                # 然后执行标准流程
                async for chunk in self._stream_process({"messages": [{"role": "user", "content": query}]}, 
                                                    {"configurable": {"thread_id": thread_id}}):
                    yield chunk
            else:
                # 标准复杂度，直接执行标准流程
                async for chunk in self._stream_process({"messages": [{"role": "user", "content": query}]}, 
                                                    {"configurable": {"thread_id": thread_id}}):
                    yield chunk

    def close(self):
        """关闭资源"""
        # 调用父类方法
        super().close()
        
        # 关闭各种工具资源
        if hasattr(self, 'local_tool'):
            self.local_tool.close()
        if hasattr(self, 'global_tool'):
            self.global_tool.close()
        if hasattr(self, 'research_tool'):
            self.research_tool.close()