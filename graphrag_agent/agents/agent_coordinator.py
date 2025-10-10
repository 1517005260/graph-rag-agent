from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import time
import json
import re

from graphrag_agent.models.get_models import get_llm_model, get_stream_llm_model, get_embeddings_model
from graphrag_agent.search.tool.deeper_research_tool import DeeperResearchTool
from graphrag_agent.search.tool.local_search_tool import LocalSearchTool
from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool
from graphrag_agent.search.tool.reasoning.chain_of_exploration import ChainOfExplorationSearcher
from graphrag_agent.search.tool.reasoning.thinking import ThinkingEngine


class GraphRAGAgentCoordinator:
    """
    多Agent协作系统协调器
    
    协调多个专用Agent共同解决复杂问题，实现Fusion GraphRAG的多Agent协同架构
    """
    
    def __init__(self, llm=None):
        # 初始化语言模型
        self.llm = llm or get_llm_model()
        self.stream_llm = get_stream_llm_model()
        self.embeddings = get_embeddings_model()
        
        # 创建专用Agent
        self.retrieval_planner = self._create_retrieval_planner()
        self.local_searcher = self._create_local_searcher()
        self.global_searcher = self._create_global_searcher()
        self.explorer = self._create_explorer()
        self.chain_explorer = self._create_chain_explorer()
        self.synthesizer = self._create_synthesizer()
        self.thinking_engine = self._create_thinking_engine()
        
        # 执行记录
        self.execution_trace = []
        self.performance_metrics = {}
    
    def _create_retrieval_planner(self):
        """创建检索计划生成器Agent"""
        class RetrievalPlannerAgent:
            def __init__(self, llm):
                self.llm = llm
                self.name = "retrieval_planner"
                self.description = "负责分析查询并生成最佳检索计划的Agent"
            
            def plan(self, query: str) -> Dict[str, Any]:
                """分析查询并生成检索计划"""
                prompt = f"""
                分析以下查询，创建一个全面的检索计划以获取所需信息。
                
                查询: "{query}"
                
                请考虑:
                1. 查询的复杂度和所需的检索深度
                2. 可能涉及的知识领域和关键实体
                3. 是否需要全局概览或具体细节
                4. 需要进行的探索步骤
                5. 是否需要追踪实体间的关系路径
                6. 查询是否涉及时间信息
                
                以JSON格式返回检索计划，包括:
                - complexity_assessment: 查询复杂度(0-1)
                - knowledge_areas: 涉及的知识领域
                - key_entities: 关键实体
                - requires_global_view: 是否需要全局概览
                - requires_path_tracking: 是否需要实体关系路径追踪
                - has_temporal_aspects: 是否包含时间相关内容
                - tasks: 检索任务列表，每个任务包含:
                  * type: 任务类型(local_search/global_search/exploration/chain_exploration)
                  * query: 具体的查询内容
                  * priority: 优先级(1-5)
                  * entities: 相关实体(用于chain_exploration类型)
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # 提取JSON
                    import re
                    import json
                    json_match = re.search(r'({.*})', content, re.DOTALL)
                    if json_match:
                        plan = json.loads(json_match.group(1))
                        return plan
                    else:
                        # 如果无法解析，返回基础计划
                        return {
                            "complexity_assessment": 0.5,
                            "requires_global_view": False,
                            "requires_path_tracking": False,
                            "has_temporal_aspects": False,
                            "tasks": [
                                {"type": "local_search", "query": query, "priority": 3}
                            ]
                        }
                except Exception as e:
                    print(f"计划生成失败: {str(e)}")
                    # 返回默认计划
                    return {
                        "complexity_assessment": 0.5,
                        "requires_global_view": False,
                        "requires_path_tracking": False,
                        "has_temporal_aspects": False,
                        "tasks": [
                            {"type": "local_search", "query": query, "priority": 3}
                        ]
                    }
        
        return RetrievalPlannerAgent(self.llm)
    
    def _create_local_searcher(self):
        """创建本地搜索Agent"""
        return LocalSearchTool()
    
    def _create_global_searcher(self):
        """创建全局搜索Agent"""
        return GlobalSearchTool()
    
    def _create_explorer(self):
        """创建探索Agent"""
        return DeeperResearchTool()
    
    def _create_chain_explorer(self):
        """创建Chain of Exploration Agent"""
        # 获取图数据库连接
        from graphrag_agent.config.neo4jdb import get_db_manager
        db_manager = get_db_manager()
        graph = db_manager.get_graph()

        # 创建Chain of Exploration搜索器
        return ChainOfExplorationSearcher(graph, self.llm, self.embeddings)
    
    def _create_thinking_engine(self):
        """创建思考引擎"""
        return ThinkingEngine(self.llm)
    
    def _create_synthesizer(self):
        """创建结果合成Agent"""
        class SynthesizerAgent:
            def __init__(self, llm):
                self.llm = llm
                self.name = "synthesizer"
                self.description = "负责整合所有检索结果并生成最终答案的Agent"
            
            def synthesize(self, query: str, results: Dict[str, List], plan: Dict[str, Any],
                          thinking_process: str = None) -> str:
                """整合结果并生成答案"""
                # 构建提示
                prompt = f"""
                基于以下检索结果，回答用户的问题。
                
                用户问题: "{query}"
                
                ## 检索计划
                {json.dumps(plan, ensure_ascii=False, indent=2)}
                """
                
                # 添加思考过程
                if thinking_process:
                    prompt += f"""
                    ## 思考过程
                    {thinking_process}
                    """
                
                # 添加检索结果
                prompt += f"""
                ## 本地检索结果
                {self._format_results(results.get('local', []))}
                
                ## 全局检索结果
                {self._format_results(results.get('global', []))}
                
                ## 探索结果
                {self._format_results(results.get('exploration', []))}
                
                ## Chain of Exploration结果
                {self._format_coe_results(results.get('chain_exploration', []))}
                
                请提供一个全面、准确的回答。确保:
                1. 综合所有相关信息
                2. 解决问题的核心
                3. 说明清晰，逻辑严密
                4. 适当引用信息来源
                5. 结构清晰，使用段落和标题组织内容
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    return f"合成回答时出错: {str(e)}"
            
            def _format_results(self, results: List) -> str:
                """格式化结果列表"""
                if not results:
                    return "无相关结果"
                
                formatted = []
                for i, result in enumerate(results):
                    formatted.append(f"结果 {i+1}:\n{result}\n")
                return "\n".join(formatted)
                
            def _format_coe_results(self, results: List) -> str:
                """格式化Chain of Exploration结果"""
                if not results:
                    return "无Chain of Exploration结果"
                    
                formatted = []
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        # 提取路径信息
                        path_info = "探索路径:\n"
                        for step in result.get('exploration_path', [])[:5]:  # 只显示前5步
                            path_info += f"- 步骤{step.get('step')}: {step.get('node_id')} ({step.get('reasoning', '无理由')})\n"
                        
                        # 提取内容
                        content_info = "发现内容:\n"
                        for j, content in enumerate(result.get('content', [])[:3]):  # 只显示前3个内容
                            text = content.get('text', '')[:200]  # 限制长度
                            content_info += f"  内容{j+1}: {text}...\n"
                            
                        formatted.append(f"探索结果 {i+1}:\n{path_info}\n{content_info}\n")
                    else:
                        formatted.append(f"探索结果 {i+1}:\n{str(result)[:500]}...\n")
                
                return "\n".join(formatted)
        
        return SynthesizerAgent(self.llm)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理查询，协调多个Agent完成任务
        
        参数:
            query: 用户查询
            
        返回:
            Dict: 包含最终答案和处理记录的字典
        """
        start_time = time.time()
        self.execution_trace = []
        
        # 1. 生成检索计划
        self._log_step("generating_plan", "生成检索计划")
        retrieval_plan = self.retrieval_planner.plan(query)
        self._log_step("plan_generated", "检索计划已生成", retrieval_plan)
        
        # 初始化思考引擎
        self._log_step("thinking_init", "初始化思考引擎")
        self.thinking_engine.initialize_with_query(query)
        
        # 1.5 如果问题复杂度高，生成初步思考
        complexity = retrieval_plan.get("complexity_assessment", 0)
        if complexity > 0.7:
            self._log_step("initial_thinking", "生成初步思考")
            initial_thinking = self.thinking_engine.generate_initial_thinking()
            self._log_step("initial_thinking_complete", "完成初步思考", {"thinking": initial_thinking})
        else:
            initial_thinking = None
        
        # 2. 根据计划执行搜索任务
        local_results = []
        global_results = []
        exploration_results = []
        chain_exploration_results = []
        
        # 按优先级排序任务
        tasks = sorted(retrieval_plan.get("tasks", []), 
                     key=lambda x: x.get("priority", 3), 
                     reverse=True)
        
        for task in tasks:
            task_type = task.get("type", "")
            task_query = task.get("query", query)
            
            # 添加任务进度到思考引擎
            self.thinking_engine.add_reasoning_step(f"执行任务: {task_type} - {task_query}")
            
            if task_type == "local_search":
                self._log_step("local_search", f"执行本地搜索: {task_query}")
                try:
                    result = self.local_searcher.search(task_query)
                    if result:
                        local_results.append(result)
                        # 添加结果到思考引擎
                        self.thinking_engine.add_reasoning_step(f"本地搜索结果摘要:\n{self._summarize_result(result)}")
                    self._log_step("local_search_completed", "本地搜索完成")
                except Exception as e:
                    self._log_step("local_search_error", f"本地搜索出错: {str(e)}")
                    
            elif task_type == "global_search":
                self._log_step("global_search", f"执行全局搜索: {task_query}")
                try:
                    result = self.global_searcher.search(task_query)
                    if result:
                        global_results.append(result)
                        # 添加结果到思考引擎
                        self.thinking_engine.add_reasoning_step(f"全局搜索结果摘要:\n{self._summarize_result(result)}")
                    self._log_step("global_search_completed", "全局搜索完成")
                except Exception as e:
                    self._log_step("global_search_error", f"全局搜索出错: {str(e)}")
                    
            elif task_type == "exploration":
                self._log_step("exploration", f"执行深度探索: {task_query}")
                try:
                    result = self.explorer.search(task_query)
                    if result:
                        exploration_results.append(result)
                        # 添加结果到思考引擎
                        self.thinking_engine.add_reasoning_step(f"深度探索结果摘要:\n{self._summarize_result(result)}")
                    self._log_step("exploration_completed", "深度探索完成")
                except Exception as e:
                    self._log_step("exploration_error", f"探索出错: {str(e)}")
                    
            elif task_type == "chain_exploration":
                self._log_step("chain_exploration", f"执行Chain of Exploration: {task_query}")
                try:
                    # 获取相关实体
                    entities = task.get("entities", [])
                    if not entities:
                        # 如果任务没有指定实体，尝试从其他结果中提取
                        entities = self._extract_entities_from_results(
                            local_results, global_results, exploration_results
                        )
                    
                    # 至少需要一个起始实体
                    if not entities:
                        self._log_step("chain_exploration_warning", "未找到起始实体，跳过Chain of Exploration")
                        continue
                        
                    # 执行Chain of Exploration
                    result = self.chain_explorer.explore(
                        task_query, 
                        entities[:3],  # 使用前3个实体作为起点
                        max_steps=3
                    )
                    
                    if result:
                        chain_exploration_results.append(result)
                        # 添加结果到思考引擎
                        path_summary = "探索路径:\n"
                        for step in result.get('exploration_path', [])[:5]:
                            path_summary += f"- 步骤{step.get('step')}: {step.get('node_id')} ({step.get('reasoning', '无理由')})\n"
                        self.thinking_engine.add_reasoning_step(f"Chain of Exploration结果:\n{path_summary}")
                    self._log_step("chain_exploration_completed", "Chain of Exploration完成")
                except Exception as e:
                    self._log_step("chain_exploration_error", f"Chain of Exploration出错: {str(e)}")
        
        # 3. 如果问题复杂度高，生成最终思考
        if complexity > 0.7:
            self._log_step("final_thinking", "生成最终思考")
            # 告诉思考引擎基于搜索结果更新想法
            self.thinking_engine.add_reasoning_step("基于所有搜索结果，更新我的思考")
            updated_thinking = self.thinking_engine.update_thinking_based_on_verification([])
            self._log_step("final_thinking_complete", "完成最终思考", {"thinking": updated_thinking})
            
            # 获取完整思考过程
            thinking_process = self.thinking_engine.get_full_thinking()
        else:
            thinking_process = None
        
        # 4. 整合所有结果
        all_results = {
            "local": local_results,
            "global": global_results,
            "exploration": exploration_results,
            "chain_exploration": chain_exploration_results
        }
        
        # 5. 合成最终答案
        self._log_step("synthesizing", "合成最终答案")
        final_answer = self.synthesizer.synthesize(query, all_results, retrieval_plan, thinking_process)
        self._log_step("synthesis_completed", "答案合成完成")
        
        # 记录总耗时
        total_time = time.time() - start_time
        self.performance_metrics["total_time"] = total_time
        
        return {
            "answer": final_answer,
            "plan": retrieval_plan,
            "results": all_results,
            "thinking": thinking_process,
            "execution_trace": self.execution_trace,
            "metrics": self.performance_metrics
        }
    
    def _summarize_result(self, result):
        """生成结果摘要"""
        if not result:
            return "无结果"
            
        if isinstance(result, str):
            # 限制长度
            if len(result) > 500:
                return result[:500] + "..."
            return result
        elif isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)[:500] + "..."
        elif isinstance(result, list):
            return str(result)[:500] + "..."
        else:
            return str(result)[:500] + "..."
    
    def _extract_entities_from_results(self, local_results, global_results, exploration_results):
        """从现有结果中提取实体"""
        entities = set()
        
        # 提取实体的简单启发式方法
        entity_patterns = [
            r'实体\s*[:：]\s*([^,，\n]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',  # 英文专有名词
            r'【([^】]+)】',  # 中文方括号内容
            r'"([^"]+)"'  # 引号内容
        ]
        
        # 处理所有结果
        all_text = []
        all_text.extend(local_results)
        all_text.extend(global_results)
        
        for result in exploration_results:
            if isinstance(result, str):
                all_text.append(result)
                
        # 从文本中提取实体
        for text in all_text:
            if not isinstance(text, str):
                continue
                
            for pattern in entity_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # 清理并添加实体
                    entity = match.strip()
                    if len(entity) > 1 and len(entity) < 30:  # 合理长度限制
                        entities.add(entity)
        
        return list(entities)
    
    async def process_query_stream(self, query: str):
        """
        流式处理查询，返回处理过程和结果
        
        参数:
            query: 用户查询
            
        返回:
            AsyncGenerator: 流式返回处理过程和结果
        """
        # 1. 生成检索计划
        yield "**正在分析问题和制定检索计划**...\n\n"
        retrieval_plan = self.retrieval_planner.plan(query)
        
        # 提取和显示计划摘要
        complexity = retrieval_plan.get("complexity_assessment", 0.5)
        requires_global = retrieval_plan.get("requires_global_view", False)
        requires_path = retrieval_plan.get("requires_path_tracking", False)
        has_temporal = retrieval_plan.get("has_temporal_aspects", False)
        knowledge_areas = retrieval_plan.get("knowledge_areas", [])
        
        plan_summary = f"**检索计划制定完成**\n"
        plan_summary += f"- 复杂度评估: {complexity:.2f}\n"
        plan_summary += f"- 需要全局视图: {'是' if requires_global else '否'}\n"
        plan_summary += f"- 需要关系路径追踪: {'是' if requires_path else '否'}\n"
        plan_summary += f"- 包含时间相关内容: {'是' if has_temporal else '否'}\n"
        
        if knowledge_areas:
            plan_summary += f"- 涉及知识领域: {', '.join(knowledge_areas[:3])}\n"
        
        yield plan_summary + "\n"
        
        # 初始化思考引擎
        self.thinking_engine.initialize_with_query(query)
        
        # 如果复杂度高，添加初步思考
        if complexity > 0.7:
            yield "**正在进行初步思考分析**...\n\n"
            initial_thinking = self.thinking_engine.generate_initial_thinking()
            
            # 返回思考摘要
            thinking_lines = initial_thinking.split('\n')
            if len(thinking_lines) > 5:
                thinking_summary = '\n'.join(thinking_lines[:5]) + "...\n"
            else:
                thinking_summary = initial_thinking + "\n"
                
            yield thinking_summary + "\n"
        
        # 2. 根据计划执行搜索任务
        tasks = sorted(retrieval_plan.get("tasks", []), 
                     key=lambda x: x.get("priority", 3), 
                     reverse=True)
        
        all_results = {"local": [], "global": [], "exploration": [], "chain_exploration": []}
        
        for i, task in enumerate(tasks):
            task_type = task.get("type", "")
            task_query = task.get("query", query)
            
            task_msg = f"**执行任务 {i+1}/{len(tasks)}**: {task_type} - {task_query}\n"
            yield task_msg
            
            # 添加任务到思考引擎
            self.thinking_engine.add_reasoning_step(f"执行任务: {task_type} - {task_query}")
            
            try:
                if task_type == "local_search":
                    result = await self._async_local_search(task_query)
                    if result:
                        all_results["local"].append(result)
                        self.thinking_engine.add_reasoning_step(f"本地搜索结果摘要:\n{self._summarize_result(result)}")
                        yield "✓ 本地搜索完成\n\n"
                        
                elif task_type == "global_search":
                    result = await self._async_global_search(task_query)
                    if result:
                        all_results["global"].append(result)
                        self.thinking_engine.add_reasoning_step(f"全局搜索结果摘要:\n{self._summarize_result(result)}")
                        yield "✓ 全局搜索完成\n\n"
                        
                elif task_type == "exploration":
                    yield "**开始深度探索**...\n"
                    result = await self._async_exploration(task_query)
                    if result:
                        all_results["exploration"].append(result)
                        self.thinking_engine.add_reasoning_step(f"深度探索结果摘要:\n{self._summarize_result(result)}")
                        yield "✓ 深度探索完成\n\n"
                        
                elif task_type == "chain_exploration":
                    yield "**开始Chain of Exploration**...\n"
                    # 获取相关实体
                    entities = task.get("entities", [])
                    if not entities:
                        # 如果任务没有指定实体，尝试从其他结果中提取
                        entities = self._extract_entities_from_results(
                            all_results["local"], 
                            all_results["global"], 
                            all_results["exploration"]
                        )
                        
                        if entities:
                            yield f"- 从已有结果中提取实体: {', '.join(entities[:3])}" + ("..." if len(entities) > 3 else "") + "\n"
                    
                    # 至少需要一个起始实体
                    if not entities:
                        yield "⚠️ 未找到起始实体，跳过Chain of Exploration\n\n"
                        continue
                        
                    # 执行Chain of Exploration
                    result = await self._async_chain_exploration(task_query, entities[:3])
                    if result:
                        all_results["chain_exploration"].append(result)
                        
                        # 添加结果到思考引擎
                        path_summary = "探索路径:\n"
                        for step in result.get('exploration_path', [])[:5]:
                            path_summary += f"- 步骤{step.get('step')}: {step.get('node_id')} ({step.get('reasoning', '无理由')})\n"
                        self.thinking_engine.add_reasoning_step(f"Chain of Exploration结果:\n{path_summary}")
                        
                        # 显示探索路径摘要
                        if "exploration_path" in result:
                            yield "- 探索路径:\n"
                            for step in result["exploration_path"][:5]:
                                yield f"  • 步骤{step.get('step')}: {step.get('node_id')}\n"
                            
                        if "content" in result:
                            yield f"- 找到 {len(result['content'])} 条相关内容\n"
                            
                        yield "✓ Chain of Exploration完成\n\n"
            
            except Exception as e:
                yield f"❌ {task_type}任务执行失败: {str(e)}\n\n"
        
        # 如果复杂度高，生成最终思考
        if complexity > 0.7:
            yield "**正在基于所有搜索结果进行最终思考**...\n\n"
            self.thinking_engine.add_reasoning_step("基于所有搜索结果，更新我的思考")
            updated_thinking = self.thinking_engine.update_thinking_based_on_verification([])
            
            # 返回思考摘要
            thinking_lines = updated_thinking.split('\n')
            if len(thinking_lines) > 5:
                thinking_summary = '\n'.join(thinking_lines[:5]) + "...\n"
            else:
                thinking_summary = updated_thinking + "\n"
                
            yield thinking_summary + "\n"
        
        # 3. 合成最终答案
        yield "**正在整合所有检索结果，生成最终答案**...\n\n"
        
        # 获取思考过程
        thinking_process = self.thinking_engine.get_full_thinking() if complexity > 0.7 else None
        
        final_answer = await self._async_synthesize(query, all_results, retrieval_plan, thinking_process)
        
        # 清理答案 - 删除"引用数据"部分保留干净的回答
        clean_answer = final_answer
        ref_index = final_answer.find("#### 引用数据")
        if ref_index > 0:
            clean_answer = final_answer[:ref_index].strip()
        
        yield f"\n\n{clean_answer}"
    
    async def _async_local_search(self, query):
        """异步执行本地搜索"""
        def sync_search():
            return self.local_searcher.search(query)
        return await asyncio.get_event_loop().run_in_executor(None, sync_search)
    
    async def _async_global_search(self, query):
        """异步执行全局搜索"""
        def sync_search():
            return self.global_searcher.search(query)
        return await asyncio.get_event_loop().run_in_executor(None, sync_search)
    
    async def _async_exploration(self, query):
        """异步执行探索"""
        def sync_explore():
            return self.explorer.search(query)
        return await asyncio.get_event_loop().run_in_executor(None, sync_explore)
    
    async def _async_chain_exploration(self, query, entities):
        """异步执行Chain of Exploration"""
        def sync_explore():
            return self.chain_explorer.explore(query, entities, max_steps=3)
        return await asyncio.get_event_loop().run_in_executor(None, sync_explore)
    
    async def _async_synthesize(self, query, results, plan, thinking_process=None):
        """异步合成答案"""
        def sync_synthesize():
            return self.synthesizer.synthesize(query, results, plan, thinking_process)
        return await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)
    
    def _log_step(self, step_type: str, description: str, data: Any = None):
        """记录执行步骤"""
        self.execution_trace.append({
            "type": step_type,
            "description": description,
            "timestamp": time.time(),
            "data": data
        })