from typing import Dict,Any, List, AsyncGenerator
import json
import time
import traceback
import asyncio
import re
import os

from langchain_core.tools import BaseTool

from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
from graphrag_agent.graph.core import connection_manager
from graphrag_agent.config.prompts import (
    system_template_build_graph,
    human_template_build_graph
)
from graphrag_agent.config.settings import (
    entity_types,
    relationship_types
)
from graphrag_agent.config.prompts import RELEVANT_EXTRACTION_PROMPT
from graphrag_agent.search.tool.reasoning.prompts import kb_prompt
from graphrag_agent.graph.extraction.entity_extractor import EntityRelationExtractor
from graphrag_agent.search.tool.deep_research_tool import DeepResearchTool
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool
from graphrag_agent.search.tool.reasoning.community_enhance import CommunityAwareSearchEnhancer
from graphrag_agent.search.tool.reasoning.thinking import ThinkingEngine
from graphrag_agent.search.tool.reasoning.kg_builder import DynamicKnowledgeGraphBuilder
from graphrag_agent.search.tool.reasoning.evidence import EvidenceChainTracker
from graphrag_agent.search.tool.reasoning.chain_of_exploration import ChainOfExplorationSearcher
from graphrag_agent.search.tool.reasoning.validator import complexity_estimate
from graphrag_agent.search.tool.deeper_research import (
    enhance_search_with_coe,
    create_multiple_reasoning_branches,
    detect_and_resolve_contradictions,
    generate_citations,
    merge_reasoning_branches,
)

class DeeperResearchTool:
    """
    增强版深度研究工具
    
    整合社区感知、动态知识图谱和Chain of Exploration等功能，
    提供更全面的深度研究能力，并充分利用所有高级推理功能
    """
    
    def __init__(self, config=None, llm=None, embeddings=None, graph=None):
        """
        初始化增强版深度研究工具
        
        Args:
            config: 配置参数
            llm: 语言模型
            embeddings: 嵌入模型
            graph: 图数据库连接
        """
        # 关键词缓存
        self._keywords_cache = {}

        # 初始化基础组件
        self.llm = llm or get_llm_model()
        self.embeddings = embeddings or get_embeddings_model()
        self.graph = graph or connection_manager.get_connection()

        self.hybrid_tool = HybridSearchTool()
        
        # 初始化增强模块
        # 1. 社区感知搜索增强器
        self.community_search = CommunityAwareSearchEnhancer(
            self.graph, 
            self.embeddings, 
            self.llm
        )
        
        # 2. 动态知识图谱构建器
        self.knowledge_builder = DynamicKnowledgeGraphBuilder(
            self.graph,
            EntityRelationExtractor(self.llm, system_template_build_graph, human_template_build_graph, entity_types, relationship_types),
        )
        
        # 3. Chain of Exploration检索器
        self.chain_explorer = ChainOfExplorationSearcher(
            self.graph,
            self.llm,
            self.embeddings
        )
        
        # 4. 证据链跟踪器
        self.evidence_tracker = EvidenceChainTracker()
        
        # 5. 继承原有的深度研究工具功能
        self.deep_research = DeepResearchTool()

        # 6. 查询生成器
        self.query_generator = self.deep_research.query_generator
        
        # 缓存设置
        self.enable_cache = True
        self.cache_dir = "./cache/deeper_research"
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
                
        # 添加执行日志容器
        self.execution_logs = []
        
        # 添加性能指标跟踪
        self.performance_metrics = {"total_time": 0}
        
        # 记录当前查询的上下文信息
        self.current_query_context = {}
        
        # 记录已探索的查询分支
        self.explored_branches = {}
        
        # 共享思考引擎实例，避免重复初始化
        if hasattr(self.deep_research, 'thinking_engine'):
            self.thinking_engine = self.deep_research.thinking_engine
        else:
            self.thinking_engine = ThinkingEngine(self.llm)
            self.deep_research.thinking_engine = self.thinking_engine
        
        # 添加各种缓存字典
        self._search_cache = {}
        self._thinking_cache = {}
        self._contradiction_cache = {}
        self._hypotheses_cache = {}
        self._counter_cache = {}
        self._coe_cache = {}
        self._specific_coe_cache = {}
        self._contradiction_detailed_cache = {}
        self._stream_search_cache = {}
        self._stream_thinking_cache = {}
        self._subquery_cache = {}
    
    def _log(self, message):
        """记录执行日志"""
        self.execution_logs.append(message)
        # print(message)  # 可选：同时打印到控制台
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词"""
        # 检查缓存
        if query in self._keywords_cache:
            return self._keywords_cache[query]

        keywords = self.hybrid_tool.extract_keywords(query)
        
        # 缓存结果
        self._keywords_cache[query] = keywords
        return keywords
    
    
    def thinking(self, query: str) -> Dict[str, Any]:
        """
        执行增强版深度研究推理过程
        
        Args:
            query: 用户问题
                    
        Returns:
            Dict: 包含思考过程和最终答案的字典
        """
        # 清空执行日志
        self.execution_logs = []
        self._log(f"\n[深度研究] 开始处理查询: {query}")

        self._keywords_cache = {}
        
        # 检查思考缓存
        if hasattr(self, '_thinking_cache') and query in self._thinking_cache:
            return self._thinking_cache[query]
        
        # 提取关键词
        keywords = self.extract_keywords(query)
        self._log(f"\n[深度研究] 提取关键词: {keywords}")
        
        # 开始新的查询跟踪
        query_id = self.evidence_tracker.start_new_query(query, keywords)
        self.current_query_context = {"query_id": query_id}
        
        # 重置分支信息
        self.explored_branches = {}
        
        # 步骤1: 使用QueryGenerator生成初始子查询
        if not hasattr(self, '_subquery_cache'):
            self._subquery_cache = {}
            
        if query in self._subquery_cache:
            initial_sub_queries = self._subquery_cache[query]
        else:
            initial_sub_queries = self.query_generator.generate_sub_queries(query)
            self._subquery_cache[query] = initial_sub_queries
            
        self._log(f"\n[深度研究] 生成了{len(initial_sub_queries)}个初始子查询: {initial_sub_queries}")
        
        # 记录为推理步骤
        subquery_step_id = self.evidence_tracker.add_reasoning_step(
            query_id,
            "initial_query_decomposition",
            f"将原始问题分解为{len(initial_sub_queries)}个子问题: {initial_sub_queries}"
        )
        
        # 步骤2: 社区感知和Chain of Exploration增强 - 添加缓存
        if not hasattr(self, '_search_cache'):
            self._search_cache = {}
            
        search_cache_key = f"enhanced_search:{query}"
        if search_cache_key in self._search_cache:
            enhanced_context = self._search_cache[search_cache_key]
        else:
            self._log(f"\n[深度研究] 开始社区感知与Chain of Exploration分析")
            enhanced_context = enhance_search_with_coe(self, query, keywords)
            self._search_cache[search_cache_key] = enhanced_context
            
        community_context = enhanced_context
        
        # 步骤3: 使用社区信息增强搜索策略
        search_strategy = community_context.get("search_strategy", {})
        follow_up_queries = search_strategy.get("follow_up_queries", [])
        
        # 合并社区生成的后续查询和初始子查询
        all_queries = list(set(initial_sub_queries + follow_up_queries))
        
        if follow_up_queries:
            self._log(f"\n[深度研究] 社区分析生成的后续查询: {follow_up_queries}")
        
        # 添加增强搜索策略的推理步骤
        strategy_step_id = self.evidence_tracker.add_reasoning_step(
            query_id, 
            "knowledge_community_analysis", 
            f"基于社区分析，识别了关键实体和相关查询策略。将探索以下后续查询: {all_queries}"
        )
        
        # 记录社区信息作为证据
        community_summaries = []
        if "community_info" in community_context and "summaries" in community_context["community_info"]:
            for summary_obj in community_context["community_info"]["summaries"]:
                if isinstance(summary_obj, dict) and "summary" in summary_obj:
                    community_summaries.append(summary_obj["summary"])
                else:
                    community_summaries.append(str(summary_obj))
                    
        if community_summaries:
            self._log(f"\n[深度研究] 找到 {len(community_summaries)} 个相关社区")
            for i, summary in enumerate(community_summaries):
                self._log(f"\n[深度研究] 社区 {i+1} 摘要: {summary[:100]}...")
                self.evidence_tracker.add_evidence(
                    strategy_step_id,
                    f"community_summary_{i}",
                    summary,
                    "community_knowledge"
                )
        
        # 步骤4: 构建初始知识图谱
        initial_entities = search_strategy.get("focus_entities", [])
        discovered_entities = search_strategy.get("discovered_entities", [])
        
        # 合并实体
        graph_entities = list(set(initial_entities + discovered_entities))
        
        if graph_entities:
            # 检查图谱缓存
            graph_cache_key = f"graph:{query}:{','.join(graph_entities[:5])}"
            if not hasattr(self, '_graph_cache'):
                self._graph_cache = {}
                
            if graph_cache_key not in self._graph_cache:
                self._log(f"\n[深度研究] 构建知识图谱，关注实体: {graph_entities}")
                self.knowledge_builder.build_query_graph(query, graph_entities, depth=2)  # 增加探索深度到2
                self._graph_cache[graph_cache_key] = True
            
            # 获取核心实体
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
            central_entity_ids = [e["id"] for e in central_entities]
            
            # 添加知识图谱分析步骤
            kg_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                "knowledge_graph_analysis",
                f"构建了初始知识图谱，识别出核心实体: {central_entity_ids}"
            )
            
            # 记录知识图谱信息作为证据
            kg_info = {
                "entity_count": self.knowledge_builder.knowledge_graph.number_of_nodes(),
                "relation_count": self.knowledge_builder.knowledge_graph.number_of_edges(),
                "central_entities": central_entity_ids
            }
            self._log(f"\n[深度研究] 知识图谱统计: {kg_info['entity_count']} 个实体, {kg_info['relation_count']} 个关系")
            
            self.evidence_tracker.add_evidence(
                kg_step_id,
                "knowledge_graph",
                json.dumps(kg_info),
                "graph_structure"
            )
        
        # 步骤5: 整合探索结果到深度研究思考过程
        exploration_results = community_context.get("exploration_results", {})
        exploration_path = exploration_results.get("exploration_path", [])
        
        # 初始化结果容器
        think = ""
        self.deep_research.all_retrieved_info = []
        
        # 收集从探索中获取的内容
        exploration_content = []
        for content_item in exploration_results.get("content", []):
            if "text" in content_item:
                exploration_content.append(content_item["text"])
        
        # 初始化思考引擎
        self._log(f"\n[深度研究] 初始化思考引擎")
        if not hasattr(self.thinking_engine, 'initialized_for') or self.thinking_engine.initialized_for != query:
            self.thinking_engine.initialize_with_query(query)
            self.thinking_engine.initialized_for = query
        
        # 添加已有的社区信息和知识图谱信息作为上下文
        context_message = ""
        
        # 添加初始子查询分解
        if initial_sub_queries:
            context_message += "为了全面解答这个问题，我将从以下几个方面进行研究：\n"
            for i, sub_q in enumerate(initial_sub_queries, 1):
                context_message += f"{i}. {sub_q}\n"
            context_message += "\n"
        
        if community_summaries:
            context_message += "找到了相关的知识社区信息:\n"
            context_message += "\n".join(community_summaries[:2])  # 只使用前两个
            
        # 添加探索路径信息
        if exploration_path:
            context_message += "\n\n通过Chain of Exploration探索路径发现:\n"
            path_description = []
            for i, step in enumerate(exploration_path[:5]):  # 限制路径长度
                if i > 0:  # 跳过起始实体
                    path_description.append(f"- 步骤{step['step']}: 实体 {step['node_id']} ({step['reasoning']})")
            context_message += "\n".join(path_description)
            
        # 如果有探索内容，添加
        if exploration_content:
            context_message += "\n\n探索获取的相关内容:\n"
            for i, content in enumerate(exploration_content[:3]):  # 限制内容数量
                context_message += f"内容{i+1}: {content[:200]}...\n\n"
            
        # 如果有跟进查询，添加到思考过程
        if follow_up_queries:
            context_message += "\n\n可能的深入探索方向:\n"
            context_message += "\n".join([f"- {q}" for q in follow_up_queries])
        
        # 将上下文添加到思考引擎
        if context_message:
            self.thinking_engine.add_reasoning_step(context_message)
            think += context_message + "\n\n"
        
        # 步骤6: 生成多个假设并创建多个推理分支  
        if complexity_estimate(query) > 0.7:  # 对复杂查询应用多分支推理
            self._log("\n[深度研究] 生成多个假设进行分支推理")
            
            # 检查假设缓存
            if not hasattr(self, '_hypothesis_cache'):
                self._hypothesis_cache = {}
                
            if query in self._hypothesis_cache:
                hypotheses = self._hypothesis_cache[query]
            else:
                # 生成假设并缓存
                hypotheses = self.query_generator.generate_multiple_hypotheses(query, self.llm)
                self._hypothesis_cache[query] = hypotheses
            
            if hypotheses:
                # 创建多个推理分支
                branch_results = create_multiple_reasoning_branches(self, query_id, hypotheses)
                
                # 添加分支概述
                branch_overview = "\n## 推理分支假设\n"
                for branch_name, info in branch_results.items():
                    branch_overview += f"- {branch_name}: {info['hypothesis']}\n"
                
                self.thinking_engine.add_reasoning_step(branch_overview)
                think += branch_overview + "\n\n"
        
        # 迭代思考过程
        for iteration in range(self.deep_research.max_iterations):
            self._log(f"\n[深度研究] 开始第{iteration + 1}轮迭代\n")
            
            # 跟踪迭代步骤
            iteration_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                f"iteration_{iteration}",
                f"\n开始第 {iteration + 1} 轮迭代思考\n"
            )
            
            # 检查是否达到最大迭代次数
            if iteration >= self.deep_research.max_iterations - 1:
                summary_think = f"\n搜索次数已达上限。不允许继续搜索。\n"
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(summary_think)
                think += self.thinking_engine.remove_result_tags(summary_think)
                self._log("\n[深度研究] 达到最大迭代次数，结束搜索")
                break

            # 更新消息历史，请求继续推理
            self.thinking_engine.update_continue_message()
            
            # 确定本轮迭代要处理的查询
            queries_to_process = []
            
            if iteration == 0 and initial_sub_queries:
                # 第一轮迭代使用预先生成的子查询
                queries_to_process = initial_sub_queries[:2]  # 限制首轮使用的子查询数量
                self._log(f"\n[深度研究] 使用初始子查询进行首轮搜索: {queries_to_process}")
            else:
                # 非首轮，使用思考引擎生成下一步查询
                result = self.thinking_engine.generate_next_query()
                
                # 处理生成结果
                if result["status"] == "empty":
                    self._log("\n[深度研究] 生成的思考内容为空")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "generation_result",
                        "生成的思考内容为空",
                        "reasoning_status"
                    )
                    
                    # 尝试使用QueryGenerator的多假设生成功能寻找新方向
                    if iteration > 0:
                        if not hasattr(self, '_iteration_hypotheses_cache'):
                            self._iteration_hypotheses_cache = {}
                        
                        iter_cache_key = f"iter_hyp:{query}:{iteration}"
                        if iter_cache_key in self._iteration_hypotheses_cache:
                            hypotheses = self._iteration_hypotheses_cache[iter_cache_key]
                        else:
                            hypotheses = self.query_generator.generate_multiple_hypotheses(query, self.llm)
                            self._iteration_hypotheses_cache[iter_cache_key] = hypotheses
                            
                        if hypotheses:
                            self._log(f"\n[深度研究] 生成了{len(hypotheses)}个新假设，尝试从新角度探索")
                            queries_to_process = hypotheses
                            hypothesis_think = "尝试从新的角度思考这个问题:\n" + "\n".join([f"- {h}" for h in hypotheses])
                            self.thinking_engine.add_reasoning_step(hypothesis_think)
                            think += hypothesis_think
                        else:
                            continue
                    else:
                        continue
                elif result["status"] == "error":
                    self._log(f"\n[深度研究] 生成查询出错: {result.get('error', '未知错误')}")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "generation_error",
                        f"生成查询出错: {result.get('error', '未知错误')}",
                        "reasoning_status"
                    )
                    break
                elif result["status"] == "answer_ready":
                    self._log("\n[深度研究] AI认为已有足够信息生成答案")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "reasoning_complete",
                        "AI认为已有足够信息生成答案",
                        "reasoning_status"
                    )
                    break
                    
                # 获取生成的思考内容
                query_think = result["content"]
                think += self.thinking_engine.remove_query_tags(query_think)
                
                # 获取搜索查询
                queries_to_process = result["queries"]
            
            # 如果当前迭代没有查询，且我们已经检索到一些信息，尝试使用QueryGenerator生成跟进查询
            if not queries_to_process and self.deep_research.all_retrieved_info:
                # 检查跟进查询缓存
                followup_cache_key = f"followup:{query}:{len(self.deep_research.all_retrieved_info)}"
                if not hasattr(self, '_followup_cache'):
                    self._followup_cache = {}
                    
                if followup_cache_key in self._followup_cache:
                    followup_queries = self._followup_cache[followup_cache_key]
                else:
                    followup_queries = self.query_generator.generate_followup_queries(
                        query, self.deep_research.all_retrieved_info
                    )
                    self._followup_cache[followup_cache_key] = followup_queries
                
                if followup_queries:
                    self._log(f"\n[深度研究] 生成了{len(followup_queries)}个跟进查询")
                    queries_to_process = followup_queries
                    followup_think = "\n考虑到已发现的信息，我需要进一步探索以下问题:\n"
                    for i, fq in enumerate(followup_queries, 1):
                        followup_think += f"{i}. {fq}\n"
                    self.thinking_engine.add_reasoning_step(followup_think)
                    think += followup_think
                    
                    # 记录为推理步骤
                    followup_step_id = self.evidence_tracker.add_reasoning_step(
                        query_id,
                        "followup_queries_generation",
                        f"基于已收集的信息生成跟进查询: {followup_queries}"
                    )
            
            # 如果没有生成搜索查询但不是第一轮，考虑结束
            if not queries_to_process:
                if not self.deep_research.all_retrieved_info and iteration == 0:
                    # 如果第一轮没检索到任何信息，强制使用原始查询
                    queries_to_process = [query]
                    self._log("\n[深度研究] 没有检索到信息，使用原始查询")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "query_fallback",
                        "没有检索到信息，使用原始查询",
                        "reasoning_status"
                    )
                else:
                    # 已有信息，结束迭代
                    self._log("\n[深度研究] 没有生成新查询且已有信息，结束迭代")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "iteration_complete",
                        "没有生成新查询且已有信息，结束迭代",
                        "reasoning_status"
                    )
                    
                    # 在结束前进行矛盾检测  
                    if not hasattr(self, '_contradiction_cache'):
                        self._contradiction_cache = {}
                        
                    if query_id in self._contradiction_cache:
                        contradiction_result = self._contradiction_cache[query_id]
                    else:
                        contradiction_result = detect_and_resolve_contradictions(self, query_id)
                        self._contradiction_cache[query_id] = contradiction_result
                    
                    if contradiction_result["contradictions"]:
                        think += "\n## 矛盾检测\n"
                        think += f"发现 {len(contradiction_result['contradictions'])} 个信息矛盾。\n"
                        for i, contradiction in enumerate(contradiction_result["contradictions"]):
                            if contradiction["type"] == "numerical":
                                think += f"{i+1}. 数值矛盾: {contradiction.get('context', '')}\n"
                            else:
                                think += f"{i+1}. 语义矛盾: {contradiction.get('analysis', '')}\n"
                    
                    break
            
            # 处理每个搜索查询
            for search_query in queries_to_process:
                self._log(f"\n[深度研究] 执行查询: {search_query}")
                
                # 记录查询步骤
                search_step_id = self.evidence_tracker.add_reasoning_step(
                    query_id,
                    search_query,
                    f"执行查询: {search_query}"
                )
                
                # 检查是否已执行过相同查询
                if self.thinking_engine.has_executed_query(search_query):
                    self._log(f"\n[深度研究] 已搜索过该查询: {search_query}")
                    summary_think = f"\n已搜索过该查询。请参考前面的结果。\n"
                    self.thinking_engine.add_reasoning_step(summary_think)
                    self.thinking_engine.add_human_message(summary_think)
                    think += self.thinking_engine.remove_result_tags(summary_think)
                    
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        "duplicate_query",
                        "已搜索过该查询，跳过重复执行",
                        "search_status"
                    )
                    continue
                
                # 记录已执行查询
                self.thinking_engine.add_executed_query(search_query)
                
                # 将搜索查询添加到消息历史
                self.thinking_engine.add_ai_message(f"{search_query}")
                think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                
                # 执行实际搜索 - 添加缓存避免重复搜索
                search_results_cache_key = f"search_results:{search_query}"
                if not hasattr(self, '_search_results_cache'):
                    self._search_results_cache = {}
                    
                if search_results_cache_key in self._search_results_cache:
                    kbinfos = self._search_results_cache[search_results_cache_key]
                    self._log(f"\n[深度研究] 使用缓存的搜索结果: {search_query}")
                else:
                    self._log(f"\n[KB检索] 开始增强搜索: {search_query}")
                    
                    # 提取搜索查询的关键词
                    search_keywords = self.extract_keywords(search_query)
                    
                    # 使用Chain of Exploration增强搜索
                    enhanced_search_context = enhance_search_with_coe(self, search_query, search_keywords)
                    exploration_content = []
                    
                    # 从Chain of Exploration结果中提取内容
                    if "exploration_results" in enhanced_search_context:
                        for content_item in enhanced_search_context["exploration_results"].get("content", []):
                            if "text" in content_item:
                                exploration_content.append({
                                    "chunk_id": content_item.get("id", "exploration_content"),
                                    "text": content_item["text"],
                                    "content_with_weight": content_item["text"],
                                    "weight": 1.0
                                })
                    
                    # 原始搜索，合并结果
                    kbinfos = self.deep_research.dual_searcher.search(search_query)
                    
                    # 将探索内容加入结果
                    if exploration_content:
                        if "chunks" not in kbinfos:
                            kbinfos["chunks"] = []
                        kbinfos["chunks"].extend(exploration_content)
                    
                    # 缓存搜索结果
                    self._search_results_cache[search_results_cache_key] = kbinfos
                
                # 记录搜索结果基本信息
                chunks_count = len(kbinfos.get("chunks", []))
                entities_count = len(kbinfos.get("entities", []))
                rels_count = len(kbinfos.get("relationships", []))
                self._log(f"\n[KB检索] 结果: {chunks_count}个chunks, {entities_count}个实体, {rels_count}个关系")
                
                # 为查询结果更新知识图谱
                if "chunks" in kbinfos:
                    for chunk in kbinfos.get("chunks", []):
                        chunk_id = chunk.get("chunk_id", "")
                        chunk_text = chunk.get("text", "")
                        
                        if chunk_id and chunk_text:
                            # 记录chunk作为证据
                            self.evidence_tracker.add_evidence(
                                search_step_id,
                                chunk_id,
                                chunk_text,
                                "document_chunk"
                            )
                            
                            # 尝试从chunk中提取实体和关系
                            # 减少实体提取，只对部分内容执行
                            if (hasattr(self.knowledge_builder, 'extractor') and 
                                self.knowledge_builder.extractor and 
                                chunks_count < 5):  # 只有当chunks数量较少时才提取
                                try:
                                    extract_cache_key = f"extract:{chunk_id}"
                                    if not hasattr(self, '_extract_cache'):
                                        self._extract_cache = {}
                                        
                                    if extract_cache_key not in self._extract_cache:
                                        self.knowledge_builder.extract_subgraph_from_chunk(
                                            chunk_text, 
                                            chunk_id
                                        )
                                        self._extract_cache[extract_cache_key] = True
                                except Exception as e:
                                    self._log(f"\n[深度研究] 提取实体关系失败: {e}")
                
                # 检查搜索结果是否为空
                has_results = (
                    kbinfos.get("chunks", []) or 
                    kbinfos.get("entities", []) or 
                    kbinfos.get("relationships", [])
                )
                
                if not has_results:
                    self._log(f"\n[KB检索] 没有找到与'{search_query}'相关的信息")
                    no_result_msg = f"\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n"
                    self.thinking_engine.add_reasoning_step(no_result_msg)
                    self.thinking_engine.add_human_message(no_result_msg)
                    think += self.thinking_engine.remove_result_tags(no_result_msg)
                    
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        "no_results",
                        f"没有找到与查询相关的信息: {search_query}",
                        "search_result"
                    )
                    continue
                
                # 正常处理有结果的情况
                truncated_prev_reasoning = self.thinking_engine.prepare_truncated_reasoning()
                
                # 构建提取相关信息的提示
                kb_prompt_result = "\n".join(kb_prompt(kbinfos, 4096))
                
                # 检查提取缓存
                extract_cache_key = f"extract_info:{search_query}:{truncated_prev_reasoning[:100]}"
                if not hasattr(self, '_extraction_cache'):
                    self._extraction_cache = {}
                    
                if extract_cache_key in self._extraction_cache:
                    summary_think = self._extraction_cache[extract_cache_key]
                    self._log(f"\n[深度研究] 使用缓存的提取结果")
                else:
                    extract_prompt = RELEVANT_EXTRACTION_PROMPT.format(
                        prev_reasoning=truncated_prev_reasoning,
                        search_query=search_query,
                        document=kb_prompt_result
                    )
                    
                    # 使用LLM提取有用信息
                    self._log(f"\n[深度研究] 分析检索结果，提取有用信息")
                    extraction_msg = self.llm.invoke([
                        {"role": "system", "content": extract_prompt},
                        {"role": "user", "content": f'基于当前的搜索查询"{search_query}"和前面的推理步骤，分析每个知识来源并找出有用信息。'}
                    ])
                    
                    summary_think = extraction_msg.content if hasattr(extraction_msg, 'content') else str(extraction_msg)
                    # 缓存提取结果
                    self._extraction_cache[extract_cache_key] = summary_think
                
                # 保存重要信息
                has_useful_info = (
                    "**Final Information**" in summary_think and 
                    "No helpful information found" not in summary_think
                )
                
                if has_useful_info:
                    useful_info = summary_think.split("**Final Information**")[1].strip()
                    self.deep_research.all_retrieved_info.append(useful_info)
                    
                    # 记录有用信息作为证据
                    self._log(f"\n[深度研究] 发现有用信息: {useful_info}")
                    self.evidence_tracker.add_evidence_with_confidence(
                        search_step_id,
                        f"useful_info_{search_query}",
                        useful_info,
                        "extracted_knowledge",
                        confidence=0.85,  # 高可信度
                        metadata={"query": search_query}
                    )
                else:
                    self._log(f"\n[深度研究] 未从检索结果中提取到有用信息")
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        f"no_useful_info_{search_query}",
                        "未从检索结果中提取到有用信息",
                        "extraction_status"
                    )
                
                # 更新推理历史
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(summary_think)
                think += self.thinking_engine.remove_result_tags(summary_think)
            
            # 在每轮迭代结束后，评估是否需要继续搜索
            if iteration > 0 and self.deep_research.all_retrieved_info:
                # 检查是否需要继续生成查询的缓存
                gap_check_key = f"gap_check:{query}:{len(self.deep_research.all_retrieved_info)}"
                if not hasattr(self, '_gap_check_cache'):
                    self._gap_check_cache = {}
                    
                if gap_check_key in self._gap_check_cache:
                    gap_needed = self._gap_check_cache[gap_check_key]
                else:
                    # 类似于DeepSearch中的_generate_gap_queries方法，检查是否需要继续生成查询
                    gap_needed = len(self.query_generator.generate_followup_queries(query, self.deep_research.all_retrieved_info)) > 0
                    self._gap_check_cache[gap_check_key] = gap_needed
                    
                if not gap_needed:
                    self._log("\n[深度研究] 已收集足够信息，无需进一步查询")
                    reflection_think = "\n已收集到足够的信息，可以开始整合分析了。"
                    self.thinking_engine.add_reasoning_step(reflection_think)
                    think += reflection_think
                    break
        
        # 步骤7: 合并推理分支  
        if self.explored_branches:
            self._log("\n[深度研究] 合并多个推理分支")
            merged_results = merge_reasoning_branches(self, query_id)
            
            if merged_results:
                think += f"\n{merged_results}\n"
                self.thinking_engine.add_reasoning_step(merged_results)
        
        # 步骤8: 矛盾检测与处理  
        if not hasattr(self, '_contradiction_cache'):
            self._contradiction_cache = {}
            
        if query_id in self._contradiction_cache:
            contradiction_result = self._contradiction_cache[query_id]
        else:
            contradiction_result = detect_and_resolve_contradictions(self, query_id)
            self._contradiction_cache[query_id] = contradiction_result
            
        if contradiction_result["contradictions"]:
            contradiction_analysis = "\n## 信息矛盾分析\n"
            contradiction_analysis += f"发现 {len(contradiction_result['contradictions'])} 个信息矛盾：\n"
            
            for i, contradiction in enumerate(contradiction_result["contradictions"]):
                if contradiction["type"] == "numerical":
                    contradiction_analysis += f"{i+1}. 数值矛盾: 在\"{contradiction.get('context', '')}\"中，"
                    contradiction_analysis += f"发现值 {contradiction.get('value1')} 和 {contradiction.get('value2')}\n"
                else:
                    contradiction_analysis += f"{i+1}. 语义矛盾: {contradiction.get('analysis', '')}\n"
            
            # 添加矛盾分析到思考过程
            think += contradiction_analysis
            self.thinking_engine.add_reasoning_step(contradiction_analysis)
        
        # 生成最终答案
        # 确保至少执行了一次搜索
        if not self.deep_research.thinking_engine.executed_search_queries:
            self._log("\n[深度研究] 未执行任何有效搜索，无法生成答案")
            result = {
                "thinking_process": think,
                "answer": f"抱歉，我无法回答关于'{query}'的问题，因为没有找到相关信息。",
                "reference": {"chunks": [], "doc_aggs": []},
                "retrieved_info": [],
                "reasoning_chain": self.evidence_tracker.get_reasoning_chain(query_id),
                "execution_logs": self.execution_logs,
            }
            
            # 缓存思考结果
            if not hasattr(self, '_thinking_cache'):
                self._thinking_cache = {}
            self._thinking_cache[query] = result
            
            return result
        
        # 使用检索到的信息生成答案
        retrieved_content = "\n\n".join(self.deep_research.all_retrieved_info)
        self._log("\n[深度研究] 生成最终答案")
        
        # 检查最终答案缓存
        final_answer_key = f"final_answer:{query}:{len(self.deep_research.all_retrieved_info)}"
        if not hasattr(self, '_final_answer_cache'):
            self._final_answer_cache = {}
            
        if final_answer_key in self._final_answer_cache:
            final_answer = self._final_answer_cache[final_answer_key]
        else:
            # 增强答案生成过程，整合知识图谱和社区分析结果
            enhanced_prompt = f"""
            用户问题：{query}
            
            我已经通过多种检索方法收集了以下信息：
            
            {retrieved_content}
            
            此外，我通过知识图谱分析发现了以下关键实体和关系：
            """
            
            # 添加知识图谱分析结果
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
            if central_entities:
                enhanced_prompt += "\n核心实体及其重要性：\n"
                for entity in central_entities:
                    entity_id = entity.get("id", "")
                    importance = entity.get("centrality", entity.get("degree", 0))
                    entity_type = entity.get("type", "unknown")
                    properties = entity.get("properties", {})
                    description = properties.get("description", "无描述")
                    
                    enhanced_prompt += f"- {entity_id} (重要性: {importance:.3f}, 类型: {entity_type}): {description}\n"
            
            # 添加社区分析结果
            if community_summaries:
                enhanced_prompt += "\n来自相关知识社区的见解：\n"
                for i, summary in enumerate(community_summaries[:2]):
                    enhanced_prompt += f"- 社区{i+1}: {summary[:200]}...\n"
            
            # 添加探索路径分析
            if exploration_path:
                enhanced_prompt += "\n知识探索路径分析：\n"
                path_summary = []
                current_entity = None
                for step in exploration_path:
                    if step["step"] > 0:  # 跳过起始实体
                        if current_entity != step["node_id"]:
                            current_entity = step["node_id"]
                            path_summary.append(f"实体 {current_entity}: {step['reasoning']}")
                
                enhanced_prompt += "\n".join(path_summary[:3]) + "\n"
            
            # 添加初始子查询分析结果
            if initial_sub_queries:
                enhanced_prompt += "\n问题分解与深入分析：\n"
                for i, subq in enumerate(initial_sub_queries, 1):
                    related_info = []
                    for info in self.deep_research.all_retrieved_info:
                        if any(term.lower() in info.lower() for term in subq.lower().split()):
                            related_info.append(info[:100] + "..." if len(info) > 100 else info)
                    
                    if related_info:
                        enhanced_prompt += f"- 子问题{i}: {subq} - 找到{len(related_info)}条相关信息\n"
                    else:
                        enhanced_prompt += f"- 子问题{i}: {subq} - 未找到直接相关信息\n"
            
            # 添加矛盾分析结果  
            if contradiction_result["contradictions"]:
                enhanced_prompt += "\n信息矛盾分析：\n"
                for i, contradiction in enumerate(contradiction_result["contradictions"][:3]):
                    if contradiction["type"] == "numerical":
                        enhanced_prompt += f"- 矛盾{i+1}: 在数值上存在不一致，一处显示 {contradiction.get('value1')}，另一处显示 {contradiction.get('value2')}\n"
                    else:
                        enhanced_prompt += f"- 矛盾{i+1}: {contradiction.get('analysis', '')}\n"
            
            # 请求生成最终答案
            enhanced_prompt += """
            请基于以上所有信息，生成一个全面深入的回答。回答应该:
            1. 直接回答用户问题
            2. 结构清晰，逻辑性强
            3. 整合所有相关信息，包括知识图谱和社区分析的见解
            4. 如有必要，指出信息中的不确定性或矛盾
            5. 对问题的不同方面进行全面分析
            """
            
            try:
                # 使用增强提示生成最终答案
                final_response = self.llm.invoke(enhanced_prompt)
                enhanced_answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
                
                # 将增强答案与思考过程结合
                final_answer = f"<think>{think}</think>\n\n{enhanced_answer}"
                
                # 生成引用标记  
                cited_answer = generate_citations(self, enhanced_answer, query_id)
                
                # 如果引用生成成功，使用带引用的答案
                if len(cited_answer) > len(enhanced_answer):
                    final_answer = f"<think>{think}</think>\n\n{cited_answer}"
                
                # 缓存最终答案
                self._final_answer_cache[final_answer_key] = final_answer
                
            except Exception as e:
                # 如果增强生成失败，回退到原始方法
                self._log(f"\n[深度研究] 增强答案生成失败: {e}，回退到标准方法")
                final_answer = self.deep_research._generate_final_answer(query, retrieved_content, think)
                # 依然缓存结果，防止再次失败
                self._final_answer_cache[final_answer_key] = final_answer
        
        # 获取知识图谱中的核心实体
        central_entities = []
        if hasattr(self.knowledge_builder, 'knowledge_graph') and self.knowledge_builder.knowledge_graph.nodes:
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
        
        # 记录最终回答步骤
        final_step_id = self.evidence_tracker.add_reasoning_step(
            query_id,
            "final_answer",
            "生成最终答案"
        )
        
        self.evidence_tracker.add_evidence(
            final_step_id,
            "final_answer_evidence",
            retrieved_content,
            "synthesized_knowledge"
        )
        
        # 获取推理摘要  
        reasoning_summary = self.evidence_tracker.summarize_reasoning(query_id)
        
        # 生成证据来源统计  
        evidence_stats = self.evidence_tracker.get_evidence_source_stats(query_id)
        
        # 返回结果
        result = {
            "thinking_process": think,
            "answer": final_answer,
            "reference": {"chunks": [], "doc_aggs": []},
            "retrieved_info": self.deep_research.all_retrieved_info,
            "reasoning_chain": self.evidence_tracker.get_reasoning_chain(query_id),
            "reasoning_summary": reasoning_summary,    
            "evidence_stats": evidence_stats,    
            "knowledge_graph": {
                "entity_count": getattr(self.knowledge_builder.knowledge_graph, "number_of_nodes", lambda: 0)(),
                "relation_count": getattr(self.knowledge_builder.knowledge_graph, "number_of_edges", lambda: 0)(),
                "central_entities": central_entities
            },
            "initial_sub_queries": initial_sub_queries,
            "contradictions": contradiction_result["contradictions"],    
            "execution_logs": self.execution_logs,
        }
        
        # 缓存思考结果
        if not hasattr(self, '_thinking_cache'):
            self._thinking_cache = {}
        self._thinking_cache[query] = result
        
        return result
    
    def search(self, query_input: Any) -> str:
        """
        执行增强版深度研究搜索
        
        Args:
            query_input: 搜索查询或包含查询的字典
                    
        Returns:
            str: 搜索结果
        """
        overall_start = time.time()
        
        # 记录开始搜索
        self._log(f"\n[深度搜索] 开始处理查询...")
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"\n[深度搜索] 解析后的查询: {query}")
        
        # 检查多级缓存
        # 1. 首先检查deeperresearch缓存
        cache_key = f"deeper:{query}"
        if hasattr(self, '_search_result_cache') and cache_key in self._search_result_cache:
            self._log(f"\n[深度搜索] 内部缓存命中，返回缓存结果")
            return self._search_result_cache[cache_key]
            
        # 2. 然后检查deep_research缓存
        deep_cache_key = f"deep:{query}"
        cached_result = self.deep_research.cache_manager.get(deep_cache_key)
        if cached_result:
            self._log(f"\n[深度搜索] 缓存命中，返回缓存结果")
            
            # 同时更新内部缓存
            if not hasattr(self, '_search_result_cache'):
                self._search_result_cache = {}
            self._search_result_cache[cache_key] = cached_result
            
            return cached_result
        
        try:
            # 执行思考过程
            self._log(f"\n[深度搜索] 开始执行思考过程")
            result = self.thinking(query)
            answer = result["answer"]
            chunk_info = result.get("reference", {})
            
            # 格式化参考资料
            references = []
            if "doc_aggs" in chunk_info:
                for doc in chunk_info["doc_aggs"]:
                    doc_id = doc.get("doc_id", "")
                    if doc_id and doc_id not in references:
                        references.append(doc_id)
            
            # 添加引用信息
            if references and "{'data': {'Chunks':" not in answer:
                ref_str = ", ".join([f"'{ref}'" for ref in references[:5]])
                answer += f"\n\n{{'data': {{'Chunks':[{ref_str}] }} }}"
            
            # 验证答案质量
            validation_results = self.deep_research.validator.validate(query, answer)
            if validation_results["passed"]:
                self._log(f"\n[深度搜索] 答案验证通过，缓存结果")
                self.deep_research.cache_manager.set(deep_cache_key, answer)
                
                # 更新内部缓存
                if not hasattr(self, '_search_result_cache'):
                    self._search_result_cache = {}
                self._search_result_cache[cache_key] = answer
            else:
                self._log(f"\n[深度搜索] 答案验证失败，不缓存")
            
            # 记录总时间
            total_time = time.time() - overall_start
            self._log(f"\n[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
            
            return answer
                    
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"抱歉，处理您的问题时遇到了错误: {str(e)}"
    
    def get_tool(self):
        """获取搜索工具"""
        class DeeperResearchRetrievalTool(BaseTool):
            name : str = "deeper_research"
            description : str = "增强版深度研究工具：通过社区感知和知识图谱分析，结合多轮推理和搜索解决复杂问题。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeeperResearchRetrievalTool()
    
    def get_thinking_tool(self):
        """获取思考过程可见的工具版本"""
        class DeepThinkingTool(BaseTool):
            name : str = "deep_thinking"
            description : str = "深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> Dict:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 执行思考过程
                return self.thinking(tk_query)
            
            def _arun(self_tool, query: Any) -> Dict:
                raise NotImplementedError("异步执行未实现")
        
        return DeepThinkingTool()
    
    def get_stream_tool(self):
        """获取流式搜索工具"""        
        class DeeperResearchStreamTool(BaseTool):
            name : str = "deeper_research_stream"
            description : str = "增强版流式深度研究工具：通过社区感知和知识图谱分析，结合多轮推理和搜索解决复杂问题，支持流式输出。"
            
            def _run(self_tool, query: Any) -> AsyncGenerator:
                return self.search_stream(query)
            
            async def _arun(self_tool, query: Any) -> AsyncGenerator:
                return await self.search_stream(query)
        
        return DeeperResearchStreamTool()
    
    def get_exploration_tool(self):
        """获取专注于知识图谱探索的工具"""
        class KnowledgeExplorationTool(BaseTool):
            name : str = "knowledge_exploration"
            description : str = "知识图谱探索工具：专注于从起始实体出发，探索知识图谱，发现潜在关联。"
            
            def _run(self_tool, query: Any) -> Dict:
                if isinstance(query, dict) and "entities" in query:
                    entities = query["entities"]
                    search_query = query.get("query", "")
                else:
                    # 提取关键词作为起始实体
                    search_query = str(query)
                    keywords = self.extract_keywords(search_query)
                    entities = keywords.get("high_level", []) + keywords.get("low_level", [])
                    entities = entities[:3]  # 最多使用3个实体
                
                # 执行探索
                if entities:
                    return self.chain_explorer.explore(search_query, entities, max_steps=3)
                else:
                    return {"status": "error", "message": "未找到起始实体"}
            
            async def _arun(self_tool, query: Any) -> Dict:
                return self._run(query)
        
        return KnowledgeExplorationTool()
    
    def get_reasoning_analysis_tool(self):
        """获取推理链分析工具"""        
        class ReasoningAnalysisTool(BaseTool):
            name : str = "reasoning_analysis"
            description : str = "推理链分析工具：分析推理过程中的证据、矛盾和支持度。"
            
            def _run(self_tool, query_id: str) -> Dict:
                # 如果没有提供查询ID，使用当前上下文的
                if not query_id and hasattr(self, 'current_query_context'):
                    query_id = self.current_query_context.get("query_id", "")
                
                if not query_id:
                    return {"status": "error", "message": "未找到有效的查询ID"}
                
                # 获取完整推理链
                reasoning_chain = self.evidence_tracker.get_reasoning_chain(query_id)
                
                # 检测矛盾
                contradiction_result = detect_and_resolve_contradictions(self, query_id)
                
                # 生成推理摘要
                reasoning_summary = self.evidence_tracker.summarize_reasoning(query_id)
                
                # 获取证据统计
                evidence_stats = self.evidence_tracker.get_evidence_source_stats(query_id)
                
                return {
                    "reasoning_chain": reasoning_chain,
                    "contradictions": contradiction_result["contradictions"],
                    "summary": reasoning_summary,
                    "evidence_stats": evidence_stats
                }
            
            async def _arun(self_tool, query_id: str) -> Dict:
                return self._run(query_id)
        
        return ReasoningAnalysisTool()
        
    async def _async_enhance_search(self, query, keywords):
        """异步执行社区感知搜索增强"""
        def enhance_wrapper():
            return enhance_search_with_coe(self, query, keywords)
        
        return await asyncio.get_event_loop().run_in_executor(None, enhance_wrapper)

    async def _async_build_graph(self, query, entities):
        """异步构建知识图谱"""
        def build_wrapper():
            return self.knowledge_builder.build_query_graph(query, entities, depth=1)
        
        return await asyncio.get_event_loop().run_in_executor(None, build_wrapper)
    
    async def _async_detect_contradictions(self, query_id):
        """异步检测矛盾"""
        def detect_wrapper():
            return detect_and_resolve_contradictions(self, query_id)
        
        return await asyncio.get_event_loop().run_in_executor(None, detect_wrapper)
    
    async def search_stream(self, query_input: Any) -> AsyncGenerator[str, None]:
        """
        执行带流式输出的增强深度研究
        
        Args:
            query_input: 查询或包含查询的字典
                
        Yields:
            流式内容
        """
        overall_start = time.time()
        
        # 记录开始搜索
        self._log(f"\n[深度搜索] 开始处理查询...")
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"\n[深度搜索] 解析后的查询: {query}")
        
        # 检查缓存
        cache_key = f"deeper:{query}"
        cached_result = self.deep_research.cache_manager.get(cache_key)
        if cached_result:
            self._log(f"\n[深度搜索] 缓存命中，分块返回缓存结果")
            # 分块返回缓存结果 - 更自然的分块
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 80:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        try:
            # 执行思考过程流
            full_response = ""
            thinking_content = ""
            last_chunk = None
            
            # 提示用户处理开始
            yield "\n**开始深度分析您的问题**...\n"
            
            # 评估查询复杂度以决定是否采用多假设思考
            complexity = complexity_estimate(query)
            if complexity > 0.7:
                yield "\n**检测到复杂查询，激活深度思考模式**...\n"
            
            # 使用异步探索器
            keywords = self.extract_keywords(query)
            if keywords.get("high_level", []):
                yield f"\n**使用链式探索方法搜索知识**...\n"
                
                # 异步启动探索任务，但不等待它完成
                # 我们会在后台处理结果
                asyncio.create_task(self.chain_explorer.explore_async(
                    query, 
                    keywords.get("high_level", [])[:3], 
                    max_steps=3
                ))
            
            async for chunk in self.thinking_stream(query):
                if isinstance(chunk, dict) and "answer" in chunk:
                    # 这是最终结果对象
                    full_response = chunk["answer"]
                    thinking_content = chunk["thinking"]
                    last_chunk = chunk
                    
                    # 在最终答案之前添加矛盾检测结果
                    if hasattr(self, 'current_query_context') and self.current_query_context.get("query_id"):
                        # 检测矛盾
                        query_id = self.current_query_context.get("query_id")
                        contradiction_result = await self._async_detect_contradictions(query_id)
                        
                        if contradiction_result["contradictions"]:
                            yield "\n**信息一致性分析**：发现信息中存在一些不一致之处。在综合答案时已考虑这些因素。\n\n"
                    
                    # 清理最终答案以移除思考过程部分
                    clean_answer = re.sub(r'<think>.*?</think>\s*', '', full_response, flags=re.DOTALL)
                    
                    # 添加推理摘要统计
                    if hasattr(self, 'current_query_context') and self.current_query_context.get("query_id"):
                        query_id = self.current_query_context.get("query_id")
                        reasoning_summary = self.evidence_tracker.summarize_reasoning(query_id)
                        
                        # 在答案末尾添加小字体的推理统计
                        if reasoning_summary and reasoning_summary.get("steps_count", 0) > 0:
                            stats = f"\n\n<small>*推理过程包含 {reasoning_summary.get('steps_count', 0)} 个步骤，"
                            stats += f"使用了 {reasoning_summary.get('evidence_count', 0)} 个证据，"
                            stats += f"耗时 {reasoning_summary.get('duration_seconds', 0):.1f} 秒*</small>"
                            clean_answer += stats
                    
                    # 缓存结果
                    self.deep_research.cache_manager.set(cache_key, clean_answer)
                    
                    yield clean_answer
                else:
                    # 正常返回流式块
                    yield chunk
            
            # 记录总时间
            total_time = time.time() - overall_start
            self._log(f"\n[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
                
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}"
            self._log(error_msg)
            traceback.print_exc()
            yield error_msg
            
    async def thinking_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        执行带流式输出的增强深度研究
        
        Args:
            query: 用户问题
                    
        Yields:
            思考步骤和最终答案
        """
        overall_start = time.time()
        
        # 清空执行日志
        self.execution_logs = []
        self._log(f"\n[深度研究] 开始处理查询: {query}")
        
        # 向用户发送初始状态消息
        yield "\n**正在分析您的问题**...\n"
        
        # 提取关键词
        keywords = self.extract_keywords(query)
        self._log(f"\n[深度研究] 提取关键词: {keywords}")
        
        # 开始新的查询跟踪
        query_id = self.evidence_tracker.start_new_query(query, keywords)
        self.current_query_context = {"query_id": query_id}
        
        # 重置分支信息
        self.explored_branches = {}
        
        # 使用QueryGenerator生成初始子查询
        yield "\n**正在将问题分解为子问题**...\n"
        
        # 异步生成子查询
        def generate_sub_queries():
            return self.query_generator.generate_sub_queries(query)
        
        initial_sub_queries = await asyncio.get_event_loop().run_in_executor(None, generate_sub_queries)
        self._log(f"\n[深度研究] 生成了{len(initial_sub_queries)}个初始子查询: {initial_sub_queries}")
        
        # 显示子查询
        sub_queries_msg = "\n**问题已分解为以下子问题**:\n"
        for i, subq in enumerate(initial_sub_queries, 1):
            sub_queries_msg += f"{i}. {subq}\n"
        yield sub_queries_msg
        
        # 记录为推理步骤
        subquery_step_id = self.evidence_tracker.add_reasoning_step(
            query_id,
            "initial_query_decomposition",
            f"将原始问题分解为{len(initial_sub_queries)}个子问题: {initial_sub_queries}"
        )
        
        # 步骤1: 社区感知增强
        yield "\n**正在分析相关知识社区**...\n"
        enhanced_context = await self._async_enhance_search(query, keywords)
        
        # 步骤2: 使用社区信息增强搜索策略
        search_strategy = enhanced_context.get("search_strategy", {})
        follow_up_queries = search_strategy.get("follow_up_queries", [])
        
        # 合并社区生成的后续查询和初始子查询
        all_queries = list(set(initial_sub_queries + follow_up_queries))
        
        if follow_up_queries:
            query_msg = f"\n**发现潜在的深入探索方向**: {', '.join(follow_up_queries[:2])}\n"
            self._log(query_msg)
            yield query_msg
        
        # 添加增强搜索策略的推理步骤
        strategy_step_id = self.evidence_tracker.add_reasoning_step(
            query_id, 
            "knowledge_community_analysis", 
            f"基于社区分析，识别了关键实体和相关查询策略。将探索以下后续查询: {all_queries}"
        )
        
        # 记录社区信息作为证据
        community_summaries = []
        if "community_info" in enhanced_context and "summaries" in enhanced_context["community_info"]:
            for summary_obj in enhanced_context["community_info"]["summaries"]:
                if isinstance(summary_obj, dict) and "summary" in summary_obj:
                    community_summaries.append(summary_obj["summary"])
                else:
                    community_summaries.append(str(summary_obj))
                    
        if community_summaries:
            comm_msg = f"\n**找到 {len(community_summaries)} 个相关知识社区**\n"
            self._log(comm_msg)
            yield comm_msg
            
            for i, summary in enumerate(community_summaries[:2]):
                short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                self._log(f"\n[深度研究] 社区 {i+1} 摘要: {short_summary}")
                
                self.evidence_tracker.add_evidence(
                    strategy_step_id,
                    f"community_summary_{i}",
                    summary,
                    "community_knowledge"
                )
        
        # 步骤3: 构建初始知识图谱
        initial_entities = search_strategy.get("focus_entities", [])
        discovered_entities = search_strategy.get("discovered_entities", [])
        
        # 合并实体
        graph_entities = list(set(initial_entities + discovered_entities))
        
        if graph_entities:
            kg_msg = f"\n**正在构建相关知识图谱**...\n"
            self._log(kg_msg)
            yield kg_msg
            
            # 异步构建知识图谱
            await self._async_build_graph(query, graph_entities)
            
            # 获取核心实体
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
            central_entity_ids = [e["id"] for e in central_entities]
            
            if central_entity_ids:
                central_msg = f"\n**识别出核心相关实体**: {', '.join(central_entity_ids[:3])}\n"
                yield central_msg
            
            # 添加知识图谱分析步骤
            kg_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                "knowledge_graph_analysis",
                f"构建了初始知识图谱，识别出核心实体: {central_entity_ids}"
            )
            
            # 记录知识图谱信息作为证据
            kg_info = {
                "entity_count": self.knowledge_builder.knowledge_graph.number_of_nodes(),
                "relation_count": self.knowledge_builder.knowledge_graph.number_of_edges(),
                "central_entities": central_entity_ids
            }
            
            self.evidence_tracker.add_evidence(
                kg_step_id,
                "knowledge_graph",
                json.dumps(kg_info),
                "graph_structure"
            )
        
        # 步骤4: 生成多个假设  
        complexity = complexity_estimate(query)
        if complexity > 0.7:  # 对复杂查询应用多假设思考
            yield "\n**生成多个思考假设**...\n"
            
            # 异步生成假设
            def generate_hypotheses():
                return self.query_generator.generate_multiple_hypotheses(query, self.llm)
            
            hypotheses = await asyncio.get_event_loop().run_in_executor(None, generate_hypotheses)
            
            if hypotheses:
                hypothesis_msg = f"\n**探索 {len(hypotheses)} 个可能假设**\n"
                for i, hyp in enumerate(hypotheses[:2]):
                    hypothesis_msg += f"- 假设 {i+1}: {hyp}\n"
                yield hypothesis_msg
                
                # 创建多个推理分支
                branch_results = create_multiple_reasoning_branches(self, query_id, hypotheses)
                
                # 对第一个分支应用反事实分析
                if branch_results and "branch_1" in branch_results:
                    yield "\n**进行反事实分析**...\n"
        
        # 步骤5: 使用深度研究工具的流式思考过程
        yield "\n**正在进行深度研究**...\n"
        
        # 调用deep_research工具的流式API，但传递我们增强的上下文
        # 准备上下文消息
        context_message = ""
        
        # 添加初始子查询分解
        if initial_sub_queries:
            context_message += "为了全面解答这个问题，我将从以下几个方面进行研究：\n"
            for i, sub_q in enumerate(initial_sub_queries, 1):
                context_message += f"{i}. {sub_q}\n"
            context_message += "\n"
            
        if community_summaries:
            context_message += "找到了相关的知识社区信息:\n"
            context_message += "\n".join(community_summaries[:2])  # 只使用前两个
            
        # 添加探索路径信息
        exploration_path = enhanced_context.get("exploration_results", {}).get("exploration_path", [])
        if exploration_path:
            context_message += "\n\n通过Chain of Exploration探索路径发现:\n"
            path_description = []
            for i, step in enumerate(exploration_path[:5]):  # 限制路径长度
                if i > 0:  # 跳过起始实体
                    path_description.append(f"- 步骤{step['step']}: 实体 {step['node_id']} ({step['reasoning']})")
            context_message += "\n".join(path_description)
        
        # 如果有跟进查询，添加到思考过程
        if follow_up_queries:
            context_message += "\n\n可能的深入探索方向:\n"
            context_message += "\n".join([f"- {q}" for q in follow_up_queries])
            
        # 将增强上下文添加到思考引擎
        if context_message:
            self.deep_research.thinking_engine.initialize_with_query(query)
            self.deep_research.thinking_engine.add_reasoning_step(context_message)
            yield context_message
            
        # 执行深度研究迭代
        think = context_message
        
        # 初始化思考引擎并添加上下文
        self.deep_research.thinking_engine.initialize_with_query(query)
        self.deep_research.thinking_engine.add_reasoning_step(context_message)
        
        # 迭代思考过程
        for iteration in range(self.deep_research.max_iterations):
            # 发送迭代进度
            iter_msg = f"\n\n**正在进行第{iteration + 1}轮思考**...\n\n"
            yield iter_msg
            self._log(f"\n[深度研究] 开始第{iteration + 1}轮迭代\n")
            
            # 跟踪迭代步骤
            iteration_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                f"iteration_{iteration}",
                f"\n开始第 {iteration + 1} 轮迭代思考\n"
            )
            
            # 检查是否达到最大迭代次数
            if iteration >= self.deep_research.max_iterations - 1:
                limit_msg = "**\n已达到最大搜索次数限制，准备生成最终答案**...\n"
                yield limit_msg
                self._log("\n[深度研究] 达到最大迭代次数，结束搜索")
                
                summary_think = f"\n搜索次数已达上限。不允许继续搜索。\n"
                self.deep_research.thinking_engine.add_reasoning_step(summary_think)
                self.deep_research.thinking_engine.add_human_message(summary_think)
                think += self.deep_research.thinking_engine.remove_result_tags(summary_think)
                break
            
            # 更新消息历史，请求继续推理
            self.deep_research.thinking_engine.update_continue_message()
            
            # 让事件循环有机会执行其他任务
            await asyncio.sleep(0)
            
            # 确定当前迭代要处理的查询
            queries_to_process = []
            
            if iteration == 0 and initial_sub_queries:
                # 第一轮迭代使用预先生成的子查询
                queries_to_process = initial_sub_queries[:2]  # 限制首轮使用的子查询数量
                yield "\n**开始根据分解的子问题进行初始搜索**...\n"
            else:
                # 非首轮，使用思考引擎生成下一步查询
                result = await self.deep_research._async_generate_next_query()
                
                # 处理生成结果
                if result["status"] == "empty":
                    empty_msg = "未能产生新的思考角度，尝试从新角度探索问题..."
                    self._log(empty_msg)
                    yield empty_msg
                    
                    # 异步生成多假设
                    def generate_hypotheses():
                        return self.query_generator.generate_multiple_hypotheses(query, self.llm)
                    
                    hypotheses = await asyncio.get_event_loop().run_in_executor(None, generate_hypotheses)
                    
                    if hypotheses:
                        self._log(f"\n[深度研究] 生成了{len(hypotheses)}个新假设，尝试从新角度探索")
                        queries_to_process = hypotheses
                        query_think = "尝试从新的角度思考这个问题:\n" + "\n".join([f"- {h}" for h in hypotheses])
                        self.deep_research.thinking_engine.add_reasoning_step(query_think)
                        think += query_think
                        yield "\n**尝试从新角度思考问题**:\n" + query_think
                    else:
                        continue
                elif result["status"] == "error":
                    error_msg = f"生成查询时遇到错误: {result.get('error', '未知错误')}"
                    self._log(error_msg)
                    yield error_msg
                    
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "generation_error",
                        f"生成查询出错: {result.get('error', '未知错误')}",
                        "reasoning_status"
                    )
                    break
                elif result["status"] == "answer_ready":
                    ready_msg = "已收集足够信息，准备生成答案..."
                    self._log(ready_msg)
                    yield ready_msg
                    
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "reasoning_complete",
                        "AI认为已有足够信息生成答案",
                        "reasoning_status"
                    )
                    break
                            
                else:
                    # 获取生成的思考内容
                    query_think = result["content"]
                    think_part = self.deep_research.thinking_engine.remove_query_tags(query_think)
                    think += think_part
                    
                    # 分组返回思考内容，提高可读性
                    thoughts = re.split(r'(\n\n)', think_part)
                    thought_buffer = ""
                    
                    for part in thoughts:
                        thought_buffer += part
                        if len(thought_buffer) >= 80 or "\n\n" in thought_buffer:
                            yield thought_buffer
                            thought_buffer = ""
                            await asyncio.sleep(0.01)
                            
                    if thought_buffer:
                        yield thought_buffer
                    
                    # 获取搜索查询
                    queries_to_process = result["queries"]
                
            # 如果当前迭代没有查询，且我们已经检索到一些信息，尝试生成跟进查询
            if not queries_to_process and self.deep_research.all_retrieved_info and iteration > 0:
                yield "\n**基于已发现的信息生成后续查询**...\n"
                
                # 异步生成跟进查询
                def generate_followup():
                    return self.query_generator.generate_followup_queries(query, self.deep_research.all_retrieved_info)
                
                followup_queries = await asyncio.get_event_loop().run_in_executor(None, generate_followup)
                
                if followup_queries:
                    self._log(f"\n[深度研究] 生成了{len(followup_queries)}个跟进查询")
                    queries_to_process = followup_queries
                    followup_think = "\n考虑到已发现的信息，我需要进一步探索以下问题:\n"
                    for i, fq in enumerate(followup_queries, 1):
                        followup_think += f"{i}. {fq}\n"
                    self.deep_research.thinking_engine.add_reasoning_step(followup_think)
                    think += followup_think
                    yield followup_think
                    
                    # 记录为推理步骤
                    followup_step_id = self.evidence_tracker.add_reasoning_step(
                        query_id,
                        "followup_queries_generation",
                        f"基于已收集的信息生成跟进查询: {followup_queries}"
                    )
            
            # 如果没有生成搜索查询但不是第一轮，考虑结束
            if not queries_to_process:
                if not self.deep_research.all_retrieved_info and iteration == 0:
                    # 如果第一轮没检索到任何信息，强制使用原始查询
                    queries_to_process = [query]
                    no_info_msg = "\n\n**没有检索到信息，尝试使用原始问题直接查询**...\n\n"
                    self._log(no_info_msg)
                    yield no_info_msg
                    
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "query_fallback",
                        "没有检索到信息，使用原始查询",
                        "reasoning_status"
                    )
                else:
                    # 已有信息，结束迭代
                    end_msg = "\n\n**没有发现新的查询角度，基于已有信息生成回答**...\n\n"
                    self._log(end_msg)
                    yield end_msg
                    
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "iteration_complete",
                        "没有生成新查询且已有信息，结束迭代",
                        "reasoning_status"
                    )
                    break
            
            # 处理每个搜索查询
            for search_query in queries_to_process:
                search_start_msg = f"\n**正在搜索: {search_query}**\n"
                self._log(search_start_msg)
                yield search_start_msg
                    
                # 检查是否已执行过相同查询
                if self.deep_research.thinking_engine.has_executed_query(search_query):
                    dupe_msg = f"\n**已搜索过类似查询，跳过重复执行**\n"
                    self._log(dupe_msg)
                    yield dupe_msg
                    
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "duplicate_query",
                        "已搜索过该查询，跳过重复执行",
                        "search_status"
                    )
                    continue
                    
                # 记录已执行查询
                self.deep_research.thinking_engine.add_executed_query(search_query)
                    
                # 将搜索查询添加到消息历史
                self.deep_research.thinking_engine.add_ai_message(f"{search_query}")
                think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                    
                # 让事件循环有机会执行其他任务
                await asyncio.sleep(0)
                    
                # 执行实际搜索 - 同时应用Chain of Exploration增强
                yield "\n**正在查询知识库**...\n"
                
                # 记录查询步骤
                search_step_id = self.evidence_tracker.add_reasoning_step(
                    query_id,
                    search_query,
                    f"执行查询: {search_query}"
                )
                
                # 提取搜索查询的关键词
                search_keywords = self.extract_keywords(search_query)
                
                # 使用Chain of Exploration增强搜索
                enhanced_search_context = await self._async_enhance_search(search_query, search_keywords)
                exploration_content = []
                
                # 从Chain of Exploration结果中提取内容
                if "exploration_results" in enhanced_search_context:
                    for content_item in enhanced_search_context["exploration_results"].get("content", []):
                        if "text" in content_item:
                            exploration_content.append({
                                "chunk_id": content_item.get("id", "exploration_content"),
                                "text": content_item["text"],
                                "content_with_weight": content_item["text"],
                                "weight": 1.0
                            })
                
                # 原始搜索，合并结果
                kbinfos = await self.deep_research._async_search(search_query)
                
                # 将探索内容加入结果
                if exploration_content:
                    if "chunks" not in kbinfos:
                        kbinfos["chunks"] = []
                    kbinfos["chunks"].extend(exploration_content)
                    
                # 检查搜索结果是否为空
                has_results = (
                    kbinfos.get("chunks", []) or 
                    kbinfos.get("entities", []) or 
                    kbinfos.get("relationships", [])
                )
                    
                if not has_results:
                    no_result_msg = f"\n**没有找到与{search_query}相关的信息，尝试其他角度**...\n"
                    self._log(no_result_msg)
                    yield no_result_msg
                    
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        "no_results",
                        f"没有找到与查询相关的信息: {search_query}",
                        "search_result"
                    )
                    
                    self.deep_research.thinking_engine.add_reasoning_step(f"\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n")
                    self.deep_research.thinking_engine.add_human_message(f"\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n")
                    think += no_result_msg
                    continue
                    
                # 正常处理有结果的情况
                truncated_prev_reasoning = self.deep_research.thinking_engine.prepare_truncated_reasoning()
                    
                # 为查询结果更新知识图谱
                if "chunks" in kbinfos:
                    for chunk in kbinfos.get("chunks", []):
                        chunk_id = chunk.get("chunk_id", "")
                        chunk_text = chunk.get("text", "")
                        
                        if chunk_id and chunk_text:
                            # 记录chunk作为证据
                            self.evidence_tracker.add_evidence(
                                search_step_id,
                                chunk_id,
                                chunk_text,
                                "document_chunk"
                            )
                            
                            # 尝试从chunk中提取实体和关系
                            if hasattr(self.knowledge_builder, 'extractor') and self.knowledge_builder.extractor:
                                try:
                                    self.knowledge_builder.extract_subgraph_from_chunk(
                                        chunk_text, 
                                        chunk_id
                                    )
                                except Exception as e:
                                    self._log(f"\n[深度研究] 提取实体关系失败: {e}")
                    
                # 构建提取相关信息的提示
                kb_prompt_result = "\n".join(kb_prompt(kbinfos, 4096))
                    
                # 告知用户正在分析结果
                yield "\n**正在分析搜索结果**...\n"
                    
                # 使用异步LLM提取有用信息
                summary_think = await self.deep_research._async_extract_info(search_query, truncated_prev_reasoning, kb_prompt_result)
                    
                # 保存重要信息
                has_useful_info = (
                    "**Final Information**" in summary_think and 
                    "No helpful information found" not in summary_think
                )
                    
                if has_useful_info:
                    useful_info = summary_think.split("**Final Information**")[1].strip()
                    self.deep_research.all_retrieved_info.append(useful_info)
                    info_msg = f"发现有用信息: {useful_info[:100]}..."
                    self._log(info_msg)
                    yield "\n**找到相关信息！**\n"
                    
                    # 记录有用信息作为证据
                    self.evidence_tracker.add_evidence_with_confidence(
                        search_step_id,
                        f"useful_info_{search_query}",
                        useful_info,
                        "extracted_knowledge",
                        confidence=0.85,  # 高可信度
                        metadata={"query": search_query}
                    )
                else:
                    no_useful_msg = "**\n未从搜索结果中发现特别有价值的信息**\n"
                    self._log(no_useful_msg)
                    yield no_useful_msg
                    
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        f"no_useful_info_{search_query}",
                        "未从检索结果中提取到有用信息",
                        "extraction_status"
                    )
                    
                # 更新推理历史
                self.deep_research.thinking_engine.add_reasoning_step(summary_think)
                self.deep_research.thinking_engine.add_human_message(summary_think)
                think += self.deep_research.thinking_engine.remove_result_tags(summary_think)
                    
                # 分组返回处理后的思考内容
                result_parts = re.split(r'(\n\n)', self.deep_research.thinking_engine.remove_result_tags(summary_think))
                result_buffer = ""
                
                for part in result_parts:
                    result_buffer += part
                    if len(result_buffer) >= 80 or "\n\n" in result_buffer:
                        yield result_buffer
                        result_buffer = ""
                        await asyncio.sleep(0.01)
                        
                if result_buffer:
                    yield result_buffer
            
            # 在每轮迭代结束后，评估是否需要继续搜索
            if iteration > 0 and self.deep_research.all_retrieved_info:
                # 异步判断是否需要继续生成查询
                def check_gap_needed():
                    return len(self.query_generator.generate_followup_queries(query, self.deep_research.all_retrieved_info)) > 0
                
                gap_needed = await asyncio.get_event_loop().run_in_executor(None, check_gap_needed)
                
                if not gap_needed:
                    reflection_msg = "\n**已收集到足够的信息，可以开始整合分析了**\n"
                    self._log(reflection_msg)
                    yield reflection_msg
                    self.deep_research.thinking_engine.add_reasoning_step("\n已收集到足够的信息，可以开始整合分析了。")
                    think += "\n已收集到足够的信息，可以开始整合分析了。"
                    break
        
        # 步骤6: 合并推理分支
        if self.explored_branches:
            merge_msg = "\n**正在合并多个推理分支**...\n"
            self._log(merge_msg)
            yield merge_msg
            
            merged_results = merge_reasoning_branches(self, query_id)
            if merged_results:
                self.deep_research.thinking_engine.add_reasoning_step(merged_results)
                think += merged_results
        
        # 步骤7: 矛盾检测与处理
        if self.deep_research.thinking_engine.executed_search_queries:
            yield "\n**检测信息一致性**...\n"
            
            # 异步检测矛盾
            contradiction_result = await self._async_detect_contradictions(query_id)
            if contradiction_result["contradictions"]:
                contradiction_msg = f"\n**发现 {len(contradiction_result['contradictions'])} 个信息不一致**\n"
                self._log(contradiction_msg)
                yield contradiction_msg
                
                contradiction_analysis = "\n## 信息矛盾分析\n"
                contradiction_analysis += f"发现 {len(contradiction_result['contradictions'])} 个信息矛盾：\n"
                
                for i, contradiction in enumerate(contradiction_result["contradictions"]):
                    if contradiction["type"] == "numerical":
                        contradiction_analysis += f"{i+1}. 数值矛盾: 在\"{contradiction.get('context', '')}\"中，"
                        contradiction_analysis += f"发现值 {contradiction.get('value1')} 和 {contradiction.get('value2')}\n"
                    else:
                        contradiction_analysis += f"{i+1}. 语义矛盾: {contradiction.get('analysis', '')}\n"
                
                # 添加矛盾分析到思考过程
                self.deep_research.thinking_engine.add_reasoning_step(contradiction_analysis)
                think += contradiction_analysis
                yield contradiction_analysis
        
        # 确保至少执行了一次搜索
        if not self.deep_research.thinking_engine.executed_search_queries:
            no_search_msg = f"\n**无法找到与{query}相关的信息，尝试给出基础回答**...\n"
            yield no_search_msg
            return
        
        # 生成最终答案
        yield "\n**正在根据所有收集的信息生成最终答案**...\n"
        
        # 使用异步生成最终答案
        def generate_final_answer():
            try:
                # 使用检索到的信息生成答案
                retrieved_content = "\n\n".join(self.deep_research.all_retrieved_info)
                
                # 增强答案生成过程，整合知识图谱和社区分析结果
                enhanced_prompt = f"""
                用户问题：{query}
                
                我已经通过多种检索方法收集了以下信息：
                
                {retrieved_content}
                
                此外，我通过知识图谱分析发现了以下关键实体和关系：
                """
                
                # 添加知识图谱分析结果
                central_entities = self.knowledge_builder.get_central_entities(limit=5)
                if central_entities:
                    enhanced_prompt += "\n核心实体及其重要性：\n"
                    for entity in central_entities:
                        entity_id = entity.get("id", "")
                        importance = entity.get("centrality", entity.get("degree", 0))
                        entity_type = entity.get("type", "unknown")
                        properties = entity.get("properties", {})
                        description = properties.get("description", "无描述")
                        
                        enhanced_prompt += f"- {entity_id} (重要性: {importance:.3f}, 类型: {entity_type}): {description}\n"
                
                # 添加社区分析结果
                if community_summaries:
                    enhanced_prompt += "\n来自相关知识社区的见解：\n"
                    for i, summary in enumerate(community_summaries[:2]):
                        enhanced_prompt += f"- 社区{i+1}: {summary[:200]}...\n"
                
                # 添加探索路径分析
                if exploration_path:
                    enhanced_prompt += "\n知识探索路径分析：\n"
                    path_summary = []
                    current_entity = None
                    for step in exploration_path:
                        if step["step"] > 0:  # 跳过起始实体
                            if current_entity != step["node_id"]:
                                current_entity = step["node_id"]
                                path_summary.append(f"实体 {current_entity}: {step['reasoning']}")
                    
                    enhanced_prompt += "\n".join(path_summary[:3]) + "\n"
                
                # 添加初始子查询分析结果
                if initial_sub_queries:
                    enhanced_prompt += "\n问题分解与深入分析：\n"
                    for i, subq in enumerate(initial_sub_queries, 1):
                        related_info = []
                        for info in self.deep_research.all_retrieved_info:
                            if any(term.lower() in info.lower() for term in subq.lower().split()):
                                related_info.append(info[:100] + "..." if len(info) > 100 else info)
                        
                        if related_info:
                            enhanced_prompt += f"- 子问题{i}: {subq} - 找到{len(related_info)}条相关信息\n"
                        else:
                            enhanced_prompt += f"- 子问题{i}: {subq} - 未找到直接相关信息\n"
                
                # 添加矛盾分析结果
                contradiction_result = detect_and_resolve_contradictions(self, query_id)
                if contradiction_result["contradictions"]:
                    enhanced_prompt += "\n信息矛盾分析：\n"
                    for i, contradiction in enumerate(contradiction_result["contradictions"][:3]):
                        if contradiction["type"] == "numerical":
                            enhanced_prompt += f"- 矛盾{i+1}: 在数值上存在不一致，一处显示 {contradiction.get('value1')}，另一处显示 {contradiction.get('value2')}\n"
                        else:
                            enhanced_prompt += f"- 矛盾{i+1}: {contradiction.get('analysis', '')}\n"
                
                # 请求生成最终答案
                enhanced_prompt += """
                请基于以上所有信息，生成一个全面深入的回答。回答应该:
                1. 直接回答用户问题
                2. 结构清晰，逻辑性强
                3. 整合所有相关信息，包括知识图谱和社区分析的见解
                4. 如有必要，指出信息中的不确定性或矛盾
                5. 对问题的不同方面进行全面分析
                """
                
                # 使用增强提示生成最终答案
                final_response = self.llm.invoke(enhanced_prompt)
                enhanced_answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
                
                # 将增强答案与思考过程结合
                final_answer = f"<think>{think}</think>\n\n{enhanced_answer}"
                
                # 生成引用标记
                cited_answer = generate_citations(self, enhanced_answer, query_id)
                
                # 如果引用生成成功，使用带引用的答案
                if len(cited_answer) > len(enhanced_answer):
                    final_answer = f"<think>{think}</think>\n\n{cited_answer}"
                
                return final_answer
            except Exception as e:
                # 如果增强生成失败，回退到原始方法
                self._log(f"\n[深度研究] 增强答案生成失败: {e}，回退到标准方法")
                return self.deep_research._generate_final_answer(query, retrieved_content, think)
        
        # 异步执行答案生成
        final_answer = await asyncio.get_event_loop().run_in_executor(None, generate_final_answer)
        
        # 获取知识图谱中的核心实体
        central_entities = []
        if hasattr(self.knowledge_builder, 'knowledge_graph') and self.knowledge_builder.knowledge_graph.nodes:
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
        
        # 记录最终回答步骤
        final_step_id = self.evidence_tracker.add_reasoning_step(
            query_id,
            "final_answer",
            "生成最终答案"
        )
        
        self.evidence_tracker.add_evidence(
            final_step_id,
            "final_answer_evidence",
            "\n\n".join(self.deep_research.all_retrieved_info),
            "synthesized_knowledge"
        )
        
        # 获取推理摘要
        reasoning_summary = self.evidence_tracker.summarize_reasoning(query_id)
        
        # 在最终答案前增加子问题分析概述
        if initial_sub_queries and "<think>" in final_answer and "</think>" in final_answer:
            # 提取思考过程之外的部分
            clean_answer = re.sub(r'<think>.*?</think>\s*', '', final_answer, flags=re.DOTALL)
            
            # 构建子问题概述
            subq_overview = "\n\n**问题分析概述**:\n"
            for i, subq in enumerate(initial_sub_queries, 1):
                related_info_count = 0
                for info in self.deep_research.all_retrieved_info:
                    if any(term.lower() in info.lower() for term in subq.lower().split()):
                        related_info_count += 1
                
                subq_overview += f"- 子问题 {i}: {subq}"
                if related_info_count > 0:
                    subq_overview += f" - 找到 {related_info_count} 条相关信息\n"
                else:
                    subq_overview += " - 未找到直接相关信息\n"
            
            # 添加推理统计
            if reasoning_summary:
                subq_overview += f"\n<small>*分析过程包含 {reasoning_summary.get('steps_count', 0)} 个步骤，"
                subq_overview += f"使用了 {reasoning_summary.get('evidence_count', 0)} 个证据来源，"
                subq_overview += f"耗时 {reasoning_summary.get('duration_seconds', 0):.1f} 秒*</small>"
            
            # 重新组合答案
            enhanced_final = clean_answer + subq_overview
            
            # 返回最终答案及思考过程
            yield {"answer": f"<think>{think}</think>\n\n{enhanced_final}", "thinking": think}
        else:
            # 返回原始答案
            yield {"answer": final_answer, "thinking": think}
        
    def close(self):
        """关闭资源"""
        # 关闭deep_research的资源
        if hasattr(self, 'deep_research'):
            self.deep_research.close()
