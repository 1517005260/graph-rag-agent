from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import asyncio
import pandas as pd
import re

class ChainOfExplorationSearcher:
    """
    增强版Chain of Exploration检索器
    
    实现多步自主探索图谱的能力，具有适应性搜索宽度、记忆机制和路径优化功能
    """
    
    def __init__(self, graph, llm, embeddings_model):
        """
        初始化Chain of Exploration检索器
        
        Args:
            graph: 图数据库连接
            llm: 语言模型
            embeddings_model: 向量嵌入模型
        """
        self.graph = graph
        self.llm = llm
        self.embeddings = embeddings_model
        self.visited_nodes = set()
        self.exploration_path = []
        self.exploration_memory = {}  # 存储已探索路径的记忆
        self.performance_metrics = {}
    
    def explore(self, query: str, starting_entities: List[str], max_steps: int = 5, exploration_width: int = 3):
        """
        从起始实体开始探索图谱
        
        Args:
            query: 用户查询
            starting_entities: 起始实体列表
            max_steps: 最大探索步数
            exploration_width: 基础探索宽度
            
        Returns:
            Dict: 探索结果
        """
        start_time = time.time()
        
        if not starting_entities:
            return {
                "entities": [],
                "relationships": [],
                "content": [],
                "exploration_path": []
            }
            
        # 重置状态
        self.visited_nodes = set(starting_entities)
        self.exploration_path = []
        query_embedding = self.embeddings.embed_query(query)
        
        # 添加起始节点到探索路径
        for entity in starting_entities:
            self.exploration_path.append({
                "step": 0,
                "node_id": entity,
                "action": "start",
                "reasoning": "初始实体"
            })
        
        # 根据查询内容生成探索策略
        exploration_strategy = self._generate_exploration_strategy(query, starting_entities)
        self.performance_metrics["strategy_generation_time"] = time.time() - start_time
        
        current_entities = starting_entities
        results = {
            "entities": [],
            "relationships": [],
            "content": [],
            "communities": []
        }
        
        # 多步探索
        for step in range(max_steps):
            step_start_time = time.time()
            
            if not current_entities:
                break
                
            # 1. 找出邻居节点
            neighbors = self._get_neighbors(current_entities)
            if not neighbors:
                break
                
            # 2. 动态宽度控制
            current_width = self._calculate_adaptive_width(
                step, 
                query, 
                neighbors, 
                base_width=exploration_width
            )
            
            # 3. 评估每个邻居与查询的相关性
            scored_neighbors = self._score_neighbors_enhanced(
                neighbors, 
                query, 
                query_embedding,
                exploration_strategy
            )
            
            # 4. 让LLM决定探索方向
            next_entities, reasoning = self._decide_next_step_with_memory(
                query, 
                current_entities, 
                scored_neighbors, 
                current_width,
                step
            )
            
            # 5. 更新已访问节点
            new_entities = [e for e in next_entities if e not in self.visited_nodes]
            self.visited_nodes.update(new_entities)
            
            # 6. 获取新发现实体的内容
            entity_info = self._get_entity_info(new_entities)
            results["entities"].extend(entity_info)
            
            # 7. 收集关系信息
            rel_info = self._get_relationship_info(new_entities)
            results["relationships"].extend(rel_info)
            
            # 8. 收集内容信息（如chunk）
            content_info = self._get_content_info(new_entities)
            results["content"].extend(content_info)
            
            # 9. 尝试获取所属社区信息
            community_info = self._get_community_info(new_entities)
            if community_info:
                results["communities"].extend(community_info)
            
            # 10. 记录探索路径
            for entity in new_entities:
                self.exploration_path.append({
                    "step": step + 1,
                    "node_id": entity,
                    "action": "explore",
                    "reasoning": reasoning
                })
            
            # 11. 更新当前实体
            current_entities = new_entities
            
            # 记录每步耗时
            self.performance_metrics[f"step_{step+1}_time"] = time.time() - step_start_time
        
        # 根据查询对所有收集的内容进行最终排序
        results["content"] = self._rank_content_by_relevance(query_embedding, results["content"])
        
        # 添加探索路径、数据统计和性能指标
        results["exploration_path"] = self.exploration_path
        results["visited_entities"] = list(self.visited_nodes)
        results["statistics"] = {
            "entity_count": len(results["entities"]),
            "relationship_count": len(results["relationships"]),
            "content_count": len(results["content"]),
            "path_length": len(self.exploration_path)
        }
        
        # 记录总耗时
        self.performance_metrics["total_time"] = time.time() - start_time
        results["performance_metrics"] = self.performance_metrics
        
        return results
    
    def _generate_exploration_strategy(self, query: str, starting_entities: List[str]) -> Dict[str, Any]:
        """
        为查询生成探索策略
        
        Args:
            query: 查询字符串
            starting_entities: 起始实体
            
        Returns:
            Dict: 探索策略
        """
        prompt = f"""
        为以下查询生成图谱探索策略，从给定的起始实体开始探索:
        
        查询: "{query}"
        起始实体: {starting_entities}
        
        请提供以下信息:
        1. 探索重点: 探索应该关注哪些类型的关系和实体?
        2. 终止条件: 什么情况下应该终止特定方向的探索?
        3. 重要程度评分: 为不同类型的关系提供重要性权重(0-1)
        
        以JSON格式返回结果:
        {{
            "focus_relations": ["关系类型1", "关系类型2", ...],
            "focus_entity_types": ["实体类型1", "实体类型2", ...],
            "avoid_relations": ["应避免的关系类型1", ...],
            "termination_conditions": ["条件1", ...],
            "relation_weights": {{"关系类型1": 0.9, "关系类型2": 0.7, ...}}
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取JSON部分
            import re
            import json
            
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                strategy = json.loads(json_match.group(0))
                return strategy
            
            # 如果无法解析，返回默认策略
            return {
                "focus_relations": [],
                "focus_entity_types": [],
                "avoid_relations": [],
                "termination_conditions": [],
                "relation_weights": {}
            }
        except Exception as e:
            print(f"生成探索策略失败: {e}")
            # 默认策略
            return {
                "focus_relations": [],
                "focus_entity_types": [],
                "avoid_relations": [],
                "termination_conditions": [],
                "relation_weights": {}
            }
    
    def _calculate_adaptive_width(self, step, query, neighbors, base_width=3):
        """
        根据查询复杂度和当前步骤动态调整探索宽度
        
        Args:
            step: 当前步骤
            query: 查询字符串
            neighbors: 邻居节点列表
            base_width: 基础宽度
            
        Returns:
            int: 调整后的宽度
        """
        # 步骤越深，宽度越小，避免指数爆炸
        step_factor = max(0.5, 1.0 - step * 0.2)
        
        # 邻居节点数量因素 - 邻居越多，宽度越大但有上限
        neighbor_factor = min(1.5, len(neighbors) / 10)
        
        # 查询复杂度因素
        complexity_factor = self._estimate_query_complexity(query)
        
        # 计算最终宽度
        adjusted_width = int(base_width * step_factor * neighbor_factor * complexity_factor)
        
        # 确保宽度在合理范围内
        return max(1, min(5, adjusted_width))
    
    def _estimate_query_complexity(self, query):
        """
        估计查询复杂度
        
        Args:
            query: 查询字符串
            
        Returns:
            float: 复杂度评分(0.5-1.5)
        """
        # 基于查询长度、问号数量和关键词数量的简单启发式方法
        length_factor = min(1.5, len(query) / 50)
        question_marks = query.count("?") + query.count("？")
        question_factor = 1.0 + (question_marks * 0.1)
        
        # 识别复杂问题的关键词
        complexity_indicators = [
            "为什么", "如果", "原因", "关系", "比较", "区别",
            "影响", "分析", "评估", "预测"
        ]
        
        # 检查关键词
        indicator_count = sum(1 for indicator in complexity_indicators if indicator.lower() in query.lower())
        indicator_factor = 1.0 + (indicator_count * 0.1)
        
        # 综合评分,基础值0.5,最大1.5
        complexity = 0.5 + (length_factor * 0.3 + question_factor * 0.3 + indicator_factor * 0.4) / 3
        
        return min(1.5, complexity)
    
    def _get_neighbors(self, entities):
        """
        获取实体的邻居节点
        
        Args:
            entities: 实体ID列表
            
        Returns:
            List: 邻居节点列表
        """
        try:
            query = """
            MATCH (e:__Entity__)-[r]-(neighbor:__Entity__)
            WHERE e.id IN $entity_ids AND NOT neighbor.id IN $visited_ids
            RETURN neighbor.id AS id, neighbor.description AS description,
                   type(r) AS relation_type, startNode(r).id AS source,
                   endNode(r).id AS target,
                   CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1.0 END AS weight
            LIMIT 100
            """
            
            params = {
                "entity_ids": entities, 
                "visited_ids": list(self.visited_nodes)
            }
            
            result = self.graph.query(query, params=params)
            
            # 结果为空的处理
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换为列表格式
            if isinstance(result, pd.DataFrame):
                neighbors_list = result.to_dict('records')
                return neighbors_list
            else:
                return result
                
        except Exception as e:
            print(f"获取邻居节点失败: {e}")
            return []
    
    def _score_neighbors_enhanced(self, neighbors, query, query_embedding, exploration_strategy):
        """
        增强版邻居评分，考虑策略权重、相似度和关系权重
        
        Args:
            neighbors: 邻居节点列表
            query: 查询字符串
            query_embedding: 查询嵌入向量
            exploration_strategy: 探索策略
            
        Returns:
            List: 评分后的邻居列表
        """
        scored_neighbors = []
        relation_weights = exploration_strategy.get("relation_weights", {})
        focus_relations = exploration_strategy.get("focus_relations", [])
        focus_entity_types = exploration_strategy.get("focus_entity_types", [])
        avoid_relations = exploration_strategy.get("avoid_relations", [])
        
        for neighbor in neighbors:
            # 构建描述文本
            description = neighbor.get('description', '')
            relation_type = neighbor.get('relation_type', '')
            neighbor_id = neighbor.get('id', '')
            
            # 初始权重 - 基础值0.5
            base_weight = 0.5
            
            try:
                # 计算语义相似度
                if description:
                    neighbor_embedding = self.embeddings.embed_query(description)
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(neighbor_embedding).reshape(1, -1)
                    )[0][0]
                else:
                    similarity = 0.0
                    
                # 获取关系权重
                relation_weight = relation_weights.get(relation_type, 1.0)
                
                # 计算策略匹配分数
                strategy_score = 1.0
                
                # 增加关注的关系类型的权重
                if relation_type in focus_relations:
                    strategy_score += 0.5
                
                # 检查实体类型
                entity_type = self._get_entity_type(neighbor_id)
                if entity_type in focus_entity_types:
                    strategy_score += 0.3
                
                # 降低需要避免的关系类型的权重
                if relation_type in avoid_relations:
                    strategy_score -= 0.5
                
                # 添加来自图的原始权重
                graph_weight = float(neighbor.get('weight', 1.0))
                
                # 计算最终得分(语义相似度*策略分数*关系权重*图权重)
                final_score = similarity * strategy_score * relation_weight * graph_weight
                
                # 添加到评分列表
                scored_neighbors.append({
                    "id": neighbor_id,
                    "description": description,
                    "relation_type": relation_type,
                    "source": neighbor.get('source', ''),
                    "target": neighbor.get('target', ''),
                    "similarity": similarity,
                    "strategy_score": strategy_score,
                    "relation_weight": relation_weight,
                    "graph_weight": graph_weight,
                    "final_score": final_score
                })
            except Exception as e:
                print(f"计算节点相似度失败: {e}")
                
        # 按最终得分排序
        return sorted(scored_neighbors, key=lambda x: x['final_score'], reverse=True)
    
    def _get_entity_type(self, entity_id):
        """
        获取实体类型
        
        Args:
            entity_id: 实体ID
            
        Returns:
            str: 实体类型
        """
        try:
            query = """
            MATCH (e:__Entity__ {id: $entity_id})
            RETURN labels(e) AS types
            """
            
            result = self.graph.query(query, params={"entity_id": entity_id})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return "unknown"
                
            if isinstance(result, pd.DataFrame):
                types = result.iloc[0]['types']
                # 过滤掉 "__Entity__" 标签
                entity_types = [t for t in types if t != "__Entity__"]
                return entity_types[0] if entity_types else "unknown"
            else:
                # 处理其他结果格式
                return "unknown"
        except Exception as e:
            print(f"获取实体类型失败: {e}")
            return "unknown"
    
    def _decide_next_step_with_memory(self, query, current_entities, scored_neighbors, width, current_step):
        """
        让LLM决定下一步探索方向，考虑已探索的记忆
        
        Args:
            query: 查询字符串
            current_entities: 当前实体列表
            scored_neighbors: 评分后的邻居列表
            width: 探索宽度
            current_step: 当前步骤
            
        Returns:
            Tuple[List[str], str]: 下一步实体列表和推理过程
        """
        memory_key = f"{query}_{','.join(sorted(current_entities))}"
        
        # 检查是否有记忆
        if memory_key in self.exploration_memory:
            remembered = self.exploration_memory[memory_key]
            # 检查记忆是否过期(根据步骤差异判断)
            if remembered["step"] == current_step:
                return remembered["entities"], remembered["reasoning"]
        
        # 构建提示
        prompt = f"""
        我正在使用Chain of Exploration方法探索知识图谱，以回答问题: "{query}"
        
        当前探索的实体有:
        {', '.join(current_entities)}
        
        当前是探索的第{current_step+1}步，需要决定下一步探索哪些实体。
        
        下面是一些可能的下一步探索选项(已按综合得分排序):
        """
        
        # 添加前10个最相关的选项(或全部如果少于10个)
        top_options = scored_neighbors[:10] if len(scored_neighbors) > 10 else scored_neighbors
        for i, neighbor in enumerate(top_options):
            prompt += f"{i+1}. {neighbor['id']} (得分: {neighbor['final_score']:.2f})\n"
            prompt += f"   - 描述: {neighbor['description']}\n"
            prompt += f"   - 关系类型: {neighbor['relation_type']} (连接到: {neighbor['source'] if neighbor['target'] in current_entities else neighbor['target']})\n\n"
            
        prompt += f"""
        请选择最多{width}个最有价值的实体继续探索。你的选择应该:
        1. 平衡相关性和覆盖广度
        2. 避免过于相似的实体
        3. 考虑探索多种关系类型的可能性
        4. 优先选择有助于回答问题的实体
        
        要求回复格式:
        ```
        实体: [实体1, 实体2, ...]
        推理: 你的选择理由...
        ```
        """
        
        try:
            # 调用LLM决策
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析结果
            entities_match = re.search(r'实体:\s*\[(.*?)\]', content, re.DOTALL)
            reasoning_match = re.search(r'推理:(.*?)($|```)', content, re.DOTALL)
            
            selected_entities = []
            reasoning = "无具体推理过程"
            
            if entities_match:
                entities_str = entities_match.group(1).strip()
                # 处理实体列表
                if entities_str:
                    # 分割并清理实体
                    entities = [e.strip().strip('"\'') for e in entities_str.split(',')]
                    selected_entities = [e for e in entities if e]
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # 如果没有解析出实体，使用得分最高的几个
            if not selected_entities:
                selected_entities = [n['id'] for n in scored_neighbors[:width]]
                reasoning = "基于相似度得分自动选择"
            
            # 保存到记忆
            self.exploration_memory[memory_key] = {
                "entities": selected_entities,
                "reasoning": reasoning,
                "step": current_step
            }
            
            return selected_entities, reasoning
                
        except Exception as e:
            print(f"LLM决策失败: {e}")
            # 出错时使用简单启发式方法
            fallback_entities = [n['id'] for n in scored_neighbors[:width]]
            fallback_reasoning = "决策过程出错，默认选择得分最高的实体"
            return fallback_entities, fallback_reasoning
    
    def _get_entity_info(self, entities):
        """
        获取实体详细信息
        
        Args:
            entities: 实体ID列表
            
        Returns:
            List: 实体信息列表
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN e.id AS id, e.description AS description,
                   labels(e) AS types
            """
            
            result = self.graph.query(query, params={"entity_ids": entities})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取实体信息失败: {e}")
            return []
    
    def _get_relationship_info(self, entities):
        """
        获取实体关系信息
        
        Args:
            entities: 实体ID列表
            
        Returns:
            List: 关系信息列表
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1.id IN $entity_ids AND e2.id IN $visited_ids
            RETURN startNode(r).id AS source, endNode(r).id AS target,
                   type(r) AS type, r.description AS description,
                   CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1.0 END AS weight
            """
            
            result = self.graph.query(
                query, 
                params={
                    "entity_ids": entities,
                    "visited_ids": list(self.visited_nodes)
                }
            )
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取关系信息失败: {e}")
            return []
    
    def _get_content_info(self, entities):
        """
        获取与实体相关的内容信息
        
        Args:
            entities: 实体ID列表
            
        Returns:
            List: 内容信息列表
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS id, c.text AS text
            LIMIT 20
            """
            
            result = self.graph.query(query, params={"entity_ids": entities})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取内容信息失败: {e}")
            return []
    
    def _get_community_info(self, entities):
        """
        获取实体所属社区信息
        
        Args:
            entities: 实体ID列表
            
        Returns:
            List: 社区信息列表
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS community_id, c.summary AS summary
            """
            
            result = self.graph.query(query, params={"entity_ids": entities})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取社区信息失败: {e}")
            return []
    
    def _rank_content_by_relevance(self, query_embedding, content_list):
        """
        根据与查询的相关性排序内容
        
        Args:
            query_embedding: 查询嵌入向量
            content_list: 内容列表
            
        Returns:
            List: 排序后的内容列表
        """
        if not content_list:
            return []
            
        scored_content = []
        
        for content in content_list:
            text = content.get("text", "")
            
            if not text:
                continue
                
            try:
                # 计算文本嵌入
                text_embedding = self.embeddings.embed_query(text)
                
                # 计算相似度
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(text_embedding).reshape(1, -1)
                )[0][0]
                
                # 添加相似度分数
                scored_item = content.copy()
                scored_item["relevance_score"] = similarity
                scored_content.append(scored_item)
                
            except Exception as e:
                print(f"计算内容相似度失败: {e}")
                scored_content.append(content)
        
        # 按相关性排序
        return sorted(scored_content, key=lambda x: x.get("relevance_score", 0), reverse=True)

    async def explore_async(self, query: str, starting_entities: List[str], max_steps: int = 5):
        """
        异步执行探索过程
        
        Args:
            query: 用户查询
            starting_entities: 起始实体列表
            max_steps: 最大探索步数
            
        Returns:
            Dict: 探索结果
            AsyncGenerator: 进度更新生成器
        """
        async def progress_generator():
            """生成进度更新"""
            yield {"status": "started", "message": "开始探索过程"}
            
            # 等待探索开始
            await asyncio.sleep(0.1)
            
            for step in range(max_steps):
                if step in self.progress_updates:
                    yield self.progress_updates[step]
                    
                await asyncio.sleep(0.5)
                
            # 最终更新
            if "final" in self.progress_updates:
                yield self.progress_updates["final"]
        
        # 初始化进度更新存储
        self.progress_updates = {}
        
        # 创建任务
        exploration_task = asyncio.create_task(self._explore_async_impl(
            query, starting_entities, max_steps
        ))
        
        # 返回结果和进度生成器
        return await exploration_task, progress_generator()
        
    async def _explore_async_impl(self, query, starting_entities, max_steps):
        """异步探索实现"""
        # 包装同步方法
        def sync_explore():
            return self.explore(query, starting_entities, max_steps)
            
        # 更新进度
        self.progress_updates[0] = {
            "status": "exploring", 
            "step": 0,
            "message": f"开始从{len(starting_entities)}个起始实体探索"
        }
        
        # 执行同步探索
        result = await asyncio.get_event_loop().run_in_executor(None, sync_explore)
        
        # 更新最终进度
        self.progress_updates["final"] = {
            "status": "completed",
            "message": f"探索完成，共发现{len(result.get('entities', []))}个实体，{len(result.get('content', []))}条内容"
        }
        
        return result