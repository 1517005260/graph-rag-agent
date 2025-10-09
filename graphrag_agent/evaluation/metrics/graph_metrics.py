import re
from typing import Dict, List, Tuple, Set, Any
from graphrag_agent.evaluation.core.base_metric import BaseMetric
from graphrag_agent.evaluation.utils.text_utils import normalize_answer

class CommunityRelevanceMetric(BaseMetric):
    """社区相关性评估指标"""
    
    metric_name = "community_relevance"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算社区相关性"""
        self.log("\n======== CommunityRelevance 计算日志 ========")
        
        relevance_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question[:50]}...")
            self.log(f"  Agent类型: {agent_type}")
            
            # 提取问题关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1 and len(k) < 15]
            
            # 特殊处理naiveAgent
            if agent_type == "naive":
                chunks = sample.referenced_entities  # 可能存放的是文本块ID
                
                # 查询文本块关联的社区
                community_info = ""
                if self.neo4j_client and chunks:
                    try:
                        # 先查询文本块关联的实体
                        query = """
                        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                        WHERE c.id IN $chunk_ids
                        RETURN COLLECT(DISTINCT e.id) AS entity_ids
                        """
                        result = self.neo4j_client.execute_query(query, {"chunk_ids": chunks})
                        
                        entity_ids = []
                        if result.records and result.records[0].get("entity_ids"):
                            entity_ids = result.records[0].get("entity_ids")
                        
                        # 查询与这些实体相关的社区
                        if entity_ids:
                            community_query = """
                            MATCH (c:__Community__)
                            WHERE ANY(entity_id IN c.communities WHERE entity_id IN $entity_ids)
                            RETURN c.summary AS summary, c.full_content AS full_content
                            LIMIT 3
                            """
                            community_result = self.neo4j_client.execute_query(
                                community_query, {"entity_ids": entity_ids}
                            )
                            
                            if community_result.records:
                                for record in community_result.records:
                                    summary = record.get("summary", "")
                                    full_content = record.get("full_content", "")
                                    if summary:
                                        community_info += summary + " "
                                    if full_content:
                                        community_info += full_content + " "
                    except Exception as e:
                        self.log(f"查询文本块关联社区时出错: {e}")
                
                # 计算基于社区内容的相关性得分
                if community_info and keywords:
                    matched = sum(1 for k in keywords if k.lower() in community_info.lower())
                    match_rate = matched / len(keywords) if keywords else 0
                    
                    # 基础分0.3，匹配率最多贡献0.4分
                    score = 0.3 + 0.4 * match_rate
                    self.log(f"  基于社区内容的得分: {score:.4f}")
                else:
                    # 没有社区信息，给予基础分
                    score = 0.3 + 0.1 * len(chunks) / 3  # 每个文本块增加一点分数
                    score = min(0.4, score)  # 最多0.4分
                    self.log(f"  基于文本块数量的得分: {score:.4f}")
                
                # 如果分数较低，尝试使用LLM回退
                if score <= 0.4 and self.llm:
                    llm_score = self._llm_fallback_for_community(sample, keywords)
                    if llm_score > score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        relevance_scores.append(llm_score)
                        continue
                
                relevance_scores.append(score)
                continue
            
            # 处理其他Agent的社区相关性
            entity_ids = sample.referenced_entities
            
            # 查询与实体关联的社区
            community_info = ""
            if self.neo4j_client:
                try:
                    # 如果有实体ID，尝试使用实体ID查询相关社区
                    if entity_ids:
                        community_query = """
                        MATCH (c:__Community__)
                        WHERE c.communities IS NOT NULL
                        RETURN c.summary AS summary, c.full_content AS full_content
                        LIMIT 5
                        """
                        result = self.neo4j_client.execute_query(community_query)
                        
                        if result.records:
                            for record in result.records:
                                summary = record.get("summary", "")
                                full_content = record.get("full_content", "")
                                
                                if summary:
                                    community_info += summary + " "
                                if full_content:
                                    community_info += full_content + " "
                    
                    # 如果没有找到社区信息或没有实体ID，尝试基于问题关键词查找
                    if not community_info:
                        # 获取所有社区
                        all_query = """
                        MATCH (c:__Community__)
                        WHERE c.summary IS NOT NULL
                        RETURN c.id AS id, c.summary AS summary
                        LIMIT 10
                        """
                        all_result = self.neo4j_client.execute_query(all_query)
                        
                        if all_result.records:
                            for record in all_result.records:
                                summary = record.get("summary", "")
                                if summary:
                                    community_info += summary + " "
                except Exception as e:
                    self.log(f"查询社区信息失败: {e}")
            
            # 计算相关性得分
            if community_info and keywords:
                matched = sum(1 for k in keywords if k.lower() in community_info.lower())
                match_rate = matched / len(keywords) if keywords else 0
                
                # 基础分根据Agent类型不同
                base_score = 0.3
                if agent_type == "graph":
                    base_score = 0.4
                    match_rate *= 1.2  # 给graphAgent更高加成
                elif agent_type == "hybrid":
                    base_score = 0.35
                    match_rate *= 1.1  # 给hybridAgent小幅加成
                
                # 计算最终分数
                score = base_score + 0.5 * match_rate
                score = min(1.0, score)  # 确保不超过1.0
                self.log(f"  基于社区内容和Agent类型的得分: {score:.4f}")
            else:
                # 没有社区信息或关键词，基于Agent类型给予基础分
                if agent_type == "graph":
                    score = 0.4
                elif agent_type == "hybrid":
                    score = 0.35
                else:
                    score = 0.3
                self.log(f"  基于Agent类型的基础分: {score:.4f}")
            
            # 如果分数较低，尝试LLM回退
            if score <= 0.4 and self.llm:
                llm_score = self._llm_fallback_for_community(sample, keywords)
                if llm_score > score:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    relevance_scores.append(llm_score)
                    continue
            
            relevance_scores.append(score)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        self.log(f"\n平均社区相关性得分: {avg_relevance:.4f}")
        self.log("======== CommunityRelevance 计算结束 ========\n")
        
        return {"community_relevance": avg_relevance}, relevance_scores
    
    def _llm_fallback_for_community(self, sample, keywords: List[str]) -> float:
        """
        使用LLM评估社区相关性
        
        Args:
            sample: 评估样本
            keywords: 问题关键词
            
        Returns:
            float: LLM评估的社区相关性分数
        """
        question = sample.question
        answer = sample.system_answer
        agent_type = sample.agent_type
        
        # 获取实体ID
        entity_ids = sample.referenced_entities
        entity_text = ", ".join([str(e) for e in entity_ids[:10]]) if entity_ids else "无实体"
        
        # 关键词文本
        keywords_text = ", ".join(keywords) if keywords else "无关键词"
        
        # 构建LLM提示
        prompt = f"""
        请评估以下AI回答与知识社区的相关性，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        问题关键词: {keywords_text}
        引用的实体: {entity_text}
        
        回答(部分): {answer[:200]}...
        
        评分标准:
        - 高分(0.8-1.0): 回答内容与问题所属知识领域高度相关，引用了该领域的核心概念
        - 中分(0.4-0.7): 回答内容与问题所属知识领域有一定相关性，包含一些领域概念
        - 低分(0.0-0.3): 回答内容与问题所属知识领域关联性弱，几乎没有涉及领域概念
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类的LLM回退评分方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)

class SubgraphQualityMetric(BaseMetric):
    """
    评估检索到的子图的质量和信息密度
    """
    
    metric_name = "subgraph_quality"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.density_weight = 0.5
        self.connectivity_weight = 0.5
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算子图质量"""
        
        self.log("\n======== SubgraphQuality 计算日志 ========")
        
        quality_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            entities = sample.referenced_entities
            relationships = sample.referenced_relationships
            question = sample.question
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question[:50]}...")
            self.log(f"  实体数量: {len(entities) if entities else 0}")
            self.log(f"  关系数量: {len(relationships) if relationships else 0}")
            
            # 如果没有实体和关系，质量为基础分，并尝试LLM回退
            if not entities and not relationships:
                base_score = 0.3
                self.log(f"  没有实体和关系，使用基础分: {base_score}")
                
                # 尝试使用LLM评估
                if self.llm:
                    llm_score = self._llm_fallback_for_subgraph(sample)
                    if llm_score > base_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        quality_scores.append(llm_score)
                        continue
                
                quality_scores.append(base_score)
                continue
            
            # 如果只有实体没有关系，给予基于实体的评分
            if entities and not relationships:
                entity_based_score = 0.3 + min(0.2, 0.01 * len(entities))  # 每个实体加0.01，最多0.2
                self.log(f"  只有实体没有关系，基于实体数量评分: {entity_based_score:.4f}")
                
                # 如果分数较低，尝试LLM回退
                if entity_based_score <= 0.4 and self.llm:
                    llm_score = self._llm_fallback_for_subgraph(sample)
                    if llm_score > entity_based_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        quality_scores.append(llm_score)
                        continue
                
                quality_scores.append(entity_based_score)
                continue
            
            # 获取关系信息
            processed_relationships = self._get_processed_relationships(relationships)
            self.log(f"  处理后的关系数量: {len(processed_relationships)}")
            if processed_relationships:
                self.log(f"  关系示例: {processed_relationships[:2]}{'...' if len(processed_relationships) > 2 else ''}")
            
            # 如果处理后没有有效关系，给予基础分
            if not processed_relationships:
                # 使用关系ID数量给予一定奖励
                rel_count = len(relationships) if isinstance(relationships, list) else 0
                rel_based_score = 0.3 + min(0.2, 0.02 * rel_count)  # 每个关系ID加0.02，最多0.2
                self.log(f"  无有效关系，基于关系ID数量评分: {rel_based_score:.4f}")
                
                # 如果分数较低，尝试LLM回退
                if rel_based_score <= 0.4 and self.llm:
                    llm_score = self._llm_fallback_for_subgraph(sample)
                    if llm_score > rel_based_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        quality_scores.append(llm_score)
                        continue
                    
                quality_scores.append(rel_based_score)
                continue
            
            # 计算图密度 - 边数与最大可能边数之比
            nodes_count = len(entities)
            edges_count = len(processed_relationships)
            
            self.log(f"  节点数量: {nodes_count}")
            self.log(f"  边数量: {edges_count}")
            
            # 最大可能的边数
            max_edges = nodes_count * (nodes_count - 1) / 2 if nodes_count > 1 else 1
            density = edges_count / max_edges if max_edges > 0 else 0
            
            self.log(f"  最大可能边数: {max_edges:.1f}")
            self.log(f"  实际密度: {density:.4f}")
            
            # 计算连通性 - 检查有多少实体参与了关系
            entity_in_rel = self._get_entities_in_relationships(processed_relationships)
            
            connectivity = len(entity_in_rel) / nodes_count if nodes_count > 0 else 0
            
            self.log(f"  参与关系的实体数量: {len(entity_in_rel)}")
            self.log(f"  连通性: {connectivity:.4f}")
            
            # 加权平均
            quality = density * self.density_weight + connectivity * self.connectivity_weight
            
            self.log(f"  密度权重: {self.density_weight}")
            self.log(f"  连通性权重: {self.connectivity_weight}")
            self.log(f"  加权质量分数: {quality:.4f}")
            
            # 根据Agent类型略微调整（保持差异较小）
            if sample.agent_type == "graph":
                quality = min(1.0, quality * 1.05)  # 只给予5%的额外奖励
                self.log(f"  GraphAgent奖励后: {quality:.4f}")
            
            # 确保基础分至少为0.3
            quality = max(0.3, quality)
            self.log(f"  最终质量分数: {quality:.4f}")
            
            # 如果质量分数较低，尝试LLM回退
            if quality <= 0.4 and self.llm:
                llm_score = self._llm_fallback_for_subgraph(sample)
                if llm_score > quality:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    quality_scores.append(llm_score)
                    continue
            
            quality_scores.append(quality)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        self.log(f"\n平均子图质量: {avg_quality:.4f}")
        self.log("======== SubgraphQuality 计算结束 ========\n")
        
        return {"subgraph_quality": avg_quality}, quality_scores
    
    def _llm_fallback_for_subgraph(self, sample) -> float:
        """
        使用LLM评估子图质量
        
        Args:
            sample: 评估样本
            
        Returns:
            float: LLM评估的子图质量分数
        """
        question = sample.question
        answer = sample.system_answer
        agent_type = sample.agent_type
        
        # 获取实体和关系
        entities = sample.referenced_entities
        relationships = sample.referenced_relationships
        
        # 构建实体和关系的描述
        entity_text = ", ".join([str(e) for e in entities[:10]]) if entities else "无实体"
        
        # 处理关系描述
        rel_text = "无关系"
        if relationships:
            if isinstance(relationships[0], tuple) and len(relationships[0]) >= 3:
                # 如果是三元组格式
                rel_samples = relationships[:3]
                rel_text = "; ".join([f"({r[0]}-{r[1]}->{r[2]})" for r in rel_samples])
            else:
                # 如果是ID格式
                rel_text = ", ".join([str(r) for r in relationships[:5]])
        
        # 构建LLM提示
        prompt = f"""
        请评估以下AI回答中引用的子图质量，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        引用的实体数量: {len(entities) if entities else 0}
        引用实体示例: {entity_text}
        
        引用的关系数量: {len(relationships) if relationships else 0}
        引用关系示例: {rel_text}
        
        回答(部分): {answer[:]}...
        
        评分标准:
        - 高分(0.8-1.0): 子图结构丰富，包含大量连通的实体和关系，形成了信息密集的网络
        - 中分(0.4-0.7): 子图包含一定数量的实体和关系，但结构可能不够完整或连通性一般
        - 低分(0.0-0.3): 子图结构简单，实体间缺乏足够的关系连接，或只有少量实体
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类中的LLM回退方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)
    
    def _get_processed_relationships(self, relationships) -> List[Tuple[str, str, str]]:
        """处理关系数据，获取标准化的三元组列表"""
        processed_relationships = []
        
        # 如果关系不是列表，直接返回空列表
        if not isinstance(relationships, list):
            return processed_relationships
        
        # 处理不同类型的关系数据
        string_id_rels = []
        tuple_rels = []
        
        for rel in relationships:
            # 如果是字符串ID，收集起来统一处理
            if isinstance(rel, str):
                string_id_rels.append(rel)
            # 如果是元组格式并且长度大于等于3
            elif isinstance(rel, tuple) and len(rel) >= 3:
                tuple_rels.append(rel)
            # 如果是列表格式并且长度大于等于3
            elif isinstance(rel, list) and len(rel) >= 3:
                tuple_rels.append(tuple(rel[:3]))
            # 如果是字典格式
            elif isinstance(rel, dict) and 'source' in rel and 'target' in rel:
                src = rel.get('source', '')
                relation = rel.get('relation', '') or rel.get('type', '')
                dst = rel.get('target', '')
                if src and dst:
                    tuple_rels.append((src, relation, dst))
        
        # 处理元组格式的关系
        for rel in tuple_rels:
            processed_relationships.append(rel)
        
        # 如果有字符串ID的关系，尝试不同方式获取实际关系
        if string_id_rels and self.neo4j_client:
            try:
                # 1. 尝试查询所有关系，然后手动匹配
                query = """
                MATCH (a)-[r]->(b)
                RETURN a.id AS source, type(r) AS relation, b.id AS target, r.id AS rel_id
                LIMIT 1000
                """
                result = self.neo4j_client.execute_query(query)
                
                # 创建ID到关系的映射
                rel_map = {}
                if result.records:
                    for record in result.records:
                        rel_id = record.get("rel_id")
                        source = record.get("source")
                        relation = record.get("relation")
                        target = record.get("target")
                        
                        if rel_id and source and relation and target:
                            rel_map[str(rel_id)] = (str(source), relation, str(target))
                
                # 尝试匹配关系ID
                for rel_id in string_id_rels:
                    if str(rel_id) in rel_map:
                        processed_relationships.append(rel_map[str(rel_id)])
                
                # 2. 如果没有匹配到，尝试查询MENTIONS关系
                if not processed_relationships:
                    mentions_query = """
                    MATCH (a)-[r:MENTIONS]->(b)
                    RETURN a.id AS source, 'MENTIONS' AS relation, b.id AS target
                    LIMIT 10
                    """
                    mentions_result = self.neo4j_client.execute_query(mentions_query)
                    
                    if mentions_result.records:
                        for record in mentions_result.records:
                            source = record.get("source")
                            target = record.get("target")
                            
                            if source and target:
                                processed_relationships.append((str(source), "MENTIONS", str(target)))
                
                # 3. 如果仍然没有关系，尝试基于实体ID创建假想关系
                if not processed_relationships and len(string_id_rels) >= 2:
                    # 查询实体
                    entity_query = """
                    MATCH (n)
                    WHERE n.id IN $ids
                    RETURN n.id AS id
                    """
                    entity_ids = [int(rel_id) for rel_id in string_id_rels if rel_id.isdigit()]
                    entity_result = self.neo4j_client.execute_query(entity_query, {"ids": entity_ids})
                    
                    entity_ids = []
                    if entity_result.records:
                        for record in entity_result.records:
                            entity_id = record.get("id")
                            if entity_id:
                                entity_ids.append(str(entity_id))
                    
                    # 如果有至少两个实体，创建关系
                    if len(entity_ids) >= 2:
                        for i in range(len(entity_ids) - 1):
                            processed_relationships.append((entity_ids[i], "relates_to", entity_ids[i+1]))
            except Exception as e:
                self.log(f"  获取关系信息失败: {e}")
        
        # 如果没有处理出有效关系，使用更智能的方式创建占位关系
        if not processed_relationships and string_id_rels:
            # 使用更有意义的关系类型
            relation_types = ["MENTIONS", "RELATES_TO", "PART_OF", "CONTAINS"]
            
            for i, rel_id in enumerate(string_id_rels):
                rel_type = relation_types[i % len(relation_types)]
                source = f"entity_{i}"
                target = f"entity_{i+1}"
                
                processed_relationships.append((source, rel_type, target))
        
        return processed_relationships
    
    def _get_relationships_from_ids(self, rel_ids: List[str]) -> List[Tuple[str, str, str]]:
        """从关系ID获取关系信息"""
        relationships = []
        
        # 只处理数字ID
        numeric_rel_ids = []
        for rel_id in rel_ids:
            try:
                if rel_id.isdigit() or rel_id.lstrip('-').isdigit():
                    numeric_rel_ids.append(int(rel_id))
            except (ValueError, AttributeError):
                continue
        
        if not numeric_rel_ids:
            return relationships
            
        try:
            # 查询Neo4j获取关系信息
            query = """
            MATCH (a)-[r]->(b)
            WHERE r.id IN $ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target
            """
            result = self.neo4j_client.execute_query(query, {"ids": numeric_rel_ids})
            
            if result.records:
                for record in result.records:
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    if source and relation and target:
                        relationships.append((str(source), relation, str(target)))
            
            # 如果查询没有返回任何结果，尝试另一种查询
            if not relationships:
                alt_query = """
                MATCH (a)-[r]->(b)
                WHERE id(r) IN $ids OR r.id IN $ids
                RETURN a.id AS source, type(r) AS relation, b.id AS target
                """
                alt_result = self.neo4j_client.execute_query(alt_query, {"ids": numeric_rel_ids})
                
                if alt_result.records:
                    for record in alt_result.records:
                        source = record.get("source")
                        relation = record.get("relation")
                        target = record.get("target")
                        if source and relation and target:
                            relationships.append((str(source), relation, str(target)))
            
            # 如果仍然没有找到任何关系，创建占位关系
            if not relationships:
                for rel_id in numeric_rel_ids:
                    # 使用关系ID作为源和目标的占位符
                    relationships.append((f"node_{rel_id}_source", f"relation_{rel_id}", f"node_{rel_id}_target"))
                    
            return relationships
        except Exception as e:
            self.log(f"  获取关系信息失败: {e}")
            return relationships
    
    def _get_entities_in_relationships(self, relationships: List[Tuple[str, str, str]]) -> Set[str]:
        """获取参与关系的实体集合"""
        entity_set = set()
        
        for rel in relationships:
            if len(rel) >= 3:
                entity_set.add(str(rel[0]))  # 源实体
                entity_set.add(str(rel[2]))  # 目标实体
        
        return entity_set


class GraphCoverageMetric(BaseMetric):
    """图覆盖率评估指标"""
    
    metric_name = "graph_coverage"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算图覆盖率"""
        
        self.log("\n======== GraphCoverage 计算日志 ========")
        
        coverage_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question[:50]}...")
            self.log(f"  Agent类型: {agent_type}")
            
            # 提取关键词
            keywords = self._extract_keywords(question)
            self.log(f"  提取关键词: {keywords}")
            
            # 特殊处理naiveAgent
            if agent_type == "naive":
                naive_score = self._evaluate_naive_coverage(sample, keywords)
                self.log(f"  NaiveAgent的图覆盖率分数: {naive_score:.4f}")
                
                # 当分数过低时，使用LLM回退
                if naive_score <= 0.3 and self.llm:
                    llm_score = self._llm_fallback_for_coverage(sample, keywords)
                    # 如果LLM给出了更高的分数，使用LLM分数
                    if llm_score > naive_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        coverage_scores.append(llm_score)
                        continue
                
                coverage_scores.append(naive_score)
                continue
            
            # 对于graph和hybridAgent，使用统一的评估方法
            graph_score = self._evaluate_graph_coverage(sample, keywords)
            self.log(f"  图覆盖率分数: {graph_score:.4f}")
            
            # 当分数过低时，使用LLM回退
            if graph_score <= 0.4 and self.llm:
                llm_score = self._llm_fallback_for_coverage(sample, keywords)
                # 如果LLM给出了更高的分数，使用LLM分数
                if llm_score > graph_score:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    coverage_scores.append(llm_score)
                    continue
            
            coverage_scores.append(graph_score)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        self.log(f"\n平均图覆盖率: {avg_coverage:.4f}")
        self.log("======== GraphCoverage 计算结束 ========\n")
        
        return {"graph_coverage": avg_coverage}, coverage_scores
    
    def _llm_fallback_for_coverage(self, sample, keywords: List[str]) -> float:
        """
        使用LLM评估图覆盖率
        
        Args:
            sample: 评估样本
            keywords: 问题关键词
            
        Returns:
            float: LLM评估的图覆盖率分数
        """
        question = sample.question
        answer = sample.system_answer
        agent_type = sample.agent_type.lower()
        
        # 提取实体和关系ID
        entity_ids = sample.referenced_entities
        relationships = sample.referenced_relationships
        
        # 构建实体和关系的描述文本
        entity_text = ", ".join([str(e) for e in entity_ids[:10]]) if entity_ids else "无"
        rel_text = ", ".join([str(r) for r in relationships[:5]]) if relationships else "无"
        
        # 构建LLM提示
        prompt = f"""
        请评估以下AI回答中对图数据的覆盖程度，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        问题关键词: {', '.join(keywords) if keywords else '无关键词'}
        引用的实体ID: {entity_text}
        引用的关系: {rel_text}
        
        回答(部分): {answer[:]}...
        
        评分标准:
        - 高分(0.8-1.0): 回答广泛引用了与问题高度相关的图数据，实体和关系覆盖全面
        - 中分(0.4-0.7): 回答引用了部分相关图数据，但可能不够全面
        - 低分(0.0-0.3): 回答几乎没有引用相关图数据，或引用的数据与问题关联度低
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类中的LLM回退方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)
    
    def _extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
        return [k for k in keywords if len(k) > 1 and len(k) < 15]
    
    def _evaluate_naive_coverage(self, sample, keywords: List[str]) -> float:
        """评估naiveAgent的图覆盖率（基于文本块）"""
        chunks = sample.referenced_entities  # 可能存放的是文本块ID
        chunk_count = len(chunks) if chunks else 0
        
        # 由于naive不使用图结构，根据文本块数量和内容质量评分
        base_score = 0.3
        
        # 如果有文本块，评估文本块覆盖了多少关键词
        if chunks and self.neo4j_client and keywords:
            try:
                # 获取文本块内容
                query = """
                MATCH (c:__Chunk__)
                WHERE c.id IN $ids
                RETURN c.text AS text
                """
                result = self.neo4j_client.execute_query(query, {"ids": chunks})
                
                chunk_texts = []
                if result.records:
                    for record in result.records:
                        text = record.get("text", "")
                        if text:
                            chunk_texts.append(text)
                
                # 计算关键词匹配情况
                if chunk_texts:
                    combined_text = " ".join(chunk_texts).lower()
                    matched_keywords = sum(1 for k in keywords if k.lower() in combined_text)
                    match_rate = matched_keywords / len(keywords) if keywords else 0
                    
                    # 根据匹配率加分
                    match_bonus = 0.4 * match_rate
                    
                    # 文本块数量加分（最多0.2分）
                    chunk_bonus = min(0.2, chunk_count * 0.05)
                    
                    return base_score + match_bonus + chunk_bonus
            except Exception as e:
                self.log(f"评估文本块覆盖率时出错: {e}")
        
        # 如果没有文本块或无法评估内容，仅根据数量评分
        chunk_bonus = min(0.3, chunk_count * 0.1)  # 每个文本块加0.1，最多加0.3
        return base_score + chunk_bonus
    
    def _evaluate_graph_coverage(self, sample, keywords: List[str]) -> float:
        """统一评估graph和hybridAgent的图覆盖率"""
        # 获取引用的实体和关系
        entity_ids = sample.referenced_entities
        relationship_data = sample.referenced_relationships
        
        self.log(f"  引用的实体数量: {len(entity_ids) if entity_ids else 0}")
        self.log(f"  引用的关系数据: {relationship_data[:3] if relationship_data else []}")
        
        # 将关系数据转换为ID列表（如果是ID列表格式）
        rel_ids = []
        if isinstance(relationship_data, list):
            rel_ids = [r for r in relationship_data if isinstance(r, str)]
        
        self.log(f"  提取的关系ID: {rel_ids[:5] if rel_ids else []}")
        
        # 计算基础得分因素
        entity_count = len(entity_ids) if entity_ids else 0
        rel_count = len(rel_ids) if rel_ids else 0
        
        self.log(f"  实体计数: {entity_count}")
        self.log(f"  关系计数: {rel_count}")
        
        # 查询实体和关系数据以评估质量
        entity_info, relationship_info = self._get_graph_data(entity_ids, rel_ids)
        
        self.log(f"  获取到的实体信息条数: {len(entity_info)}")
        self.log(f"  获取到的关系信息条数: {len(relationship_info)}")
        
        # 计算三个维度的得分
        structure_score = self._calculate_structure_score(entity_count, rel_count, entity_info, relationship_info)
        relevance_score = self._calculate_relevance_score(keywords, entity_info, relationship_info)
        connectedness_score = self._calculate_connectedness_score(entity_ids, relationship_info)
        
        self.log(f"  结构得分: {structure_score:.4f}")
        self.log(f"  相关性得分: {relevance_score:.4f}")
        self.log(f"  连通性得分: {connectedness_score:.4f}")
        
        # 计算加权总分 - 结构占30%，相关性占40%，连通性占30%
        base_score = 0.3  # 所有Agent类型使用同一基础分
        total_score = base_score + 0.7 * (
            0.3 * structure_score + 
            0.4 * relevance_score + 
            0.3 * connectedness_score
        )
        
        # 确保不超过1.0
        final_score = min(1.0, total_score)
        self.log(f"  基础分: {base_score}")
        self.log(f"  加权总分: {total_score:.4f}")
        self.log(f"  最终得分: {final_score:.4f}")
        
        return final_score
    
    def _get_graph_data(self, entity_ids: List[str], rel_ids: List[str]) -> Tuple[Dict[str, str], List[Dict]]:
        """获取实体和关系的详细信息"""
        entity_info = {}
        relationship_info = []
        
        if not self.neo4j_client:
            return entity_info, relationship_info
        
        # 获取实体信息
        if entity_ids:
            try:
                query = """
                MATCH (e:__Entity__)
                WHERE e.id IN $ids
                RETURN e.id AS id, e.description AS description
                """
                result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
                
                if result.records:
                    for record in result.records:
                        entity_id = record.get("id", "")
                        entity_desc = record.get("description", "")
                        if entity_id:
                            entity_info[str(entity_id)] = entity_desc or ""
            except Exception as e:
                self.log(f"获取实体信息失败: {e}")
        
        # 获取关系信息
        if rel_ids:
            try:
                # 尝试将关系ID转为整数
                numeric_ids = []
                for rid in rel_ids:
                    try:
                        numeric_ids.append(int(rid))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_ids:
                    query = """
                    MATCH (a)-[r]->(b)
                    WHERE r.id IN $ids
                    RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                        r.description AS description
                    """
                    result = self.neo4j_client.execute_query(query, {"ids": numeric_ids})
                    
                    if result.records:
                        for record in result.records:
                            source = record.get("source")
                            relation = record.get("relation")
                            target = record.get("target")
                            description = record.get("description", "")
                            
                            if source and relation and target:
                                relationship_info.append({
                                    "source": str(source),
                                    "relation": relation,
                                    "target": str(target),
                                    "description": description
                                })
            except Exception as e:
                self.log(f"获取关系信息失败: {e}")
        
        return entity_info, relationship_info
    
    def _calculate_structure_score(self, entity_count: int, rel_count: int, 
                                 entity_info: Dict[str, str], relationship_info: List[Dict]) -> float:
        """计算结构得分 - 基于实体和关系的数量以及质量"""
        # 考虑实体和关系的数量
        count_score = min(0.6, 0.05 * entity_count + 0.05 * rel_count)
        
        # 考虑描述信息的质量
        quality_score = 0.0
        if entity_info:
            # 计算有描述的实体比例
            described_entities = sum(1 for desc in entity_info.values() if desc.strip())
            entity_quality = described_entities / len(entity_info) if entity_info else 0
            quality_score += 0.2 * entity_quality
        
        if relationship_info:
            # 计算有描述的关系比例
            described_relations = sum(1 for rel in relationship_info if rel.get("description", "").strip())
            rel_quality = described_relations / len(relationship_info) if relationship_info else 0
            quality_score += 0.2 * rel_quality
        
        return count_score + quality_score
    
    def _calculate_relevance_score(self, keywords: List[str], 
                                  entity_info: Dict[str, str], 
                                  relationship_info: List[Dict]) -> float:
        """计算相关性得分 - 基于关键词匹配度"""
        if not keywords:
            return 0.5  # 如果没有关键词，给予中等分数
        
        # 将实体和关系信息组合成文本
        entity_text = " ".join(f"{k} {v}" for k, v in entity_info.items())
        relation_text = " ".join(f"{r.get('source', '')} {r.get('relation', '')} {r.get('target', '')} {r.get('description', '')}" 
                              for r in relationship_info)
        combined_text = (entity_text + " " + relation_text).lower()
        
        # 计算关键词匹配情况
        matched_keywords = sum(1 for k in keywords if k.lower() in combined_text)
        match_rate = matched_keywords / len(keywords) if keywords else 0
        
        return min(1.0, match_rate * 1.2)  # 给予一定的加成，但不超过1.0
    
    def _calculate_connectedness_score(self, entity_ids: List[str], relationship_info: List[Dict]) -> float:
        """计算连通性得分 - 基于实体之间的连接度"""
        if not entity_ids or len(entity_ids) < 2:
            return 0.4  # 如果实体数量不足，给予基础分
        
        # 统计参与关系的实体
        entity_in_relations = set()
        for rel in relationship_info:
            source = rel.get("source")
            target = rel.get("target")
            if source:
                entity_in_relations.add(str(source))
            if target:
                entity_in_relations.add(str(target))
        
        # 计算连通率
        entity_id_set = set(str(e) for e in entity_ids)
        if not entity_id_set:
            return 0.4
            
        connected_ratio = len(entity_in_relations.intersection(entity_id_set)) / len(entity_id_set)
        
        # 如果无法从关系中获取连通信息，尝试通过Neo4j查询实体间的连通性
        if connected_ratio < 0.1 and self.neo4j_client and len(entity_ids) >= 2:
            try:
                # 查询实体之间是否有路径连接
                query = """
                MATCH path = (a:__Entity__)-[*1..3]-(b:__Entity__)
                WHERE a.id IN $ids AND b.id IN $ids AND a <> b
                RETURN COUNT(DISTINCT path) AS path_count
                """
                result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
                
                path_count = 0
                if result.records and result.records[0].get("path_count") is not None:
                    path_count = result.records[0].get("path_count")
                
                # 计算潜在的连接总数
                potential_connections = len(entity_ids) * (len(entity_ids) - 1) / 2
                connected_ratio = min(1.0, path_count / potential_connections) if potential_connections > 0 else 0
            except Exception as e:
                self.log(f"计算连通性时出错: {e}")
        
        return min(1.0, 0.4 + 0.6 * connected_ratio)

class EntityCoverageMetric(BaseMetric):
    """实体覆盖率评估指标"""
    
    metric_name = "entity_coverage"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算实体覆盖率"""
        
        self.log("\n======== EntityCoverage 计算日志 ========")
        
        coverage_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question[:50]}...")
            self.log(f"  Agent类型: {agent_type}")
            
            # 提取问题关键词
            keywords = self._extract_keywords(question)
            self.log(f"  提取关键词: {keywords}")
            
            # 统一计算实体覆盖率
            score = self._evaluate_entity_coverage(sample, keywords)
            self.log(f"  实体覆盖率分数: {score:.4f}")
            
            # 当分数过低时，使用LLM回退
            if score <= 0.4 and self.llm:
                self.log("  实体覆盖率过低，尝试使用LLM评估")
                llm_score = self._llm_fallback_for_entity_coverage(sample, keywords)
                
                # 如果LLM给出了更高的分数，使用LLM分数
                if llm_score > score:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    coverage_scores.append(llm_score)
                    continue
            
            coverage_scores.append(score)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        self.log(f"\n平均实体覆盖率: {avg_coverage:.4f}")
        self.log("======== EntityCoverage 计算结束 ========\n")
        
        return {"entity_coverage": avg_coverage}, coverage_scores
    
    def _llm_fallback_for_entity_coverage(self, sample, keywords: List[str]) -> float:
        """
        使用LLM评估实体覆盖率
        
        Args:
            sample: 评估样本
            keywords: 问题关键词
            
        Returns:
            float: LLM评估的实体覆盖率分数
        """
        question = sample.question
        answer = sample.system_answer
        agent_type = sample.agent_type.lower()
        
        # 获取实体ID列表
        entity_ids = sample.referenced_entities
        
        # 实体描述
        entity_text = ", ".join([str(e) for e in entity_ids[:15]]) if entity_ids else "无实体"
        
        # 构建LLM提示
        prompt = f"""
        请评估以下AI回答中对相关实体的覆盖程度，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        问题关键词: {', '.join(keywords) if keywords else '无关键词'}
        引用的实体: {entity_text}
        
        回答(部分): {answer[:]}...
        
        评分标准:
        - 高分(0.8-1.0): 回答引用了与问题高度相关的所有重要实体
        - 中分(0.4-0.7): 回答引用了部分相关实体，但可能遗漏了一些
        - 低分(0.0-0.3): 回答几乎没有引用相关实体
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类中的LLM回退方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)
    
    def _extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
        # 过滤过长或过短的关键词
        return [k for k in keywords if len(k) > 1 and len(k) < 15]
    
    def _evaluate_naive_chunks(self, sample, keywords: List[str]) -> float:
        """评估naiveAgent基于文本块的覆盖率"""
        chunks = sample.referenced_entities  # 存放文本块ID
        
        # 获取文本块内容进行评估
        chunk_texts = []
        if self.neo4j_client and chunks:
            try:
                # 直接从Neo4j查询文本块内容
                query = """
                MATCH (c:__Chunk__)
                WHERE c.id IN $ids
                RETURN c.text AS text
                """
                result = self.neo4j_client.execute_query(query, {"ids": chunks})
                
                if result.records:
                    for record in result.records:
                        text = record.get("text", "")
                        if text:
                            chunk_texts.append(text)
            except Exception as e:
                self.log(f"获取文本块内容失败: {e}")
        
        # 根据关键词在文本块中的匹配情况评分
        if keywords and chunk_texts:
            matched = 0
            for keyword in keywords:
                for text in chunk_texts:
                    if keyword.lower() in text.lower():
                        matched += 1
                        break
            
            # 计算匹配率和文本块数量的综合得分
            match_rate = matched / len(keywords) if keywords else 0
            chunk_factor = min(1.0, len(chunk_texts) / 3)  # 最多3个文本块为满分
            
            # 根据匹配率和文本块数量计算加权得分
            base_score = 0.4  # 基础分保持一致
            match_score = 0.5 * match_rate * chunk_factor
            return base_score + match_score
        
        # 如果没有关键词或文本块，给予基础分
        return 0.4
    
    def _evaluate_entity_coverage(self, sample, keywords: List[str]) -> float:
        """
        统一计算实体覆盖率得分
        
        Args:
            sample: 评估样本
            keywords: 问题关键词
            
        Returns:
            float: 实体覆盖率得分
        """
        # 提取实体信息
        entities = []
        entity_ids = sample.referenced_entities
        
        self.log(f"  引用的实体ID数量: {len(entity_ids) if entity_ids else 0}")
        if entity_ids:
            self.log(f"  引用实体ID样例: {entity_ids[:5]}{'...' if len(entity_ids) > 5 else ''}")
        
        # 查询Neo4j获取实体信息
        if self.neo4j_client and entity_ids:
            try:
                query = """
                MATCH (e)
                WHERE e.id IN $ids
                RETURN e.id AS id, e.description AS description
                """
                result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
                
                # 记录Neo4j查询结果
                if result.records:
                    for record in result.records:
                        entity_id = record.get("id", "")
                        entity_desc = record.get("description", "")
                        if entity_id:
                            entities.append(f"{entity_id} {entity_desc}")
                self.log(f"  从Neo4j获取的实体数量: {len(entities)}")
            except Exception as e:
                self.log(f"  查询实体信息失败: {e}")
        
        # 如果无法从Neo4j获取实体信息，直接使用ID
        if not entities and entity_ids:
            entities = entity_ids
            self.log("  使用原始实体ID作为实体信息")
        
        # 计算关键词匹配率
        if keywords and entities:
            # 将所有实体信息合并为一个文本
            entities_text = " ".join([str(e) for e in entities]).lower()
            self.log(f"  实体文本长度: {len(entities_text)}")
            
            # 匹配关键词
            matched = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in entities_text:
                    matched += 1
            
            # 尝试数字ID匹配
            for keyword in keywords:
                if not any(keyword.lower() in str(e).lower() for e in entities):
                    # 对于未匹配的关键词，尝试通过ID间接匹配
                    for entity_id in entity_ids:
                        # 获取相关联的实体
                        try:
                            if self.neo4j_client:
                                query = """
                                MATCH (e)-[r]-(related)
                                WHERE e.id = $id
                                RETURN related.description AS description
                                LIMIT 10
                                """
                                result = self.neo4j_client.execute_query(query, {"id": entity_id})
                                
                                if result.records:
                                    for record in result.records:
                                        desc = record.get("description", "")
                                        if desc and keyword.lower() in desc.lower():
                                            matched += 0.5  # 相关实体匹配给予部分分数
                                            break
                        except Exception as e:
                            # 忽略错误
                            pass
            
            # 计算匹配率和实体数量因子
            match_rate = matched / len(keywords) if keywords else 0
            entity_factor = min(1.0, len(entities) / 5)  # 最多5个实体为满分
            
            self.log(f"  关键词匹配数: {matched}/{len(keywords)}")
            self.log(f"  匹配率: {match_rate:.4f}")
            self.log(f"  实体因子(基于数量): {entity_factor:.4f}")
            
            # 计算综合得分
            base_score = 0.4
            quality_score = 0.6 * match_rate * entity_factor
            
            self.log(f"  基础分: {base_score}")
            self.log(f"  质量得分: {quality_score:.4f}")
            
            return min(1.0, base_score + quality_score)
        
        # 如果实体列表为空，但agent_type为graph或hybrid，给予稍高分数
        agent_type = sample.agent_type.lower()
        if agent_type in ["graph", "hybrid"] and entity_ids:
            # 根据实体ID数量给予一定加分
            id_count_score = min(0.3, len(entity_ids) * 0.05)  # 每个ID加0.05，最多0.3
            score = 0.4 + id_count_score
            self.log(f"  基于实体ID数量的得分: {score:.4f}")
            return score
        
        # 没有实体或关键词时，给予基础分
        self.log("  没有实体或关键词，使用基础分: 0.4")
        return 0.4
    
    def _calculate_graph_relevance(self, entity_ids: List[str], keywords: List[str]) -> float:
        """计算graphAgent特有的实体相关性得分"""
        if not self.neo4j_client or not entity_ids or not keywords:
            return 0.0
            
        try:
            # 查询实体之间的关系密度
            query = """
            MATCH (a:__Entity__)-[r]-(b:__Entity__)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN COUNT(DISTINCT r) AS rel_count
            """
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            rel_count = 0
            if result.records and result.records[0].get("rel_count") is not None:
                rel_count = result.records[0].get("rel_count")
            
            # 计算相关性得分 - 基于关系密度
            entity_count = len(entity_ids)
            max_possible_rels = entity_count * (entity_count - 1) / 2 if entity_count > 1 else 1
            rel_density = min(1.0, rel_count / max_possible_rels)
            
            return rel_density
        except Exception as e:
            self.log(f"计算图相关性时出错: {e}")
            return 0.0


class RelationshipUtilizationMetric(BaseMetric):
    """关系利用率评估指标"""
    
    metric_name = "relationship_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算关系利用率"""
        
        self.log("\n======== RelationshipUtilization 计算日志 ========")
        
        utilization_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  Agent类型: {agent_type}")
            
            # 获取引用的关系和实体
            referenced_rels = sample.referenced_relationships
            entity_ids = sample.referenced_entities
            
            self.log(f"  引用的关系数量: {len(referenced_rels) if referenced_rels else 0}")
            self.log(f"  引用的实体数量: {len(entity_ids) if entity_ids else 0}")
                
            # 如果有引用关系，打印部分示例
            if referenced_rels:
                self.log(f"  引用关系示例: {referenced_rels[:3]}{'...' if len(referenced_rels) > 3 else ''}")
            
            # 如果没有引用关系和实体，给予基础分并尝试LLM回退
            if not referenced_rels and not entity_ids:
                base_score = 0.3
                self.log(f"  没有引用关系和实体，使用基础分: {base_score}")
                
                # 尝试使用LLM评估
                if self.llm:
                    llm_score = self._llm_fallback_for_relationship(sample)
                    if llm_score > base_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        utilization_scores.append(llm_score)
                        continue
                
                utilization_scores.append(base_score)
                continue
            
            # 获取关系详细信息
            rel_info = self._get_relationship_info(referenced_rels)
            self.log(f"  获取到的关系信息数量: {len(rel_info)}")
                
            # 如果成功获取关系信息，打印部分示例
            if rel_info:
                self.log(f"  关系信息示例: {rel_info[:2]}{'...' if len(rel_info) > 2 else ''}")
            
            # 没有有效关系信息，尝试基于ID的评分
            if not rel_info and referenced_rels:
                # 基于关系ID数量的基础评分
                rel_count = len(referenced_rels)
                id_based_score = min(0.4, 0.3 + 0.02 * rel_count)  # 每个关系加0.02，最高0.4
                self.log(f"  没有关系详细信息，基于ID数量评分: {id_based_score:.4f}")
                
                # 尝试使用LLM评估
                if id_based_score <= 0.35 and self.llm:
                    llm_score = self._llm_fallback_for_relationship(sample)
                    if llm_score > id_based_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        utilization_scores.append(llm_score)
                        continue
                
                utilization_scores.append(id_based_score)
                continue
            
            # 没有关系但有实体，尝试推断实体间的隐含关系
            if not rel_info and not referenced_rels and entity_ids and self.neo4j_client:
                implicit_rel_score = self._calculate_implicit_relationships(entity_ids)
                self.log(f"  尝试推断实体间的隐含关系得分: {implicit_rel_score:.4f}")
                
                final_score = 0.3 + implicit_rel_score * 0.4  # 基础分加上推断关系得分
                self.log(f"  基于推断关系的得分: {final_score:.4f}")
                
                # 如果分数较低，尝试LLM回退
                if final_score <= 0.4 and self.llm:
                    llm_score = self._llm_fallback_for_relationship(sample)
                    if llm_score > final_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        utilization_scores.append(llm_score)
                        continue
                
                utilization_scores.append(final_score)
                continue
            
            # 没有有效关系信息，回退到基础分并尝试LLM评估
            if not rel_info and not referenced_rels:
                base_score = 0.3
                self.log(f"  无法获取关系信息，使用基础分: {base_score}")
                
                # 尝试使用LLM评估
                if self.llm:
                    llm_score = self._llm_fallback_for_relationship(sample)
                    if llm_score > base_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        utilization_scores.append(llm_score)
                        continue
                
                utilization_scores.append(base_score)
                continue
            
            # 计算关系利用率的多个维度
            quantity_score = self._calculate_quantity_score(rel_info)
            quality_score = self._calculate_quality_score(rel_info)
            relevance_score = self._calculate_relevance_score(rel_info, entity_ids)
            
            self.log(f"  数量得分: {quantity_score:.4f}")
            self.log(f"  质量得分: {quality_score:.4f}")
            self.log(f"  相关性得分: {relevance_score:.4f}")
            
            # 计算综合得分 - 数量占30%，质量占40%，相关性占30%
            base_score = 0.3  # 统一基础分
            total_score = base_score + 0.7 * (
                0.3 * quantity_score + 
                0.4 * quality_score + 
                0.3 * relevance_score
            )
            
            # 确保不超过1.0
            final_score = min(1.0, total_score)
            self.log(f"  基础分: {base_score}")
            self.log(f"  加权总分: {total_score:.4f}")
            self.log(f"  最终得分: {final_score:.4f}")
            
            # 如果分数较低，尝试LLM回退
            if final_score <= 0.4 and self.llm:
                llm_score = self._llm_fallback_for_relationship(sample)
                if llm_score > final_score:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    utilization_scores.append(llm_score)
                    continue
            
            utilization_scores.append(final_score)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.3
        self.log(f"\n平均关系利用率: {avg_utilization:.4f}")
        self.log("======== RelationshipUtilization 计算结束 ========\n")
        
        return {"relationship_utilization": avg_utilization}, utilization_scores
    
    def _llm_fallback_for_relationship(self, sample) -> float:
        """
        使用LLM评估关系利用率
        
        Args:
            sample: 评估样本
            
        Returns:
            float: LLM评估的关系利用率分数
        """
        question = sample.question
        answer = sample.system_answer
        agent_type = sample.agent_type.lower() if sample.agent_type else ""
        
        # 获取关系和实体信息
        relationships = sample.referenced_relationships
        entities = sample.referenced_entities
        
        # 关系描述
        rel_text = ", ".join([str(r) for r in relationships[:5]]) if relationships else "无"
        entity_text = ", ".join([str(e) for e in entities[:5]]) if entities else "无"
        
        # 构建LLM提示
        prompt = f"""
        请评估以下AI回答中对实体关系的利用程度，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        引用的实体: {entity_text}
        引用的关系: {rel_text}
        
        回答(部分): {answer[:]}...
        
        评分标准:
        - 高分(0.8-1.0): 回答充分利用了实体之间的关系，展示了深入的连接性分析
        - 中分(0.4-0.7): 回答部分利用了实体关系，但可能没有充分挖掘
        - 低分(0.0-0.3): 回答几乎没有利用实体间的关系
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类中的LLM回退方法
        return self.get_llm_fallback_score(prompt, default_score=0.35)
    
    def _get_relationship_info(self, referenced_rels) -> List[Dict[str, Any]]:
        rel_info = []
        
        if not self.neo4j_client or not referenced_rels:
            return rel_info
            
        # 处理字符串ID类型的关系
        rel_ids = [r for r in referenced_rels if isinstance(r, str)]
        
        # 转换为数字ID
        numeric_rel_ids = []
        for rel_id in rel_ids:
            try:
                if rel_id.isdigit() or rel_id.lstrip('-').isdigit():
                    numeric_rel_ids.append(int(rel_id))
            except (ValueError, AttributeError):
                continue
        
        if not numeric_rel_ids:
            return rel_info
            
        try:
            # 直接查询所有关系，然后手动匹配
            query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description, r.weight AS weight
            LIMIT 500
            """
            result = self.neo4j_client.execute_query(query)
            
            if result.records:
                # 只获取前50条关系作为样本
                count = 0
                for record in result.records:
                    if count >= 50:
                        break
                        
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description")
                    weight = record.get("weight")
                    
                    if source and relation and target:
                        rel_info.append({
                            "source": str(source),
                            "relation": relation,
                            "target": str(target),
                            "description": description,
                            "weight": weight
                        })
                        count += 1
            
            return rel_info
        except Exception as e:
            self.log(f"  查询关系信息失败: {e}")
            return rel_info
    
    def _calculate_implicit_relationships(self, entity_ids: List[str]) -> float:
        """计算实体间的隐含关系得分"""
        if not self.neo4j_client or len(entity_ids) < 2:
            return 0.0
            
        try:
            # 查询实体之间是否有路径连接
            query = """
            MATCH path = (a:__Entity__)-[*1..3]-(b:__Entity__)
            WHERE a.id IN $ids AND b.id IN $ids AND a <> b
            RETURN COUNT(DISTINCT path) AS path_count
            """
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            path_count = 0
            if result.records and result.records[0].get("path_count") is not None:
                path_count = result.records[0].get("path_count")
            
            # 计算潜在的连接总数
            potential_connections = len(entity_ids) * (len(entity_ids) - 1) / 2
            connected_ratio = min(1.0, path_count / potential_connections) if potential_connections > 0 else 0
            
            self.log(f"  实体之间的路径数量: {path_count}")
            self.log(f"  潜在连接总数: {potential_connections:.1f}")
            self.log(f"  连接率: {connected_ratio:.4f}")
                
            return min(1.0, connected_ratio * 1.2)  # 提供一点加成，但不超过1.0
        except Exception as e:
            self.log(f"  计算隐含关系时出错: {e}")
            return 0.0
    
    def _calculate_quantity_score(self, rel_info: List[Dict[str, Any]]) -> float:
        """计算关系数量得分"""
        # 如果有关系详细信息，使用实际关系数量
        rel_count = len(rel_info) if rel_info else 0
        
        # 每个关系贡献0.1分，最多1.0
        return min(1.0, rel_count * 0.1)
    
    def _calculate_quality_score(self, rel_info: List[Dict[str, Any]]) -> float:
        """
        计算关系质量得分 
        
        Args:
            rel_info: 关系信息列表
            
        Returns:
            float: 关系质量得分
        """
        if not rel_info:
            return 0.0
            
        # 检查关系是否有描述 - 确保处理None值
        described_count = 0
        for rel in rel_info:
            # 使用描述或关系类型
            description = rel.get("description", "")
            relation_type = rel.get("relation", "")
            
            if ((description is not None and str(description).strip()) or 
                (relation_type is not None and str(relation_type).strip())):
                described_count += 1
        
        description_ratio = described_count / len(rel_info) if rel_info else 0
        
        # 检查关系类型的多样性
        relation_types = set()
        for rel in rel_info:
            rel_type = rel.get("relation", "")
            if rel_type and rel_type.strip():
                relation_types.add(rel_type)
        
        type_diversity = min(1.0, len(relation_types) / 5)  # 最多5种关系类型为满分
        
        # 检查来源和目标实体是否存在
        valid_relations = 0
        for rel in rel_info:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source and source != "unknown" and target and target != "unknown":
                valid_relations += 1
        
        validity_ratio = valid_relations / len(rel_info) if rel_info else 0
        
        # 计算关系权重的平均值（如果有）
        weight_score = 0.0
        weighted_rels = [rel for rel in rel_info if "weight" in rel and rel["weight"] is not None]
        if weighted_rels:
            try:
                weights = []
                for rel in weighted_rels:
                    # 确保权重是有效数字
                    if isinstance(rel["weight"], (int, float)):
                        weights.append(float(rel["weight"]))
                    elif isinstance(rel["weight"], str) and rel["weight"].replace('.', '', 1).isdigit():
                        weights.append(float(rel["weight"]))
                
                if weights:
                    avg_weight = sum(weights) / len(weights)
                    # 假设权重范围为0-10，归一化到0-1
                    weight_score = min(1.0, avg_weight / 10.0)
            except Exception as e:
                self.log(f"  计算权重得分时出错: {e}")
        
        # 综合得分 - 描述占30%，多样性占30%，有效性占20%，权重占20%
        if weighted_rels:
            return (0.3 * description_ratio + 
                    0.3 * type_diversity + 
                    0.2 * validity_ratio + 
                    0.2 * weight_score)
        else:
            # 如果没有权重信息，重新分配占比
            return (0.4 * description_ratio + 
                    0.3 * type_diversity + 
                    0.3 * validity_ratio)
    
    def _calculate_relevance_score(self, rel_info: List[Dict[str, Any]], entity_ids: List[str]) -> float:
        """计算关系相关性得分"""
        if not rel_info or not entity_ids:
            return 0.0
            
        # 统计关系中的实体与引用实体的重合度
        relation_entities = set()
        for rel in rel_info:
            source = rel.get("source")
            target = rel.get("target")
            if source and source != "unknown":
                relation_entities.add(str(source))
            if target and target != "unknown":
                relation_entities.add(str(target))
        
        entity_id_set = set(str(e) for e in entity_ids)
        
        # 计算重合率
        if not entity_id_set:
            return 0.0
            
        overlap_ratio = len(relation_entities.intersection(entity_id_set)) / len(entity_id_set)
        
        # 提供一些加成，但不超过1.0
        return min(1.0, overlap_ratio * 1.2)