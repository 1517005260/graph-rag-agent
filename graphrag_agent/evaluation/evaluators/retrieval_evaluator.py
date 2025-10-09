import re
import time
from typing import Dict, List, Tuple

from graphrag_agent.evaluation.core.base_evaluator import BaseEvaluator
from graphrag_agent.evaluation.core.evaluation_data import RetrievalEvaluationData, RetrievalEvaluationSample
from graphrag_agent.evaluation.preprocessing.reference_extractor import extract_references_from_answer

class GraphRAGRetrievalEvaluator(BaseEvaluator):
    """GraphRAG检索评估器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.qa_agent = config.get('qa_agent', None)
        self.entity_map = {}  # 实体ID到描述的映射
        self.relation_map = {}  # 关系ID到关系信息的映射
    
    def evaluate(self, data: RetrievalEvaluationData) -> Dict[str, float]:
        """执行评估"""
        self.log("\n======== 开始评估检索性能 ========")
        
        # 打印样本信息
        self.log(f"样本总数: {len(data.samples)}")

        # 预处理阶段 - 建立实体和关系映射
        self._prepare_entity_relation_maps()
        
        # 首先处理每个样本的数据，确保引用的实体和关系信息完整
        for i, sample in enumerate(data.samples):
            self.log(f"\n处理样本 {i+1}:")
            
            # 打印基本信息
            self.log(f"  问题: {sample.question[:50]}...")
            self.log(f"  Agent类型: {sample.agent_type}")

            # 增强实体和关系处理
            self._enhance_entity_data(sample)
            self._enhance_relation_data(sample)
            
            # 打印处理后的信息
            self.log(f"  处理后的引用实体数量: {len(sample.referenced_entities)}")
            self.log(f"  处理后的引用关系数量: {len(sample.referenced_relationships)}")
            
            # 打印回答的一部分以及从回答中提取的引用数据
            answer = sample.system_answer
            self.log(f"  回答前100字符: {answer[:100]}...")
            
            # 显示当前样本的引用实体和关系信息
            self.log(f"  当前引用实体数量: {len(sample.referenced_entities)}")
            self.log(f"  当前引用关系数量: {len(sample.referenced_relationships)}")
            
            # 提取引用数据并打印
            refs = extract_references_from_answer(answer)
            
            self.log(f"  提取的引用数据:")
            self.log(f"    实体: {refs.get('entities', [])[:5]}{'...' if len(refs.get('entities', [])) > 5 else ''}") 
            self.log(f"    关系: {refs.get('relationships', [])[:5]}{'...' if len(refs.get('relationships', [])) > 5 else ''}")
            self.log(f"    文本块: {refs.get('chunks', [])[:3]}{'...' if len(refs.get('chunks', [])) > 3 else ''}")
            
            # 1. 处理naiveAgent - 确保文本块数据正确存储
            if sample.agent_type.lower() == "naive":
                self.log("  处理NaiveAgent的引用数据...")
                
                # 将文本块ID从referenced_relationships移到referenced_entities
                if not sample.referenced_entities and isinstance(sample.referenced_relationships, list):
                    for item in sample.referenced_relationships:
                        if isinstance(item, str) and len(item) > 30:  # 长字符串可能是文本块ID
                            sample.referenced_entities.append(item)
                    sample.referenced_relationships = []
                    self.log(f"  将文本块从关系移到实体字段，现在实体数: {len(sample.referenced_entities)}")
                
                # 确保从json数据中提取的文本块ID在referenced_entities中
                for chunk_id in refs.get("chunks", []):
                    if chunk_id not in sample.referenced_entities:
                        sample.referenced_entities.append(chunk_id)
                        self.log(f"  添加文本块ID: {chunk_id[:10]}...")
            
            # 2. 处理其他Agent - 确保实体和关系ID正确存储
            else:
                self.log("  处理非NaiveAgent的引用数据...")
                
                # 更新实体ID
                added_entities = 0
                for entity_id in refs.get("entities", []):
                    if entity_id and entity_id not in sample.referenced_entities:
                        sample.referenced_entities.append(entity_id)
                        added_entities += 1
                
                # 更新关系ID
                added_relationships = 0
                for rel_id in refs.get("relationships", []):
                    if rel_id and rel_id not in sample.referenced_relationships:
                        sample.referenced_relationships.append(rel_id)
                        added_relationships += 1
                        
                self.log(f"  添加了{added_entities}个实体和{added_relationships}个关系")
                
            # 显示最终引用信息
            self.log(f"  最终引用实体数量: {len(sample.referenced_entities)}")
            self.log(f"  最终引用关系数量: {len(sample.referenced_relationships)}")
        
        # 执行评估计算
        result_dict = {}
        
        for metric_name in self.metrics:
            try:
                self.log(f"\n开始计算指标: {metric_name}")
                metric_class_name = self.metric_class[metric_name].__class__.__name__
                self.log(f"\n使用评估类: {metric_class_name}")
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                result_dict.update(metric_result)
                
                # 更新每个样本的评分
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)
                    
                self.log(f"完成指标 {metric_name} 计算，平均得分: {list(metric_result.values())[0]:.4f}")
            except Exception as e:
                import traceback
                self.log(f'评估 {metric_name} 时出错: {e}')
                self.log(traceback.format_exc())
                continue
        
        self.log("\n所有指标计算结果:")
        for metric, score in result_dict.items():
            self.log(f"  {metric}: {score:.4f}")
        
        self.log("======== 检索性能评估结束 ========\n")
        
        # 保存评估结果
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
        
        # 保存评估数据
        if self.save_data_flag:
            self.save_data(data)
        
        return result_dict
    
    def _prepare_entity_relation_maps(self):
        """准备实体和关系映射，用于快速查找"""
        self.entity_map = {}
        self.relation_map = {}
        
        if not self.neo4j_client:
            return
        
        try:
            # 获取所有实体
            entity_query = """
            MATCH (n)
            RETURN n.id AS id, n.description AS description
            LIMIT 2000
            """
            entity_result = self.neo4j_client.execute_query(entity_query)
            
            if entity_result.records:
                for record in entity_result.records:
                    ent_id = record.get("id")
                    ent_desc = record.get("description", "")
                    if ent_id:
                        self.entity_map[str(ent_id)] = ent_desc
            
            # 获取所有关系
            relation_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, type(r) AS relation, b.id AS target, r.id AS rel_id
            LIMIT 1000
            """
            relation_result = self.neo4j_client.execute_query(relation_query)
            
            if relation_result.records:
                for record in relation_result.records:
                    rel_id = record.get("rel_id")
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    
                    if rel_id and source and relation and target:
                        self.relation_map[str(rel_id)] = {
                            "source": str(source),
                            "relation": relation,
                            "target": str(target)
                        }
        except Exception as e:
            self.log(f"准备实体和关系映射时出错: {e}")

    def _enhance_entity_data(self, sample):
        """增强实体数据处理"""
        # 1. 确保实体ID是字符串
        sample.referenced_entities = [str(e) for e in sample.referenced_entities]
        
        # 2. 尝试添加实体描述
        if self.entity_map:
            enhanced_entities = []
            for ent_id in sample.referenced_entities:
                if ent_id in self.entity_map:
                    desc = self.entity_map[ent_id]
                    enhanced_entity = {
                        "id": ent_id,
                        "description": desc
                    }
                    enhanced_entities.append(enhanced_entity)
                else:
                    enhanced_entities.append({
                        "id": ent_id,
                        "description": f"实体 {ent_id}"
                    })
            
            # 将增强的实体信息保存到样本中
            sample.entity_details = enhanced_entities

    def _enhance_relation_data(self, sample):
        """增强关系数据处理"""
        # 1. 处理字符串ID的关系
        if not isinstance(sample.referenced_relationships, list):
            sample.referenced_relationships = []
            return
        
        string_rel_ids = [r for r in sample.referenced_relationships if isinstance(r, str)]
        
        # 2. 尝试使用关系映射增强关系信息
        enhanced_relations = []
        for rel_id in string_rel_ids:
            if rel_id in self.relation_map:
                rel_data = self.relation_map[rel_id]
                enhanced_relation = (
                    rel_data["source"],
                    rel_data["relation"],
                    rel_data["target"]
                )
                enhanced_relations.append(enhanced_relation)
        
        # 3. 如果成功增强了关系，更新样本
        if enhanced_relations:
            sample.enhanced_relationships = enhanced_relations
        else:
            # 使用更智能的方式创建占位关系
            relation_types = ["MENTIONS", "RELATES_TO", "PART_OF", "CONTAINS"]
            
            for i, rel_id in enumerate(string_rel_ids):
                rel_type = relation_types[i % len(relation_types)]
                source = f"entity_{i}"
                target = f"entity_{i+1}"
                
                enhanced_relations.append((source, rel_type, target))
            
            sample.enhanced_relationships = enhanced_relations
    
    def get_entities_info(self, entity_ids: List[str]) -> List[Tuple[str, str]]:
        """获取实体信息（ID和描述）"""
        if not self.neo4j_client or not entity_ids:
            return []
        
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $ids
            RETURN e.id AS id, e.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            entities_info = []
            if result.records:
                for record in result.records:
                    entity_id = record.get("id", "未知ID")
                    entity_desc = record.get("description", "")
                    # 使用实体ID和描述
                    entities_info.append((str(entity_id), entity_desc or ""))
            
            # 如果没有找到实体，返回原始ID
            if not entities_info:
                entities_info = [(eid, "") for eid in entity_ids]
                
            return entities_info
                
        except Exception as e:
            self.log(f"查询实体信息失败: {e}")
            return [(eid, "") for eid in entity_ids]

    def get_relationships_info(self, relationship_ids: List[str]) -> List[Tuple[str, str, str]]:
        """获取关系信息（源实体-关系类型-目标实体）"""
        if not self.neo4j_client or not relationship_ids:
            return []
        
        try:
            # 转换所有ID为整数
            numeric_ids = []
            for rid in relationship_ids:
                try:
                    numeric_ids.append(int(rid))
                except (ValueError, TypeError):
                    # 如果不能转换为整数，跳过
                    pass
            
            if not numeric_ids:
                # 如果没有有效的数字ID，返回空列表
                return []
            
            # 通过关系ID直接匹配关系
            query = """
            MATCH (a)-[r]->(b)
            WHERE r.id IN $ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": numeric_ids})
            
            relationships_info = []
            if result.records:
                for record in result.records:
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description", "")
                    
                    # 只有当所有值都存在时才添加关系
                    if source and relation and target:
                        # 使用关系的描述补充关系类型
                        rel_info = relation
                        if description:
                            rel_info = f"{relation}({description})"
                            
                        relationships_info.append((str(source), rel_info, str(target)))
            
            return relationships_info
                
        except Exception as e:
            self.log(f"查询关系信息失败: {e}")
            return []
        
    def evaluate_agent(self, agent_name: str, questions: List[str]) -> Dict[str, float]:
        """
        评估特定Agent的检索性能
        
        Args:
            agent_name: Agent名称 (naive, hybrid, graph, deep)
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 评估结果
        """
        agent = self.config.get_agent(agent_name)
        if not agent:
            raise ValueError(f"未找到Agent: {agent_name}")
        
        # 创建评估数据集
        eval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for question in questions:
            # 创建评估样本
            sample = RetrievalEvaluationSample(
                question=question,
                agent_type=agent_name
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 普通回答
            answer = agent.ask(question)
            
            # 计算检索时间
            retrieval_time = time.time() - start_time
            
            # 更新样本
            sample.update_system_answer(answer, agent_name)
            sample.retrieval_time = retrieval_time
            
            # 使用Neo4j获取相关图数据
            if self.neo4j_client:
                entities, relationships = self._get_relevant_graph_data(question)
                sample.update_retrieval_data(entities, relationships)
            
            # 添加到评估数据
            eval_data.append(sample)
        
        # 执行评估
        return self.evaluate(eval_data)
    
    def compare_agents(self, questions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        比较所有Agent的检索性能
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个Agent的评估结果
        """
        results = {}
        
        for agent_name in ["naive", "hybrid", "graph", "deep"]:
            agent = self.config.get_agent(agent_name)
            if agent:
                self.log(f"评估Agent: {agent_name}")
                agent_results = self.evaluate_agent(agent_name, questions)
                results[agent_name] = agent_results
                
                # 打印结果
                self.log(f"{agent_name} 评估结果:")
                for metric, score in agent_results.items():
                    self.log(f"  {metric}: {score:.4f}")
                self.log("")
        
        return results
    
    def _get_relevant_graph_data(self, question: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """从Neo4j获取与问题相关的实体和关系"""
        if not self.neo4j_client:
            return [], []
            
        try:
            # 提取问题关键词
            try:
                import jieba.analyse
                question_words = jieba.analyse.extract_tags(question, topK=5)
            except Exception as e:
                # 简单分词回退方案
                self.log(f"关键词提取失败: {e}")
                question_words = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', question)
                question_words = [w for w in question_words if len(w) > 1]
            
            entities = []
            relationships = []
            
            # 查询与关键词相关的实体 - 使用e.id和e.description
            entity_query = """
            MATCH (e:__Entity__)
            WHERE ANY(word IN $keywords WHERE 
                e.id CONTAINS word OR
                e.description CONTAINS word)
            RETURN e.id AS id
            LIMIT 15
            """
            
            entity_result = self.neo4j_client.execute_query(entity_query, {"keywords": question_words})
            
            if entity_result.records:
                for record in entity_result.records:
                    entity_id = record.get("id")
                    if entity_id:
                        entities.append(entity_id)
            
            # 如果找到实体，查询相关关系
            if entities:
                # 查询实体之间的关系
                rel_query = """
                MATCH (a:__Entity__)-[r]->(b:__Entity__)
                WHERE a.id IN $entity_ids OR b.id IN $entity_ids
                RETURN DISTINCT a.id AS source, type(r) AS relation, b.id AS target
                LIMIT 30
                """
                
                rel_result = self.neo4j_client.execute_query(rel_query, {"entity_ids": entities})
                
                if rel_result.records:
                    for record in rel_result.records:
                        source = record.get("source")
                        relation = record.get("relation")
                        target = record.get("target")
                        if source and relation and target:
                            relationships.append((source, relation, target))
            
            # 如果未找到足够实体，尝试通过文本块查找
            if len(entities) < 3:
                chunk_query = """
                MATCH (c:__Chunk__)
                WHERE ANY(word IN $keywords WHERE c.text CONTAINS word)
                RETURN c.id AS chunk_id
                LIMIT 5
                """
                
                chunk_result = self.neo4j_client.execute_query(chunk_query, {"keywords": question_words})
                
                chunk_ids = []
                if chunk_result.records:
                    for record in chunk_result.records:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            chunk_ids.append(chunk_id)
                
                # 如果找到文本块，获取相关实体
                if chunk_ids:
                    chunk_entity_query = """
                    MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.id AS entity_id
                    """
                    
                    chunk_entity_result = self.neo4j_client.execute_query(
                        chunk_entity_query, {"chunk_ids": chunk_ids}
                    )
                    
                    if chunk_entity_result.records:
                        for record in chunk_entity_result.records:
                            entity_id = record.get("entity_id")
                            if entity_id and entity_id not in entities:
                                entities.append(entity_id)
        except Exception as e:
            self.log(f"获取图数据时出错: {e}")
        
        return entities, relationships
    
    def format_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        将比较结果格式化为表格
        
        Args:
            results: 比较结果
            
        Returns:
            str: 表格字符串
        """
        # 获取所有指标
        all_metrics = set()
        for agent_results in results.values():
            all_metrics.update(agent_results.keys())
        
        # 构建表头
        header = "| 指标 | " + " | ".join(results.keys()) + " |"
        separator = "| --- | " + " | ".join(["---" for _ in results]) + " |"
        
        # 构建行
        rows = []
        for metric in sorted(all_metrics):
            row = f"| {metric} |"
            for agent in results:
                score = results[agent].get(metric, "N/A")
                if isinstance(score, float):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)
                row += f" {score_str} |"
            rows.append(row)
        
        # 拼接表格
        table = "\n".join([header, separator] + rows)
        return table