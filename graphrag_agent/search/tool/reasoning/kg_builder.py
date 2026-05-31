import networkx as nx
from typing import Dict, List
import re
import time

class DynamicKnowledgeGraphBuilder:
    """
    动态知识图谱构建器
    
    在推理过程中实时构建与问题相关的知识子图，
    支持因果推理和关系发现
    """
    
    def __init__(self, graph, entity_relation_extractor=None):
        """
        初始化动态知识图谱构建器
        
        Args:
            graph: 图数据库连接
            entity_relation_extractor: 实体关系提取器
        """
        self.graph = graph
        self.extractor = entity_relation_extractor
        self.knowledge_graph = nx.DiGraph()  # 内存中的知识图谱
        self.seed_entities = set()  # 种子实体
        
    def build_query_graph(self, 
                        query: str, 
                        entities: List[str], 
                        depth: int = 2) -> nx.DiGraph:
        """
        为查询构建动态知识图谱
        
        Args:
            query: 用户查询
            entities: 初始实体列表
            depth: 图谱探索深度
            
        Returns:
            nx.DiGraph: 构建的知识图谱
        """
        # 确保有有效的实体
        if not entities:
            return self.knowledge_graph
            
        # 重置图谱
        self.knowledge_graph = nx.DiGraph()
        self.seed_entities = set(entities)
        
        start_time = time.time()
        
        # 添加种子实体
        for entity in entities:
            self.knowledge_graph.add_node(
                entity, 
                type="seed_entity",
                properties={"source": "query"}
            )
        
        # 递归探索图谱
        self._explore_graph(entities, current_depth=0, max_depth=depth)
        
        # 添加图谱构建元数据
        self.knowledge_graph.graph['build_time'] = time.time() - start_time
        self.knowledge_graph.graph['query'] = query
        self.knowledge_graph.graph['entity_count'] = self.knowledge_graph.number_of_nodes()
        self.knowledge_graph.graph['relation_count'] = self.knowledge_graph.number_of_edges()
        
        print(f"构建查询图谱完成，包含 {self.knowledge_graph.number_of_nodes()} 个实体和 "
              f"{self.knowledge_graph.number_of_edges()} 个关系，耗时 "
              f"{time.time() - start_time:.2f}秒")
              
        return self.knowledge_graph
    
    def _explore_graph(self, entities: List[str], current_depth: int, max_depth: int):
        """
        递归探索和扩展图谱
        
        Args:
            entities: 当前层次的实体列表
            current_depth: 当前探索深度
            max_depth: 最大探索深度
        """
        if current_depth >= max_depth or not entities:
            return
            
        # 查询实体的相邻节点和关系
        try:
            # 构建查询
            query = """
            MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
            WHERE e1.id IN $entity_ids
            RETURN e1.id AS source, 
                   e2.id AS target,
                   type(r) AS relation,
                   e2.description AS target_description
            LIMIT 100
            """
            
            # 执行查询
            relationships = self.graph.query(
                query, 
                params={"entity_ids": entities}
            )
            
            # 如果没有找到关系，返回
            if not relationships:
                return
                
            # 收集新发现的实体
            new_entities = []
            
            # 添加关系到图谱
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                relation = rel['relation']
                
                # 检查目标实体是否已在图谱中
                if target not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        target,
                        type="entity",
                        properties={"description": rel.get('target_description', '')}
                    )
                    new_entities.append(target)
                    
                # 添加边
                if not self.knowledge_graph.has_edge(source, target):
                    self.knowledge_graph.add_edge(
                        source, 
                        target, 
                        type=relation
                    )
            
            # 递归探索新发现的实体
            if new_entities:
                self._explore_graph(
                    new_entities, 
                    current_depth + 1, 
                    max_depth
                )
                
        except Exception as e:
            print(f"探索图谱时出错: {e}")
    
    def build_hierarchical_graph(self, documents):
        """
        构建包含文档层级、章节和特殊元素的图谱
        
        Args:
            documents: 文档列表
        """
        # 清理原图谱
        self.knowledge_graph = nx.DiGraph()
        
        for doc in documents:
            doc_id = doc.get('id')
            # 添加文档节点
            self.knowledge_graph.add_node(
                doc_id,
                type="document",
                properties={"title": doc.get('title', ''), "type": doc.get('type', '')}
            )
            
            # 添加章节节点和关系
            for section in doc.get('sections', []):
                section_id = f"{doc_id}_section_{section.get('id')}"
                self.knowledge_graph.add_node(
                    section_id,
                    type="section",
                    properties={"title": section.get('title', ''), "content": section.get('content', '')}
                )
                # 添加文档到章节的关系
                self.knowledge_graph.add_edge(doc_id, section_id, type="HAS_SECTION")
                
                # 添加段落节点
                for i, paragraph in enumerate(section.get('paragraphs', [])):
                    para_id = f"{section_id}_para_{i}"
                    self.knowledge_graph.add_node(
                        para_id,
                        type="paragraph",
                        properties={"content": paragraph, "index": i}
                    )
                    # 添加章节到段落的关系
                    self.knowledge_graph.add_edge(section_id, para_id, type="HAS_PARAGRAPH")
                    
                # 添加特殊元素（图表、公式等）
                for element in section.get('special_elements', []):
                    element_id = f"{section_id}_{element.get('type')}_{element.get('id')}"
                    self.knowledge_graph.add_node(
                        element_id,
                        type=element.get('type'),  # 如：formula, table, figure
                        properties={"content": element.get('content', ''), "description": element.get('description', '')}
                    )
                    # 添加章节到特殊元素的关系
                    self.knowledge_graph.add_edge(section_id, element_id, type=f"HAS_{element.get('type').upper()}")
        
        print(f"构建层级图谱完成，包含 {self.knowledge_graph.number_of_nodes()} 个节点和 "
              f"{self.knowledge_graph.number_of_edges()} 个关系")
        
        return self.knowledge_graph
    
    
    def extract_subgraph_from_chunk(self, chunk_text: str, chunk_id: str) -> bool:
        """
        从文本块中提取知识子图
        
        Args:
            chunk_text: 文本块内容
            chunk_id: 文本块ID
            
        Returns:
            bool: 是否成功提取
        """
        if not self.extractor:
            return False
            
        try:
            # 使用实体关系提取器分析文本
            extraction_result = self.extractor._process_single_chunk(chunk_text)
            
            if not extraction_result:
                return False
                
            # 解析结果
            entity_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
            relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')
            
            # 提取实体
            for match in entity_pattern.findall(extraction_result):
                entity_id, entity_type, description = match
                
                # 添加到图谱
                if entity_id not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        entity_id,
                        type=entity_type,
                        properties={
                            "description": description,
                            "source": f"chunk:{chunk_id}"
                        }
                    )
            
            # 提取关系
            for match in relationship_pattern.findall(extraction_result):
                source_id, target_id, rel_type, description, weight = match
                
                # 确保节点存在
                for node_id in [source_id, target_id]:
                    if node_id not in self.knowledge_graph:
                        self.knowledge_graph.add_node(
                            node_id,
                            type="unknown",
                            properties={
                                "description": "从关系中提取的实体",
                                "source": f"chunk:{chunk_id}"
                            }
                        )
                
                # 添加关系
                self.knowledge_graph.add_edge(
                    source_id,
                    target_id,
                    type=rel_type,
                    properties={
                        "description": description,
                        "weight": float(weight),
                        "source": f"chunk:{chunk_id}"
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"从文本块提取子图时出错: {e}")
            return False
    
    def get_central_entities(self, limit: int = 5) -> List[Dict]:
        """
        获取图谱中最重要的实体
        
        Args:
            limit: 返回实体数量
            
        Returns:
            List[Dict]: 重要实体列表
        """
        if not self.knowledge_graph.nodes:
            return []
            
        try:
            # 使用PageRank算法找出重要节点
            pagerank = nx.pagerank(self.knowledge_graph)
            
            # 排序
            top_entities = sorted(
                pagerank.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # 格式化结果
            result = []
            for entity_id, score in top_entities:
                node_data = self.knowledge_graph.nodes[entity_id]
                result.append({
                    "id": entity_id,
                    "centrality": score,
                    "type": node_data.get("type", "unknown"),
                    "properties": node_data.get("properties", {})
                })
                
            return result
            
        except Exception as e:
            print(f"计算中心实体时出错: {e}")
            # 使用度中心性作为备选方案
            in_degree = dict(self.knowledge_graph.in_degree())
            out_degree = dict(self.knowledge_graph.out_degree())
            
            # 合并入度和出度
            total_degree = {
                node: in_degree.get(node, 0) + out_degree.get(node, 0)
                for node in set(in_degree) | set(out_degree)
            }
            
            # 排序
            top_entities = sorted(
                total_degree.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # 格式化结果
            result = []
            for entity_id, degree in top_entities:
                node_data = self.knowledge_graph.nodes[entity_id]
                result.append({
                    "id": entity_id,
                    "degree": degree,
                    "type": node_data.get("type", "unknown"),
                    "properties": node_data.get("properties", {})
                })
                
            return result