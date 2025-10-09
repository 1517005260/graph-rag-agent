import time
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.config.settings import (
    DISAMBIG_STRING_THRESHOLD,
    DISAMBIG_VECTOR_THRESHOLD, 
    DISAMBIG_NIL_THRESHOLD,
    DISAMBIG_TOP_K
)

class EntityDisambiguator:
    """
    实体消歧器: mention → 字符串召回 → 向量重排 → NIL检测 → canonical_id
    
    通过多阶段管道消除实体歧义，将mention映射到知识图谱中的规范实体
    """
    
    def __init__(self):
        self.graph = connection_manager.get_connection()
        self.embeddings = get_embeddings_model()
        
        # 性能统计
        self.stats = {
            'mentions_processed': 0,
            'candidates_recalled': 0,
            'nil_detected': 0,
            'disambiguated': 0
        }
    
    def string_recall(self, mention: str, top_k: int = DISAMBIG_TOP_K) -> List[Dict[str, Any]]:
        """
        阶段1: 字符串召回候选实体
        使用编辑距离和模糊匹配快速召回相似实体
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.id IS NOT NULL
        WITH e, 
             apoc.text.levenshteinSimilarity(toLower($mention), toLower(e.id)) AS similarity
        WHERE similarity >= $threshold
        RETURN e.id AS entity_id,
               e.description AS description,
               similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        results = self.graph.query(query, params={
            'mention': mention,
            'threshold': DISAMBIG_STRING_THRESHOLD,
            'top_k': top_k
        })
        
        self.stats['candidates_recalled'] += len(results)
        return results
    
    def vector_rerank(self, mention: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        阶段2: 向量重排候选实体
        使用语义相似度对候选进行重新排序
        """
        if not candidates:
            return []
        
        # 计算mention的embedding
        mention_vec = self.embeddings.embed_query(mention)
        
        # 获取候选实体的embedding
        entity_ids = [c['entity_id'] for c in candidates]
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:`__Entity__` {id: eid})
        WHERE e.embedding IS NOT NULL
        RETURN e.id AS entity_id, e.embedding AS embedding
        """
        
        embeddings_result = self.graph.query(query, params={'entity_ids': entity_ids})
        embeddings_map = {r['entity_id']: r['embedding'] for r in embeddings_result}
        
        # 计算向量相似度并重排
        reranked = []
        for candidate in candidates:
            entity_id = candidate['entity_id']
            if entity_id in embeddings_map:
                entity_vec = embeddings_map[entity_id]
                similarity = self._cosine_similarity(mention_vec, entity_vec)
                
                reranked.append({
                    **candidate,
                    'vector_similarity': similarity,
                    'combined_score': 0.4 * candidate['similarity'] + 0.6 * similarity
                })
        
        return sorted(reranked, key=lambda x: x['combined_score'], reverse=True)
    
    def nil_detection(self, mention: str, candidates: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        阶段3: NIL检测
        判断是否为未登录实体(不在知识库中的新实体)
        """
        if not candidates:
            return True, None
        
        # 检查最佳候选的分数
        best_candidate = candidates[0]
        if best_candidate.get('combined_score', 0) < DISAMBIG_NIL_THRESHOLD:
            self.stats['nil_detected'] += 1
            return True, None
        
        return False, best_candidate['entity_id']
    
    def disambiguate(self, mention: str) -> Dict[str, Any]:
        """
        完整的消歧流程
        """
        self.stats['mentions_processed'] += 1
        
        # 字符串召回
        candidates = self.string_recall(mention)
        
        if not candidates:
            return {
                'mention': mention,
                'canonical_id': None,
                'is_nil': True,
                'candidates': []
            }
        
        # 向量重排
        reranked = self.vector_rerank(mention, candidates)
        
        # NIL检测
        is_nil, canonical_id = self.nil_detection(mention, reranked)
        
        if not is_nil:
            self.stats['disambiguated'] += 1
        
        return {
            'mention': mention,
            'canonical_id': canonical_id,
            'is_nil': is_nil,
            'candidates': reranked[:3]  # 返回前3个候选
        }
    
    def batch_disambiguate(self, mentions: List[str]) -> List[Dict[str, Any]]:
        """批量消歧"""
        results = []
        for mention in mentions:
            result = self.disambiguate(mention)
            results.append(result)
        
        return results
    
    def apply_to_graph(self) -> int:
        """
        将消歧结果应用到图谱
        核心思路：找到已合并的实体组，为组内其他实体指向主实体作为canonical_id
        
        分页策略：
        - 每次都查询前batch_size个未处理的分组（WHERE canonical_id IS NULL）
        - 处理完后这些分组消失，下次查询自动返回新的batch_size个
        - 循环直到查询返回空结果
        - 无需SKIP，因为结果集自动缩小
        """
        total_updated = 0
        batch_size = 500
        processed_groups = 0
        iteration = 0
        
        print(f"开始分批处理WCC分组，每批 {batch_size} 个")
        print(f"策略：每轮查询前{batch_size}个未处理分组，处理后自动从结果集移除")
        
        while True:
            iteration += 1
            
            # 每次查询前batch_size个未处理的分组
            # WHERE canonical_id IS NULL 确保只查询未处理的
            # 无SKIP，永远从头开始查
            query = """
            MATCH (e:`__Entity__`)
            WHERE e.wcc IS NOT NULL 
            AND e.embedding IS NOT NULL
            AND e.canonical_id IS NULL
            WITH e.wcc AS community, collect(e) AS entities
            WHERE size(entities) >= 2
            WITH community, entities
            ORDER BY community
            LIMIT $limit
            UNWIND entities AS entity
            WITH community, entity, COUNT { (entity)--() } AS degree
            WITH community, collect({id: entity.id, description: entity.description, degree: degree}) AS entity_info
            RETURN community, entity_info
            """
            
            # 注意：params中只有limit，没有skip
            groups = self.graph.query(query, params={'limit': batch_size})
            
            if not groups:
                print(f"第 {iteration} 轮：查询返回0个分组，所有数据已处理完毕")
                break
            
            print(f"第 {iteration} 轮：查询返回 {len(groups)} 个待处理分组")
            
            # 处理当前批次的分组
            batch_updated = 0
            for group in groups:
                entities = group['entity_info']
                
                # 选择度数最高的作为canonical（代表性最强）
                canonical = max(entities, key=lambda x: x['degree'])
                canonical_id = canonical['id']
                
                # 其他实体指向它
                other_ids = [e['id'] for e in entities if e['id'] != canonical_id]
                
                if other_ids:
                    update_query = """
                    UNWIND $entity_ids AS eid
                    MATCH (e:`__Entity__` {id: eid})
                    SET e.canonical_id = $canonical_id,
                        e.disambiguated = true,
                        e.disambiguated_at = datetime()
                    RETURN count(e) AS updated
                    """
                    
                    result = self.graph.query(update_query, params={
                        'entity_ids': other_ids,
                        'canonical_id': canonical_id
                    })
                    
                    if result:
                        batch_updated += result[0]['updated']
            
            total_updated += batch_updated
            processed_groups += len(groups)
            
            print(f"第 {iteration} 轮：已处理完毕，更新 {batch_updated} 个实体")
            print(f"累计：已处理 {processed_groups} 个分组，更新 {total_updated} 个实体")
            
            # 如果本批返回的分组少于batch_size，说明剩余数据不足一批
            # 下一轮会返回空，循环自然结束
            if len(groups) < batch_size:
                print(f"本轮返回 {len(groups)} < {batch_size}，剩余数据不足一批")
        
        print(f"\n{'='*60}")
        print(f"消歧完成统计：")
        print(f"  总轮数: {iteration}")
        print(f"  处理的WCC分组数: {processed_groups}")
        print(f"  更新的实体数: {total_updated}")
        print(f"{'='*60}\n")
        
        # 最终验证：确保没有遗漏
        print("执行最终验证...")
        remaining_query = """
        MATCH (e:`__Entity__`)
        WHERE e.wcc IS NOT NULL 
        AND e.embedding IS NOT NULL
        AND e.canonical_id IS NULL
        WITH e.wcc AS community, collect(e) AS entities
        WHERE size(entities) >= 2
        RETURN count(DISTINCT community) AS remaining_groups,
            sum(size(entities)) AS remaining_entities
        """
        
        remaining = self.graph.query(remaining_query)
        if remaining and remaining[0]['remaining_groups'] > 0:
            print(f"验证失败：仍有 {remaining[0]['remaining_groups']} 个分组未处理！")
            print(f"\n包含 {remaining[0]['remaining_entities']} 个实体")
        else:
            print(f"验证通过：所有符合条件的WCC分组均已处理")
        
        return total_updated
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats