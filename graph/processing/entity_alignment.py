import time
from typing import List, Dict, Any, Optional

from graph.core import connection_manager
from model.get_models import get_llm_model
from config.settings import (
    ALIGNMENT_CONFLICT_THRESHOLD,
    ALIGNMENT_MIN_GROUP_SIZE
)

class EntityAligner:
    """
    实体对齐器: canonical_id分组 → 冲突检测 → 合并
    
    将具有相同canonical_id的实体对齐合并，解决冲突
    """
    
    def __init__(self):
        self.graph = connection_manager.get_connection()
        self.llm = get_llm_model()
        
        # 性能统计
        self.stats = {
            'groups_found': 0,
            'conflicts_detected': 0,
            'entities_aligned': 0
        }
    
    def group_by_canonical_id(self, skip: int = 0, limit: int = 100) -> Dict[str, List[str]]:
        """
        阶段1: 按canonical_id分组
        找出所有指向同一canonical_id的实体
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.canonical_id IS NOT NULL
        WITH e.canonical_id AS canonical_id, collect(e.id) AS entity_ids
        WHERE size(entity_ids) >= $min_size
        RETURN canonical_id, entity_ids
        ORDER BY size(entity_ids) DESC
        SKIP $skip
        LIMIT $limit
        """
        
        results = self.graph.query(query, params={
            'min_size': ALIGNMENT_MIN_GROUP_SIZE,
            'skip': skip,
            'limit': limit
        })
        
        groups = {}
        for row in results:
            canonical_id = row['canonical_id']
            entity_ids = row['entity_ids']
            # 确保canonical_id本身也在列表中
            if canonical_id not in entity_ids:
                entity_ids.append(canonical_id)
            groups[canonical_id] = entity_ids
        
        return groups
    
    def count_alignment_groups(self) -> int:
        """
        统计需要对齐的组总数
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.canonical_id IS NOT NULL
        WITH e.canonical_id AS canonical_id, collect(e.id) AS entity_ids
        WHERE size(entity_ids) >= $min_size
        RETURN count(*) AS total
        """
        
        result = self.graph.query(query, params={
            'min_size': ALIGNMENT_MIN_GROUP_SIZE
        })
        
        return result[0]['total'] if result else 0
    
    def detect_conflicts(self, canonical_id: str, entity_ids: List[str]) -> Dict[str, Any]:
        """
        阶段2: 冲突检测
        检测同一canonical_id下的实体是否存在语义冲突
        """
        # 获取实体的描述和关系，使用COUNT {}替代size()
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:`__Entity__` {id: eid})
        OPTIONAL MATCH (e)-[r]->(other)
        WITH e, collect(DISTINCT type(r)) AS rel_types, count(r) AS rel_count
        RETURN e.id AS entity_id,
               e.description AS description,
               rel_types,
               rel_count
        """
        
        entities = self.graph.query(query, params={'entity_ids': entity_ids})
        
        # 简单冲突检测: 如果关系类型差异太大，可能存在冲突
        if len(entities) < 2:
            return {'has_conflict': False, 'entities': entities}
        
        # 计算关系类型的交集比例
        all_rel_types = [set(e['rel_types']) for e in entities if e['rel_types']]
        if all_rel_types:
            intersection = set.intersection(*all_rel_types) if len(all_rel_types) > 1 else all_rel_types[0]
            union = set.union(*all_rel_types)
            
            jaccard = len(intersection) / len(union) if union else 0
            
            has_conflict = jaccard < ALIGNMENT_CONFLICT_THRESHOLD
            
            if has_conflict:
                self.stats['conflicts_detected'] += 1
            
            return {
                'has_conflict': has_conflict,
                'jaccard_similarity': jaccard,
                'entities': entities
            }
        
        return {'has_conflict': False, 'entities': entities}
    
    def resolve_conflict(self, canonical_id: str, conflict_info: Dict[str, Any]) -> str:
        """
        使用LLM解决冲突，决定保留哪个实体
        """
        entities = conflict_info['entities']
        
        # 构建LLM提示
        entity_desc = "\n".join([
            f"- {e['entity_id']}: {e['description']}, {e['rel_count']} relations"
            for e in entities
        ])
        
        prompt = f"""Given these entities that should refer to the same concept:
{entity_desc}

Which entity ID best represents the canonical form? Reply with only the entity ID."""
        
        try:
            response = self.llm.invoke(prompt)
            selected = response.content.strip()
            
            # 验证选择的ID是否在列表中
            valid_ids = [e['entity_id'] for e in entities]
            if selected in valid_ids:
                return selected
        except:
            pass
        
        # 回退: 选择关系数最多的
        return max(entities, key=lambda x: x['rel_count'])['entity_id']
    
    def merge_entities(self, canonical_id: str, entity_ids: List[str], keep_id: Optional[str] = None) -> int:
        """
        阶段3: 合并实体
        将所有实体合并到canonical实体，只保留一个
        保留原始关系类型，不丢失语义信息
        
        使用CALL子查询隔离边处理，确保即使没有边，主流程也能继续执行SET和DELETE
        """
        if not entity_ids or len(entity_ids) < 2:
            return 0
        
        # 确定保留哪个实体
        target_id = keep_id or canonical_id
        
        # 确保target在entity_ids中
        if target_id not in entity_ids:
            target_id = entity_ids[0]
        
        # 要删除的实体
        to_delete = [eid for eid in entity_ids if eid != target_id]
        
        if not to_delete:
            return 0
        
        # 合并查询：使用CALL子查询隔离边处理
        merge_query = """
        // 1. 确保目标实体存在
        MERGE (target:`__Entity__` {id: $target_id})
        
        WITH target, size($to_delete) AS deletion_count
        
        // 2. 逐个处理要删除的实体
        UNWIND $to_delete AS del_id
        MATCH (old:`__Entity__` {id: del_id})
        
        // 3. 在子查询中处理出边（不影响主流程）
        CALL {
            WITH old, target
            // 收集出边信息
            OPTIONAL MATCH (old)-[r_out]->(other)
            WHERE other.id <> $target_id
            WITH old, target, 
                type(r_out) AS rel_type, 
                other, 
                properties(r_out) AS rel_props
            WHERE rel_type IS NOT NULL AND other IS NOT NULL
            
            // 检查目标是否已有相同类型和属性的关系到该节点
            OPTIONAL MATCH (target)-[existing]->(other)
            WHERE type(existing) = rel_type
            
            WITH old, target, rel_type, other, rel_props, 
                 collect(properties(existing)) AS existing_props
            // 只有当不存在完全相同的关系时才创建（基于类型和属性）
            WHERE NOT rel_props IN existing_props
            
            CALL apoc.create.relationship(target, rel_type, rel_props, other) 
            YIELD rel
            RETURN count(rel) AS out_edges_created
        }
        
        // 4. 在子查询中处理入边（不影响主流程）
        WITH old, target, deletion_count, out_edges_created
        CALL {
            WITH old, target
            // 收集入边信息
            OPTIONAL MATCH (other)-[r_in]->(old)
            WHERE other.id <> $target_id
            WITH old, target,
                type(r_in) AS rel_type,
                other,
                properties(r_in) AS rel_props
            WHERE rel_type IS NOT NULL AND other IS NOT NULL
            
            // 检查是否已有相同类型和属性的关系从该节点到目标
            OPTIONAL MATCH (other)-[existing]->(target)
            WHERE type(existing) = rel_type
            
            WITH old, target, rel_type, other, rel_props,
                 collect(properties(existing)) AS existing_props
            // 只有当不存在完全相同的关系时才创建（基于类型和属性）
            WHERE NOT rel_props IN existing_props
            
            CALL apoc.create.relationship(other, rel_type, rel_props, target)
            YIELD rel
            RETURN count(rel) AS in_edges_created
        }
        
        // 5. 合并属性并标记（这部分始终执行，不受边处理影响）
        WITH target, old, deletion_count, out_edges_created, in_edges_created
        SET target.description = COALESCE(target.description, old.description),
            target.aligned_from = COALESCE(target.aligned_from, []) + [old.id],
            target.aligned_at = datetime(),
            target.canonical_id = $target_id
        
        // 6. 删除旧实体（始终执行）
        DETACH DELETE old

        RETURN deletion_count AS deleted, 
            sum(out_edges_created) AS total_out_edges,
            sum(in_edges_created) AS total_in_edges
        """
        
        try:
            result = self.graph.query(merge_query, params={
                'target_id': target_id,
                'to_delete': to_delete
            })
            
            if result and len(result) > 0:
                deleted = result[0].get('deleted', 0)
                out_edges = result[0].get('total_out_edges', 0)
                in_edges = result[0].get('total_in_edges', 0)
                
                self.stats['entities_aligned'] += deleted
                
                # 可选：记录详细信息
                if deleted > 0:
                    print(f"合并成功: 删除 {deleted} 个实体，转移 {out_edges} 条出边，{in_edges} 条入边")
                
                return deleted
            else:
                print(f"警告: 合并查询返回空结果，target={target_id}, to_delete={to_delete}")
                return 0
                
        except Exception as e:
            print(f"合并实体时出错: {e}, target={target_id}, to_delete={to_delete}")
            # 不抛出异常，返回0继续处理其他分组
            return 0
    
    def align_all(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        执行完整的对齐流程
        """
        start_time = time.time()
        
        # 统计总数
        total_groups = self.count_alignment_groups()
        print(f"发现 {total_groups} 个需要对齐的canonical组")
        
        self.stats['groups_found'] = total_groups
        total_merged = 0
        groups_processed = 0
        skip = 0
        while True:
            # 获取当前批次
            groups = self.group_by_canonical_id(skip=skip, limit=batch_size)
            
            if not groups:
                # 没有更多分组，退出循环
                break
            
            batch_count = len(groups)
            print(f"处理批次: skip={skip}, 获取 {batch_count} 个分组")
            
            # 处理当前批次的每个组
            for canonical_id, entity_ids in groups.items():
                # 冲突检测
                conflict_info = self.detect_conflicts(canonical_id, entity_ids)
                
                if conflict_info['has_conflict']:
                    # 解决冲突
                    keep_id = self.resolve_conflict(canonical_id, conflict_info)
                else:
                    keep_id = canonical_id
                
                # 合并实体
                merged = self.merge_entities(canonical_id, entity_ids, keep_id)
                total_merged += merged
                groups_processed += 1
            
            # 移动到下一批次
            skip += batch_size
            
            # 如果当前批次少于batch_size，说明已经是最后一批
            if batch_count < batch_size:
                break
        
        elapsed = time.time() - start_time
        
        return {
            'groups_processed': groups_processed,
            'entities_aligned': total_merged,
            'conflicts_detected': self.stats['conflicts_detected'],
            'elapsed_time': elapsed
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats