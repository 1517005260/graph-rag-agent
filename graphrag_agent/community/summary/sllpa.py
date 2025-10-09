from typing import List, Dict
from .base import BaseSummarizer
import time

from graphrag_agent.config.settings import BATCH_SIZE

class SLLPASummarizer(BaseSummarizer):
    """SLLPA算法的社区摘要生成器"""
    
    def collect_community_info(self) -> List[Dict]:
        """收集SLLPA社区信息"""
        start_time = time.time()
        print("收集SLLPA社区信息...")
        
        try:
            # 获取社区总数
            count_result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0  // SLLPA的所有社区都在level 0
            RETURN count(c) AS community_count
            """)
            
            community_count = count_result[0]['community_count'] if count_result else 0
            if not community_count:
                print("没有找到SLLPA社区")
                return []
                
            print(f"找到 {community_count} 个SLLPA社区，开始收集详细信息")
            
            if community_count > 1000:
                return self._collect_info_in_batches(community_count)
            
            result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0
            WITH c ORDER BY c.community_rank DESC NULLS LAST
            LIMIT 200
            
            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) AS nodes
            WHERE size(nodes) > 1
            
            CALL {
                WITH nodes
                MATCH (n1:__Entity__)
                WHERE n1 IN nodes
                MATCH (n2:__Entity__)
                WHERE n2 IN nodes AND id(n1) < id(n2)
                MATCH (n1)-[r]->(n2)
                WITH collect(distinct r) as relationships
                LIMIT 100
                RETURN relationships
            }
            
            RETURN c.id AS communityId,
                   [n in nodes | {
                       id: n.id, 
                       description: n.description, 
                       type: [el in labels(n) WHERE el <> '__Entity__'][0]
                   }] AS nodes,
                   [r in relationships | {
                       start: startNode(r).id, 
                       type: type(r), 
                       end: endNode(r).id, 
                       description: r.description
                   }] AS rels
            """)
            
            elapsed_time = time.time() - start_time
            print(f"收集到 {len(result)} 个SLLPA社区信息，耗时: {elapsed_time:.2f}秒")
            return result
            
        except Exception as e:
            print(f"收集SLLPA社区信息出错: {e}")
            return self._collect_info_fallback()
    
    def _collect_info_in_batches(self, total_count: int) -> List[Dict]:
        """分批收集社区信息"""
        batch_size = 50  # 默认批处理大小
        if BATCH_SIZE:
            batch_size = min(50, max(10, BATCH_SIZE // 2))  # 调整为适合社区收集的批次大小
            
        total_batches = (total_count + batch_size - 1) // batch_size
        all_results = []
        
        print(f"使用批处理收集SLLPA社区信息，共 {total_batches} 批")
        
        for batch in range(total_batches):
            if batch > 20:  # 限制批次
                print(f"已达到最大批次限制(20)，停止收集")
                break
                
            skip = batch * batch_size
            
            try:
                batch_result = self.graph.query("""
                MATCH (c:`__Community__`)
                WHERE c.level = 0
                WITH c ORDER BY c.community_rank DESC NULLS LAST
                SKIP $skip LIMIT $batch_size
                
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                
                CALL {
                    WITH nodes
                    WITH nodes[0..20] AS limited_nodes
                    MATCH (n1)-->(n2)
                    WHERE n1 IN limited_nodes AND n2 IN limited_nodes
                    RETURN collect(distinct relationship(n1, n2)) as relationships
                }
                
                RETURN c.id AS communityId,
                       [n in nodes | {
                           id: n.id, 
                           description: n.description, 
                           type: [el in labels(n) WHERE el <> '__Entity__'][0]
                       }] AS nodes,
                       [r in relationships | {
                           start: startNode(r).id, 
                           type: type(r), 
                           end: endNode(r).id, 
                           description: r.description
                       }] AS rels
                """, params={"skip": skip, "batch_size": batch_size})
                
                all_results.extend(batch_result)
                print(f"批次 {batch+1}/{total_batches} 完成，收集到 {len(batch_result)} 个社区")
                
            except Exception as e:
                print(f"批次 {batch+1} 处理出错: {e}")
                continue
        
        return all_results
    
    def _collect_info_fallback(self) -> List[Dict]:
        """备用的信息收集方法"""
        try:
            print("尝试使用简化查询收集社区信息...")
            result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0
            WITH c LIMIT 50
            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) as nodes
            WHERE size(nodes) > 1
            RETURN c.id AS communityId,
                   [n in nodes | {
                       id: n.id, 
                       description: coalesce(n.description, 'No description'), 
                       type: labels(n)[0]
                   }] AS nodes,
                   [] AS rels
            """)
            
            print(f"使用简化查询收集到 {len(result)} 个社区信息")
            return result
        except Exception as e:
            print(f"简化查询也失败: {e}")
            return []