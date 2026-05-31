from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from graphrag_agent.config.settings import GDS_CONCURRENCY

class SLLPADetector(GraphProjectionMixin, BaseCommunityDetector):
    """SLLPA算法社区检测实现"""
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行SLLPA算法检测社区"""
        if not self.G:
            raise ValueError("请先创建图投影")
            
        print("开始执行SLLPA社区检测...")
        
        try:
            # 执行SLLPA算法
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                **self._get_optimized_sllpa_params()
            )
            
            community_count = result.get('communityCount', 0)
            iterations = result.get('iterations', 0)
            
            print(f"SLLPA算法完成: {community_count} 个社区, "
                  f"{iterations} 次迭代")
            
            return {
                'communityCount': community_count,
                'iterations': iterations
            }
            
        except Exception as e:
            print(f"SLLPA算法执行失败: {e}")
            return self._execute_fallback_sllpa()
    
    def _execute_fallback_sllpa(self) -> Dict[str, Any]:
        """执行备用SLLPA算法"""
        print("尝试使用备用参数...")
        
        try:
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                maxIterations=50,  # 减少迭代次数
                minAssociationStrength=0.2,  # 提高阈值
                concurrency=1  # 单线程执行
            )
            
            return {
                'communityCount': result.get('communityCount', 0),
                'iterations': result.get('iterations', 0),
                'note': '使用了备用参数'
            }
        except Exception as e:
            raise ValueError(f"SLLPA算法执行失败: {e}")
    
    def _get_optimized_sllpa_params(self) -> Dict[str, Any]:
        """获取优化的SLLPA参数"""
        if self.memory_mb > 32 * 1024:  # >32GB
            return {
                'maxIterations': 100,
                'minAssociationStrength': 0.05,
                'concurrency': GDS_CONCURRENCY
            }
        elif self.memory_mb > 16 * 1024:  # >16GB
            return {
                'maxIterations': 80,
                'minAssociationStrength': 0.08,
                'concurrency': max(1, GDS_CONCURRENCY - 1)
            }
        else:  # 小内存
            return {
                'maxIterations': 50,
                'minAssociationStrength': 0.1,
                'concurrency': max(1, GDS_CONCURRENCY // 2)
            }
    
    def save_communities(self) -> Dict[str, int]:
        """保存SLLPA算法结果"""
        print("开始保存SLLPA社区检测结果...")
        
        try:
            # 创建约束
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )
            
            # 保存社区
            result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL
            WITH count(e) AS entities_with_communities
            
            CALL {
                WITH entities_with_communities
                MATCH (e:`__Entity__`)
                WHERE e.communityIds IS NOT NULL
                WITH collect(e) AS entities
                CALL {
                    WITH entities
                    UNWIND entities AS e
                    UNWIND range(0, size(e.communityIds) - 1, 1) AS index
                    MERGE (c:`__Community__` {id: '0-'+toString(e.communityIds[index])})
                    ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                }
                RETURN count(*) AS processed_count
            }
            
            RETURN CASE 
                WHEN entities_with_communities > 0 THEN entities_with_communities 
                ELSE 0 
            END AS total_count
            """)
            
            total_count = result[0]['total_count'] if result else 0
            print(f"已保存 {total_count} 个SLLPA社区关系")
            
            return {'saved_communities': total_count}
            
        except Exception as e:
            print(f"保存SLLPA社区结果失败: {e}")
            return self._save_communities_fallback()
    
    def _save_communities_fallback(self) -> Dict[str, int]:
        """备用社区保存方法"""
        print("尝试使用简化方法保存社区...")
        
        try:
            result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL AND size(e.communityIds) > 0
            WITH e, e.communityIds[0] AS primary_community
            MERGE (c:`__Community__` {id: '0-' + toString(primary_community)})
            ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) as count
            """)
            
            count = result[0]['count'] if result else 0
            print(f"使用简化方法保存了 {count} 个社区关系")
            
            return {
                'saved_communities': count,
                'note': '使用了简化保存方法'
            }
        except Exception as e:
            raise ValueError(f"无法保存社区结果: {e}")