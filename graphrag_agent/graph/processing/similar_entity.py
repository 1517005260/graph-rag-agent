import time
from graphdatascience import GraphDataScience
from typing import Tuple, List, Any, Dict
from dataclasses import dataclass

from graphrag_agent.config.settings import (
    similarity_threshold,
    BATCH_SIZE,
    GDS_MEMORY_LIMIT,
    NEO4J_CONFIG,
    SIMILAR_ENTITY_SETTINGS,
)
from graphrag_agent.graph.core import connection_manager, timer, get_performance_stats, print_performance_stats

@dataclass
class GDSConfig:
    """Neo4j GDS配置参数"""
    uri: str = NEO4J_CONFIG["uri"]
    username: str = NEO4J_CONFIG["username"]
    password: str = NEO4J_CONFIG["password"]
    similarity_threshold: float = similarity_threshold
    word_edit_distance: int = SIMILAR_ENTITY_SETTINGS["word_edit_distance"]
    batch_size: int = SIMILAR_ENTITY_SETTINGS["batch_size"]
    memory_limit: int = SIMILAR_ENTITY_SETTINGS["memory_limit"]  # 单位：GB
    top_k: int = SIMILAR_ENTITY_SETTINGS["top_k"]
    
    def __post_init__(self):
        # 如果配置文件中有设置则使用配置值
        if BATCH_SIZE:
            self.batch_size = BATCH_SIZE
        if GDS_MEMORY_LIMIT:
            self.memory_limit = GDS_MEMORY_LIMIT

class SimilarEntityDetector:
    """
    相似实体检测器，使用Neo4j GDS库实现实体相似性分析和社区识别。
    
    主要功能：
    1. 建立实体投影图
    2. 使用KNN算法识别相似实体
    3. 使用WCC算法进行社区检测
    4. 识别潜在的重复实体
    """
    
    def __init__(self, config: GDSConfig = None):
        """
        初始化相似实体检测器
        
        Args:
            config: GDS配置参数，包含连接信息和算法阈值
        """
        self.config = config or GDSConfig()
        self.gds = GraphDataScience(
            self.config.uri,
            auth=(self.config.username, self.config.password)
        )
        self.graph = connection_manager.get_connection()
        self.projection_name = "entities"
        self.G = None
        
        # 性能监控
        self.projection_time = 0
        self.knn_time = 0
        self.wcc_time = 0
        self.query_time = 0
        
        # 创建索引来优化重复实体检测
        self._create_indexes()
    
    def _create_indexes(self):
        """创建必要的索引以优化查询性能"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.wcc)"
        ]
        
        connection_manager.create_multiple_indexes(index_queries)
    
    @timer
    def create_entity_projection(self) -> Tuple[Any, Dict[str, Any]]:
        """
        创建实体的内存投影子图
        
        Returns:
            Tuple[Any, Dict[str, Any]]: 投影图对象和结果信息
        """
        start_time = time.time()
        
        # 如果已存在，先清除旧的投影
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"清除旧投影时出错 (可忽略): {e}")
        
        # 获取实体总数
        entity_count = self._get_entity_count()
        if entity_count == 0:
            print("没有找到有效的实体节点，请确保数据已经正确导入")
            return None, {"status": "error", "message": "No entities found"}
        
        # 创建新的投影图
        try:
            self.G, result = self.gds.graph.project(
                self.projection_name,          # 图名称
                "__Entity__",                  # 节点投影
                "*",                           # 关系投影（所有类型）
                nodeProperties=["embedding"]    # 配置参数
            )
        except Exception as e:
            print(f"创建投影时出错: {e}")
            # 尝试更保守的配置
            try:
                print("尝试使用保守配置重新创建投影...")
                config = {
                    "nodeProjection": {"__Entity__": {"properties": ["embedding"]}},
                    "relationshipProjection": {"*": {"orientation": "UNDIRECTED"}},
                    "nodeProperties": ["embedding"]
                }
                self.G, result = self.gds.graph.project(
                    self.projection_name,
                    config
                )
            except Exception as e2:
                print(f"二次尝试仍然失败: {e2}")
                return None, {"status": "error", "message": str(e2)}
        
        self.projection_time = time.time() - start_time
        
        if self.G:
            print(f"投影创建成功，耗时: {self.projection_time:.2f}秒")
            return self.G, result
        else:
            print("投影创建失败")
            return None, {"status": "error", "message": "Failed to create projection"}
    
    def _get_entity_count(self) -> int:
        """
        获取实体总数
        
        Returns:
            int: 实体数量
        """
        result = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL
            RETURN count(e) AS count
            """
        )
        return result[0]["count"] if result else 0
    
    @timer
    def detect_similar_entities(self) -> Dict[str, Any]:
        """
        使用KNN算法检测相似实体并创建SIMILAR关系
        
        Returns:
            Dict[str, Any]: 算法结果统计
        """
        if not self.G:
            raise ValueError("请先创建实体投影")
        
        start_time = time.time()
        print("开始检测相似实体...")
        
        try:
            top_k = max(1, self.config.top_k)
            # 使用KNN算法找出相似实体
            mutate_result = self.gds.knn.mutate(
                self.G,
                nodeProperties=['embedding'],
                mutateRelationshipType='SIMILAR',
                mutateProperty='score',
                similarityCutoff=self.config.similarity_threshold,
                topK=top_k
            )
            
            # 将KNN结果写入数据库
            write_result = self.gds.knn.write(
                self.G,
                nodeProperties=['embedding'],
                writeRelationshipType='SIMILAR',
                writeProperty='score',
                similarityCutoff=self.config.similarity_threshold,
                topK=top_k
            )
            
            self.knn_time = time.time() - start_time
            print(f"KNN完成，写入 {write_result['relationshipsWritten']} 个关系, 用时: {self.knn_time:.2f}秒")
            
            return {
                "status": "success",
                "relationshipsWritten": write_result['relationshipsWritten'],
                "knnTime": self.knn_time
            }
            
        except Exception as e:
            print(f"KNN算法执行失败: {e}")
            # 尝试使用备用参数
            try:
                print("尝试使用备用参数重新执行KNN...")
                fallback_top_k = max(1, top_k // 2)
                fallback_params = {
                    "nodeProperties": ["embedding"],
                    "writeRelationshipType": "SIMILAR",
                    "writeProperty": "score",
                    "similarityCutoff": self.config.similarity_threshold,
                    "topK": fallback_top_k,
                    "sampleRate": 0.5  # 降低采样率
                }
                
                fallback_result = self.gds.knn.write(self.G, **fallback_params)
                self.knn_time = time.time() - start_time
                
                print(f"备用KNN执行完成，写入 {fallback_result['relationshipsWritten']} 个关系, 用时: {self.knn_time:.2f}秒")
                
                return {
                    "status": "success",
                    "relationshipsWritten": fallback_result['relationshipsWritten'],
                    "knnTime": self.knn_time,
                    "note": "使用了备用参数"
                }
                
            except Exception as e2:
                print(f"备用KNN也失败了: {e2}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
    @timer
    def detect_communities(self) -> Dict[str, Any]:
        """
        使用WCC算法检测社区并将结果写入节点的wcc属性
        
        Returns:
            Dict[str, Any]: 社区检测结果统计
        """
        if not self.G:
            raise ValueError("请先创建实体投影")
        
        start_time = time.time()
        print("开始检测社区...")
        
        try:
            # 使用WCC算法
            result = self.gds.wcc.write(
                self.G,
                writeProperty="wcc",
                relationshipTypes=["SIMILAR"],
                consecutiveIds=True
            )
            
            self.wcc_time = time.time() - start_time
            
            community_count = result.get("communityCount", 0)
            print(f"社区检测完成，找到 {community_count} 个社区, 用时: {self.wcc_time:.2f}秒")
            
            return {
                "status": "success",
                "communityCount": community_count,
                "wccTime": self.wcc_time
            }
        
        except Exception as e:
            print(f"WCC算法执行失败: {e}")
            # 尝试使用备用参数
            try:
                print("尝试使用备用参数重新执行WCC...")
                fallback_result = self.gds.wcc.write(
                    self.G,
                    writeProperty="wcc",
                    relationshipTypes=["SIMILAR"]
                )
                
                self.wcc_time = time.time() - start_time
                community_count = fallback_result.get("communityCount", 0)
                
                print(f"备用WCC执行完成，找到 {community_count} 个社区, 用时: {self.wcc_time:.2f}秒")
                
                return {
                    "status": "success",
                    "communityCount": community_count,
                    "wccTime": self.wcc_time,
                    "note": "使用了备用参数"
                }
                
            except Exception as e2:
                print(f"备用WCC也失败了: {e2}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
    @timer
    def find_potential_duplicates(self) -> List[Any]:
        """
        查找潜在的重复实体
        
        Returns:
            List[Any]: 潜在重复实体的候选列表
        """
        query_start = time.time()
        
        # 查找包含多个实体的社区
        community_counts = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.wcc IS NOT NULL AND size(e.id) > 1
            WITH e.wcc AS community, count(*) AS count
            WHERE count > 1
            RETURN community, count
            ORDER BY count DESC
            """
        )
        
        if not community_counts:
            print("没有找到可能包含重复实体的社区")
            return []
        
        # 为有效社区查找潜在重复
        results = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 1  // 长度大于1个字符
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // 添加文本距离计算
            WITH distinct
                [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] 
                AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // 如果组之间有共同元素，则合并组
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, 
                index2 IN range(0, size(results)-1, 1) |
                CASE WHEN index <> index2 AND
                    size(apoc.coll.intersection(acc, results[index2])) > 0
                    THEN apoc.coll.union(acc, results[index2])
                    ELSE acc
                END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // 额外过滤
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, 
                combinedResultIndex, 
                allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """,
            params={'distance': self.config.word_edit_distance}
        )
        
        self.query_time = time.time() - query_start
        
        # 转换查询结果为简单的字符串列表列表格式
        processed_results = []
        for record in results:
            if "combinedResult" in record and isinstance(record["combinedResult"], list):
                processed_results.append(record["combinedResult"])
        
        print(f"潜在重复实体查找完成，找到 {len(processed_results)} 组候选实体, 用时: {self.query_time:.2f}秒")
        
        return processed_results
    
    def cleanup(self) -> None:
        """清理内存中的投影图"""
        if self.G:
            try:
                self.G.drop()
                print("投影图清理完成")
            except Exception as e:
                print(f"清理投影图时出错: {str(e)}")
            finally:
                self.G = None

    @timer
    def process_entities(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        执行完整的实体处理流程
        
        Returns:
            Tuple[List[Any], Dict[str, Any]]: 潜在重复实体的列表和性能统计
        """
        start_time = time.time()
        duplicates = []
        
        try:
            # 1. 创建实体投影
            self.G, projection_result = self.create_entity_projection()
            
            if not self.G:
                print("实体投影创建失败，无法继续处理")
                return [], {"status": "error", "message": "投影创建失败"}
                
            # 2. 检测相似实体
            knn_result = self.detect_similar_entities()
            
            if knn_result.get('status') == 'error':
                print(f"相似实体检测失败: {knn_result.get('message')}")
                return [], {"status": "error", "message": "相似实体检测失败"}
                
            # 3. 检测社区
            wcc_result = self.detect_communities()
            
            if wcc_result.get('status') == 'error':
                print(f"社区检测失败: {wcc_result.get('message')}")
                return [], {"status": "error", "message": "社区检测失败"}
                
            # 4. 查找潜在重复
            duplicates = self.find_potential_duplicates()
            
            total_time = time.time() - start_time
            
            # 准备性能统计
            time_records = {
                "投影时间": self.projection_time,
                "KNN时间": self.knn_time,
                "WCC时间": self.wcc_time,
                "查询时间": self.query_time
            }
            
            stats = get_performance_stats(total_time, time_records)
            stats.update({
                "status": "success",
                "候选实体组数": len(duplicates),
                "关系数量": knn_result.get('relationshipsWritten', 0),
                "社区数量": wcc_result.get('communityCount', 0)
            })
            
            print_performance_stats(stats)
            
            return duplicates, stats
            
        except Exception as e:
            print(f"实体处理过程中发生错误: {e}")
            return [], {"status": "error", "message": str(e)}
            
        finally:
            # 确保清理投影图
            self.cleanup()
