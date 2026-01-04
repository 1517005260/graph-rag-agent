"""
Neo4j 原生 Vector Search（工程级实践）

使用 db.index.vector.queryNodes 进行服务端向量搜索，
避免客户端排序，提升性能和准确性。
"""
from typing import List, Dict, Any, Optional
import time

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.config.settings import (
    CHUNK_VECTOR_INDEX,
    ENTITY_VECTOR_INDEX
)


class Neo4jVectorSearch:
    """
    Neo4j 原生向量搜索（工程级实践）

    使用 Neo4j 5.x 的 db.index.vector.queryNodes API，
    在服务端完成向量相似度计算和排序。
    """

    def __init__(self):
        """初始化 Neo4j 向量搜索"""
        self.graph = connection_manager.get_connection()

    def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        return_properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 Neo4j 原生 API 搜索相似的 Chunk 节点

        Args:
            query_embedding: 查询向量
            top_k: 返回的最大结果数
            return_properties: 需要返回的节点属性列表，None 则返回所有属性

        Returns:
            List[Dict]: 搜索结果列表，每项包含：
                - id: Chunk ID
                - text: Chunk 文本
                - score: 相似度分数（0-1）
                - 其他指定的属性
        """
        # 构建返回属性的 Cypher 语句
        if return_properties is None:
            return_properties = ["id", "text", "fileName", "position"]

        property_returns = ", ".join([f"node.{prop} AS {prop}" for prop in return_properties])

        query = f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $top_k,
            $query_embedding
        )
        YIELD node, score
        RETURN {property_returns}, score
        ORDER BY score DESC
        """

        try:
            results = self.graph.query(
                query,
                params={
                    "index_name": CHUNK_VECTOR_INDEX,
                    "top_k": top_k,
                    "query_embedding": query_embedding
                }
            )

            return results
        except Exception as e:
            print(f"❌ Neo4j vector search failed: {e}")
            # 如果索引不存在或其他错误，返回空列表
            return []

    def search_entities(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        return_properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 Neo4j 原生 API 搜索相似的 Entity 节点

        Args:
            query_embedding: 查询向量
            top_k: 返回的最大结果数
            return_properties: 需要返回的节点属性列表

        Returns:
            List[Dict]: 搜索结果列表
        """
        if return_properties is None:
            return_properties = ["id", "description"]

        property_returns = ", ".join([f"node.{prop} AS {prop}" for prop in return_properties])

        query = f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $top_k,
            $query_embedding
        )
        YIELD node, score
        RETURN {property_returns}, score
        ORDER BY score DESC
        """

        try:
            results = self.graph.query(
                query,
                params={
                    "index_name": ENTITY_VECTOR_INDEX,
                    "top_k": top_k,
                    "query_embedding": query_embedding
                }
            )

            return results
        except Exception as e:
            print(f"❌ Neo4j entity vector search failed: {e}")
            return []

    def search_chunks_with_filter(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        file_name: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        带过滤条件的 Chunk 搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回的最大结果数
            file_name: 过滤指定文件名
            min_score: 最小相似度阈值

        Returns:
            List[Dict]: 搜索结果列表
        """
        query = """
        CALL db.index.vector.queryNodes(
            $index_name,
            $top_k,
            $query_embedding
        )
        YIELD node, score
        WHERE ($file_name IS NULL OR node.fileName = $file_name)
          AND ($min_score IS NULL OR score >= $min_score)
        RETURN node.id AS id,
               node.text AS text,
               node.fileName AS fileName,
               node.position AS position,
               score
        ORDER BY score DESC
        """

        try:
            results = self.graph.query(
                query,
                params={
                    "index_name": CHUNK_VECTOR_INDEX,
                    "top_k": top_k * 2,  # 预留空间用于过滤
                    "query_embedding": query_embedding,
                    "file_name": file_name,
                    "min_score": min_score
                }
            )

            # 确保返回的结果数不超过 top_k
            return results[:top_k]
        except Exception as e:
            print(f"❌ Neo4j filtered vector search failed: {e}")
            return []

    def check_index_exists(self, index_name: str) -> bool:
        """
        检查指定的 vector index 是否存在

        Args:
            index_name: 索引名称

        Returns:
            bool: 索引是否存在
        """
        query = """
        SHOW INDEXES
        YIELD name, type
        WHERE name = $index_name AND type = 'VECTOR'
        RETURN count(*) > 0 AS exists
        """

        try:
            result = self.graph.query(
                query,
                params={"index_name": index_name}
            )
            return result[0]["exists"] if result else False
        except Exception as e:
            print(f"⚠️ Failed to check index existence: {e}")
            return False

    def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """
        获取 vector index 的统计信息

        Args:
            index_name: 索引名称

        Returns:
            Dict: 索引统计信息，包含 name, type, state, populationPercent 等
        """
        query = """
        SHOW INDEXES
        YIELD name, type, state, populationPercent, uniqueness
        WHERE name = $index_name
        RETURN name, type, state, populationPercent, uniqueness
        """

        try:
            results = self.graph.query(
                query,
                params={"index_name": index_name}
            )
            return results[0] if results else None
        except Exception as e:
            print(f"⚠️ Failed to get index stats: {e}")
            return None
