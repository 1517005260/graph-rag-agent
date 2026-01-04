"""
实体对齐 (Entity Resolution) 模块

功能：
- 基于字符串相似度识别重复实体
- 合并相似实体节点
- 保留所有关系和属性

实现方式：
- V1 (规则版): Levenshtein 距离 + 相似度阈值
- 使用 Neo4j APOC 的 mergeNodes 函数

性能优化建议：
- 生产环境建议使用 Blocking 技术（按实体类型、首字母等分块）
- 对于超大规模图谱，考虑使用 MinHash LSH 等近似算法
"""
import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

try:
    from Levenshtein import ratio as levenshtein_ratio
except ImportError:
    # 如果没有安装 python-Levenshtein，使用简单的 SequenceMatcher
    from difflib import SequenceMatcher

    def levenshtein_ratio(s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio()

from graphrag_agent.config.neo4jdb import get_db_manager

_LOGGER = logging.getLogger(__name__)


class EntityResolver:
    """
    实体对齐器

    使用字符串相似度算法识别和合并重复实体。
    """

    def __init__(self, threshold: float = 0.9, batch_size: int = 1000):
        """
        初始化实体对齐器

        Args:
            threshold: 相似度阈值（0-1），默认 0.9
            batch_size: 批处理大小，用于分批处理大量实体
        """
        self.db_manager = get_db_manager()
        self.threshold = threshold
        self.batch_size = batch_size

        _LOGGER.info(f"实体对齐器初始化：相似度阈值 = {threshold}")

    def resolve_entities(
        self,
        entity_type: Optional[str] = None,
        use_blocking: bool = True
    ) -> Dict[str, Any]:
        """
        执行实体对齐的核心逻辑

        Args:
            entity_type: 指定实体类型（None 表示所有类型）
            use_blocking: 是否使用分块策略优化性能

        Returns:
            对齐结果统计
        """
        _LOGGER.info("开始实体对齐流程...")

        # 1. 获取所有实体
        entities = self._fetch_entities(entity_type)

        if not entities:
            _LOGGER.warning("没有找到需要对齐的实体")
            return {
                "status": "success",
                "total_entities": 0,
                "merge_groups": 0,
                "merged_count": 0
            }

        _LOGGER.info(f"获取到 {len(entities)} 个实体")

        # 2. 查找相似实体组
        if use_blocking:
            merge_candidates = self._find_similar_entities_with_blocking(entities)
        else:
            merge_candidates = self._find_similar_entities_naive(entities)

        _LOGGER.info(f"发现 {len(merge_candidates)} 组相似实体")

        # 3. 在 Neo4j 中执行合并
        merged_count = 0
        for group in merge_candidates:
            try:
                self._merge_nodes_in_db(group)
                merged_count += 1
            except Exception as e:
                _LOGGER.error(f"合并实体组失败 {group}: {e}")

        result = {
            "status": "success",
            "total_entities": len(entities),
            "merge_groups": len(merge_candidates),
            "merged_count": merged_count,
            "threshold": self.threshold
        }

        _LOGGER.info(f"实体对齐完成：合并了 {merged_count} 组实体")

        return result

    def _fetch_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        从 Neo4j 获取实体列表

        Args:
            entity_type: 实体类型标签

        Returns:
            实体列表（包含 id 和 labels）
        """
        if entity_type:
            query = """
            MATCH (e:`__Entity__`)
            WHERE $entity_type IN labels(e)
            RETURN e.id as name, labels(e) as labels
            """
            params = {"entity_type": entity_type}
        else:
            query = """
            MATCH (e:`__Entity__`)
            RETURN e.id as name, labels(e) as labels
            """
            params = {}

        try:
            result = self.db_manager.graph.query(query, params)
            return [dict(record) for record in result]
        except Exception as e:
            _LOGGER.exception(f"获取实体列表失败: {e}")
            return []

    def _find_similar_entities_naive(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        朴素算法：O(N^2) 查找相似实体

        适用于小规模数据集（< 10000 个实体）

        Args:
            entities: 实体列表

        Returns:
            相似实体组列表
        """
        merge_candidates = []
        visited: Set[str] = set()

        for i in range(len(entities)):
            name1 = entities[i]['name']

            if name1 in visited:
                continue

            group = [name1]

            for j in range(i + 1, len(entities)):
                name2 = entities[j]['name']

                if name2 in visited:
                    continue

                # 计算相似度
                similarity = levenshtein_ratio(name1, name2)

                if similarity >= self.threshold:
                    group.append(name2)
                    visited.add(name2)

            if len(group) > 1:
                merge_candidates.append(group)
                visited.add(name1)

        return merge_candidates

    def _find_similar_entities_with_blocking(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        分块策略：按首字母分组，降低比较次数

        适用于中等规模数据集（10000 - 100000 个实体）

        Args:
            entities: 实体列表

        Returns:
            相似实体组列表
        """
        # 按首字母分块
        blocks = defaultdict(list)

        for entity in entities:
            name = entity['name']
            if not name:
                continue

            # 使用首字母作为块键（支持中文和英文）
            key = name[0].lower()
            blocks[key].append(entity)

        _LOGGER.info(f"使用分块策略，共 {len(blocks)} 个块")

        # 对每个块内部执行相似度比较
        all_merge_candidates = []

        for block_key, block_entities in blocks.items():
            if len(block_entities) < 2:
                continue

            _LOGGER.debug(f"处理块 '{block_key}'，包含 {len(block_entities)} 个实体")

            # 对块内实体执行朴素算法
            block_candidates = self._find_similar_entities_naive(block_entities)
            all_merge_candidates.extend(block_candidates)

        return all_merge_candidates

    def _merge_nodes_in_db(self, entity_names: List[str]):
        """
        调用 Neo4j APOC 合并节点

        使用 apoc.refactor.mergeNodes 合并多个节点为一个。
        - 保留第一个节点
        - 合并所有关系
        - 合并所有属性

        Args:
            entity_names: 要合并的实体名称列表

        Raises:
            Exception: 如果合并失败
        """
        if len(entity_names) < 2:
            _LOGGER.warning(f"实体组少于 2 个，跳过合并: {entity_names}")
            return

        _LOGGER.info(f"合并实体组: {entity_names} -> {entity_names[0]}")

        # APOC mergeNodes 语法
        cypher = """
        MATCH (e:`__Entity__`)
        WHERE e.id IN $names
        WITH collect(e) as nodes
        CALL apoc.refactor.mergeNodes(nodes, {
            properties: 'combine',
            mergeRels: true
        })
        YIELD node
        RETURN node.id as merged_name
        """

        try:
            result = self.db_manager.graph.query(
                cypher,
                params={"names": entity_names}
            )

            if result:
                merged_name = result[0].get('merged_name')
                _LOGGER.info(f"成功合并为: {merged_name}")
            else:
                _LOGGER.warning(f"合并操作未返回结果: {entity_names}")

        except Exception as e:
            _LOGGER.exception(f"合并实体失败 {entity_names}: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取当前图谱的实体统计信息

        Returns:
            统计信息
        """
        query = """
        MATCH (e:`__Entity__`)
        RETURN count(e) as total_entities,
               count(DISTINCT labels(e)) as entity_types
        """

        try:
            result = self.db_manager.graph.query(query)
            if result:
                return dict(result[0])
            else:
                return {"total_entities": 0, "entity_types": 0}
        except Exception as e:
            _LOGGER.exception(f"获取统计信息失败: {e}")
            return {"total_entities": 0, "entity_types": 0, "error": str(e)}

    def preview_duplicates(
        self,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        预览可能重复的实体（不执行合并）

        Args:
            entity_type: 实体类型
            limit: 返回结果数量限制

        Returns:
            可能重复的实体组列表
        """
        entities = self._fetch_entities(entity_type)

        if not entities:
            return []

        merge_candidates = self._find_similar_entities_with_blocking(entities)

        # 返回前 N 组
        results = []
        for group in merge_candidates[:limit]:
            results.append({
                "entities": group,
                "count": len(group),
                "merge_to": group[0]
            })

        return results
