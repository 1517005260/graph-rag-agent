"""
语义缓存 (Semantic Cache)

基于向量相似度的查询缓存系统，用于检测语义相似的问题并返回缓存的答案。

特点：
- 使用余弦相似度计算查询语义相似性
- 内存 LRU 缓存（生产环境可替换为 Redis Vector）
- 可配置相似度阈值
- 自动限制缓存大小
- 线程安全

使用方式：
    from server.utils.semantic_cache import get_semantic_cache

    cache = get_semantic_cache()

    # 检查缓存
    cached = cache.get(query_embedding)
    if cached:
        return cached

    # 添加到缓存
    cache.add(query_embedding, response)
"""

import numpy as np
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque
import logging

_LOGGER = logging.getLogger(__name__)


class SemanticCache:
    """
    语义缓存类

    基于向量相似度的查询缓存，通过比较查询向量的余弦相似度来判断是否命中缓存。

    Attributes:
        threshold: 相似度阈值 (0-1)，默认 0.95
        max_size: 最大缓存条目数，默认 1000
        cache: 缓存存储 (使用 deque 实现 LRU)
        hits: 缓存命中次数
        misses: 缓存未命中次数
    """

    def __init__(self, threshold: float = 0.95, max_size: int = 1000):
        """
        初始化语义缓存

        Args:
            threshold: 相似度阈值，值越高越严格（默认 0.95）
            max_size: 最大缓存条目数（默认 1000）
        """
        self.threshold = threshold
        self.max_size = max_size

        # 使用 deque 实现 LRU（最近最少使用）
        # 每个元素是一个字典：{'embedding': np.array, 'response': str, 'timestamp': datetime, 'hit_count': int}
        self.cache: deque = deque(maxlen=max_size)

        # 统计信息
        self.hits = 0
        self.misses = 0

        # 线程锁，确保线程安全
        self._lock = threading.Lock()

        _LOGGER.info(f"语义缓存已初始化: threshold={threshold}, max_size={max_size}")

    def get(self, query_embedding: List[float]) -> Optional[str]:
        """
        根据查询向量查找缓存

        计算查询向量与缓存中所有向量的余弦相似度，如果找到相似度超过阈值的条目，
        则返回缓存的响应。采用 LRU 策略，命中的条目会被移到队列末尾。

        Args:
            query_embedding: 查询的向量表示

        Returns:
            如果找到相似查询，返回缓存的响应；否则返回 None
        """
        if not self.cache:
            self.misses += 1
            return None

        with self._lock:
            query_vec = np.array(query_embedding)
            query_norm = np.linalg.norm(query_vec)

            if query_norm == 0:
                _LOGGER.warning("查询向量范数为 0，无法计算相似度")
                self.misses += 1
                return None

            best_similarity = -1
            best_item = None
            best_index = -1

            # 遍历缓存，寻找最相似的条目
            for idx, item in enumerate(self.cache):
                cache_vec = item['embedding']
                cache_norm = np.linalg.norm(cache_vec)

                if cache_norm == 0:
                    continue

                # 计算余弦相似度
                similarity = np.dot(query_vec, cache_vec) / (query_norm * cache_norm)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_item = item
                    best_index = idx

            # 检查是否超过阈值
            if best_similarity >= self.threshold:
                self.hits += 1

                # 更新命中次数和时间戳
                best_item['hit_count'] += 1
                best_item['last_hit'] = datetime.now()

                # LRU 策略：将命中的条目移到队列末尾（最近使用）
                self.cache.remove(best_item)
                self.cache.append(best_item)

                _LOGGER.info(
                    f"✅ 语义缓存命中！相似度: {best_similarity:.4f}, "
                    f"累计命中 {best_item['hit_count']} 次"
                )

                return best_item['response']
            else:
                self.misses += 1
                _LOGGER.debug(f"语义缓存未命中，最高相似度: {best_similarity:.4f}")
                return None

    def add(self, query_embedding: List[float], response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加查询-响应对到缓存

        Args:
            query_embedding: 查询的向量表示
            response: 对应的响应
            metadata: 可选的元数据（如查询文本、Agent 类型等）
        """
        with self._lock:
            cache_item = {
                'embedding': np.array(query_embedding),
                'response': response,
                'timestamp': datetime.now(),
                'last_hit': datetime.now(),
                'hit_count': 0,
                'metadata': metadata or {}
            }

            # deque 会自动处理 maxlen，超出时移除最旧的条目
            self.cache.append(cache_item)

            _LOGGER.debug(f"已添加到语义缓存，当前缓存大小: {len(self.cache)}/{self.max_size}")

    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            _LOGGER.info("语义缓存已清空")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total_requests,
                "hit_rate": f"{hit_rate:.2%}",
                "threshold": self.threshold
            }

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"SemanticCache("
            f"size={stats['size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, "
            f"misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']}"
            f")"
        )


# 全局单例
_semantic_cache_instance: Optional[SemanticCache] = None
_cache_lock = threading.Lock()


def get_semantic_cache(threshold: float = 0.95, max_size: int = 1000) -> SemanticCache:
    """
    获取语义缓存单例

    Args:
        threshold: 相似度阈值（仅在首次创建时使用）
        max_size: 最大缓存大小（仅在首次创建时使用）

    Returns:
        SemanticCache 实例
    """
    global _semantic_cache_instance

    if _semantic_cache_instance is None:
        with _cache_lock:
            if _semantic_cache_instance is None:
                _semantic_cache_instance = SemanticCache(
                    threshold=threshold,
                    max_size=max_size
                )

    return _semantic_cache_instance


def reset_semantic_cache():
    """重置语义缓存单例（主要用于测试）"""
    global _semantic_cache_instance

    with _cache_lock:
        if _semantic_cache_instance is not None:
            _semantic_cache_instance.clear()
        _semantic_cache_instance = None
