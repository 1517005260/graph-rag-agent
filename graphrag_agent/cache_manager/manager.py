import time
import os
from typing import Any, Dict, Optional, Callable
from pathlib import Path

from .strategies import CacheKeyStrategy, SimpleCacheKeyStrategy, ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy
from .backends import CacheStorageBackend, MemoryCacheBackend, HybridCacheBackend, ThreadSafeCacheBackend
from .models import CacheItem
from .vector_similarity import VectorSimilarityMatcher, get_cache_embedding_provider

from graphrag_agent.config.settings import similarity_threshold as st
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class CacheManager:
    """统一缓存管理器，提供高级缓存功能和向量相似性匹配"""
    
    def __init__(self, 
                 key_strategy: CacheKeyStrategy = None, 
                 storage_backend: CacheStorageBackend = None,
                 cache_dir: str = "./cache",
                 memory_only: bool = False,
                 max_memory_size: int = 100,
                 max_disk_size: int = 1000,
                 thread_safe: bool = True,
                 enable_vector_similarity: bool = True,
                 similarity_threshold: float = st,
                 max_vectors: int = 10000):
        """
        初始化缓存管理器
        
        参数:
            key_strategy: 缓存键策略
            storage_backend: 存储后端
            cache_dir: 缓存目录
            memory_only: 是否仅使用内存
            max_memory_size: 最大内存缓存数量
            max_disk_size: 最大磁盘缓存数量
            thread_safe: 是否线程安全
            enable_vector_similarity: 是否启用向量相似性匹配
            similarity_threshold: 向量相似度阈值
            max_vectors: 最大向量数量
        """
       # 设置缓存键策略
        self.key_strategy = key_strategy or SimpleCacheKeyStrategy()
        
        # 设置存储后端
        backend = self._create_storage_backend(
            storage_backend, memory_only, cache_dir, 
            max_memory_size, max_disk_size
        )
        
        # 如果需要线程安全，添加包装器
        self.storage = ThreadSafeCacheBackend(backend) if thread_safe else backend
        
        # 向量相似性匹配器
        self.enable_vector_similarity = enable_vector_similarity
        if enable_vector_similarity:
            # 确保缓存目录存在
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            # 创建向量索引文件路径
            vector_index_file = f"{cache_dir}/vector_index" if not memory_only else None

            # 获取配置的嵌入提供者
            embedding_provider = get_cache_embedding_provider()

            self.vector_matcher = VectorSimilarityMatcher(
                embedding_provider=embedding_provider,
                similarity_threshold=similarity_threshold,
                max_vectors=max_vectors,
                index_file=vector_index_file
            )
        else:
            self.vector_matcher = None
        
        # 性能指标收集
        self.performance_metrics = {
            'exact_hits': 0,
            'vector_hits': 0,
            'misses': 0,
            'total_queries': 0
        }
    
    def _create_storage_backend(self, storage_backend, memory_only, cache_dir, 
                              max_memory_size, max_disk_size) -> CacheStorageBackend:
        """创建存储后端"""
        if storage_backend:
            return storage_backend
        elif memory_only:
            return MemoryCacheBackend(max_size=max_memory_size)
        else:
            return HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=max_memory_size,
                disk_max_size=max_disk_size
            )
    
    def _get_consistent_key(self, query: str, **kwargs) -> str:
        """生成一致的缓存键"""
        return self.key_strategy.generate_key(query, **kwargs)
    
    def _extract_context_info(self, **kwargs) -> Dict[str, Any]:
        """提取上下文信息用于向量匹配"""
        return {
            'thread_id': kwargs.get('thread_id', 'default'),
            'keywords': kwargs.get('keywords', []),
            'low_level_keywords': kwargs.get('low_level_keywords', []),
            'high_level_keywords': kwargs.get('high_level_keywords', [])
        }
    
    def get(self, query: str, skip_validation: bool = False, **kwargs) -> Optional[Any]:
        """获取缓存内容，支持精确匹配和向量相似性匹配"""
        start_time = time.time()
        self.performance_metrics['total_queries'] += 1
        
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 首先尝试精确匹配
        cached_data = self.storage.get(key)
        if cached_data is not None:
            self.performance_metrics['exact_hits'] += 1
            cache_item = CacheItem.from_any(cached_data)
            cache_item.update_access_stats()
            
            # 验证逻辑
            if skip_validation or cache_item.is_high_quality():
                content = cache_item.get_content()
                self.performance_metrics["get_time"] = time.time() - start_time
                return content
            
            content = cache_item.get_content()
            self.performance_metrics["get_time"] = time.time() - start_time
            return content
        
        # 如果精确匹配失败且启用了向量相似性，尝试向量匹配
        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            similar_keys = self.vector_matcher.find_similar(query, context_info, top_k=3)
            
            for similar_key, similarity_score in similar_keys:
                cached_data = self.storage.get(similar_key)
                if cached_data is not None:
                    self.performance_metrics['vector_hits'] += 1
                    cache_item = CacheItem.from_any(cached_data)
                    cache_item.update_access_stats()
                    
                    # 添加相似性信息到元数据
                    cache_item.metadata['similarity_score'] = similarity_score
                    cache_item.metadata['original_query'] = query
                    cache_item.metadata['matched_via_vector'] = True
                    
                    if skip_validation or cache_item.is_high_quality():
                        content = cache_item.get_content()
                        self.performance_metrics["get_time"] = time.time() - start_time
                        return content
                    
                    content = cache_item.get_content()
                    self.performance_metrics["get_time"] = time.time() - start_time
                    return content
        
        # 未找到匹配的缓存
        self.performance_metrics['misses'] += 1
        self.performance_metrics["get_time"] = time.time() - start_time
        return None
    
    def get_fast(self, query: str, **kwargs) -> Optional[Any]:
        """快速获取高质量缓存内容"""
        start_time = time.time()
        
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is not None:
            cache_item = CacheItem.from_any(cached_data)
            
            # 只返回高质量缓存
            if cache_item.is_high_quality():
                cache_item.update_access_stats()
                
                # 更新上下文历史
                self._update_strategy_history(query, **kwargs)
                
                content = cache_item.get_content()
                self.performance_metrics["fast_get_time"] = time.time() - start_time
                return content
        
        # 尝试向量相似性匹配高质量缓存
        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            similar_keys = self.vector_matcher.find_similar(query, context_info, top_k=1)
            
            for similar_key, similarity_score in similar_keys:
                cached_data = self.storage.get(similar_key)
                if cached_data is not None:
                    cache_item = CacheItem.from_any(cached_data)
                    
                    if cache_item.is_high_quality():
                        cache_item.update_access_stats()
                        cache_item.metadata['similarity_score'] = similarity_score
                        cache_item.metadata['matched_via_vector'] = True
                        
                        content = cache_item.get_content()
                        self.performance_metrics["fast_get_time"] = time.time() - start_time
                        return content
        
        self.performance_metrics["fast_get_time"] = time.time() - start_time
        return None
    
    def set(self, query: str, result: Any, **kwargs) -> None:
        """设置缓存内容"""
        start_time = time.time()
        
        # 更新策略历史
        self._update_strategy_history(query, **kwargs)
        
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 包装缓存项
        cache_item = self._wrap_cache_item(result)
        
        # 存储缓存项
        self.storage.set(key, cache_item.to_dict())
        
        # 添加到向量索引
        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            self.vector_matcher.add_vector(key, query, context_info)
        
        self.performance_metrics["set_time"] = time.time() - start_time
    
    def _update_strategy_history(self, query: str, **kwargs):
        """更新策略历史"""
        if isinstance(self.key_strategy, (ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy)):
            thread_id = kwargs.get("thread_id", "default")
            self.key_strategy.update_history(query, thread_id)
    
    def _wrap_cache_item(self, result: Any) -> CacheItem:
        """包装缓存项"""
        if isinstance(result, dict) and "content" in result and "metadata" in result:
            return CacheItem.from_dict(result)
        else:
            return CacheItem(result)
    
    def mark_quality(self, query: str, is_positive: bool, **kwargs) -> bool:
        """标记缓存质量"""
        start_time = time.time()
        
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            self.performance_metrics["mark_time"] = time.time() - start_time
            return False
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 标记质量
        cache_item.mark_quality(is_positive)
        
        # 更新缓存
        item_dict = cache_item.to_dict()
        if is_positive and cache_item.is_high_quality():
            item_dict["metadata"]["fast_path_eligible"] = True
        
        self.storage.set(key, item_dict)
        
        self.performance_metrics["mark_time"] = time.time() - start_time
        return True
    
    def delete(self, query: str, **kwargs) -> bool:
        """删除缓存项"""
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 从向量索引中删除
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.remove_vector(key)
        
        # 删除缓存项
        return self.storage.delete(key)
    
    def clear(self) -> None:
        """清空缓存"""
        self.storage.clear()
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.clear()
    
    def flush(self) -> None:
        """强制刷新所有待写入的数据到磁盘"""
        # 刷新存储后端
        if hasattr(self.storage, 'backend') and hasattr(self.storage.backend, 'flush'):
            self.storage.backend.flush()
        elif hasattr(self.storage, 'flush'):
            self.storage.flush()
        
        # 如果是混合缓存，需要刷新磁盘缓存部分
        if hasattr(self.storage, 'backend'):
            backend = self.storage.backend
            if hasattr(backend, 'disk_cache') and hasattr(backend.disk_cache, 'flush'):
                backend.disk_cache.flush()
        elif hasattr(self.storage, 'disk_cache') and hasattr(self.storage.disk_cache, 'flush'):
            self.storage.disk_cache.flush()
        
        # 保存向量索引
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.save_index()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = self.performance_metrics.copy()
        if metrics['total_queries'] > 0:
            metrics['exact_hit_rate'] = metrics['exact_hits'] / metrics['total_queries']
            metrics['vector_hit_rate'] = metrics['vector_hits'] / metrics['total_queries']
            metrics['total_hit_rate'] = (metrics['exact_hits'] + metrics['vector_hits']) / metrics['total_queries']
            metrics['miss_rate'] = metrics['misses'] / metrics['total_queries']
        return metrics
    
    def validate_answer(self, query: str, answer: str, validator: Callable[[str, str], bool] = None, **kwargs) -> bool:
        """验证答案质量"""
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            # 如果缓存不存在，直接使用验证函数
            if validator:
                return validator(query, answer)
            return self._default_validation(query, answer)
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 检查用户验证状态
        if cache_item.metadata.get("user_verified", False):
            return True
        
        # 检查质量分数
        quality_score = cache_item.metadata.get("quality_score", 0)
        if quality_score < 0:
            return False
        
        # 如果提供了自定义验证函数，使用它
        if validator:
            return validator(query, answer)
        
        return self._default_validation(query, answer)
    
    def _default_validation(self, query: str, answer: str) -> bool:
        """默认验证逻辑"""
        # 基本验证：长度检查
        if len(answer.strip()) < 10:
            return False
        
        # 检查答案是否与查询相关
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # 至少要有一些共同词汇
        common_words = query_words.intersection(answer_words)
        if len(common_words) == 0 and len(query_words) > 2:
            return False
        
        return True
    
    def save_vector_index(self):
        """保存向量索引"""
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.save_index()