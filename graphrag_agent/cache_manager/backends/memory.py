import time
from typing import Any, Optional
from .base import CacheStorageBackend


class MemoryCacheBackend(CacheStorageBackend):
    """内存缓存后端实现"""
    
    def __init__(self, max_size: int = 100):
        """
        初始化内存缓存后端
        
        参数:
            max_size: 缓存最大项数
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}  # 用于LRU淘汰策略
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存项值，不存在则返回None
        """
        value = self.cache.get(key)
        if value is not None:
            # 更新访问时间（LRU策略）
            self.access_times[key] = time.time()
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满，删除最久未使用的项
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        参数:
            key: 缓存键
            
        返回:
            bool: 是否成功删除
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()  # 确保同时清空访问时间字典
    
    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存项"""
        if not self.access_times:
            return
        
        # 找出最旧的项
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.delete(oldest_key)  # 使用delete方法确保同时清理access_times
        
    def cleanup_unused(self) -> None:
        """清理access_times中未使用的键"""
        # 找出那些在access_times中存在但在cache中不存在的键
        unused_keys = [k for k in self.access_times if k not in self.cache]
        for key in unused_keys:
            del self.access_times[key]