import threading
from typing import Any, Optional
from .base import CacheStorageBackend


class ThreadSafeCacheBackend(CacheStorageBackend):
    """线程安全的缓存后端装饰器"""

    def __init__(self, backend: CacheStorageBackend):
        """
        初始化线程安全缓存后端
        
        参数:
            backend: 被装饰的缓存后端
        """
        self.backend = backend
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            return self.backend.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存"""
        with self.lock:
            self.backend.set(key, value)
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            return self.backend.delete(key)
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.backend.clear()