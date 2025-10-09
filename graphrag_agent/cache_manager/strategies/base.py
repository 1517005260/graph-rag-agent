from abc import ABC, abstractmethod


class CacheKeyStrategy(ABC):
    """缓存键生成策略的抽象基类"""
    
    @abstractmethod
    def generate_key(self, query: str, **kwargs) -> str:
        """生成缓存键"""
        pass