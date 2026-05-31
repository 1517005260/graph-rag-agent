import hashlib
from .base import CacheKeyStrategy


class SimpleCacheKeyStrategy(CacheKeyStrategy):
    """简单的MD5哈希缓存键策略"""
    
    def generate_key(self, query: str, **kwargs) -> str:
        """使用查询字符串的MD5哈希生成缓存键"""
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()