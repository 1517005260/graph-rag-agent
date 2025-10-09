import hashlib
from .base import CacheKeyStrategy

class GlobalCacheKeyStrategy(CacheKeyStrategy):
    """全局缓存键策略，忽略会话ID和其他上下文信息"""
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        仅使用查询内容生成缓存键，忽略线程ID和其他上下文参数
        
        参数:
            query: 查询字符串
            **kwargs: 其他参数（被忽略）
            
        返回:
            str: 生成的缓存键
        """
        # 移除可能的前缀（如"generate:"）
        if ":" in query:
            parts = query.split(":", 1)
            if len(parts) > 1:
                query = parts[1]
                
        # 直接使用查询内容的MD5哈希作为缓存键
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()