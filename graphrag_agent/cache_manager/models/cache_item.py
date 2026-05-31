import time
import json
from typing import Any, Dict, Optional


class CacheItem:
    """缓存项包装类，支持元数据和序列化"""
    
    def __init__(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """初始化缓存项"""
        self.content = content
        self.metadata = self._initialize_metadata(metadata)
    
    def _initialize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """初始化元数据，确保包含必要字段"""
        meta = metadata or {}
        
        defaults = {
            "created_at": time.time(),
            "quality_score": 0,
            "user_verified": False,
            "access_count": 0,
            "fast_path_eligible": False,
            "last_accessed": None,
            "similarity_score": None,
            "matched_via_vector": False,
            "original_query": None
        }
        
        # 合并默认值和提供的元数据
        for key, default_value in defaults.items():
            if key not in meta:
                meta[key] = default_value
        
        return meta
    
    def get_content(self) -> Any:
        """获取内容"""
        return self.content
    
    def is_high_quality(self) -> bool:
        """判断是否为高质量缓存"""
        return (self.metadata.get("user_verified", False) or 
                self.metadata.get("quality_score", 0) > 2 or
                self.metadata.get("fast_path_eligible", False))
    
    def mark_quality(self, is_positive: bool) -> None:
        """标记缓存质量"""
        if is_positive:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = current_score + 1
            self.metadata["user_verified"] = True
            self.metadata["fast_path_eligible"] = True
        else:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = max(-5, current_score - 2)  # 允许负分，但有下限
            self.metadata["fast_path_eligible"] = False
    
    def update_access_stats(self) -> None:
        """更新访问统计"""
        self.metadata["access_count"] = self.metadata.get("access_count", 0) + 1
        self.metadata["last_accessed"] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "metadata": self.metadata
        }
    
    def to_json(self, ensure_ascii: bool = False) -> str:
        """转换为JSON字符串"""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, default=str)
        except (TypeError, ValueError) as e:
            # 如果序列化失败，返回错误信息
            return json.dumps({
                "content": f"Serialization failed: {str(e)}",
                "metadata": self.metadata
            })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """从字典创建缓存项"""
        try:
            if isinstance(data, dict):
                if "content" in data and "metadata" in data:
                    metadata = data["metadata"]
                    if not isinstance(metadata, dict):
                        metadata = {}
                    return cls(data["content"], metadata)
                else:
                    # 处理简单格式
                    return cls(data)
            else:
                return cls(data)
        except Exception as e:
            # 返回错误缓存项，确保程序不会崩溃
            return cls(f"Error deserializing cache item: {str(e)}", {
                "created_at": time.time(),
                "quality_score": -10,  # 标记为低质量
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
            })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CacheItem':
        """从JSON字符串创建缓存项"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            return cls(f"JSON decode error: {str(e)}", {
                "created_at": time.time(),
                "quality_score": -10,
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
            })
    
    @classmethod
    def from_any(cls, data: Any) -> 'CacheItem':
        """从任意数据创建缓存项，具有自动类型检测"""
        if isinstance(data, cls):
            return data
        elif isinstance(data, str):
            # 尝试解析JSON
            try:
                parsed_data = json.loads(data)
                return cls.from_dict(parsed_data)
            except json.JSONDecodeError:
                # 如果不是JSON，直接作为内容
                return cls(data)
        elif isinstance(data, dict) and "content" in data:
            return cls.from_dict(data)
        else:
            return cls(data)
    
    def get_age(self) -> float:
        """获取缓存项的年龄（秒）"""
        created_at = self.metadata.get("created_at", time.time())
        return time.time() - created_at
    
    def is_expired(self, max_age: float) -> bool:
        """检查缓存项是否过期"""
        return self.get_age() > max_age
    
    def __repr__(self) -> str:
        """字符串表示"""
        content_preview = str(self.content)[:50]
        if len(str(self.content)) > 50:
            content_preview += "..."
        
        return (f"CacheItem(content='{content_preview}', "
                f"quality_score={self.metadata.get('quality_score', 0)}, "
                f"access_count={self.metadata.get('access_count', 0)})")