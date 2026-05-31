import hashlib
from .base import CacheKeyStrategy


class ContextAwareCacheKeyStrategy(CacheKeyStrategy):
    """上下文感知的缓存键策略，考虑会话历史"""
    
    def __init__(self, context_window: int = 3):
        """
        初始化上下文感知缓存键策略
        
        参数:
            context_window: 要考虑的前几条会话历史记录
        """
        self.context_window = context_window
        self.conversation_history = {}
        self.history_versions = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
        
        # 增加版本号，确保上下文变化时键也会变化
        self.history_versions[thread_id] += 1
    
    def generate_key(self, query: str, **kwargs) -> str:
        """生成上下文感知的缓存键"""
        thread_id = kwargs.get("thread_id", "default")
        
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        
        # 获取历史版本号
        version = self.history_versions.get(thread_id, 0)
        
        # 构建上下文字符串 - 包含最近的n条消息
        context = " ".join(history[-self.context_window:] if self.context_window > 0 else [])
        
        # 组合上下文、线程ID、版本和查询生成缓存键
        # 确保线程ID被包含在键中
        combined = f"thread:{thread_id}|ctx:{context}|v{version}|{query}".strip()
        return hashlib.md5(combined.encode('utf-8')).hexdigest()


class ContextAndKeywordAwareCacheKeyStrategy(CacheKeyStrategy):
    """结合上下文和关键词的缓存键策略，同时考虑会话历史和关键词"""
    
    def __init__(self, context_window: int = 3):
        """
        初始化上下文与关键词感知的缓存键策略
        
        参数:
            context_window: 要考虑的前几条会话历史记录
        """
        self.context_window = context_window
        self.conversation_history = {}
        self.history_versions = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
        
        # 增加版本号，确保上下文变化时键也会变化
        self.history_versions[thread_id] += 1
    
    def generate_key(self, query: str, **kwargs) -> str:
        """生成同时考虑上下文和关键词的缓存键"""
        thread_id = kwargs.get("thread_id", "default")
        key_parts = [f"thread:{thread_id}", query.strip()]
        
        # 添加上下文信息
        history = self.conversation_history.get(thread_id, [])
        version = self.history_versions.get(thread_id, 0)
        
        # 构建上下文字符串 - 包含最近的n条消息
        if self.context_window > 0 and history:
            context = " ".join(history[-self.context_window:])
            key_parts.append(f"ctx:{hashlib.md5(context.encode('utf-8')).hexdigest()}")
        
        # 添加版本号
        key_parts.append(f"v:{version}")
        
        # 添加低级关键词
        low_level_keywords = kwargs.get("low_level_keywords", [])
        if low_level_keywords:
            key_parts.append("low:" + ",".join(sorted(low_level_keywords)))
        
        # 添加高级关键词
        high_level_keywords = kwargs.get("high_level_keywords", [])
        if high_level_keywords:
            key_parts.append("high:" + ",".join(sorted(high_level_keywords)))
        
        # 生成最终的键
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()