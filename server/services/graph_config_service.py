"""
图谱配置服务 (GraphConfigService)
提供配置的单例管理、内存缓存和热更新机制

核心功能：
1. 单例模式 - 全局唯一实例
2. 内存缓存 - 避免频繁文件 I/O
3. 热更新 - API 修改后立即生效
4. 线程安全 - 支持并发访问
"""

import threading
from typing import Optional, Dict
from datetime import datetime

from graphrag_agent.config.graph_config_storage import get_storage
from graphrag_agent.config.graph_config_model import GraphConfig


class GraphConfigService:
    """图谱配置服务单例"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """私有构造函数，防止外部直接实例化"""
        self._cached_config: Optional[GraphConfig] = None
        self._cache_lock = threading.Lock()
        self._last_reload_time: Optional[datetime] = None

    @classmethod
    def get_instance(cls) -> 'GraphConfigService':
        """
        获取单例实例（线程安全）

        Returns:
            GraphConfigService 实例
        """
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定模式
                if cls._instance is None:
                    cls._instance = GraphConfigService()
        return cls._instance

    def get_config(self) -> Optional[GraphConfig]:
        """
        获取最新配置（优先读内存缓存，无缓存则读文件）

        Returns:
            GraphConfig 实例，如果不存在则返回 None
        """
        with self._cache_lock:
            if self._cached_config is None:
                # 缓存为空，从文件加载
                storage = get_storage()
                self._cached_config = storage.load()
                self._last_reload_time = datetime.now()

            return self._cached_config

    def update_config(self, new_config_dict: Dict) -> GraphConfig:
        """
        更新配置并刷新内存缓存

        Args:
            new_config_dict: 新配置字典

        Returns:
            保存后的 GraphConfig 实例

        Raises:
            ValueError: 配置数据验证失败
            Exception: 保存失败
        """
        with self._cache_lock:
            # 1. 校验数据（使用 Pydantic 验证）
            try:
                config_obj = GraphConfig(**new_config_dict)
            except Exception as e:
                raise ValueError(f"配置数据验证失败: {str(e)}")

            # 2. 持久化到文件
            storage = get_storage()
            success = storage.save(config_obj)

            if not success:
                raise Exception("配置保存失败")

            # 3. 刷新内存缓存
            self._cached_config = config_obj
            self._last_reload_time = datetime.now()

            print(f"[GraphConfigService] 配置已更新: {config_obj.project_name} @ {self._last_reload_time}")

            return config_obj

    def update_config_obj(self, config_obj: GraphConfig) -> GraphConfig:
        """
        直接使用 GraphConfig 对象更新配置

        Args:
            config_obj: GraphConfig 实例

        Returns:
            保存后的 GraphConfig 实例
        """
        with self._cache_lock:
            # 1. 持久化到文件
            storage = get_storage()
            success = storage.save(config_obj)

            if not success:
                raise Exception("配置保存失败")

            # 2. 刷新内存缓存
            self._cached_config = config_obj
            self._last_reload_time = datetime.now()

            print(f"[GraphConfigService] 配置已更新: {config_obj.project_name} @ {self._last_reload_time}")

            return config_obj

    def reload(self) -> Optional[GraphConfig]:
        """
        强制重新加载配置（从文件读取，刷新缓存）

        Returns:
            GraphConfig 实例，如果不存在则返回 None
        """
        with self._cache_lock:
            storage = get_storage()
            self._cached_config = storage.load()
            self._last_reload_time = datetime.now()

            if self._cached_config:
                print(f"[GraphConfigService] 配置已重载: {self._cached_config.project_name} @ {self._last_reload_time}")
            else:
                print(f"[GraphConfigService] 配置重载完成，当前无配置 @ {self._last_reload_time}")

            return self._cached_config

    def delete_config(self) -> bool:
        """
        删除配置（删除文件并清空缓存）

        Returns:
            是否删除成功
        """
        with self._cache_lock:
            # 1. 删除文件
            storage = get_storage()
            success = storage.delete()

            if success:
                # 2. 清空缓存
                self._cached_config = None
                self._last_reload_time = datetime.now()
                print(f"[GraphConfigService] 配置已删除 @ {self._last_reload_time}")

            return success

    def exists(self) -> bool:
        """
        检查配置是否存在

        Returns:
            配置是否存在
        """
        # 先检查缓存
        if self._cached_config is not None:
            return True

        # 检查文件
        storage = get_storage()
        return storage.exists()

    def get_cache_status(self) -> Dict:
        """
        获取缓存状态信息（用于调试）

        Returns:
            缓存状态字典
        """
        with self._cache_lock:
            return {
                "has_cache": self._cached_config is not None,
                "project_name": self._cached_config.project_name if self._cached_config else None,
                "last_reload_time": self._last_reload_time.isoformat() if self._last_reload_time else None,
                "domain_count": len(self._cached_config.domain_definitions) if self._cached_config else 0,
                "bridge_count": len(self._cached_config.bridge_definitions) if self._cached_config else 0
            }


# 全局便捷函数
def get_config_service() -> GraphConfigService:
    """
    获取全局配置服务实例

    Returns:
        GraphConfigService 单例
    """
    return GraphConfigService.get_instance()
