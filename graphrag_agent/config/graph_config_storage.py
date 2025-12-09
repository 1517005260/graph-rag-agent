"""
图谱配置存储管理
负责 GraphConfig 的持久化和加载
"""

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from .graph_config_model import GraphConfig, IndustryTemplate


class GraphConfigStorage:
    """图谱配置存储管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置存储

        Args:
            config_path: 配置文件路径，默认为 ./graph_config.json
        """
        if config_path is None:
            # 默认保存在项目根目录
            config_path = os.path.join(os.getcwd(), "graph_config.json")

        self.config_path = Path(config_path)
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """确保配置文件目录存在"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, config: GraphConfig) -> bool:
        """
        保存配置到文件

        Args:
            config: GraphConfig 实例

        Returns:
            bool: 是否保存成功
        """
        try:
            # 更新时间戳
            config.updated_at = datetime.now()

            # 转换为 JSON
            config_dict = config.model_dump(mode='json')

            # 写入文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def load(self) -> Optional[GraphConfig]:
        """
        从文件加载配置

        Returns:
            GraphConfig 实例，如果文件不存在或加载失败则返回 None
        """
        if not self.config_path.exists():
            return None

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            return GraphConfig(**config_dict)
        except Exception as e:
            print(f"加载配置失败: {e}")
            return None

    def exists(self) -> bool:
        """检查配置文件是否存在"""
        return self.config_path.exists()

    def delete(self) -> bool:
        """删除配置文件"""
        try:
            if self.config_path.exists():
                self.config_path.unlink()
            return True
        except Exception as e:
            print(f"删除配置失败: {e}")
            return False

    def load_template(self, template_name: str) -> Optional[GraphConfig]:
        """
        加载行业模板

        Args:
            template_name: 模板名称 ('legal', 'ecommerce', 'medical')

        Returns:
            GraphConfig 实例，如果模板不存在则返回 None
        """
        templates = IndustryTemplate.get_all_templates()
        return templates.get(template_name)

    def list_templates(self) -> dict[str, dict]:
        """
        列出所有可用模板及其基本信息

        Returns:
            模板字典，格式为 {template_name: {project_name, industry, description}}
        """
        templates = IndustryTemplate.get_all_templates()

        return {
            name: {
                "project_name": config.project_name,
                "industry": config.industry,
                "description": config.description,
                "bridge_count": len(config.bridge_definitions),
                "domain_count": len(config.domain_definitions)
            }
            for name, config in templates.items()
        }

    def save_from_template(self, template_name: str,
                          project_name: Optional[str] = None,
                          created_by: Optional[str] = None) -> bool:
        """
        从模板创建并保存配置

        Args:
            template_name: 模板名称
            project_name: 可选的项目名称，用于覆盖模板默认名称
            created_by: 创建者

        Returns:
            bool: 是否保存成功
        """
        config = self.load_template(template_name)
        if config is None:
            return False

        # 自定义项目名称
        if project_name:
            config.project_name = project_name

        # 设置创建者
        if created_by:
            config.created_by = created_by

        # 重置时间戳
        config.created_at = datetime.now()
        config.updated_at = datetime.now()

        return self.save(config)


# 全局单例
_storage_instance = None


def get_storage() -> GraphConfigStorage:
    """获取全局配置存储实例"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = GraphConfigStorage()
    return _storage_instance
