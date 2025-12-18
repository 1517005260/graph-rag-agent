"""
动态提示词构建器
根据用户配置的 GraphConfig 动态生成实体抽取和关系抽取的提示词
"""

from typing import Optional
from graphrag_agent.config.graph_config_model import GraphConfig, BridgeDefinition, DomainDefinition


class DynamicPromptBuilder:
    """动态提示词构建器"""

    def __init__(self, config: GraphConfig):
        """
        初始化构建器

        Args:
            config: GraphConfig 实例
        """
        self.config = config

    def build_system_prompt(self) -> str:
        """
        构建系统提示词（用于实体和关系抽取）

        Returns:
            完整的系统提示词
        """
        prompt_parts = []

        # 1. 项目概述
        prompt_parts.append(f"# 任务：{self.config.project_name}")
        if self.config.description:
            prompt_parts.append(f"\n{self.config.description}\n")

        # 2. 桥接点定义（全局公共概念）
        prompt_parts.append("\n## 重要：桥接点（Bridge Points）")
        prompt_parts.append("以下是连接所有领域的公共概念，必须在每个文档中提取：\n")

        for bridge in self.config.bridge_definitions:
            prompt_parts.append(f"### {bridge.name} (key: {bridge.key})")
            prompt_parts.append(f"- 描述：{bridge.description}")

            if bridge.examples:
                prompt_parts.append(f"- 示例：{', '.join(bridge.examples)}")

            if bridge.is_enum_restricted and bridge.enum_values:
                prompt_parts.append(f"- **限制**：必须从以下值中选择：{', '.join(bridge.enum_values)}")
            else:
                prompt_parts.append("- **提取规则**：根据文档内容自由提取")

            prompt_parts.append("")

        # 3. 领域定义（业务子图）
        prompt_parts.append("\n## 领域（Domains）")
        prompt_parts.append("根据文档内容，判断其所属领域，并提取相应的实体和关系：\n")

        for i, domain in enumerate(self.config.domain_definitions, 1):
            prompt_parts.append(f"### 领域 {i}：{domain.domain_name}")
            prompt_parts.append(f"- 描述：{domain.description}")
            prompt_parts.append(f"- 触发条件：{domain.trigger_condition}")

            # 实体类型
            if domain.schema.entities:
                prompt_parts.append(f"- 实体类型：{', '.join(domain.schema.entities)}")

            # 关系类型
            if domain.schema.relations:
                prompt_parts.append(f"- 关系类型：{', '.join(domain.schema.relations)}")

            # 桥接点映射
            if domain.bridge_mappings:
                prompt_parts.append("- 桥接点映射：")
                for mapping in domain.bridge_mappings:
                    bridge = self._find_bridge(mapping.bridge_key)
                    if bridge:
                        role_name = mapping.field_name or bridge.name
                        prompt_parts.append(f"  - {role_name} → {bridge.name}（{mapping.role}）")

            prompt_parts.append("")

        # 4. 抽取指导
        prompt_parts.append("\n## 抽取指导")
        prompt_parts.append("1. **首先**识别文档所属的领域（可能同时属于多个领域）")
        prompt_parts.append("2. **必须**提取所有相关的桥接点，确保不同文档可以通过桥接点关联")
        prompt_parts.append("3. **然后**根据领域的 schema 提取实体和关系")
        prompt_parts.append("4. **实体提取**：")
        prompt_parts.append("   - 提取实体时，标注其类型（来自领域 schema）")
        prompt_parts.append("   - 如果实体是桥接点，同时标注 bridge_key")
        prompt_parts.append("5. **关系提取**：")
        prompt_parts.append("   - 关系类型必须来自领域 schema")
        prompt_parts.append("   - 跨领域的关系通过桥接点连接")

        # 5. 输出格式
        prompt_parts.append("\n## 输出格式")
        prompt_parts.append("输出 JSON 格式，包含以下字段：")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        prompt_parts.append('  "domains": ["识别出的领域名称"],')
        prompt_parts.append('  "bridges": [')
        prompt_parts.append('    {"bridge_key": "bridge_xxx", "value": "提取的值", "confidence": 0.9}')
        prompt_parts.append('  ],')
        prompt_parts.append('  "entities": [')
        prompt_parts.append('    {')
        prompt_parts.append('      "name": "实体名称",')
        prompt_parts.append('      "type": "实体类型（来自领域schema）",')
        prompt_parts.append('      "bridge_key": "如果是桥接点，填写对应的key，否则为null",')
        prompt_parts.append('      "description": "实体描述",')
        prompt_parts.append('      "domain": "所属领域"')
        prompt_parts.append('    }')
        prompt_parts.append('  ],')
        prompt_parts.append('  "relationships": [')
        prompt_parts.append('    {')
        prompt_parts.append('      "source": "源实体名称",')
        prompt_parts.append('      "target": "目标实体名称",')
        prompt_parts.append('      "type": "关系类型（来自领域schema）",')
        prompt_parts.append('      "description": "关系描述",')
        prompt_parts.append('      "domain": "所属领域"')
        prompt_parts.append('    }')
        prompt_parts.append('  ]')
        prompt_parts.append('}')
        prompt_parts.append("```")

        return "\n".join(prompt_parts)

    def build_extraction_prompt(self, document_text: str) -> str:
        """
        构建抽取提示词（用于具体文档的抽取任务）

        Args:
            document_text: 文档文本

        Returns:
            完整的抽取提示词
        """
        prompt_parts = []

        prompt_parts.append("请根据系统指导，分析以下文档并提取知识图谱：\n")
        prompt_parts.append("---")
        prompt_parts.append(document_text)
        prompt_parts.append("---\n")

        prompt_parts.append("请仔细分析文档，按照以下步骤进行：")
        prompt_parts.append("1. 识别文档所属的领域")
        prompt_parts.append("2. 提取所有相关的桥接点")
        prompt_parts.append("3. 提取领域内的实体和关系")
        prompt_parts.append("\n请以 JSON 格式输出结果。")

        return "\n".join(prompt_parts)

    def build_domain_classifier_prompt(self, document_text: str) -> str:
        """
        构建领域分类提示词（用于判断文档所属领域）

        Args:
            document_text: 文档文本

        Returns:
            领域分类提示词
        """
        prompt_parts = []

        prompt_parts.append(f"# 任务：{self.config.project_name} - 领域分类\n")

        prompt_parts.append("请判断以下文档属于哪些领域（可能同时属于多个领域）：\n")

        for domain in self.config.domain_definitions:
            prompt_parts.append(f"- **{domain.domain_name}**：{domain.description}")
            prompt_parts.append(f"  触发条件：{domain.trigger_condition}\n")

        prompt_parts.append("---")
        prompt_parts.append(document_text)
        prompt_parts.append("---\n")

        prompt_parts.append('输出 JSON 格式：{"domains": ["领域1", "领域2"], "confidence": {"领域1": 0.9, "领域2": 0.7}}')

        return "\n".join(prompt_parts)

    def get_domain_schema(self, domain_name: str) -> Optional[DomainDefinition]:
        """
        获取指定领域的 schema

        Args:
            domain_name: 领域名称

        Returns:
            DomainDefinition 或 None
        """
        for domain in self.config.domain_definitions:
            if domain.domain_name == domain_name:
                return domain
        return None

    def get_bridge_by_key(self, bridge_key: str) -> Optional[BridgeDefinition]:
        """
        根据 key 获取桥接点定义

        Args:
            bridge_key: 桥接点 key

        Returns:
            BridgeDefinition 或 None
        """
        return self._find_bridge(bridge_key)

    def _find_bridge(self, bridge_key: str) -> Optional[BridgeDefinition]:
        """内部辅助方法：查找桥接点"""
        for bridge in self.config.bridge_definitions:
            if bridge.key == bridge_key:
                return bridge
        return None

    def get_all_entity_types(self) -> list[str]:
        """
        获取所有实体类型（跨所有领域）

        Returns:
            实体类型列表
        """
        entity_types = set()
        for domain in self.config.domain_definitions:
            entity_types.update(domain.schema.entities)
        return list(entity_types)

    def get_all_relationship_types(self) -> list[str]:
        """
        获取所有关系类型（跨所有领域）

        Returns:
            关系类型列表
        """
        relationship_types = set()
        for domain in self.config.domain_definitions:
            relationship_types.update(domain.schema.relations)
        return list(relationship_types)

    def get_bridge_keys(self) -> list[str]:
        """
        获取所有桥接点的 key

        Returns:
            桥接点 key 列表
        """
        return [bridge.key for bridge in self.config.bridge_definitions]


def create_prompt_builder_from_storage() -> Optional[DynamicPromptBuilder]:
    """
    从存储加载配置并创建提示词构建器

    Returns:
        DynamicPromptBuilder 实例，如果配置不存在则返回 None
    """
    from graphrag_agent.config.graph_config_storage import get_storage

    storage = get_storage()
    config = storage.load()

    if config is None:
        return None

    return DynamicPromptBuilder(config)
