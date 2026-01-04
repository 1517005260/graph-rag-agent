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

        # 4. 关键抽取规则（大幅增强中文适配性）
        prompt_parts.append("\n## 关键抽取规则 (CRITICAL RULES)")
        prompt_parts.append("1. **实体名称(name)必须保持原文**：")
        prompt_parts.append("   - 不要翻译，不要缩写，不要改写")
        prompt_parts.append("   - 例如原文是"学生处"，name必须是"学生处"，不能是"StudentOffice"或"Student Affairs Office"")
        prompt_parts.append("   - 例如原文是"国家奖学金"，name必须是"国家奖学金"，不能是"National Scholarship"")
        prompt_parts.append("")
        prompt_parts.append("2. **实体类型(type)必须使用Schema定义的类型**：")
        prompt_parts.append("   - 类型必须严格从定义的 schema 中选择（如上述领域中的实体类型）")
        prompt_parts.append("   - 禁止创造新类型，禁止使用原文中的词汇作为类型")
        prompt_parts.append("   - 类型与名称分离：type是分类标签，name是实际名称")
        prompt_parts.append("")
        prompt_parts.append("3. **桥接点(Bridge)优先**：")
        prompt_parts.append("   - 一旦发现文档内容匹配'桥接点'定义，必须提取")
        prompt_parts.append("   - 必须填入正确的 bridge_key")
        prompt_parts.append("   - 桥接点是跨领域连接的关键，不可遗漏")
        prompt_parts.append("")
        prompt_parts.append("4. **禁止臆造关系**：")
        prompt_parts.append("   - 只有当原文明确提到两个实体有关联时才提取")
        prompt_parts.append("   - 不要基于常识推理补充关系")
        prompt_parts.append("   - 关系类型必须来自 schema，不可自创")
        prompt_parts.append("")
        prompt_parts.append("5. **多值处理**：")
        prompt_parts.append("   - 如果一个实体有多个别名，提取最正式的名称作为 name")
        prompt_parts.append("   - 其他别名作为 description 的一部分")
        prompt_parts.append("")
        prompt_parts.append("6. **领域识别**：")
        prompt_parts.append("   - 首先识别文档所属的领域（可能同时属于多个领域）")
        prompt_parts.append("   - 然后根据领域的 schema 提取实体和关系")

        # 5. 中文抽取示例（Few-Shot Learning）
        prompt_parts.append("\n## 中文抽取示例 (Example)")
        prompt_parts.append("**原文**：根据《学生管理规定》，教务处负责学籍管理。")
        prompt_parts.append("")
        prompt_parts.append("**正确输出 JSON**：")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        prompt_parts.append('  "domains": ["规则库"],')
        prompt_parts.append('  "bridges": [],')
        prompt_parts.append('  "entities": [')
        prompt_parts.append('    {"name": "学生管理规定", "type": "政策", "bridge_key": null, "description": "学校规章制度", "domain": "规则库"},')
        prompt_parts.append('    {"name": "教务处", "type": "部门", "bridge_key": null, "description": "学校行政部门", "domain": "规则库"},')
        prompt_parts.append('    {"name": "学籍管理", "type": "流程", "bridge_key": null, "description": "管理活动", "domain": "规则库"}')
        prompt_parts.append('  ],')
        prompt_parts.append('  "relationships": [')
        prompt_parts.append('    {"source": "教务处", "target": "学籍管理", "type": "负责", "description": "负责该流程", "domain": "规则库"}')
        prompt_parts.append('  ]')
        prompt_parts.append("}")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("**错误示例**（请避免）：")
        prompt_parts.append("- ❌ `{\"name\": \"StudentAffairs\", \"type\": \"部门\"}` → 名称必须是中文原文")
        prompt_parts.append("- ❌ `{\"name\": \"教务处\", \"type\": \"行政部门\"}` → 类型必须来自schema，不能自创")
        prompt_parts.append("- ❌ `{\"name\": \"教务处\", \"type\": \"ORGANIZATION\"}` → 类型必须与schema一致，如果schema是中文则用中文")

        # 6. 输出格式
        prompt_parts.append("\n## 输出格式")
        prompt_parts.append("输出 JSON 格式，包含以下字段：")
        prompt_parts.append("```json")
        prompt_parts.append("{{")
        prompt_parts.append('  "domains": ["识别出的领域名称"],')
        prompt_parts.append('  "bridges": [')
        prompt_parts.append('    {{"bridge_key": "bridge_xxx", "value": "提取的值", "confidence": 0.9}}')
        prompt_parts.append('  ],')
        prompt_parts.append('  "entities": [')
        prompt_parts.append('    {{')
        prompt_parts.append('      "name": "实体名称",')
        prompt_parts.append('      "type": "实体类型（来自领域schema）",')
        prompt_parts.append('      "bridge_key": "如果是桥接点，填写对应的key，否则为null",')
        prompt_parts.append('      "description": "实体描述",')
        prompt_parts.append('      "domain": "所属领域"')
        prompt_parts.append('    }}')
        prompt_parts.append('  ],')
        prompt_parts.append('  "relationships": [')
        prompt_parts.append('   {{')
        prompt_parts.append('      "source": "源实体名称",')
        prompt_parts.append('      "target": "目标实体名称",')
        prompt_parts.append('      "type": "关系类型（来自领域schema）",')
        prompt_parts.append('      "description": "关系描述",')
        prompt_parts.append('      "domain": "所属领域"')
        prompt_parts.append('    }}')
        prompt_parts.append('  ]')
        prompt_parts.append('}}')
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

        prompt_parts.append('输出 JSON 格式：{{"domains": ["领域1", "领域2"], "confidence": {{"领域1": 0.9, "领域2": 0.7}}}}')

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
