"""
实体提取器工厂
支持动态配置和传统配置两种模式
"""

from typing import Optional
from graphrag_agent.graph.extraction.entity_extractor import EntityRelationExtractor
from graphrag_agent.prompts.dynamic_prompt_builder import create_prompt_builder_from_storage
from graphrag_agent.config.graph_config_storage import get_storage


def create_entity_extractor(
    llm,
    system_template: Optional[str] = None,
    human_template: Optional[str] = None,
    entity_types: Optional[list] = None,
    relationship_types: Optional[list] = None,
    max_workers: int = 4,
    batch_size: int = 5,
    cache_dir: str = "./cache/graph"
) -> EntityRelationExtractor:
    """
    创建实体关系提取器

    优先使用动态配置（如果存在），否则使用传统的模板配置

    Args:
        llm: 语言模型
        system_template: 系统提示模板（传统模式）
        human_template: 用户提示模板（传统模式）
        entity_types: 实体类型列表（传统模式）
        relationship_types: 关系类型列表（传统模式）
        max_workers: 并行工作线程数
        batch_size: 批处理大小
        cache_dir: 缓存目录

    Returns:
        EntityRelationExtractor 实例
    """
    # 尝试加载动态配置
    storage = get_storage()
    config = storage.load()

    if config is not None:
        # 使用动态配置
        print("✨ 检测到通用图谱配置，使用动态提示词生成")
        print(f"📋 项目: {config.project_name}")
        print(f"🏢 行业: {config.industry}")
        print(f"🔗 桥接点数量: {len(config.bridge_definitions)}")
        print(f"📦 领域数量: {len(config.domain_definitions)}")

        # 创建动态提示词构建器
        from graphrag_agent.prompts.dynamic_prompt_builder import DynamicPromptBuilder
        prompt_builder = DynamicPromptBuilder(config)

        # 生成系统提示词
        dynamic_system_template = prompt_builder.build_system_prompt()

        # 生成用户提示词（简化版，实际抽取时会用build_extraction_prompt）
        dynamic_human_template = """请分析以下文本块并提取实体和关系：

{input_text}

---
请严格按照系统指导输出 JSON 格式的结果。"""

        # 获取所有实体类型和关系类型
        all_entity_types = prompt_builder.get_all_entity_types()
        all_relationship_types = prompt_builder.get_all_relationship_types()

        # 创建提取器
        extractor = EntityRelationExtractor(
            llm=llm,
            system_template=dynamic_system_template,
            human_template=dynamic_human_template,
            entity_types=all_entity_types,
            relationship_types=all_relationship_types,
            cache_dir=cache_dir,
            max_workers=max_workers,
            batch_size=batch_size
        )

        # 在提取器上附加prompt_builder，以便后续使用
        extractor.prompt_builder = prompt_builder
        extractor.is_dynamic = True

        return extractor

    else:
        # 使用传统配置
        print("📝 使用传统配置模式")

        if system_template is None or human_template is None:
            raise ValueError("在传统模式下，必须提供 system_template 和 human_template")

        if entity_types is None or relationship_types is None:
            raise ValueError("在传统模式下，必须提供 entity_types 和 relationship_types")

        extractor = EntityRelationExtractor(
            llm=llm,
            system_template=system_template,
            human_template=human_template,
            entity_types=entity_types,
            relationship_types=relationship_types,
            cache_dir=cache_dir,
            max_workers=max_workers,
            batch_size=batch_size
        )

        extractor.is_dynamic = False

        return extractor


def get_entity_types_from_config() -> list:
    """
    从配置中获取实体类型列表

    Returns:
        实体类型列表
    """
    storage = get_storage()
    config = storage.load()

    if config is not None:
        from graphrag_agent.prompts.dynamic_prompt_builder import DynamicPromptBuilder
        prompt_builder = DynamicPromptBuilder(config)
        return prompt_builder.get_all_entity_types()
    else:
        # 回退到传统配置
        from graphrag_agent.config.settings import entity_types
        return entity_types


def get_relationship_types_from_config() -> list:
    """
    从配置中获取关系类型列表

    Returns:
        关系类型列表
    """
    storage = get_storage()
    config = storage.load()

    if config is not None:
        from graphrag_agent.prompts.dynamic_prompt_builder import DynamicPromptBuilder
        prompt_builder = DynamicPromptBuilder(config)
        return prompt_builder.get_all_relationship_types()
    else:
        # 回退到传统配置
        from graphrag_agent.config.settings import relationship_types
        return relationship_types
