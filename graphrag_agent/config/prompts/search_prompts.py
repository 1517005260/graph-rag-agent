"""
搜索工具相关提示模板统一配置。
"""

from textwrap import dedent

LOCAL_SEARCH_CONTEXT_PROMPT = dedent(
    """
    ---分析报告---
    请注意，下面提供的分析报告按**重要性降序排列**。

    {context}

    用户的问题是：
    {input}

    请使用三级标题(###)标记主题
    """
).strip()

LOCAL_SEARCH_KEYWORD_PROMPT = dedent(
    """
    你是一个专门从用户查询中提取搜索关键词的助手。你需要将关键词分为两类：
    1. 低级关键词：具体实体名称、人物、地点、具体事件等
    2. 高级关键词：主题、概念、关系类型等

    返回格式必须是JSON格式：
    {{
        "low_level": ["关键词1", "关键词2", ...],
        "high_level": ["关键词1", "关键词2", ...]
    }}

    注意：
    - 每类提取3-5个关键词即可
    - 不要添加任何解释或其他文本，只返回JSON
    - 如果某类无关键词，则返回空列表
    """.strip()
)

GLOBAL_SEARCH_MAP_PROMPT = dedent(
    """
    ---数据表格---
    {context_data}

    用户的问题是：
    {question}
    """
).strip()

GLOBAL_SEARCH_REDUCE_PROMPT = dedent(
    """
    ---分析报告---
    {report_data}

    用户的问题是：
    {question}
    """
).strip()

GLOBAL_SEARCH_KEYWORD_PROMPT = dedent(
    """
    你是一个专门从用户查询中提取搜索关键词的助手。提取最相关的关键词，这些关键词将用于在知识库中查找信息。

    请返回一个关键词列表，格式为JSON数组：
    ["关键词1", "关键词2", ...]

    注意：
    - 提取5-8个关键词即可
    - 不要添加任何解释或其他文本，只返回JSON数组
    - 关键词应该是名词短语、概念或专有名词
    """
).strip()

HYBRID_TOOL_QUERY_PROMPT = dedent(
    """
    ---分析报告---
    请注意，以下内容组合了低级详细信息和高级主题概念。

    ## 低级内容（实体详细信息）:
    {low_level}

    ## 高级内容（主题和概念）:
    {high_level}

    用户的问题是：
    {query}

    请综合利用上述信息回答问题，确保回答全面且有深度。
    回答格式应包含：
    1. 主要内容（使用清晰的段落展示）
    2. 在末尾标明引用的数据来源
    """
).strip()

NAIVE_SEARCH_QUERY_PROMPT = dedent(
    """
    ---文档片段---
    {context}

    问题：
    {query}
    """
).strip()
