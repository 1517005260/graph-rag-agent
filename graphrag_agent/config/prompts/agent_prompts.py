"""
Agent层提示模板集中定义，保持不同Agent之间的提示内容一致。
"""

from textwrap import dedent

GRAPH_AGENT_KEYWORD_PROMPT = dedent(
    """
    提取以下查询的关键词:
    查询: {query}

    请提取两类关键词:
    1. 低级关键词: 具体实体、名称、术语
    2. 高级关键词: 主题、概念、领域

    以JSON格式返回。
    """
).strip()

GRAPH_AGENT_GENERATE_PROMPT = dedent(
    """
    ---分析报告---
    请注意，下面提供的分析报告按**重要性降序排列**。

    {context}

    用户的问题是：
    {question}

    请严格按照以下格式输出回答：
    1. 使用三级标题(###)标记主题
    2. 主要内容用清晰的段落展示
    3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
    """
).strip()

GRAPH_AGENT_REDUCE_PROMPT = dedent(
    """
    ---分析报告---
    {report_data}

    用户的问题是：
    {question}
    """
).strip()

DEEP_RESEARCH_THINKING_SUMMARY_PROMPT = dedent(
    """
    以下是对问题的思考过程:

    {thinking}

    原始问题是:
    {question}

    请生成一个全面、有深度的回答，不要重复思考过程，直接给出最终综合结论。
    结论应该清晰、直接地回答问题，包含相关的事实和见解。
    如果有不同的观点或矛盾的信息，请指出并提供平衡的视角。
    """
).strip()

EXPLORATION_SUMMARY_PROMPT = dedent(
    """
    基于以下知识图谱探索路径和发现的内容，生成一个关于"{query}"的综合性摘要:

    探索路径:
    {path_summary}

    关键内容:
    {content_summary}

    请提供一个全面、有深度的分析，包括关键发现、关联关系和见解。
    """
).strip()

CONTRADICTION_IMPACT_PROMPT = dedent(
    """
    在回答关于"{query}"的问题时，发现以下信息矛盾:

    {contradictions_text}

    请分析这些矛盾对最终答案可能产生的影响，以及如何在存在这些矛盾的情况下给出最准确的回答。
    """
).strip()

HYBRID_AGENT_GENERATE_PROMPT = dedent(
    """
    ---分析报告---
    以下是检索到的相关信息，按重要性排序：

    {context}

    用户的问题是：
    {question}

    请以清晰、全面的方式回答问题，确保：
    1. 回答结合了检索到的低级（实体细节）和高级（主题概念）信息
    2. 使用三级标题(###)组织内容，增强可读性
    3. 结尾处用"#### 引用数据"标记引用来源
    """
).strip()

NAIVE_RAG_HUMAN_PROMPT = dedent(
    """
    ---检索结果---
    {context}

    问题：
    {question}
    """
)
