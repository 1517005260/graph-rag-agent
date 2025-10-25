from textwrap import dedent

from graphrag_agent.config.settings import KB_NAME

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
MAX_SEARCH_LIMIT = 5

REASON_PROMPT = (
    f"你是一个推理助手，可以使用搜索工具来搜索{KB_NAME}相关问题，帮助你准确回答用户的问题。你有特殊工具：\n\n"
    f"- 要执行搜索：请写 {BEGIN_SEARCH_QUERY} 你的查询内容 {END_SEARCH_QUERY}。\n"
    f"然后，系统会搜索并分析相关内容，然后以 {BEGIN_SEARCH_RESULT} ...搜索结果... {END_SEARCH_RESULT} 的格式提供有用信息。\n\n"
    f"你可以根据需要重复搜索过程。最大搜索次数限制为 {MAX_SEARCH_LIMIT} 次。\n\n"
    "获得所有需要的信息后，继续你的推理。\n\n"
    "-- 示例 1 --\n"
    "问题: \"电影《大白鲨》和《皇家赌场》的导演是否来自同一个国家？\"\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}《大白鲨》的导演是谁？{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n《大白鲨》的导演是史蒂文·斯皮尔伯格...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理。\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}史蒂文·斯皮尔伯格来自哪里？{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n史蒂文·艾伦·斯皮尔伯格是美国电影制作人...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理...\n\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}《皇家赌场》的导演是谁？{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n《皇家赌场》是2006年由马丁·坎贝尔执导的特工片...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理...\n\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}马丁·坎贝尔来自哪里？{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n马丁·坎贝尔是一位新西兰电影导演...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理...\n\n"
    "助手:\n"
    "现在我有足够的信息回答这个问题。\n\n"
    "**回答**：不，《大白鲨》的导演史蒂文·斯皮尔伯格来自美国，而《皇家赌场》的导演马丁·坎贝尔来自新西兰。他们不是来自同一个国家。\n\n"
    "-- 示例 2 --\n"
    "问题: \"深圳市对新能源汽车有哪些补贴政策？\"\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}深圳市新能源汽车补贴政策{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n深圳市对新能源汽车的补贴政策包括购置补贴和充电设施补贴...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理。\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}深圳市新能源汽车购置补贴具体标准{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n深圳市新能源乘用车补贴标准为每车最高不超过2万元...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理...\n\n"
    "助手:\n"
    f"    {BEGIN_SEARCH_QUERY}深圳市新能源汽车充电设施补贴政策{END_SEARCH_QUERY}\n\n"
    "用户:\n"
    f"    {BEGIN_SEARCH_RESULT}\n深圳市对新建充电桩给予每千瓦400元的补贴...\n{END_SEARCH_RESULT}\n\n"
    "继续用新信息进行推理...\n\n"
    "助手:\n"
    "现在我有足够的信息回答这个问题。\n\n"
    "**回答**：深圳市对新能源汽车的补贴政策主要包括两个方面：1）购置补贴：每辆新能源乘用车最高不超过2万元；2）充电设施补贴：对新建充电桩给予每千瓦400元的补贴。\n\n"
    "**记住**：\n"
    f"- 你有一个知识库可以搜索，只需提供适当的搜索查询。\n"
    f"- 使用 {BEGIN_SEARCH_QUERY} 请求数据库搜索，以 {END_SEARCH_QUERY} 结束。\n"
    "- 查询语言必须与'问题'或'搜索结果'使用相同的语言。\n"
    "- 如果找不到有用信息，重写搜索查询以使用更少和更精确的关键词。\n"
    "- 完成搜索后，继续你的推理以回答问题。\n\n"
    "请回答以下问题。应该逐步思考来解决它。\n\n"
)

RELEVANT_EXTRACTION_PROMPT = """**任务说明：**

    你需要根据以下输入来阅读和分析搜索到的内容：**之前的推理步骤**、**当前搜索查询**和**搜索到的内容**。你的目标是从**搜索到的内容**中提取与**当前搜索查询**相关的有用信息，并将这些信息无缝整合到**之前的推理步骤**中，以继续为原始问题推理。

    **指导原则：**

    1. **分析搜索到的内容：**
    - 仔细审查搜索结果的内容。
    - 识别与**当前搜索查询**相关的事实信息，这些信息能够帮助推理过程解答原始问题。

    2. **提取相关信息：**
    - 选择那些能直接推进**之前推理步骤**的信息。
    - 确保提取的信息准确且相关。

    3. **输出格式：**
    - **如果搜索内容对当前查询提供了有用信息：** 以`**Final Information**`开头呈现信息，如下所示。
    - 输出的语言**必须**与'搜索查询'或'搜索内容'的语言一致。\n"
    **Final Information**

    [有用的信息]

    - **如果搜索内容没有为当前查询提供任何有用信息：** 输出以下文本：

    **Final Information**

    No helpful information found.

    **输入：**
    - **之前的推理步骤：**  
    {prev_reasoning}

    - **当前搜索查询：**  
    {search_query}

    - **搜索到的内容：**  
    {document}

    """

SUB_QUERY_PROMPT = """为了更全面地回答这个问题，请将原始问题分解为最多三个子问题。返回一个字符串列表。
        如果这是一个非常简单的问题，不需要分解，则在列表中只保留一个原始问题。

        请确保每个子问题都是明确的、能够独立回答的，且与原始问题直接相关。

        原始问题: {original_query}

        示例输入:
        "深圳市政府对新能源汽车有哪些补贴政策？"

        示例输出:
        [
            "深圳市政府对新能源乘用车有哪些补贴政策？",
            "深圳市政府对新能源商用车有哪些补贴政策？",
            "深圳市新能源汽车补贴政策在2023年有哪些变化？"
        ]

        请按照Python列表格式提供您的回答:
        """

FOLLOWUP_QUERY_PROMPT = """根据原始查询和已检索的信息，判断是否需要额外的搜索查询。如果需要进一步研究，请提供最多2个搜索查询。如果不需要进一步研究，返回空列表。

        原始查询: {original_query}

        已检索的信息: 
        {retrieved_info}

        请考虑以下因素：
        1. 现有信息是否完整回答了原始查询
        2. 是否存在信息缺口或未解决的疑问
        3. 是否需要获取更具体或最新的信息

        请只返回有效的Python字符串列表格式，不要包含任何其他文本。
        """

FINAL_ANSWER_PROMPT = """基于以下思考过程和检索到的信息，回答用户问题。提供详细、准确、全面的回答，引用相关来源。

        用户问题: "{query}"

        检索到的信息:
        {retrieved_content}

        思考过程:
        {thinking_process}

        请生成综合性的最终回答，不需要解释你的思考过程，直接给出结论。确保答案清晰、有条理，并包含相关的重要信息。
        
        格式要求：
        1. 使用简洁明了的语言
        2. 适当使用标题结构化信息
        3. 不要说"根据检索的信息"等表达
        4. 直接给出确定的答案，不要使用"可能"、"或许"等不确定表达（除非确实存在不确定性）
        """

INITIAL_THINKING_PROMPT = dedent(
    """
    请对问题进行深入思考，考虑以下方面:
    1. 问题的核心是什么？
    2. 需要哪些信息来回答这个问题？
    3. 有哪些可能的思考方向？

    请提供你的推理过程，不需要立即给出答案。
    """
).strip()

HYPOTHESIS_GENERATION_PROMPT = dedent(
    """
    基于初步思考：
    {initial_thinking}

    生成3-5个合理的假设，每个假设应该：
    1. 解释已有的观察结果
    2. 可以被进一步验证
    3. 相互之间有所不同

    以JSON格式返回：
    [
        {{"hypothesis": "...", "reasoning": "..."}},
        ...
    ]
    """
).strip()

HYPOTHESIS_VERIFICATION_PROMPT = dedent(
    """
    请验证以下假设：

    假设: {hypothesis_text}
    理由: {reasoning_text}

    请考虑:
    1. 这个假设是否符合已知信息?
    2. 有哪些证据支持或反对这个假设?
    3. 这个假设是否有逻辑漏洞?
    4. 是否需要更多信息来验证这个假设?

    请提供详细的验证分析。
    """
).strip()

VERIFICATION_STATUS_PROMPT = dedent(
    """
    根据以下验证分析，该假设的状态是什么?

    验证分析:
    {verification}

    请从以下三个选项中选择一个:
    - supported: 假设被证据支持
    - rejected: 假设被证据反驳
    - uncertain: 证据不足，无法确定

    只返回一个状态单词，不要包含解释。
    """
).strip()

UPDATE_THINKING_PROMPT = dedent(
    """
    基于以下验证结果汇总，请更新你的思考过程:

    {verification_summary}

    请综合考虑所有被支持的假设，并解释为什么拒绝其他假设。
    提供一个更新后的、连贯的思考过程。
    """
).strip()

COUNTERFACTUAL_ANALYSIS_PROMPT = dedent(
    """
    假设以下情况为真:
    {hypothesis}

    基于这个假设，请重新思考问题。即使这个假设与事实不符，也请认真推理。
    分析如果这个假设为真，会导致什么结论?
    """
).strip()

COUNTERFACTUAL_COMPARISON_PROMPT = dedent(
    """
    请比较原始推理和反事实假设下的推理:

    原始推理基于已知事实。
    反事实推理基于假设: {hypothesis}

    这种对比揭示了什么关键见解?
    这是否帮助我们更好地理解问题?
    """
).strip()

SEARCH_RESULT_COMPARISON_PROMPT = dedent(
    """
    请评估以下两个搜索结果，判断哪个更具体、包含更多有价值的信息，特别是具体数字、规定或明确标准。

    原始查询: {query}

    结果1:
    {text1}

    结果2:
    {text2}

    请评估哪个结果包含更具体的信息（如明确的规定、数字、标准等）。
    只返回以下三个选项之一：
    - "precise"：如果结果1更具体、更有价值
    - "kb"：如果结果2更具体、更有价值
    - "both"：如果两者具有相当的价值或包含不同的有价值信息

    只返回选项，不要包含其他解释。
    """
).strip()

SEARCH_MULTI_HYPOTHESIS_PROMPT = dedent(
    """
    为以下问题生成2-3个可能的假设，这些假设应该代表不同角度或思路：

    问题: "{query}"

    每个假设应该:
    1. 不同于其他假设
    2. 提供一种可能的思考方向
    3. 有助于深入分析问题

    以列表形式返回假设，每个假设简短明了。
    """
).strip()

__all__ = [
    "BEGIN_SEARCH_QUERY",
    "END_SEARCH_QUERY",
    "BEGIN_SEARCH_RESULT",
    "END_SEARCH_RESULT",
    "MAX_SEARCH_LIMIT",
    "REASON_PROMPT",
    "RELEVANT_EXTRACTION_PROMPT",
    "SUB_QUERY_PROMPT",
    "FOLLOWUP_QUERY_PROMPT",
    "FINAL_ANSWER_PROMPT",
    "INITIAL_THINKING_PROMPT",
    "HYPOTHESIS_GENERATION_PROMPT",
    "HYPOTHESIS_VERIFICATION_PROMPT",
    "VERIFICATION_STATUS_PROMPT",
    "UPDATE_THINKING_PROMPT",
    "COUNTERFACTUAL_ANALYSIS_PROMPT",
    "COUNTERFACTUAL_COMPARISON_PROMPT",
    "SEARCH_RESULT_COMPARISON_PROMPT",
    "SEARCH_MULTI_HYPOTHESIS_PROMPT",
]
