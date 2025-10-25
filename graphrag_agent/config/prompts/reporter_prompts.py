"""
Reporter层Prompt模板集合

包含纲要生成、段落写作、一致性校验和引用格式化的Prompt模板
"""

# 功能: 根据PlanSpec和ExecutionRecord生成报告纲要
OUTLINE_PROMPT = '''你是一个报告规划助手。你需要根据用户查询、执行计划和收集的证据，生成一份结构清晰的报告纲要。

**用户查询**: {query}

**任务执行概要**:
{plan_summary}

**收集的证据** (共{evidence_count}条，格式示例：`result_id | granularity | source | 摘要`):
{evidence_summary}

**报告类型**: {report_type}
- **short_answer**: 简洁的问答式回复（1-3段，200-500字）
- **long_document**: 结构化长文档（多章节，1000-5000字）

**纲要生成原则**:

**对于short_answer**:
- 结构简单: 引言 -> 核心内容 -> 结论
- 直接回答问题，避免冗余

**对于long_document**:
- 结构完整: 标题 -> 摘要 -> 引言 -> 主体(多章节) -> 结论 -> 参考文献
- 每个章节有清晰的主题和子主题
- 章节之间逻辑连贯，层层递进

**章节设计要点**:
1. 每个章节对应一个核心问题或主题
2. 章节标题准确反映内容
3. 标注每个章节需要使用的证据来源（使用 `result_id` 标识）
4. 预估每个章节的字数

**示例1 - short_answer纲要**:
查询: "孙悟空的师父是谁?"
纲要:
```json
{{
  "report_type": "short_answer",
  "title": "孙悟空的师父",
  "sections": [
    {{
      "section_id": "s1",
      "title": "答案",
      "summary": "直接回答：菩提祖师",
      "evidence_ids": ["evidence_001", "evidence_002"],
      "estimated_words": 200
    }},
    {{
      "section_id": "s2",
      "title": "补充信息",
      "summary": "菩提祖师传授的本领和相关背景",
      "evidence_ids": ["evidence_003"],
      "estimated_words": 150
    }}
  ],
  "total_estimated_words": 350
}}
```

**示例2 - long_document纲要**:
查询: "分析悟空传中孙悟空的成长轨迹"
纲要:
```json
{{
  "report_type": "long_document",
  "title": "《悟空传》中孙悟空的成长轨迹分析",
  "abstract": "本文深入分析《悟空传》中孙悟空从天真少年到反抗者的思想转变过程，探讨关键事件对其性格塑造的影响。",
  "sections": [
    {{
      "section_id": "s1",
      "title": "引言：《悟空传》的叙事背景",
      "summary": "介绍《悟空传》的创作背景和与传统西游记的差异",
      "evidence_ids": ["evidence_001", "evidence_002"],
      "estimated_words": 300
    }},
    {{
      "section_id": "s2",
      "title": "第一阶段：天真无邪的花果山时期",
      "summary": "分析孙悟空早期性格特点和生活状态",
      "evidence_ids": ["evidence_003", "evidence_004", "evidence_005"],
      "estimated_words": 600
    }},
    {{
      "section_id": "s3",
      "title": "第二阶段：拜师学艺与觉醒",
      "summary": "菩提祖师的教导和孙悟空第一次面对规则约束",
      "evidence_ids": ["evidence_006", "evidence_007"],
      "estimated_words": 500
    }},
    {{
      "section_id": "s4",
      "title": "第三阶段：大闹天宫与反抗精神的形成",
      "summary": "分析孙悟空与天庭冲突的深层原因和思想转变",
      "evidence_ids": ["evidence_008", "evidence_009", "evidence_010"],
      "estimated_words": 800
    }},
    {{
      "section_id": "s5",
      "title": "结论：从个体反抗到存在主义思考",
      "summary": "总结孙悟空成长的核心主题和文学意义",
      "evidence_ids": ["evidence_011"],
      "estimated_words": 400
    }}
  ],
  "total_estimated_words": 2600
}}
```

现在请根据以上信息生成报告纲要，严格按照JSON格式输出：

**报告纲要**:
```json
'''


# 功能: 逐章节写作，生成带引用的段落内容
SECTION_WRITE_PROMPT = '''你是一个专业的技术写作助手。你需要根据纲要和证据，撰写报告的一个章节。

**报告整体纲要**:
{outline}

**当前章节**:
- 章节ID: {section_id}
- 章节标题: {section_title}
- 章节摘要: {section_summary}
- 预估字数: {estimated_words}

**可用证据**:
{evidence_list}

**写作要求**:
1. **结构清晰**: 每个段落一个中心论点
2. **证据充分**: 每个论断必须有证据支持
3. **引用规范**: 使用 `[result_id]` 标注证据，如"孙悟空拜菩提祖师为师[result_abc123]"
4. **逻辑连贯**: 段落之间过渡自然
5. **语言专业**: 准确、简洁、避免口语化
6. **字数控制**: 接近预估字数，不要过度偏离
7. **禁止额外标题**: 章节正文内部不要再使用 `#`/`##`/`###` 等 Markdown 标题，不要重复章节标题，也不要提前输出结论类标题

**引用格式**:
- 在需要引用的地方添加 `[result_id]`
- `result_id` 对应 `evidence_list` 中的 `RetrievalResult.result_id`
- 同一证据可以多次引用

**示例输出**:

孙悟空离开花果山后，历经千山万水来到斜月三星洞，拜菩提祖师为师[result_node_03]。在这一阶段，他开始接触到修行的纪律和天地间的规则。菩提祖师传授了他七十二变和筋斗云等神通[result_chunk_04]，这不仅让孙悟空获得了强大的能力，也第一次让他意识到力量的边界和责任。

然而，当孙悟空在师兄弟面前卖弄本领时，菩提祖师严厉惩罚并将其逐出师门[result_chunk_05]。这一事件成为孙悟空思想转变的起点——他开始质疑权威和规则的合理性。正如小说中所描述的："他不明白为什么有了本事就不能用，为什么师父给了力量又要禁锢它。"[result_quote_06] 这种困惑为后来的反抗天庭埋下了伏笔。

从心理发展的角度看，这一阶段孙悟空经历了从无知到觉醒的关键转变。他开始独立思考，不再盲从权威[result_analysis_07]。这种觉醒不仅是个人成长的必然，也是《悟空传》主题——个体自由与制度约束冲突——的核心体现。

---

现在请根据以上信息撰写当前章节，使用Markdown格式输出：
'''


# 功能: 校验报告的事实一致性、引用准确性和逻辑连贯性
CONSISTENCY_CHECK_PROMPT = '''你是一个质量校验助手。你需要检查报告内容的一致性、准确性和完整性。

**报告内容**:
{report_content}

**原始证据列表**:
{evidence_list}

**校验项目**:

**1. 事实一致性检查**:
- 报告中的陈述是否与证据一致？
- 是否存在与证据矛盾的内容？
- 是否有过度推断或缺乏依据的结论？

**2. 引用准确性检查**:
- 所有引用标记[result_id]是否都有对应的证据？
- 引用的内容是否准确反映了证据的原意？
- 是否有遗漏重要证据未引用？

**3. 逻辑连贯性检查**:
- 段落之间是否衔接自然？
- 论证逻辑是否严密？
- 是否存在自相矛盾的表述？

**4. 冗余与重复检查**:
- 是否有重复表达相同内容的段落？
- 是否有可以合并的相似内容？

**5. 完整性检查**:
- 是否遗漏了纲要中计划的重要内容？
- 是否回答了用户的核心问题？

**判断标准**:
- **一致性好** (is_consistent: true): 无明显事实错误、引用准确、逻辑清晰
- **需要修正** (is_consistent: false): 存在事实错误、引用不当或逻辑混乱

**示例1 - 一致性良好**:
报告: "孙悟空拜菩提祖师为师[result_node_01]，学习了七十二变[result_chunk_02]。"
证据: result_node_01 => "孙悟空在斜月三星洞拜菩提祖师为师", result_chunk_02 => "菩提祖师传授七十二变"
检查结果:
```json
{{
  "is_consistent": true,
  "issues": [],
  "corrections": []
}}
```

**示例2 - 需要修正**:
报告: "孙悟空拜唐僧为师[result_node_01]，学习了七十二变。"
证据: result_node_01 => "孙悟空在斜月三星洞拜菩提祖师为师"
检查结果:
```json
{{
  "is_consistent": false,
  "issues": [
    {{
      "type": "事实错误",
      "location": "第1段第1句",
      "description": "师父错误：报告称'唐僧'，证据显示应为'菩提祖师'",
      "severity": "high"
    }},
    {{
      "type": "引用缺失",
      "location": "第1段第1句",
      "description": "'七十二变'的陈述缺少引用标记",
      "severity": "medium"
    }}
  ],
  "corrections": [
    {{
      "original": "孙悟空拜唐僧为师[result_node_01]，学习了七十二变。",
      "corrected": "孙悟空拜菩提祖师为师[result_node_01]，学习了七十二变[result_chunk_02]。",
      "reason": "修正师父名称错误，并补充引用"
    }}
  ]
}}
```

现在请检查以上报告，严格按照JSON格式输出：

**一致性检查结果**:
```json
'''


# 功能: 生成规范化的引用列表
CITATION_FORMAT_PROMPT = '''你是一个引用格式化助手。你需要将检索结果转换为规范的参考文献列表。

**检索结果**（每项均为`RetrievalResult`的序列化结果）:
{retrieval_results}

**引用格式**: {citation_style}
- **default**: 默认简洁格式
- **apa**: APA学术格式
- **mla**: MLA学术格式

**格式示例**:

**default格式**:
```markdown
## 参考文献

[result_chunk_123] 来源: chunk:doc_123_chunk_5 (社区:人物社区) (置信度:0.95)
    内容摘要: 孙悟空在斜月三星洞拜菩提祖师为师...

[result_entity_456] 来源: entity:孙悟空 (知识图谱实体) (置信度:0.88)
    关系: 孙悟空 -师徒-> 菩提祖师

[result_path_001] 来源: subgraph:path_001 (关系路径探索) (置信度:0.82)
    路径: 孙悟空 -> 大闹天宫 -> 天庭 -> 如来佛祖
```

**apa格式**:
```markdown
## References

[result_chunk_123] chunk:doc_123_chunk_5 (2024). Character relationships in Journey to the West. Retrieved from 悟空传文本数据库.

[result_entity_456] entity:孙悟空 (2024). Knowledge graph entity. Neo4j graph database.
```

现在请生成引用列表，使用Markdown格式输出：
'''


# Map-Reduce 辅助模板
EVIDENCE_MAP_PROMPT = """
你是一个信息提取专家。请从以下证据中提取关键信息，用于撰写章节「{section_title}」。

**章节目标**: {section_goal}

**证据列表**:
{evidence_list}

**任务**:
1. 列出3-5个关键论点（key_points）
2. 识别所有提及的实体（entities）
3. 生成200字以内的摘要文本（summary_text）

**输出格式**（JSON）:
{{
    "key_points": ["论点1", "论点2"],
    "entities": ["实体1", "实体2"],
    "summary_text": "摘要文本..."
}}

如果无法提取信息，请返回空的JSON对象。
""".strip()


SECTION_REDUCE_PROMPT = """
你是一个技术写作专家。请基于以下证据摘要，撰写章节「{section_title}」的内容。

**章节要求**:
- 目标: {section_goal}
- 预估字数: {estimated_words}

**证据摘要**（已经过预处理）:
{evidence_summaries}

**写作要求**:
1. 整合所有关键论点，去除冗余
2. 保持逻辑连贯，使用过渡句
3. 引用格式：[证据ID]
4. 字数控制在 {estimated_words} ± 20% 范围内

输出Markdown格式的章节内容，不要使用代码块。
""".strip()


INTERMEDIATE_SUMMARY_PROMPT = """
你正在为章节「{section_title}」进行资料整合。请基于以下摘要生成更精炼的中间摘要。

**章节目标**: {section_goal}

**输入摘要**:
{evidence_summaries}

请输出JSON格式：
{{
    "key_points": ["要点1", "要点2"],
    "entities": ["实体1", "实体2"],
    "summary_text": "整合后的摘要"
}}
""".strip()


MERGE_PROMPT = """
请融合两个摘要，输出整合结果的JSON。

**章节**: {section_title}

左侧摘要:
{left_summary}

右侧摘要:
{right_summary}

请输出JSON：
{{
    "key_points": ["要点1", "要点2"],
    "entities": ["实体1", "实体2"],
    "summary_text": "合并后的摘要"
}}
""".strip()


REFINE_PROMPT = """
请根据新证据精炼当前章节草稿。

**当前草稿**:
{current_draft}

**新证据摘要**:
{new_evidence}

任务：
1. 判断新证据是否提供了新信息
2. 如有新信息，将其自然融入草稿
3. 如与现有内容冲突，保留更可靠的版本
4. 保持章节「{section_title}」的主题一致性

输出更新后的章节内容（Markdown格式），不要使用代码块。
""".strip()


INTRO_PROMPT = """
你需要为报告「{report_title}」撰写引言。

**原始查询/任务**: {query}

**章节概要**:
{section_summaries}

请以150-200字撰写引言，概述报告背景与结构。输出Markdown段落，不要使用列表或代码块。
""".strip()


CONCLUSION_PROMPT = """
你需要为报告「{report_title}」撰写结论。

**章节内容摘要**:
{section_content}

共使用证据数量: {evidence_count}

请总结关键发现并给出建议，150-200字，输出Markdown段落。
""".strip()


TERMINOLOGY_PROMPT = """
请从以下文本中提取不超过10个关键术语及其解释，输出JSON对象：
{section_text}
""".strip()
