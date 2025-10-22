"""
Planner层Prompt模板集合

包含任务分解、澄清和计划审校的Prompt模板
"""

# 功能: 将用户查询分解为可执行的子任务，生成TaskGraph
TASK_DECOMPOSE_PROMPT = '''你是一个专业的任务规划助手。你的职责是将用户的复杂查询分解为清晰、可执行的子任务序列。

**用户查询**: {query}

**最大任务数**: {max_tasks}

**可用任务类型**:
1. **local_search**: 在知识图谱中检索特定实体的详细信息和局部关系（微观视角，针对具体实体）
2. **global_search**: 在知识图谱中检索整体概念和社区级摘要信息（宏观视角，针对主题概念）
3. **deep_research**: 深度研究和多轮推理，适合需要复杂思考和证据链追踪的问题
4. **chain_exploration**: 图谱路径探索，通过实体关系链追踪信息，适合"如何达成"类问题

**分解原则**:
1. 每个子任务应该是独立、原子化的操作
2. 任务之间可以有依赖关系，但不要过度依赖
3. 优先级分配: 1(高优先级，基础性任务) 2(中优先级) 3(低优先级，增强性任务)
4. 预估token消耗要合理（简单查询300-500，复杂查询500-1000，深度研究1000-2000）
5. 避免创建冗余任务
6. 初始化每个任务的状态为 "pending"

**示例1 - 简单查询**:
查询: "孙悟空的师父是谁?"
分解结果:
```json
{{
  "nodes": [
    {{
      "task_id": "task_001",
      "task_type": "local_search",
      "description": "在知识图谱中检索孙悟空的师徒关系",
      "priority": 1,
      "estimated_tokens": 400,
      "depends_on": [],
      "entities": ["孙悟空"],
      "status": "pending"
    }}
  ],
  "execution_mode": "sequential"
}}
```

**示例2 - 复杂查询**:
查询: "分析悟空传中孙悟空与天庭的冲突根源和发展过程"
分解结果:
```json
{{
  "nodes": [
    {{
      "task_id": "task_001",
      "task_type": "local_search",
      "description": "检索孙悟空的基本信息和早期经历",
      "priority": 1,
      "estimated_tokens": 500,
      "depends_on": [],
      "entities": ["孙悟空"],
      "status": "pending"
    }},
    {{
      "task_id": "task_002",
      "task_type": "local_search",
      "description": "检索天庭的统治结构和规则体系",
      "priority": 1,
      "estimated_tokens": 500,
      "depends_on": [],
      "entities": ["天庭"],
      "status": "pending"
    }},
    {{
      "task_id": "task_003",
      "task_type": "chain_exploration",
      "description": "追踪孙悟空与天庭之间的对抗关系路径",
      "priority": 2,
      "estimated_tokens": 800,
      "depends_on": ["task_001", "task_002"],
      "entities": ["孙悟空", "天庭"],
      "status": "pending"
    }},
    {{
      "task_id": "task_004",
      "task_type": "deep_research",
      "description": "深度分析冲突的根源和演变逻辑",
      "priority": 2,
      "estimated_tokens": 1200,
      "depends_on": ["task_003"],
      "status": "pending"
    }},
    {{
      "task_id": "task_005",
      "task_type": "global_search",
      "description": "获取悟空传的整体主题和叙事结构",
      "priority": 3,
      "estimated_tokens": 600,
      "depends_on": [],
      "status": "pending"
    }}
  ],
  "execution_mode": "adaptive"
}}
```

现在请针对以下查询生成任务分解方案，严格按照JSON格式输出：

**查询**: {query}

**任务分解方案**:
```json
'''


# 功能: 检测查询模糊性，生成澄清问题
CLARIFY_PROMPT = '''你是一个查询澄清助手。你的职责是识别用户查询中的模糊性和不确定性，并生成针对性的澄清问题。

**用户查询**: {query}

**领域背景**: {domain}

**需要检查的模糊性类型**:
1. **领域范围不明确**: 查询涉及多个可能的领域或主题
2. **时间范围缺失**: 查询涉及时间信息但未指定具体时期
3. **粒度不清晰**: 用户需要概览还是详细信息不明确
4. **实体歧义**: 提到的实体名称可能有多种含义
5. **意图不明**: 用户是想知道"是什么"、"为什么"还是"如何做"

**判断标准**:
- 如果查询清晰明确，无需澄清，返回 `needs_clarification: false`
- 如果存在以上任一模糊性，返回 `needs_clarification: true` 并列出具体问题

**示例1 - 需要澄清**:
查询: "悟空的对手"
分析: 存在多重模糊性
- 实体歧义: "悟空"可能指原著孙悟空或《悟空传》中的角色
- 粒度不清: 是指所有对手还是主要对手
- 时间范围: 是指整个故事还是某个阶段

输出:
```json
{{
  "needs_clarification": true,
  "questions": [
    "您是想了解《悟空传》中的孙悟空，还是西游记原著中的孙悟空？",
    "您希望了解哪个阶段的对手？(1)大闹天宫前 (2)大闹天宫期间 (3)取经路上",
    "您需要全部对手列表还是只关注主要对手？"
  ],
  "ambiguity_types": ["实体歧义", "时间范围缺失", "粒度不清晰"]
}}
```

**示例2 - 无需澄清**:
查询: "《悟空传》中孙悟空在大闹天宫时的主要对手是谁?"
分析: 查询明确清晰
- 领域明确: 《悟空传》
- 时间明确: 大闹天宫时
- 粒度明确: 主要对手
- 实体明确: 孙悟空

输出:
```json
{{
  "needs_clarification": false,
  "questions": [],
  "ambiguity_types": []
}}
```

现在请分析以下查询，严格按照JSON格式输出：

**查询**: {query}

**澄清分析结果**:
```json
'''


# 功能: 审校TaskGraph，生成完整的PlanSpec
PLAN_REVIEW_PROMPT = '''你是一个计划审校助手。你的职责是审核任务分解方案(TaskGraph)，确保其合理性和可执行性，并生成完整的执行计划(PlanSpec)。

**用户原始查询**: {query}

**重写后的查询**: {refined_query}

**任务图(TaskGraph)**:
```json
{task_graph}
```

**用户确认的假设条件**: {assumptions}

**审校检查项**:
1. **任务数量**: 是否在合理范围内（建议1-8个任务）
2. **依赖关系**: 是否存在循环依赖或无法满足的依赖
3. **任务类型匹配**: 每个任务的类型是否适合其描述的操作
4. **优先级合理性**: 基础任务应该是高优先级
5. **Token预估**: 总token消耗是否在预算内（建议 < 10000）
6. **执行模式**: sequential(简单查询) vs adaptive(复杂查询) 的选择是否恰当
7. **状态字段**: 每个任务的初始 `status` 应为 `"pending"`

**生成内容**:
1. **问题陈述(ProblemStatement)**: 清晰描述用户要解决的问题
2. **验收标准(AcceptanceCriteria)**: 定义任务完成的标准
3. **审校意见**: 指出潜在问题并给出修改建议

**示例输出**:
```json
{{
  "problem_statement": {{
    "original_query": "分析悟空传中孙悟空的成长轨迹",
    "refined_query": "分析《悟空传》中孙悟空从天真少年到反抗者的思想转变和关键事件",
    "background_info": "《悟空传》是今何在的网络小说，重新解构了西游记的故事",
    "user_intent": "理解角色发展和主题深意"
  }},
  "acceptance_criteria": {{
    "completion_conditions": [
      "覆盖孙悟空成长的主要阶段",
      "提供关键事件和转折点的证据",
      "分析思想变化的内在逻辑"
    ],
    "quality_requirements": [
      "引用原文证据支持分析",
      "逻辑连贯且有深度",
      "避免主观臆断"
    ],
    "min_evidence_count": 5,
    "min_confidence": 0.75
  }},
  "validation_results": {{
    "is_valid": true,
    "issues": [],
    "suggestions": [
      "建议在task_003之后增加一个global_search任务，获取悟空传的整体叙事结构"
    ],
    "estimated_total_tokens": 3200,
    "estimated_time_minutes": 5
  }}
}}
```

现在请审校以上任务图并生成完整计划，严格按照JSON格式输出：

**PlanSpec**:
```json
'''
