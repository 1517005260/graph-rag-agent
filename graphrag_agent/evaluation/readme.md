# GraphRAG 评估系统

这是一个用于评估不同类型 RAG Agent 性能的全面评估系统，包括传统向量检索（Naive RAG）、图检索（Graph RAG）、混合检索（Hybrid RAG）、融合检索（Fusion RAG）和深度研究（Deep Research）Agent。

## 目录结构

```
evaluator/
├── __init__.py                  # 模块初始化文件，包含调试模式设置
├── core/                        # 核心组件
│   ├── __init__.py              # 核心组件初始化文件
│   ├── base_evaluator.py        # 评估器基类
│   ├── base_metric.py           # 评估指标基类
│   └── evaluation_data.py       # 评估数据结构定义
├── evaluator_config/            # 配置相关
│   ├── __init__.py              # 配置模块初始化
│   ├── agent_evaluation_config.py # Agent评估配置
│   └── evaluatorConfig.py       # 评估器配置类
├── evaluators/                  # 具体评估器实现
│   ├── __init__.py              # 评估器模块初始化
│   ├── answer_evaluator.py      # 答案评估器
│   ├── composite_evaluator.py   # 组合评估器
│   └── retrieval_evaluator.py   # 检索评估器
├── metrics/                     # 各类评估指标
│   ├── __init__.py              # 指标模块初始化
│   ├── answer_metrics.py        # 答案质量评估指标
│   ├── deep_search_metrics.py   # 深度研究评估指标
│   ├── graph_metrics.py         # 图评估指标
│   ├── llm_metrics.py           # LLM评估指标
│   └── retrieval_metrics.py     # 检索性能评估指标
├── preprocessing/               # 预处理工具
│   ├── __init__.py              # 预处理模块初始化
│   ├── reference_extractor.py   # 引用数据提取器
│   └── text_cleaner.py          # 文本清理工具
├── test/                        # 测试脚本
│   ├── answer.json              # 测试用标准答案
│   ├── evaluate_all_agents.py   # 评估所有Agent的脚本
│   ├── evaluate_deep_agent.py   # 评估Deep Agent脚本
│   ├── evaluate_fusion_agent.py # 评估Fusion Agent脚本
│   ├── evaluate_graph_agent.py  # 评估Graph Agent脚本
│   ├── evaluate_hybrid_agent.py # 评估Hybrid Agent脚本
│   ├── evaluate_naive_agent.py  # 评估Naive Agent脚本
│   ├── questions.json           # 测试用问题
│   └── readme.md                # 测试脚本使用说明
└── utils/                       # 工具函数
    ├── __init__.py              # 工具模块初始化
    ├── data_utils.py            # 数据处理工具
    ├── eval_utils.py            # 评估工具函数
    ├── logging_utils.py         # 日志工具
    └── text_utils.py            # 文本处理工具
```

## 系统实现思路

GraphRAG 评估系统基于组合式设计，采用基类与派生类的层次架构，通过灵活的配置系统和可扩展的评估指标实现对多种 RAG Agent 的全面评估。

### 核心架构

系统围绕以下几个核心组件构建：

1. **基类与接口**：
   - `BaseEvaluator`：所有评估器的基类，定义评估流程和结果处理
   - `BaseMetric`：所有评估指标的基类，定义指标计算接口

2. **数据结构**：
   - `AnswerEvaluationSample/Data`：用于答案评估的数据结构
   - `RetrievalEvaluationSample/Data`：用于检索评估的数据结构

3. **评估器实现**：
   - `AnswerEvaluator`：专注评估回答质量
   - `GraphRAGRetrievalEvaluator`：专注评估检索性能
   - `CompositeGraphRAGEvaluator`：组合评估器，协调多维度评估

4. **评估指标体系**：按照不同维度划分为多类评估指标，支持规则评分和LLM回退评分

### 核心功能实现

#### 1. 组合评估器 (CompositeGraphRAGEvaluator)

组合评估器是系统的核心，它负责协调不同维度的评估并生成最终报告：

```python
# 使用标准答案评估Agent
results = evaluator.evaluate_with_golden_answers(agent_name, questions, golden_answers)

# 仅评估检索性能
results = evaluator.evaluate_retrieval_only(agent_name, questions)

# 比较多个Agent的性能
results = evaluator.compare_agents_with_golden_answers(questions, golden_answers)
```

核心方法包括：
- `evaluate_with_golden_answers`: 使用标准答案评估特定Agent
- `evaluate_retrieval_only`: 仅评估Agent的检索性能
- `compare_agents_with_golden_answers`: 比较多个Agent的性能
- `format_comparison_table`: 将比较结果格式化为表格

#### 2. 评估指标计算

系统实现了多种评估指标，每个指标都是`BaseMetric`的派生类，核心方法是`calculate_metric`：

```python
def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
    """
    计算评估指标
    
    Args:
        data: 评估数据对象
        
    Returns:
        Tuple[Dict, List]: 评估结果和每个样本的评分
    """
```

标准的指标计算流程为：
1. 对每个样本执行评估计算
2. 获取各样本的得分
3. 计算平均分作为总体得分
4. 返回总体得分和各样本得分

#### 3. LLM回退机制

对于难以通过规则评分的情况，系统实现了LLM回退机制：

```python
def get_llm_fallback_score(self, prompt: str, default_score: float = 0.5) -> float:
    """
    使用LLM进行回退评分
    
    Args:
        prompt: 提示文本
        default_score: 默认分数，当LLM评分失败时返回
        
    Returns:
        float: LLM评分结果或默认分数
    """
```

#### 4. 引用数据提取与处理

系统能够从回答中提取引用的实体、关系和文本块信息，便于评估检索性能：

```python
def extract_references_from_answer(answer: str) -> Dict[str, Any]:
    """
    从回答中提取引用数据
    
    Args:
        answer: AI生成的回答
        
    Returns:
        Dict: 包含entities, relationships, chunks等信息的字典
    """
```

## 评估维度

系统实现了四个主要评估维度，共20+种评估指标：

### 1. 答案质量评估

这一维度主要关注系统生成回答的质量和准确性，包括多个方面：

#### 1.1 精确匹配 (ExactMatch)
评估系统回答与标准答案的匹配程度：
- 实现原理：对答案进行标准化预处理，移除标点、冠词等，计算匹配度
- 计算方法：完全匹配得1.0分，不匹配时计算内容相似度，并可使用LLM评估语义等价性
- 源代码：`metrics/answer_metrics.py` 中的 `ExactMatch` 类

#### 1.2 F1分数 (F1Score)
平衡精确率和召回率的评估：
- 实现原理：对答案进行中文分词，过滤停用词，计算词汇重叠度
- 计算方法：使用精确率（共有词/预测词数）和召回率（共有词/标准词数）计算F1值
- 特点：能更准确反映部分正确的情况，对非完全匹配更友好
- 源代码：`metrics/answer_metrics.py` 中的 `F1Score` 类

#### 1.3 回答连贯性 (ResponseCoherence)
评估回答的结构完整性和逻辑流畅程度：
- 实现原理：分析答案结构特征（段落数、标题、句子数），并使用LLM评估
- 评估要点：结构组织、逻辑连贯性、思路清晰度
- 源代码：`metrics/llm_metrics.py` 中的 `ResponseCoherence` 类

#### 1.4 事实一致性 (FactualConsistency)
评估回答是否包含自相矛盾的信息：
- 实现原理：提取回答中的关键信息点，评估信息间的逻辑一致性
- 评估要点：内部逻辑一致性、信息准确性、无矛盾内容
- 源代码：`metrics/llm_metrics.py` 中的 `FactualConsistency` 类

#### 1.5 回答全面性 (ComprehensiveAnswerMetric)
评估回答是否涵盖问题的各个方面：
- 实现原理：使用LLM评估回答是否全面解决问题的所有关键方面
- 评估要点：信息覆盖度、细节丰富程度、是否遗漏关键信息
- 源代码：`metrics/llm_metrics.py` 中的 `ComprehensiveAnswerMetric` 类

#### 1.6 LLM综合评估 (LLMGraphRagEvaluator)
使用LLM多维度评估回答质量：
- 实现原理：LLM从四个方面评估：全面性、相关性、增强理解能力、直接性
- 评估特点：通过加权计算生成综合分数，可根据需求调整各维度权重
- 源代码：`metrics/llm_metrics.py` 中的 `LLMGraphRagEvaluator` 类

### 2. 检索性能评估

这一维度关注系统检索信息的效果和效率：

#### 2.1 检索精确率 (RetrievalPrecision)
评估检索到的实体与最终引用实体的匹配程度：
- 实现原理：比较检索到的实体集合与答案中引用的实体集合的重叠度
- 计算方法：`matched / 引用实体总数`，考虑直接匹配和数字ID匹配两种情况
- 源代码：`metrics/retrieval_metrics.py` 中的 `RetrievalPrecision` 类

#### 2.2 检索利用率 (RetrievalUtilization)
评估系统对检索结果的有效利用程度：
- 实现原理：分析答案中引用实体在检索结果中的覆盖情况
- 计算方法：检查引用实体在检索实体中的覆盖率，同时考虑部分匹配和相似性
- 源代码：`metrics/retrieval_metrics.py` 中的 `RetrievalUtilization` 类

#### 2.3 检索延迟 (RetrievalLatency)
评估检索过程的时间效率：
- 实现原理：记录从提问到获得回答的时间（秒）
- 特点：直接报告原始时间值，不进行打分转换，越低越好
- 源代码：`metrics/retrieval_metrics.py` 中的 `RetrievalLatency` 类

#### 2.4 文本块利用率 (ChunkUtilization)
评估对检索到的文本块内容的使用程度：
- 实现原理：从Neo4j获取引用文本块内容，提取关键短语，计算在回答中的使用比例
- 适用场景：主要适用于基于文本块的NaiveAgent
- 计算方法：文本块关键短语在回答中的出现比例
- 源代码：`metrics/retrieval_metrics.py` 中的 `ChunkUtilization` 类

### 3. 图评估

这一维度专注于评估系统对知识图谱的利用效果：

#### 3.1 实体覆盖率 (EntityCoverageMetric)
评估检索实体对问题相关知识点的覆盖程度：
- 实现原理：获取实体信息和描述，提取问题关键词，计算关键词在实体信息中的匹配率
- 计算要点：考虑实体数量和匹配质量，使用Neo4j查询实体详细信息
- 源代码：`metrics/graph_metrics.py` 中的 `EntityCoverageMetric` 类

#### 3.2 图覆盖率 (GraphCoverageMetric)
评估检索到的知识图谱对问题领域的覆盖程度：
- 实现原理：从三个维度评估：结构得分、相关性得分和连通性得分
- 结构得分：基于实体和关系的数量及质量
- 相关性得分：关键词在图数据中的匹配率
- 连通性得分：实体之间的连接度（路径数量）
- 源代码：`metrics/graph_metrics.py` 中的 `GraphCoverageMetric` 类

#### 3.3 关系利用率 (RelationshipUtilizationMetric)
评估系统对实体间关系的利用程度：
- 实现原理：从三个维度综合评估：数量得分、质量得分和相关性得分
- 数量得分：基于关系数量计算
- 质量得分：基于关系描述质量和多样性
- 相关性得分：关系中实体与引用实体的重合度
- 源代码：`metrics/graph_metrics.py` 中的 `RelationshipUtilizationMetric` 类

#### 3.4 社区相关性 (CommunityRelevanceMetric)
评估检索到的知识社区对问题领域的相关程度：
- 实现原理：查询实体关联的社区信息，计算问题关键词在社区信息中的匹配率
- 特点：根据Agent类型给予不同基础分和加成，对不同类型的Agent有差异化评估
- 源代码：`metrics/graph_metrics.py` 中的 `CommunityRelevanceMetric` 类

#### 3.5 子图质量 (SubgraphQualityMetric)
评估检索到的子图的信息密度和结构质量：
- 实现原理：计算图密度（边数与最大可能边数之比）和连通性（参与关系的实体比例）
- 计算方法：加权平均密度和连通性得分
- 源代码：`metrics/graph_metrics.py` 中的 `SubgraphQualityMetric` 类

### 4. 深度研究评估

这一维度评估系统的深度推理和研究能力，主要针对Deep Research Agent：

#### 4.1 推理连贯性 (ReasoningCoherence)
评估深度研究Agent推理过程的连贯性和逻辑性：
- 实现原理：分析思考过程的结构和连贯性，评估搜索查询与推理步骤的关联度
- 评估要点：思考过程结构化程度、逻辑清晰度、步骤衔接自然度
- 源代码：`metrics/deep_search_metrics.py` 中的 `ReasoningCoherence` 类

#### 4.2 推理深度 (ReasoningDepth)
评估深度研究Agent推理过程的深度和复杂性：
- 实现原理：分析搜索查询数量和多样性、思考过程的层次深度、信息提取质量
- 评估要点：查询深度、思考层次、信息整合复杂度
- 源代码：`metrics/deep_search_metrics.py` 中的 `ReasoningDepth` 类

#### 4.3 迭代改进 (IterativeImprovementMetric)
评估深度研究Agent在多轮推理中的改进程度：
- 实现原理：比较首次和最终查询的质量差异、评估信息提取的迭代改进
- 评估要点：查询策略演进、信息质量提升、自我纠错能力
- 源代码：`metrics/deep_search_metrics.py` 中的 `IterativeImprovementMetric` 类

#### 4.4 知识图谱利用率 (KnowledgeGraphUtilizationMetric)
评估深度研究Agent对知识图谱信息的利用程度：
- 实现原理：分析思考过程中对图谱概念的引用频率、实体和社区信息整合质量
- 特点：专为DeeperResearchTool设计，评估图谱知识在深度推理中的应用
- 源代码：`metrics/deep_search_metrics.py` 中的 `KnowledgeGraphUtilizationMetric` 类

## 使用方法

评估系统提供了多种使用方式，支持单独评估某种Agent或比较多个Agent的性能：

```python
# 评估单个Agent
evaluate_agent(
    agent_type="graph",
    questions=questions,
    golden_answers=golden_answers,
    save_dir="./results",
    metrics=["em", "f1", "retrieval_precision"]
)

# 比较多个Agent
evaluator = CompositeGraphRAGEvaluator(config)
results = evaluator.compare_agents_with_golden_answers(questions, golden_answers)
```

各类评估脚本位于`test/`目录下，可以参考`test/readme.md`了解详细用法。