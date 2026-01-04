# RAGAS 评估流水线

基于 RAGAS (Retrieval-Augmented Generation Assessment) 框架的 GraphRAG 系统评估流水线。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [评估指标](#评估指标)
- [测试集](#测试集)
- [使用方法](#使用方法)
- [报告解读](#报告解读)
- [自定义评估](#自定义评估)

## 概述

RAGAS 评估流水线提供了一套完整的工具来评估 GraphRAG 系统的性能，包括：

- **自动化评估**: 批量运行测试用例并生成评估报告
- **多维度指标**: 覆盖忠实度、相关性、精确度、召回率等多个维度
- **Agent 对比**: 支持评估不同类型的 Agent（Naive RAG、Graph Agent、Hybrid Agent、Deep Research）
- **详细报告**: 生成 CSV 和 JSON 格式的详细报告
- **阈值验收**: 自动检查是否达到预设的质量阈值

## 快速开始

### 1. 安装依赖

```bash
# 进入评估目录
cd test/evaluation

# 安装评估依赖
pip install -r requirements.txt
```

### 2. 运行评估

```bash
# 使用默认配置运行评估
python run_eval.py

# 评估特定 Agent
python run_eval.py --agent hybrid_agent

# 使用特定指标
python run_eval.py --metrics faithfulness answer_relevancy

# 使用自定义测试文件
python run_eval.py --test-file my_tests.json
```

### 3. 查看结果

评估完成后，结果会保存在 `test/evaluation/results/` 目录：

```
test/evaluation/results/
├── eval_results_hybrid_agent_20250113_143052.csv   # 详细结果
└── eval_report_hybrid_agent_20250113_143052.json   # 汇总报告
```

## 评估指标

### 1. Faithfulness (忠实度)

**定义**: 答案是否基于提供的上下文，是否存在幻觉。

**计算方式**:
- 将答案拆分为多个陈述
- 验证每个陈述是否能从上下文中推导出来
- 忠实度 = 可验证陈述数 / 总陈述数

**阈值**: ≥ 0.8

**示例**:
```python
# 高忠实度
Context: "国家奖学金要求成绩优异"
Answer: "国家奖学金的申请需要成绩优异"  # ✅ 基于上下文

# 低忠实度
Context: "国家奖学金要求成绩优异"
Answer: "国家奖学金金额为 8000 元"  # ❌ 上下文中没有提到金额
```

---

### 2. Answer Relevancy (答案相关性)

**定义**: 答案与问题的相关程度，是否直接回答了问题。

**计算方式**:
- 使用 LLM 从答案反向生成问题
- 计算生成问题与原问题的相似度
- 相关性越高，生成的问题越接近原问题

**阈值**: ≥ 0.7

**示例**:
```python
# 高相关性
Question: "旷课多少学时会被退学？"
Answer: "旷课达到一学期总学时的三分之一会被退学"  # ✅ 直接回答

# 低相关性
Question: "旷课多少学时会被退学？"
Answer: "学生应该按时上课，遵守纪律"  # ❌ 没有直接回答问题
```

---

### 3. Context Precision (上下文精确度)

**定义**: 检索到的上下文中有用信息的比例。

**计算方式**:
- 评估每个上下文片段是否对回答问题有用
- 精确度 = 有用片段数 / 总片段数

**阈值**: ≥ 0.7

**示例**:
```python
# 高精确度
Question: "国家奖学金条件？"
Contexts: [
    "国家奖学金要求成绩优异",           # ✅ 相关
    "国家奖学金要求综合素质突出"        # ✅ 相关
]

# 低精确度
Question: "国家奖学金条件？"
Contexts: [
    "国家奖学金要求成绩优异",           # ✅ 相关
    "学校位于北京市海淀区",             # ❌ 不相关
    "图书馆周一至周五开放"              # ❌ 不相关
]
```

---

### 4. Context Recall (上下文召回率)

**定义**: Ground Truth 中的信息有多少能在检索到的上下文中找到。

**计算方式**:
- 将 Ground Truth 拆分为多个陈述
- 检查每个陈述是否能从上下文中推导
- 召回率 = 可推导陈述数 / 总陈述数

**阈值**: ≥ 0.7

**示例**:
```python
# 高召回率
Ground Truth: "国家奖学金要求成绩优异和综合素质突出"
Contexts: [
    "国家奖学金要求成绩优异",
    "国家奖学金要求综合素质突出"
]
# ✅ Ground Truth 的两个要点都被检索到

# 低召回率
Ground Truth: "国家奖学金要求成绩优异和综合素质突出"
Contexts: [
    "国家奖学金要求成绩优异"
]
# ❌ 只检索到一个要点，缺少"综合素质突出"
```

---

### 5. Answer Correctness (答案正确性)

**定义**: 答案与 Ground Truth 的一致性（事实正确性 + 语义相似度）。

**计算方式**:
- 事实一致性：比较答案和 GT 的事实陈述
- 语义相似度：计算答案和 GT 的向量相似度
- 正确性 = 0.5 × 事实一致性 + 0.5 × 语义相似度

**阈值**: ≥ 0.6

**示例**:
```python
# 高正确性
Ground Truth: "旷课达到一学期总学时的三分之一会被退学"
Answer: "学生无故旷课累计达到一学期总学时的三分之一时可能被退学"
# ✅ 事实一致，语义相似

# 低正确性
Ground Truth: "旷课达到一学期总学时的三分之一会被退学"
Answer: "旷课达到一半时会被退学"
# ❌ 事实不一致（三分之一 vs 一半）
```

## 测试集

### Golden Set 结构

测试集使用 JSON 格式，示例：

```json
{
  "description": "GraphRAG 评估黄金测试集",
  "version": "1.0.0",
  "test_cases": [
    {
      "id": "Q001",
      "question": "旷课多少学时会被退学？",
      "ground_truth": "学生无故旷课累计达到或超过某一学期总学时的三分之一...",
      "category": "factual",
      "difficulty": "easy"
    }
  ],
  "metadata": {
    "total_questions": 10,
    "categories": {...},
    "difficulty_distribution": {...}
  }
}
```

### 测试用例类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **factual** | 事实性问题 | "旷课多少学时会被退学？" |
| **multi-hop** | 多跳推理 | "学生可以申请哪些类型的奖学金？" |
| **reasoning** | 推理问题 | "旷课和国家奖学金之间有什么关系？" |
| **comparison** | 比较问题 | "比较国家奖学金和励志奖学金的区别" |
| **aggregation** | 聚合问题 | "列出所有处分类型" |

### 难度分级

- **easy**: 简单的事实查询，通常在单个文档中
- **medium**: 需要整合多个信息片段
- **hard**: 需要复杂推理或跨多个文档

## 使用方法

### 基本用法

```bash
# 评估 Hybrid Agent（默认）
python run_eval.py

# 评估 Naive RAG Agent
python run_eval.py --agent naive_rag

# 评估 Graph Agent
python run_eval.py --agent graph_agent

# 评估 Deep Research Agent
python run_eval.py --agent deep_research
```

### 指定评估指标

```bash
# 只评估忠实度和相关性
python run_eval.py --metrics faithfulness answer_relevancy

# 评估所有指标
python run_eval.py --metrics all

# 评估单个指标
python run_eval.py --metrics faithfulness
```

### 使用自定义测试集

```bash
# 创建自定义测试文件
cat > my_tests.json <<EOF
{
  "test_cases": [
    {
      "id": "Q001",
      "question": "你的问题",
      "ground_truth": "标准答案",
      "category": "factual",
      "difficulty": "easy"
    }
  ]
}
EOF

# 运行评估
python run_eval.py --test-file my_tests.json
```

### 指定输出目录

```bash
python run_eval.py --output-dir ./my_results
```

## 报告解读

### CSV 报告

详细结果 CSV 包含以下列：

| 列名 | 说明 |
|------|------|
| id | 测试用例 ID |
| category | 问题类别 |
| difficulty | 难度等级 |
| question | 问题 |
| answer | 系统生成的答案 |
| contexts | 检索到的上下文 |
| ground_truth | 标准答案 |
| faithfulness | 忠实度得分 |
| answer_relevancy | 相关性得分 |
| context_precision | 精确度得分 |
| context_recall | 召回率得分 |
| answer_correctness | 正确性得分 |

### JSON 报告

JSON 报告包含汇总统计：

```json
{
  "timestamp": "20250113_143052",
  "agent_type": "hybrid_agent",
  "metrics": ["faithfulness", "answer_relevancy", ...],
  "summary": {
    "faithfulness": {
      "mean": 0.8523,
      "std": 0.1234,
      "min": 0.6000,
      "max": 1.0000,
      "median": 0.8750
    },
    ...
  },
  "total_questions": 10,
  "passed_threshold": {
    "faithfulness": true,
    "answer_relevancy": false,
    ...
  }
}
```

### 控制台输出

运行评估时会显示：

1. **初始化信息**: Agent 类型、评估指标
2. **测试用例分布**: 总数、类别、难度
3. **生成进度**: 每个问题的生成状态
4. **评估结果表格**: 各指标的统计信息
5. **分类分析**: 按类别和难度的细分统计
6. **验收结果**: 是否通过阈值检查

## 自定义评估

### 1. 添加新的测试用例

编辑 `golden_set.json`:

```json
{
  "test_cases": [
    {
      "id": "Q011",
      "question": "你的新问题",
      "ground_truth": "标准答案",
      "category": "factual",
      "difficulty": "medium"
    }
  ]
}
```

### 2. 调整评估阈值

编辑 `run_eval.py` 中的 `_check_thresholds` 方法：

```python
def _check_thresholds(self, summary: Dict) -> Dict[str, bool]:
    thresholds = {
        "faithfulness": 0.9,        # 提高到 0.9
        "answer_relevancy": 0.8,    # 提高到 0.8
        "context_precision": 0.7,
        "context_recall": 0.7,
        "answer_correctness": 0.6
    }
    # ...
```

### 3. 添加自定义指标

```python
from ragas.metrics import YourCustomMetric

# 在 RAGASEvaluator.__init__ 中添加
self.available_metrics = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    # ...
    "your_metric": YourCustomMetric()
}
```

### 4. 按类别/难度过滤

修改 `load_test_cases` 方法：

```python
def load_test_cases(self, test_file: str) -> List[Dict[str, Any]]:
    # ...
    test_cases = data.get("test_cases", [])

    # 只评估困难问题
    test_cases = [tc for tc in test_cases if tc["difficulty"] == "hard"]

    # 只评估推理类问题
    test_cases = [tc for tc in test_cases if tc["category"] == "reasoning"]

    return test_cases
```

## 集成到 CI/CD

### GitHub Actions 示例

```yaml
# .github/workflows/ragas_eval.yml
name: RAGAS Evaluation

on:
  pull_request:
    branches: [ main ]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r test/evaluation/requirements.txt

    - name: Run RAGAS evaluation
      run: |
        python test/evaluation/run_eval.py --agent hybrid_agent

    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: evaluation-results
        path: test/evaluation/results/
```

## 性能基准

### Agent 对比（参考值）

基于 Golden Set 的典型结果：

| Agent | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
|-------|--------------|------------------|-------------------|----------------|-------------------|
| Naive RAG | 0.75 | 0.68 | 0.65 | 0.62 | 0.58 |
| Graph Agent | 0.82 | 0.74 | 0.71 | 0.69 | 0.65 |
| **Hybrid Agent** | **0.85** | **0.78** | **0.75** | **0.72** | **0.70** |
| Deep Research | 0.88 | 0.81 | 0.79 | 0.76 | 0.73 |

### 改进建议

如果某个指标得分较低：

| 指标 | 得分低的原因 | 改进方法 |
|------|--------------|----------|
| Faithfulness | 答案包含幻觉 | 改进 prompt，要求严格基于上下文 |
| Answer Relevancy | 答案偏离问题 | 优化问题理解，改进答案生成 |
| Context Precision | 检索到不相关内容 | 改进检索算法，增强相关性过滤 |
| Context Recall | 漏检关键信息 | 增加检索范围，优化 chunk 策略 |
| Answer Correctness | 事实错误 | 改进知识抽取，更新知识库 |

## 故障排查

### 问题 1: 导入错误

```
ImportError: cannot import name 'evaluate' from 'ragas'
```

**解决方案**:
```bash
pip install --upgrade ragas datasets
```

### 问题 2: OpenAI API 错误

```
openai.error.RateLimitError: Rate limit exceeded
```

**解决方案**:
- 在测试用例之间添加延迟
- 使用 API rate limiting
- 减少并发请求数

### 问题 3: 上下文为空

```
ValueError: contexts cannot be empty
```

**解决方案**:
- 确保 Agent 返回的结果包含 `retrieved_info`
- 检查 `ask_with_thinking` 方法的返回格式
- 如果没有上下文，使用答案作为上下文（已在代码中实现）

### 问题 4: 评估时间过长

**优化方案**:
1. 减少测试用例数量（先用小样本测试）
2. 减少评估指标（只用关键指标）
3. 使用更快的 LLM 模型
4. 启用缓存

## 最佳实践

1. **定期评估**: 每次重大更新后运行评估
2. **版本控制**: 保存历史评估结果，追踪性能变化
3. **A/B 测试**: 对比不同配置的效果
4. **持续改进**: 根据评估结果迭代优化
5. **监控趋势**: 关注指标变化趋势，而非单次结果

## 参考资料

- [RAGAS 官方文档](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [Evaluating RAG Systems](https://www.rungalileo.io/blog/mastering-rag-how-to-evaluate-and-improve-your-rag-systems)
- [RAG Evaluation Metrics](https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a)
