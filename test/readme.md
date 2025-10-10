# Test 模块

## 目录结构

```
test/
├── search_with_stream.py    # 各种 Agent 的流式输出测试
└── search_without_stream.py # 各种 Agent 的标准输出测试
```

## 模块说明

Test 模块提供了对多种智能代理 (Agent) 的测试接口，支持流式和非流式两种输出模式，便于开发者比较不同代理的性能和效果差异。

### 支持的 Agent 类型

1. **DeepResearchAgent**：深度研究代理，能够进行深入分析并展示思考过程
2. **NaiveRagAgent**：基础检索增强生成代理
3. **GraphAgent**：基于知识图谱的代理
4. **HybridAgent**：混合型代理，结合多种策略
5. **FusionGraphRAGAgent**：融合图谱和检索增强的代理

### 两种测试模式

1. **非流式模式** (`search_without_stream.py`)：
   - 一次性返回完整回答
   - 支持获取执行轨迹，方便调试和分析
   - 适合批量测试和结果比较

2. **流式模式** (`search_with_stream.py`)：
   - 支持逐字输出，提升实时用户体验
   - 可选择是否显示思考过程
   - 适合实际应用场景

### 核心功能

1. **思考过程展示**：支持可视化代理的思考和推理过程
2. **执行轨迹追踪**：记录代理决策链和执行步骤
3. **多类型输出格式**：支持纯文本、字典和流式三种输出格式
4. **批量测试**：可一次运行所有代理测试，便于对比

### 使用示例

```python
# 非流式测试示例
from graphrag_agent.agents.deep_research_agent import DeepResearchAgent

agent = DeepResearchAgent()
result = agent.ask("优秀学生要如何申请")
print(result)

# 带思考过程的测试
result_with_thinking = agent.ask_with_thinking("优秀学生要如何申请")
print(result_with_thinking.get('answer'))

# 流式测试示例 (异步)
import asyncio

async def test_stream():
    agent = DeepResearchAgent()
    async for chunk in agent.ask_stream("优秀学生要如何申请", show_thinking=True):
        # 处理字典类型的最终答案
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            # 输出文本块
            print(chunk, end="", flush=True)

asyncio.run(test_stream())
```

### 测试配置

测试模块默认使用以下查询进行所有代理的比较测试：
- 测试查询: "优秀学生要如何申请"
- 可根据需要修改测试脚本中的查询内容，进行不同场景的测试