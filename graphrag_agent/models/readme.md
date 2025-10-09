# Model 模块

## 目录结构

```
model/
├── __init__.py          # 模块初始化文件
├── get_models.py        # 模型获取和初始化功能
└── test_stream_model.py # 流式模型测试
```

## 模块说明

Model 模块负责初始化和管理各种语言模型，主要基于 LangChain 框架实现了对 OpenAI API 的集成，支持普通调用和流式输出两种模式。

### 核心功能

1. **模型初始化**：支持从环境变量加载配置参数，灵活配置模型行为

2. **支持的模型类型**：
   - 嵌入模型 (Embeddings)：用于文本向量化
   - 对话模型 (LLM)：支持普通和流式两种输出方式

3. **流式输出支持**：通过 AsyncIteratorCallbackHandler 实现逐字输出能力

### 核心函数

- `get_embeddings_model()`：初始化并返回文本嵌入模型，用于向量化查询和文档
- `get_llm_model()`：初始化并返回标准对话模型，用于一次性生成完整回答
- `get_stream_llm_model()`：初始化并返回流式对话模型，支持逐字输出，提升用户体验

### 使用方法

```python
# 标准模型使用示例
from model.get_models import get_llm_model

llm = get_llm_model()
response = llm.invoke("你好")
print(response)

# 流式模型使用示例(注：由于langchain bug，这里需要改源码才能使用，本项目采用模拟流式输出)
import asyncio
from model.get_models import get_stream_llm_model
from langchain_core.messages import HumanMessage

async def main():
    chat = get_stream_llm_model()
    messages = [HumanMessage(content="Tell me a short joke.")]
    async for chunk in chat.astream(messages):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### 配置说明

模块依赖以下环境变量：
- `OPENAI_API_KEY`: OpenAI API 密钥
- `OPENAI_BASE_URL`: API 基础 URL (可配置为代理或自定义端点)
- `OPENAI_EMBEDDINGS_MODEL`: 使用的嵌入模型名称
- `OPENAI_LLM_MODEL`: 使用的语言模型名称
- `TEMPERATURE`: 模型温度参数
- `MAX_TOKENS`: 最大生成token数