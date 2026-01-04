# 统一返回结构规范（工程级实践）

## 🎯 设计目标

解决以下问题：
1. **Tool/Retriever/Agent 返回格式不统一** - 导致下游处理复杂
2. **Frontend 解析 JSON 不可靠** - LLM 生成的 JSON 格式可能不正确
3. **检索结果 vs 生成回答混淆** - 缺乏明确的数据结构

## 📐 架构原则

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Tool      │─────▶│   Agent     │─────▶│  Frontend   │
│ returns dict│      │ dict→prompt │      │ show answer │
│   (JSON)    │      │    →LLM     │      │  (no parse) │
└─────────────┘      └─────────────┘      └─────────────┘
```

**核心分工：**
- **Tool**: 返回标准化 dict（SearchResponse）
- **Agent**: 解析 dict，构建 prompt，调用 LLM
- **Frontend**: 直接展示 answer 字段（不解析 JSON）

## 📦 数据结构

### SearchResponse

```python
{
  "answer": str,           # LLM 生成的回答文本
  "references": {          # 引用的资源
    "chunks": List[str],       # 文本块 ID 列表
    "entities": List[str],     # 实体 ID 列表
    "communities": List[str],  # 社区 ID 列表
    "relationships": List[str] # 关系 ID 列表
  },
  "meta": {                # 元数据
    "retriever": str,          # 检索器类型: naive | graph | hybrid | deep_research
    "search_time": float,      # 搜索耗时（秒）
    "llm_time": float,         # LLM 生成耗时（秒）
    "total_time": float,       # 总耗时（秒）
    "scores": List[float],     # 相似度分数列表
    "top_k": int,              # 检索的文档数量
    "cache_hit": bool,         # 是否命中缓存
    "timestamp": str           # ISO 格式时间戳
  }
}
```

### 完整示例

```json
{
  "answer": "根据检索结果，学生旷课达到20学时将被退学处理。",
  "references": {
    "chunks": ["chunk_001", "chunk_045", "chunk_089"],
    "entities": ["entity_student_001", "entity_regulation_002"],
    "communities": [],
    "relationships": ["rel_001"]
  },
  "meta": {
    "retriever": "naive",
    "search_time": 0.25,
    "llm_time": 1.5,
    "total_time": 1.75,
    "scores": [0.89, 0.76, 0.65],
    "top_k": 10,
    "cache_hit": false,
    "timestamp": "2025-12-17T10:30:00"
  }
}
```

## 🔧 使用方法

### 1. 在 Tool 中构建响应

```python
from graphrag_agent.search.response_models import ResponseBuilder

def search(self, query: str) -> Dict[str, Any]:
    # 初始化构建器
    builder = ResponseBuilder(retriever_name="naive")

    # 执行搜索
    results = self.vector_search.search_chunks(...)

    # 添加引用信息
    chunk_ids = [item["id"] for item in results]
    scores = [item["score"] for item in results]

    builder.add_chunks(chunk_ids)
    builder.add_scores(scores)
    builder.set_top_k(self.top_k)

    # 生成回答
    answer = self.llm.invoke(...)

    # 记录性能指标
    builder.set_timing(
        search_time=0.5,
        llm_time=1.2,
        total_time=1.7
    )

    # 返回标准化 dict
    return builder.build_dict(answer=answer)
```

### 2. 错误处理

```python
from graphrag_agent.search.response_models import create_error_response

try:
    results = self.search(query)
except Exception as e:
    return create_error_response(
        retriever_name="naive",
        error_message=str(e),
        error_type="search_error"
    )
```

### 3. Tool._run() 接口设计

```python
def get_tool(self) -> BaseTool:
    class NaiveRetrievalTool(BaseTool):
        def _run(self, query: str) -> str:
            # 内部使用结构化 dict
            response_dict = self.search(query)

            # ✅ 对外只返回 answer 纯文本
            # Agent 可以直接使用，无需解析 JSON
            return response_dict.get("answer", "搜索失败")
```

**关键设计原则：**
- **内部**：`search()` 返回结构化 dict（类型安全、便于测试和扩展）
- **对外**：`_run()` 返回纯文本 answer（避免 Agent 层类型错误）

这样避免了：
- ❌ `sequence item 0: expected str instance, dict found`
- ❌ Agent 解析 JSON 失败
- ❌ LLM 生成的 JSON 格式不正确

### 4. Agent 中使用

```python
# Agent 调用 Tool（现在返回纯文本）
answer_text = self.tool._run(query)  # 返回纯文本，可直接使用

# 无需解析 JSON，直接使用
print(answer_text)
```

### 5. 需要访问结构化数据时

如果 Agent 需要访问 references 或 meta 信息：

```python
# 直接调用 search() 方法（返回完整 dict）
response_dict = self.naive_tool.search(query)

# 访问所有字段
answer = response_dict["answer"]
chunks = response_dict["references"]["chunks"]
search_time = response_dict["meta"]["search_time"]
```

### 6. Frontend 中使用

```python
# Frontend 直接展示 answer
response = agent.ask(query)  # Agent 已经处理好了
print(response)  # 直接展示，不需要解析 JSON
```

## 🚀 迁移指南

### 旧代码（❌）

```python
def search(self, query: str) -> str:
    # ...
    answer = self.llm.invoke(...)

    # 手动拼接 JSON
    chunk_refs = ", ".join([f"'{id}'" for id in chunk_ids])
    answer += f"\n\n{{'data': {{'Chunks':[{chunk_refs}] }} }}"

    return answer
```

### 新代码（✅）

```python
from graphrag_agent.search.response_models import ResponseBuilder

def search(self, query: str) -> Dict[str, Any]:
    builder = ResponseBuilder(retriever_name="naive")

    # ...
    builder.add_chunks(chunk_ids)
    answer = self.llm.invoke(...)

    # 返回标准化 dict
    return builder.build_dict(answer=answer)
```

## 📊 优势对比

| 特性 | 旧方案 | 新方案（工程级） |
|------|--------|------------------|
| **返回格式** | 字符串拼接 JSON | 标准化 Pydantic 模型 |
| **类型安全** | ❌ 无 | ✅ 强类型检查 |
| **可扩展性** | ❌ 难以扩展 | ✅ 易于添加字段 |
| **错误处理** | ❌ 手动处理 | ✅ 统一错误响应 |
| **性能指标** | ❌ 分散记录 | ✅ 集中管理 |
| **缓存兼容** | ❌ 格式不一致 | ✅ 标准格式 |
| **Frontend 解析** | ❌ 需要解析 JSON | ✅ 直接使用 answer |

## 🔄 兼容性

### 缓存兼容

新代码能处理旧格式缓存：

```python
cached_result = self.cache_manager.get(cache_key)
if cached_result:
    # 新格式：dict
    if isinstance(cached_result, dict) and "answer" in cached_result:
        return cached_result
    # 旧格式：str（兼容处理）
    builder.set_cache_hit(True)
    return builder.build_dict(answer=str(cached_result))
```

### Tool 接口兼容

LangChain BaseTool 要求 `_run()` 返回 `str`，所以在 `get_tool()` 中序列化：

```python
def _run(self, query: str) -> str:
    response_dict = self.search(query)  # Dict
    return json.dumps(response_dict, ensure_ascii=False)  # JSON 字符串
```

## 📝 TODO：后续工作

1. **其他 Search Tool 迁移**：
   - `GraphSearchTool`
   - `HybridSearchTool`
   - `DeepResearchTool`

2. **Agent 层适配**：
   - 修改 Agent 解析 Tool 返回值的逻辑
   - 使用 `response["answer"]` 而不是解析 LLM 输出的 JSON

3. **Frontend 简化**：
   - 移除 JSON 解析逻辑
   - 直接展示 Agent 返回的 answer

4. **文档更新**：
   - 更新 CLAUDE.md
   - 更新各模块的 README

## 🛠️ 已完成

- ✅ `graphrag_agent/search/response_models.py` - 定义统一返回结构
- ✅ `graphrag_agent/search/tool/naive_search_tool.py` - 重构为使用统一返回结构
- ✅ 提供完整的使用示例和迁移指南

## 📚 相关文件

- `graphrag_agent/search/response_models.py` - 数据模型定义
- `graphrag_agent/search/tool/naive_search_tool.py` - 参考实现
- `graphrag_agent/search/tool/base.py` - BaseSearchTool
