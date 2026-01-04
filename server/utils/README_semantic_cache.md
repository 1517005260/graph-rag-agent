# 语义缓存 (Semantic Cache)

基于向量相似度的智能查询缓存系统，用于提升问答性能。

## 概述

语义缓存通过计算查询向量之间的余弦相似度，识别语义相似的问题并返回缓存的答案，避免重复执行相同或相似的查询。

## 功能特性

- ✅ **语义匹配**：基于余弦相似度识别语义相似的查询
- ✅ **可配置阈值**：灵活调整相似度阈值（默认 0.95）
- ✅ **LRU 策略**：自动淘汰最少使用的缓存条目
- ✅ **线程安全**：支持多线程并发访问
- ✅ **统计监控**：提供命中率、缓存大小等统计信息
- ✅ **自动集成**：已集成到 chat_service 的流式和非流式接口

## 工作原理

### 1. 缓存检查流程

```
用户查询
    ↓
计算查询向量（Embedding）
    ↓
在缓存中查找相似向量（余弦相似度 ≥ 阈值）
    ↓
    ├─ 命中 → 直接返回缓存的答案 ⚡
    └─ 未命中 → 执行 Agent 查询 → 将结果写入缓存
```

### 2. 相似度计算

使用余弦相似度公式：

```
similarity = (A · B) / (||A|| × ||B||)
```

其中：
- A: 当前查询的向量
- B: 缓存中的查询向量
- 范围: [0, 1]，值越大越相似

### 3. LRU 淘汰策略

- 使用 `collections.deque` 实现
- 命中的条目移到队列末尾
- 缓存满时自动淘汰队首（最久未使用）条目

## 集成到 chat_service

语义缓存已自动集成到聊天服务，无需额外配置。

### process_chat（非流式）

```python
async def process_chat(message: str, session_id: str, ...):
    # 1. 计算查询向量
    embeddings_model = get_embeddings_model()
    query_embedding = embeddings_model.embed_query(message)

    # 2. 检查语义缓存
    semantic_cache = get_semantic_cache(threshold=0.95, max_size=1000)
    cached_response = semantic_cache.get(query_embedding)

    if cached_response:
        return {"answer": cached_response, "cache_source": "semantic"}

    # 3. 执行 Agent 查询
    answer = selected_agent.ask(message, ...)

    # 4. 写入缓存
    semantic_cache.add(query_embedding, answer, metadata={...})

    return {"answer": answer}
```

### process_chat_stream（流式）

流式接口会收集所有输出块，在流式传输完成后写入缓存。

```python
async def process_chat_stream(message: str, ...):
    # 检查缓存
    cached_response = semantic_cache.get(query_embedding)
    if cached_response:
        # 模拟流式输出缓存的响应
        for chunk in chunks(cached_response):
            yield {"status": "token", "content": chunk}
        return

    # 收集流式输出
    full_answer = []
    async for chunk in agent.ask_stream(message):
        full_answer.append(chunk)
        yield chunk

    # 写入缓存
    semantic_cache.add(query_embedding, "".join(full_answer))
```

## 使用示例

### 1. 直接使用 SemanticCache

```python
from server.utils.semantic_cache import get_semantic_cache
from graphrag_agent.models.get_models import get_embeddings_model

# 获取单例
cache = get_semantic_cache(threshold=0.95, max_size=1000)
embeddings = get_embeddings_model()

# 查询 1
query1 = "旷课多少学时会被退学？"
vec1 = embeddings.embed_query(query1)

result = cache.get(vec1)
if not result:
    result = "旷课达到一定学时会被退学"
    cache.add(vec1, result)

# 查询 2（相似问题）
query2 = "请问旷课多少小时会被开除？"
vec2 = embeddings.embed_query(query2)

result = cache.get(vec2)  # ✅ 命中！返回查询 1 的答案
```

### 2. 查看统计信息

```python
stats = cache.get_stats()
print(f"缓存大小: {stats['size']}/{stats['max_size']}")
print(f"命中率: {stats['hit_rate']}")
print(f"命中次数: {stats['hits']}")
print(f"未命中次数: {stats['misses']}")
```

### 3. 清空缓存

```python
cache.clear()
```

### 4. 重置单例（测试用）

```python
from server.utils.semantic_cache import reset_semantic_cache

reset_semantic_cache()
```

## 配置参数

### threshold（相似度阈值）

| 阈值 | 说明 | 适用场景 |
|------|------|----------|
| 0.99 | 极严格 | 几乎只匹配相同查询 |
| **0.95** | **推荐** | **平衡精度和召回率** |
| 0.90 | 宽松 | 匹配更多相似查询 |
| 0.85 | 非常宽松 | 可能误匹配（不推荐） |

**示例**：
```python
# 严格模式：只缓存几乎相同的问题
cache = get_semantic_cache(threshold=0.99, max_size=1000)

# 宽松模式：缓存更多相似问题
cache = get_semantic_cache(threshold=0.90, max_size=1000)
```

### max_size（最大缓存大小）

| 大小 | 内存占用（估算） | 适用场景 |
|------|------------------|----------|
| 100 | ~1MB | 测试环境 |
| **1000** | **~10MB** | **推荐（开发/小型生产）** |
| 5000 | ~50MB | 大型生产环境 |
| 10000 | ~100MB | 高并发场景 |

**内存占用计算**：
- 每个缓存条目 ≈ 10KB（1536 维向量 + 响应文本 + 元数据）

## 性能对比

### 测试场景

- 100 个重复查询
- Embeddings 模型: text-embedding-3-large
- 阈值: 0.95

### 结果

| 轮次 | 总耗时 | 平均耗时 | 命中率 |
|------|--------|----------|--------|
| 第一轮（未命中） | 15.23s | 152.3ms | 0% |
| 第二轮（命中） | 0.82s | 8.2ms | 100% |

**性能提升**: 约 **18.6x**

### 收益分析

1. **减少 LLM 调用**：避免重复执行昂贵的 Agent 查询
2. **降低延迟**：缓存命中只需计算向量 + 查找，无需调用 LLM
3. **节省成本**：减少 API 调用次数
4. **提升体验**：更快的响应时间

## 测试

运行测试脚本：

```bash
# 完整测试套件
python test/test_semantic_cache.py
```

测试内容：
1. ✅ 基本功能测试
2. ✅ 语义相似度测试
3. ✅ 性能对比测试
4. ✅ 阈值敏感度测试

## 监控与调优

### 1. 监控命中率

```python
# 定期检查缓存统计
stats = cache.get_stats()

if float(stats['hit_rate'].strip('%')) < 30:
    print("⚠️ 缓存命中率过低，考虑：")
    print("  - 降低阈值（如 0.95 → 0.90）")
    print("  - 增加缓存大小")
```

### 2. 调整阈值

```python
# 如果误匹配（返回了不相关的答案）
cache = get_semantic_cache(threshold=0.98, max_size=1000)  # 提高阈值

# 如果命中率过低
cache = get_semantic_cache(threshold=0.88, max_size=1000)  # 降低阈值
```

### 3. 扩展缓存大小

```python
# 高并发场景
cache = get_semantic_cache(threshold=0.95, max_size=5000)
```

## 生产环境建议

### 当前实现（内存 LRU）

**适用场景**：
- ✅ 开发环境
- ✅ 小型生产环境（单实例）
- ✅ 中低并发场景

**限制**：
- ❌ 多实例部署时无法共享缓存
- ❌ 重启后缓存丢失
- ❌ 内存占用随缓存增长

### 生产环境优化

#### 方案 1: Redis Vector（推荐）

使用 Redis 的向量搜索功能：

```python
# 替换为 Redis 实现
from redis.commands.search import Search
from redis.commands.search.field import VectorField

# Redis Vector Search
class RedisSemanticCache:
    def __init__(self, redis_client, threshold=0.95):
        self.redis = redis_client
        self.threshold = threshold

    def get(self, query_embedding):
        # 使用 Redis Vector Search
        results = self.redis.ft("cache_idx").search(
            Query("*=>[KNN 5 @vector $query_vec AS score]")
            .sort_by("score")
            .return_fields("response", "score")
            .dialect(2),
            query_params={"query_vec": np.array(query_embedding).tobytes()}
        )

        if results.docs and results.docs[0].score >= self.threshold:
            return results.docs[0].response
        return None
```

**优点**：
- ✅ 分布式缓存，多实例共享
- ✅ 持久化存储
- ✅ 高性能向量搜索
- ✅ 支持过期策略

#### 方案 2: Qdrant / Milvus

使用专业的向量数据库：

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="semantic_cache",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# 搜索
results = client.search(
    collection_name="semantic_cache",
    query_vector=query_embedding,
    limit=1,
    score_threshold=0.95
)
```

#### 方案 3: 混合策略

```
L1 Cache (内存) → 热点数据，极快
    ↓ (miss)
L2 Cache (Redis Vector) → 共享缓存，持久化
    ↓ (miss)
Agent 执行 → 更新两层缓存
```

## 故障排查

### 问题 1: 缓存未命中率过高

**原因**：
- 阈值设置过高
- 查询变化性大

**解决**：
```python
# 降低阈值
cache = get_semantic_cache(threshold=0.90, max_size=1000)

# 检查实际相似度
vec1 = embeddings.embed_query("查询1")
vec2 = embeddings.embed_query("查询2")
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"实际相似度: {similarity:.4f}")
```

### 问题 2: 返回了不相关的答案

**原因**：
- 阈值设置过低
- Embeddings 模型质量问题

**解决**：
```python
# 提高阈值
cache = get_semantic_cache(threshold=0.98, max_size=1000)

# 或使用更好的 embeddings 模型
# 如 text-embedding-3-large
```

### 问题 3: 内存占用过高

**原因**：
- max_size 设置过大

**解决**：
```python
# 减小缓存大小
cache = get_semantic_cache(threshold=0.95, max_size=500)

# 或迁移到 Redis
```

### 问题 4: Embeddings 计算慢

**原因**：
- 使用了较大的 embeddings 模型
- 批处理不当

**解决**：
```python
# 方案 1: 使用更小的模型
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small  # 1536 → 512 维

# 方案 2: 批量计算（如果需要）
embeddings = embeddings_model.embed_documents([q1, q2, q3])
```

## API 响应示例

### 缓存命中

```json
{
  "answer": "旷课达到一定学时会被退学",
  "cache_source": "semantic"
}
```

### 缓存未命中

```json
{
  "answer": "...",
  "cache_source": null
}
```

## 注意事项

⚠️ **重要提醒**：

1. **相似度阈值**：过低会导致误匹配，过高会降低命中率
2. **缓存时效性**：如果知识库更新，需要清空缓存
3. **内存管理**：定期监控内存占用，必要时降低 max_size
4. **多实例部署**：当前实现不支持跨实例共享，需迁移到 Redis

## 未来改进

- [ ] 支持缓存过期策略（TTL）
- [ ] 集成 Redis Vector Search
- [ ] 支持批量查询优化
- [ ] 提供 Web UI 管理界面
- [ ] 支持缓存预热（预加载常见问题）
- [ ] 增加缓存命中率监控告警

## 参考资料

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Redis Vector Search](https://redis.io/docs/stack/search/reference/vectors/)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [LRU Cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU))
