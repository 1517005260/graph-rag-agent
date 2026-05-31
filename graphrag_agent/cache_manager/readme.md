# 缓存管理系统

GraphRAG Agent 的缓存管理系统，提供多层次缓存策略、向量语义匹配和智能质量管理。

## 系统特性

- 多种缓存键策略：简单哈希、上下文感知、关键词增强、全局共享
- 灵活的存储后端：内存、磁盘、混合模式
- 向量语义匹配：基于 FAISS 的语义相似性缓存检索
- 质量管理：自动质量评分、用户反馈标记、高质量缓存优先
- 线程安全：支持多线程并发访问
- 性能监控：命中率统计、延迟追踪

## 项目结构

```
cache_manager/
├── __init__.py                     # 模块入口，导出主要类和接口
├── manager.py                      # 统一缓存管理器，核心接口
├── model_cache.py                  # 模型缓存管理，预加载嵌入模型
├── backends/                       # 存储后端实现
│   ├── __init__.py
│   ├── base.py                     # 存储后端抽象基类
│   ├── memory.py                   # 内存缓存后端（LRU策略）
│   ├── disk.py                     # 磁盘缓存后端（持久化存储）
│   ├── hybrid.py                   # 混合缓存后端（内存+磁盘）
│   └── thread_safe.py              # 线程安全装饰器
├── models/                         # 数据模型
│   ├── __init__.py
│   └── cache_item.py               # 缓存项模型，包含元数据管理
├── strategies/                     # 缓存键生成策略
│   ├── __init__.py
│   ├── base.py                     # 策略抽象基类
│   ├── simple.py                   # 简单 MD5 哈希策略
│   ├── context_aware.py            # 上下文感知策略及关键词增强策略
│   └── global_strategy.py          # 全局缓存策略
└── vector_similarity/              # 向量相似性匹配
    ├── __init__.py
    ├── embeddings.py               # 文本嵌入向量提供者（OpenAI/SentenceTransformer）
    └── matcher.py                  # FAISS 向量相似性匹配器
```

## 快速开始

### 基本使用

```python
from graphrag_agent.cache_manager import CacheManager

# 创建缓存管理器（使用默认配置）
cache = CacheManager()

# 存储缓存
cache.set("什么是Python?", "Python是一种高级编程语言...")

# 获取缓存
result = cache.get("什么是Python?")
print(result)  # 输出: Python是一种高级编程语言...

# 语义相似查询也能命中缓存（需启用向量相似性）
result = cache.get("Python编程语言是什么?")

# 标记缓存质量
cache.mark_quality("什么是Python?", is_positive=True)

# 获取性能指标
metrics = cache.get_metrics()
print(f"缓存命中率: {metrics.get('total_hit_rate', 0):.2%}")
```

## 核心接口

### CacheManager 主要方法

#### 缓存读取操作

```python
# 1. 标准获取 - 支持精确匹配和向量相似性匹配
result = cache.get(
    query: str,                  # 查询文本
    skip_validation: bool = False,  # 是否跳过质量验证
    **kwargs                     # 上下文参数（thread_id, keywords等）
) -> Any

# 2. 快速获取 - 仅返回高质量缓存
result = cache.get_fast(
    query: str,
    **kwargs
) -> Any
```

#### 缓存写入操作

```python
# 设置缓存
cache.set(
    query: str,      # 查询文本
    result: Any,     # 缓存内容
    **kwargs         # 上下文参数
) -> None

# 删除缓存项
success = cache.delete(query: str, **kwargs) -> bool

# 清空所有缓存
cache.clear() -> None
```

#### 质量管理

```python
# 标记缓存质量（正面/负面反馈）
success = cache.mark_quality(
    query: str,
    is_positive: bool,  # True表示高质量，False表示低质量
    **kwargs
) -> bool

# 验证答案质量（支持自定义验证函数）
is_valid = cache.validate_answer(
    query: str,
    answer: str,
    validator: Callable[[str, str], bool] = None,
    **kwargs
) -> bool
```

#### 性能监控与维护

```python
# 获取性能指标
metrics = cache.get_metrics() -> Dict[str, Any]
# 返回示例: {
#   'exact_hits': 10,      # 精确匹配命中次数
#   'vector_hits': 5,      # 向量匹配命中次数
#   'misses': 3,           # 未命中次数
#   'total_queries': 18,   # 总查询次数
#   'exact_hit_rate': 0.56,
#   'vector_hit_rate': 0.28,
#   'total_hit_rate': 0.83,
#   'miss_rate': 0.17
# }

# 强制刷新到磁盘
cache.flush() -> None

# 保存向量索引
cache.save_vector_index() -> None
```

## 配置选项

### 环境变量配置

缓存系统通过环境变量（.env文件）进行配置：

```bash
# 缓存根目录
CACHE_ROOT=./cache

# 模型缓存目录
MODEL_CACHE_ROOT=./cache

# 缓存目录
CACHE_DIR=./cache

# 基本配置
CACHE_MEMORY_ONLY=false              # 是否仅使用内存缓存
CACHE_MAX_MEMORY_SIZE=100            # 内存缓存最大项目数
CACHE_MAX_DISK_SIZE=1000             # 磁盘缓存最大项目数
CACHE_THREAD_SAFE=true               # 是否启用线程安全

# 向量相似性配置
CACHE_ENABLE_VECTOR_SIMILARITY=true  # 是否启用向量相似性匹配
CACHE_SIMILARITY_THRESHOLD=0.8       # 相似度阈值（0-1）
CACHE_MAX_VECTORS=10000              # 最大向量数量

# 嵌入提供者配置
CACHE_EMBEDDING_PROVIDER=sentence_transformer  # openai 或 sentence_transformer
CACHE_SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2  # SentenceTransformer 模型名称
```

### 代码配置示例

```python
from graphrag_agent.cache_manager import (
    CacheManager,
    ContextAwareCacheKeyStrategy,
    GlobalCacheKeyStrategy,
    HybridCacheBackend
)

# 完整配置示例
cache = CacheManager(
    # 缓存键策略
    key_strategy=ContextAwareCacheKeyStrategy(context_window=3),

    # 存储配置
    cache_dir="./my_cache",           # 缓存目录
    memory_only=False,                # 是否仅使用内存
    max_memory_size=200,              # 内存缓存最大项目数
    max_disk_size=5000,               # 磁盘缓存最大项目数

    # 安全性
    thread_safe=True,                 # 是否线程安全

    # 向量相似性
    enable_vector_similarity=True,    # 启用向量匹配
    similarity_threshold=0.85,        # 相似度阈值
    max_vectors=20000                 # 最大向量数量
)
```

## 缓存策略

系统提供四种缓存键生成策略，适用于不同场景。

### 1. 简单策略（SimpleCacheKeyStrategy）

基于查询文本的 MD5 哈希，适用于无上下文关联的独立查询。

```python
from graphrag_agent.cache_manager import CacheManager, SimpleCacheKeyStrategy

cache = CacheManager(key_strategy=SimpleCacheKeyStrategy())

# 使用
cache.set("Python是什么?", "Python是一种编程语言")
result = cache.get("Python是什么?")  # 精确匹配才能命中
```

**适用场景**：
- 无状态API
- 独立查询系统
- 不需要考虑上下文的缓存

### 2. 上下文感知策略（ContextAwareCacheKeyStrategy）

考虑会话历史，适用于对话系统和多轮交互场景。

```python
from graphrag_agent.cache_manager import CacheManager, ContextAwareCacheKeyStrategy

cache = CacheManager(
    key_strategy=ContextAwareCacheKeyStrategy(
        context_window=3  # 考虑最近3轮对话
    )
)

# 使用时需要提供 thread_id 区分不同会话
cache.set("继续", "继续前面的讨论...", thread_id="user_123")
result = cache.get("继续", thread_id="user_123")
```

**工作原理**：
- 维护每个会话的历史记录（通过 thread_id 区分）
- 缓存键包含：thread_id + 最近N轮对话 + 当前查询
- 自动更新历史版本号

**适用场景**：
- 聊天机器人
- 多轮对话系统
- 需要上下文理解的问答系统

### 3. 上下文与关键词感知策略（ContextAndKeywordAwareCacheKeyStrategy）

同时考虑上下文和关键词，提供更精确的缓存匹配。

```python
from graphrag_agent.cache_manager import CacheManager, ContextAndKeywordAwareCacheKeyStrategy

cache = CacheManager(
    key_strategy=ContextAndKeywordAwareCacheKeyStrategy(
        context_window=3
    )
)

# 使用关键词增强缓存键
cache.set(
    "分析数据",
    "数据分析结果...",
    thread_id="user_123",
    low_level_keywords=["pandas", "numpy"],      # 低级关键词（具体技术）
    high_level_keywords=["数据科学", "机器学习"]  # 高级关键词（领域概念）
)

# 查询时提供相同的关键词才能精确命中
result = cache.get(
    "分析数据",
    thread_id="user_123",
    low_level_keywords=["pandas", "numpy"],
    high_level_keywords=["数据科学", "机器学习"]
)
```

**工作原理**：
- 缓存键包含：thread_id + 上下文哈希 + 版本号 + 低级关键词 + 高级关键词
- 关键词会被排序以保证一致性
- 适合需要细粒度控制的场景

**适用场景**：
- 技术文档问答
- 领域特定的智能助手
- 需要精确主题匹配的系统

### 4. 全局策略（GlobalCacheKeyStrategy）

忽略上下文，全局共享缓存，适用于通用知识查询。

```python
from graphrag_agent.cache_manager import CacheManager, GlobalCacheKeyStrategy

cache = CacheManager(key_strategy=GlobalCacheKeyStrategy())

# 所有用户共享缓存，不区分会话
cache.set("地球的半径是多少?", "约6371公里")
result = cache.get("地球的半径是多少?")  # 任何用户都能获取
```

**适用场景**：
- 百科知识问答
- 静态内容缓存
- 跨用户共享的通用查询

## 存储后端

### 1. 内存缓存（MemoryCacheBackend）

基于 LRU（最近最少使用）策略的纯内存缓存。

```python
cache = CacheManager(
    memory_only=True,
    max_memory_size=1000
)
```

**特点**：
- 读写速度最快
- 进程重启后数据丢失
- 适合临时缓存和测试环境

### 2. 磁盘缓存（DiskCacheBackend）

持久化到磁盘的缓存后端。

```python
from graphrag_agent.cache_manager import DiskCacheBackend, CacheManager

disk_backend = DiskCacheBackend(
    cache_dir="./large_cache",
    max_size=50000,
    batch_size=20,           # 批量写入大小
    flush_interval=60.0      # 自动刷新间隔（秒）
)

cache = CacheManager(storage_backend=disk_backend)
```

**特点**：
- 数据持久化
- 支持大容量存储
- 批量写入提升性能

### 3. 混合缓存（HybridCacheBackend）- 推荐

结合内存和磁盘的两层缓存架构。

```python
cache = CacheManager(
    memory_only=False,
    max_memory_size=200,    # 内存中保留200个高质量缓存
    max_disk_size=5000,     # 磁盘最多存储5000个缓存
    cache_dir="./cache"
)
```

**工作原理**：
- 内存层：存储高质量、高访问频率的缓存（基于质量分数）
- 磁盘层：存储所有缓存，提供持久化
- 自动同步：内存和磁盘之间自动数据同步

**特点**：
- 兼顾速度和容量
- 智能缓存提升
- 生产环境推荐

## 向量相似性匹配

基于 FAISS 的语义相似性搜索，允许语义相近的查询命中缓存。

### 基本使用

```python
cache = CacheManager(
    enable_vector_similarity=True,
    similarity_threshold=0.8  # 相似度阈值（0-1）
)

# 存储缓存
cache.set("Python是什么?", "Python是一种高级编程语言...")

# 语义相似的查询也能命中缓存
result = cache.get("什么是Python编程语言?")  # 可能返回上面的缓存
result = cache.get("请介绍Python")           # 可能返回上面的缓存
```

### 嵌入提供者

系统支持两种嵌入提供者：

#### 1. SentenceTransformer（默认）

```bash
# 环境变量配置
CACHE_EMBEDDING_PROVIDER=sentence_transformer
CACHE_SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
```

**特点**：
- 本地运行，无需API调用
- 速度快，成本低
- 支持多种预训练模型

**常用模型**：
- `all-MiniLM-L6-v2`：小型、快速（默认）
- `paraphrase-multilingual-MiniLM-L12-v2`：多语言支持
- `all-mpnet-base-v2`：高质量，速度较慢

#### 2. OpenAI Embeddings

```bash
# 环境变量配置
CACHE_EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
```

**特点**：
- 高质量嵌入
- 需要API调用，有成本
- 适合对准确性要求高的场景

### 相似度阈值调优

```python
# 高阈值（0.9）- 严格匹配
cache = CacheManager(
    enable_vector_similarity=True,
    similarity_threshold=0.9  # 只有非常相似的查询才命中
)

# 中等阈值（0.8）- 平衡模式（推荐）
cache = CacheManager(
    enable_vector_similarity=True,
    similarity_threshold=0.8  # 平衡准确率和召回率
)

# 低阈值（0.7）- 宽松匹配
cache = CacheManager(
    enable_vector_similarity=True,
    similarity_threshold=0.7  # 更多语义相关的查询都能命中
)
```

### 向量索引管理

```python
# 手动保存向量索引
cache.save_vector_index()

# 向量索引会自动持久化到：{cache_dir}/vector_index.faiss
```

## 高级用法

### 1. 批量操作与性能优化

```python
# 批量设置缓存
queries_and_results = [
    ("查询1", "结果1"),
    ("查询2", "结果2"),
    ("查询3", "结果3")
]

for query, result in queries_and_results:
    cache.set(query, result)

# 强制刷新到磁盘（批量操作后建议调用）
cache.flush()

# 定期保存向量索引
cache.save_vector_index()
```

### 2. 自定义验证器

```python
def custom_validator(query: str, answer: str) -> bool:
    """自定义答案验证逻辑"""
    # 长度检查
    if len(answer) < 20:
        return False

    # 内容检查
    if "错误" in answer or "抱歉" in answer:
        return False

    # 相关性检查
    query_keywords = set(query.lower().split())
    answer_keywords = set(answer.lower().split())
    if len(query_keywords & answer_keywords) == 0:
        return False

    return True

# 使用自定义验证器
is_valid = cache.validate_answer("测试查询", "测试答案", custom_validator)
```

### 3. 监控缓存性能

```python
import time

# 执行一些缓存操作
start = time.time()
for i in range(100):
    cache.set(f"query_{i}", f"result_{i}")
    cache.get(f"query_{i}")

# 查看性能统计
metrics = cache.get_metrics()

print(f"总查询次数: {metrics['total_queries']}")
print(f"精确命中次数: {metrics['exact_hits']}")
print(f"向量命中次数: {metrics['vector_hits']}")
print(f"未命中次数: {metrics['misses']}")
print(f"总命中率: {metrics.get('total_hit_rate', 0):.2%}")
print(f"精确命中率: {metrics.get('exact_hit_rate', 0):.2%}")
print(f"向量命中率: {metrics.get('vector_hit_rate', 0):.2%}")
print(f"未命中率: {metrics.get('miss_rate', 0):.2%}")

# 平均响应时间（如果有）
if 'get_time' in metrics:
    print(f"最后查询耗时: {metrics['get_time']:.4f}秒")
```

### 4. 多线程环境

```python
import threading
from graphrag_agent.cache_manager import CacheManager

# 创建线程安全的缓存管理器
cache = CacheManager(thread_safe=True)

def worker_function(worker_id):
    """工作线程函数"""
    thread_id = f"thread_{worker_id}"

    # 每个线程使用自己的 thread_id
    for i in range(10):
        query = f"worker_{worker_id}_query_{i}"
        result = f"worker_{worker_id}_result_{i}"

        cache.set(query, result, thread_id=thread_id)
        retrieved = cache.get(query, thread_id=thread_id)

        assert retrieved == result

# 创建多个工作线程
threads = []
for i in range(10):
    t = threading.Thread(target=worker_function, args=(i,))
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()

print("多线程测试完成")
print(f"性能指标: {cache.get_metrics()}")
```

### 5. 质量反馈循环

```python
def ask_with_feedback(query: str, generate_answer_fn):
    """带反馈的查询函数"""
    # 先尝试从缓存获取
    cached = cache.get(query)

    if cached is not None:
        # 询问用户反馈
        user_satisfied = input(f"缓存答案: {cached}\n满意吗？(y/n): ").lower() == 'y'

        # 标记质量
        cache.mark_quality(query, is_positive=user_satisfied)

        if user_satisfied:
            return cached

    # 生成新答案
    answer = generate_answer_fn(query)

    # 存入缓存
    cache.set(query, answer)

    # 初始标记为正面（可选）
    cache.mark_quality(query, is_positive=True)

    return answer
```

### 6. 模型缓存预加载

```python
from graphrag_agent.cache_manager import initialize_model_cache, ensure_model_cache_dir

# 预加载嵌入模型（首次运行或部署时）
initialize_model_cache()

# 确保模型缓存目录存在
cache_dir = ensure_model_cache_dir()
print(f"模型缓存目录: {cache_dir}")
```

## 系统架构

### 配置来源

缓存系统通过多层配置系统管理：

```python
# graphrag_agent/config/settings.py
CACHE_SETTINGS = {
    "dir": CACHE_DIR,                              # 从环境变量 CACHE_DIR 读取
    "memory_only": False,                          # 从 CACHE_MEMORY_ONLY 读取
    "max_memory_size": 100,                        # 从 CACHE_MAX_MEMORY_SIZE 读取
    "max_disk_size": 1000,                         # 从 CACHE_MAX_DISK_SIZE 读取
    "thread_safe": True,                           # 从 CACHE_THREAD_SAFE 读取
    "enable_vector_similarity": True,              # 从 CACHE_ENABLE_VECTOR_SIMILARITY 读取
    "similarity_threshold": similarity_threshold,  # 继承自知识图谱配置
    "max_vectors": 10000,                          # 从 CACHE_MAX_VECTORS 读取
}
```

### 缓存项数据模型

```python
# CacheItem 结构
{
    "content": Any,           # 实际缓存内容
    "metadata": {
        "created_at": float,  # 创建时间戳
        "last_accessed": float,  # 最后访问时间
        "access_count": int,  # 访问次数
        "quality_score": int,  # 质量分数（正反馈+1，负反馈-1）
        "user_verified": bool,  # 用户是否验证
        "fast_path_eligible": bool,  # 是否适合快速路径
        "similarity_score": float,  # 向量匹配时的相似度（仅向量匹配时有）
        "matched_via_vector": bool,  # 是否通过向量匹配（仅向量匹配时有）
        "original_query": str  # 原始查询（仅向量匹配时有）
    }
}
```