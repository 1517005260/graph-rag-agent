# 缓存管理系统

支持多种存储后端、智能缓存策略和向量相似性匹配。

## 项目结构

```
CacheManage/
├── __init__.py                     # 模块入口，导出主要类和接口
├── manager.py                      # 统一缓存管理器，核心接口
├── backends/                       # 存储后端实现
│   ├── __init__.py                
│   ├── base.py                     # 存储后端抽象基类
│   ├── memory.py                   # 内存缓存后端 (LRU策略)
│   ├── disk.py                     # 磁盘缓存后端 (持久化存储)
│   ├── hybrid.py                   # 混合缓存后端 (内存+磁盘)
│   └── thread_safe.py              # 线程安全装饰器
├── models/                         # 数据模型
│   ├── __init__.py                
│   └── cache_item.py               # 缓存项模型，包含元数据管理
├── strategies/                     # 缓存键生成策略
│   ├── __init__.py                
│   ├── base.py                     # 策略抽象基类
│   ├── simple.py                   # 简单MD5哈希策略
│   ├── context_aware.py            # 上下文感知策略
│   └── global_strategy.py          # 全局缓存策略
└── vector_similarity/              # 向量相似性匹配
    ├── __init__.py                
    ├── embeddings.py               # 文本嵌入向量提供者
    └── matcher.py                  # FAISS向量相似性匹配器
```

### 基本使用

```python
from CacheManage import CacheManager

# 创建缓存管理器 (使用默认配置)
cache = CacheManager()

# 存储缓存
cache.set("什么是Python?", "Python是一种高级编程语言...")

# 获取缓存
result = cache.get("什么是Python?")
print(result)  # 输出: Python是一种高级编程语言...

# 标记缓存质量
cache.mark_quality("什么是Python?", is_positive=True)
```

## 核心接口

### CacheManager 主要方法

#### 基本缓存操作

```python
# 获取缓存 (支持向量相似性匹配)
result = cache.get(query: str, skip_validation: bool = False, **kwargs) -> Any

# 快速获取高质量缓存
result = cache.get_fast(query: str, **kwargs) -> Any

# 设置缓存
cache.set(query: str, result: Any, **kwargs) -> None

# 删除缓存
success = cache.delete(query: str, **kwargs) -> bool

# 清空所有缓存
cache.clear() -> None
```

#### 质量管理

```python
# 标记缓存质量 (正面/负面反馈)
success = cache.mark_quality(query: str, is_positive: bool, **kwargs) -> bool

# 验证答案质量
is_valid = cache.validate_answer(query: str, answer: str, validator: Callable = None, **kwargs) -> bool
```

#### 性能监控

```python
# 获取性能指标
metrics = cache.get_metrics() -> Dict[str, Any]
# 返回: {'exact_hits': 10, 'vector_hits': 5, 'misses': 3, 'total_queries': 18, ...}

# 强制刷新到磁盘
cache.flush() -> None
```

## 配置选项

### 创建缓存管理器

```python
from CacheManage import (
    CacheManager, 
    ContextAwareCacheKeyStrategy,
    MemoryCacheBackend,
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
    similarity_threshold=0.8,         # 相似度阈值
    max_vectors=10000                 # 最大向量数量
)
```

## 缓存策略

### 1. 简单策略 (SimpleCacheKeyStrategy)

适用于无上下文关联的查询缓存。

```python
from CacheManage import CacheManager, SimpleCacheKeyStrategy

cache = CacheManager(key_strategy=SimpleCacheKeyStrategy())
```

### 2. 上下文感知策略 (ContextAwareCacheKeyStrategy)

考虑会话历史，适用于对话系统。

```python
from CacheManage import CacheManager, ContextAwareCacheKeyStrategy

cache = CacheManager(
    key_strategy=ContextAwareCacheKeyStrategy(context_window=3)
)

# 使用时需要提供 thread_id
cache.set("继续", "继续前面的讨论...", thread_id="user_123")
result = cache.get("继续", thread_id="user_123")
```

### 3. 上下文与关键词感知策略 (ContextAndKeywordAwareCacheKeyStrategy)

同时考虑上下文和关键词，提供更精确的缓存匹配。

```python
from CacheManage import CacheManager, ContextAndKeywordAwareCacheKeyStrategy

cache = CacheManager(
    key_strategy=ContextAndKeywordAwareCacheKeyStrategy()
)

# 使用关键词增强缓存键
cache.set(
    "分析数据", 
    "数据分析结果...", 
    thread_id="user_123",
    low_level_keywords=["pandas", "numpy"],
    high_level_keywords=["数据科学", "机器学习"]
)
```

### 4. 全局策略 (GlobalCacheKeyStrategy)

忽略上下文，全局共享缓存。

```python
from CacheManage import CacheManager, GlobalCacheKeyStrategy

cache = CacheManager(key_strategy=GlobalCacheKeyStrategy())
```

## 存储后端

### 1. 内存缓存 (MemoryCacheBackend)

```python
cache = CacheManager(memory_only=True, max_memory_size=1000)
```

### 2. 混合缓存 (HybridCacheBackend) - 推荐

```python
cache = CacheManager(
    memory_only=False,
    max_memory_size=200,    # 内存中保留200个高质量缓存
    max_disk_size=5000,     # 磁盘最多存储5000个缓存
    cache_dir="./cache"
)
```

### 3. 自定义后端

```python
from CacheManage import CacheManager, DiskCacheBackend

# 使用纯磁盘缓存
disk_backend = DiskCacheBackend(
    cache_dir="./large_cache",
    max_size=50000,
    batch_size=20,
    flush_interval=60.0
)

cache = CacheManager(storage_backend=disk_backend)
```

## 向量相似性匹配

启用向量相似性后，系统可以找到语义相似的缓存项：

```python
# 存储缓存
cache.set("Python是什么?", "Python是一种编程语言...")

# 语义相似的查询也能命中缓存
result = cache.get("什么是Python编程语言?")  # 可能返回上面的缓存
```

### 自定义相似度阈值

```python
cache = CacheManager(
    enable_vector_similarity=True,
    similarity_threshold=0.85,  # 提高阈值，要求更高相似度
    max_vectors=20000
)
```

## 高级用法

### 1. 批量操作和性能优化

```python
# 批量设置缓存
queries_and_results = [
    ("查询1", "结果1"),
    ("查询2", "结果2"),
    ("查询3", "结果3")
]

for query, result in queries_and_results:
    cache.set(query, result)

# 强制刷新到磁盘
cache.flush()
```

### 2. 自定义验证器

```python
def custom_validator(query: str, answer: str) -> bool:
    """自定义答案验证逻辑"""
    if len(answer) < 20:
        return False
    if "错误" in answer:
        return False
    return True

# 使用自定义验证器
is_valid = cache.validate_answer("测试查询", "测试答案", custom_validator)
```

### 3. 监控缓存性能

```python
# 执行一些缓存操作...
cache.set("query1", "result1")
cache.get("query1")
cache.get("nonexistent_query")

# 查看性能统计
metrics = cache.get_metrics()
print(f"命中率: {metrics.get('total_hit_rate', 0):.2%}")
print(f"精确命中: {metrics['exact_hits']}")
print(f"向量命中: {metrics['vector_hits']}")
print(f"缓存未命中: {metrics['misses']}")
```

### 4. 多线程环境

```python
import threading
from CacheManage import CacheManager

# 创建线程安全的缓存管理器
cache = CacheManager(thread_safe=True)

def worker_function(thread_id):
    # 每个线程使用自己的 thread_id
    cache.set(f"query_{thread_id}", f"result_{thread_id}", thread_id=thread_id)
    result = cache.get(f"query_{thread_id}", thread_id=thread_id)

# 创建多个线程
threads = []
for i in range(10):
    t = threading.Thread(target=worker_function, args=(f"thread_{i}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## 配置文件

系统依赖 `config/settings.py` 中的配置：

```python
# config/settings.py
similarity_threshold = 0.8  # 默认相似度阈值
```

## 最佳实践

1. **选择合适的策略**: 
   - 对话系统使用 `ContextAwareCacheKeyStrategy`
   - 无状态API使用 `SimpleCacheKeyStrategy` 或 `GlobalCacheKeyStrategy`

2. **合理配置缓存大小**:
   - 内存缓存保持在100-500个项目
   - 磁盘缓存可以设置更大容量

3. **启用向量相似性**:
   - 对于自然语言查询，强烈推荐启用
   - 调整相似度阈值以平衡准确性和召回率

4. **定期维护**:
   - 使用 `flush()` 确保数据持久化
   - 监控 `get_metrics()` 了解缓存效果

5. **质量管理**:
   - 积极使用 `mark_quality()` 标记高质量缓存
   - 利用 `get_fast()` 优先获取高质量缓存

## 故障排除

### 常见问题

1. **向量相似性不工作**
   ```python
   # 确保安装了必要依赖
   pip install sentence-transformers faiss-cpu
   
   # 检查配置
   cache = CacheManager(enable_vector_similarity=True)
   ```

2. **磁盘缓存不持久化**
   ```python
   # 确保调用 flush()
   cache.flush()
   
   # 或者在程序结束时清理
   import atexit
   atexit.register(cache.flush)
   ```

3. **线程安全问题**
   ```python
   # 确保启用线程安全
   cache = CacheManager(thread_safe=True)
   ```

## 性能建议

- 内存缓存: 适合频繁访问的小数据集
- 混合缓存: 推荐用于生产环境，平衡性能和容量
- 磁盘缓存: 适合大容量、持久化需求
- 向量相似性: 会增加一定开销，但显著提高命中率