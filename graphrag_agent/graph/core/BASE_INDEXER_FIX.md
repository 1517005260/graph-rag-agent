# BaseIndexer 并行处理顺序错乱修复

## 问题分析

### 原始代码存在的问题

在 `graphrag_agent/graph/core/base_indexer.py` 的 `process_in_parallel` 方法中，存在两个严重问题：

```python
# ❌ 原始存在问题的逻辑
def process_in_parallel(self, items: List[Any], process_func) -> List[Any]:
    max_workers = min(self.max_workers, CONFIG_MAX_WORKERS)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_func, item): i
            for i, item in enumerate(items)
        }

        # 问题点1：as_completed 按完成顺序返回，而不是提交顺序
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)  # 直接追加导致顺序与 items 不一致
            except Exception as e:
                print(f"并行处理出错: {e}")
                # 问题点2：如果出错，列表长度变短，进一步破坏对齐

    return results
```

### 后果分析

#### 1. **顺序错乱 (Critical)**

**场景示例**：
```python
items = ["文本A", "文本B", "文本C"]
# 如果 "文本B" 处理最快，"文本A" 处理最慢
results = [向量B, 向量C, 向量A]  # ❌ 顺序完全错乱
```

**影响**：
- 如果 `items` 是 `[文本A, 文本B]`，而 `文本B` 处理得快，`results` 会变成 `[向量B, 向量A]`
- 如果后续代码假设 `results[0]` 对应 `文本A`，**整个图谱的数据关联就全错了**
- 实体和关系映射错位，导致知识图谱结构完全损坏

**严重性**：🔥🔥🔥
- 数据损坏：向量与实体对应关系错误
- 难以发现：不会抛出异常，但数据悄悄损坏
- 影响范围：所有使用并行处理的索引构建流程

#### 2. **长度不一致 (High)**

**场景示例**：
```python
items = ["文本A", "文本B", "文本C"]
# 假设处理 "文本B" 时失败
results = [向量A, 向量C]  # ❌ 只有2个结果，无法知道哪个失败了
```

**影响**：
- 当前的 `try...except` 块在发生异常时**没有向 results 追加任何内容**
- 如果 100 个任务中有 1 个失败，返回的列表只有 99 项
- **无法知道是哪一项失败了**，完全无法做后续的索引对齐
- 后续代码尝试访问 `results[99]` 会导致 `IndexError`

**严重性**：🔥🔥
- 数据丢失：无法追踪哪些项目处理失败
- 索引错位：后续代码无法正确对齐数据
- 调试困难：异常被吞没，无法定位失败的具体项目

## 解决方案

### 修复策略

采用 **"预分配列表 + 按索引填充"** 方案：

1. **预分配固定长度列表**：`results = [None] * len(items)`
2. **记录索引映射**：`future_to_index = {future: i for i, item in enumerate(items)}`
3. **按索引归位**：`results[index] = result`
4. **异常时保留占位符**：`results[index] = None`

### 修复后的代码

```python
def process_in_parallel(self, items: List[Any], process_func) -> List[Any]:
    """
    并行处理项目（修复顺序错乱问题）

    使用预分配列表 + 按索引填充的方式确保：
    1. 结果顺序与输入 items 严格一致
    2. 即使部分任务失败，列表长度也保持一致（失败位置为 None）

    Args:
        items: 待处理项目列表
        process_func: 处理单个项目的函数

    Returns:
        List[Any]: 处理结果列表，顺序与 items 一致。如果某项处理失败，该位置为 None。
    """
    if not items:
        return []

    max_workers = min(self.max_workers, CONFIG_MAX_WORKERS)
    # 1. 预分配固定长度的列表，确保索引对齐
    results = [None] * len(items)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 记录 Future -> Index 的映射
        future_to_index = {
            executor.submit(process_func, item): i
            for i, item in enumerate(items)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                # 2. 按原始索引归位
                results[index] = result
            except Exception as e:
                print(f"并行处理出错 (索引 {index}): {e}")
                # 出错时保留 None，确保列表长度不变
                results[index] = None

    return results
```

### 方案优势

| 特性 | 原始代码 | 修复后代码 |
|------|---------|-----------|
| **顺序保证** | ❌ 按完成顺序，乱序 | ✅ 严格按输入顺序 |
| **长度保证** | ❌ 异常时长度变短 | ✅ 始终等于 len(items) |
| **错误追踪** | ❌ 无法知道哪项失败 | ✅ 失败位置为 None，可精确定位 |
| **数据对齐** | ❌ results[i] 不对应 items[i] | ✅ results[i] 严格对应 items[i] |
| **性能影响** | - | ✅ 预分配列表，O(1) 赋值，无额外开销 |

### 与 executor.map 的对比

**为什么不用 `executor.map`？**

```python
# 方案B: 使用 executor.map
results = list(executor.map(process_func, items))
```

**问题**：
- `executor.map` 在遇到**第一个异常**时会直接抛出，导致后续任务被取消
- 无法灵活处理部分失败的情况（例如：跳过失败项继续处理）
- 不适合需要容错的场景

**当前方案优势**：
- 即使部分任务失败，其他任务仍继续执行
- 可以在结果中明确标记失败位置（None）
- 调用者可以选择如何处理失败项（跳过、重试、记录日志等）

## 影响范围

### 受影响的模块

虽然 `process_in_parallel` 方法在当前代码库中尚未被直接调用，但作为 `BaseIndexer` 的公共方法，潜在受影响的模块包括：

1. **`ChunkIndexer`** (`graphrag_agent/graph/indexing/chunk_indexer.py`)
   - 继承自 `BaseIndexer`
   - 未来如果使用 `process_in_parallel` 进行向量化，会受影响

2. **`EntityIndexer`** (`graphrag_agent/graph/indexing/entity_indexer.py`)
   - 继承自 `BaseIndexer`
   - 未来如果使用 `process_in_parallel` 进行实体处理，会受影响

3. **自定义索引器**
   - 任何继承 `BaseIndexer` 并使用 `process_in_parallel` 的自定义代码

### 向后兼容性

**调用者需要注意**：

修复后，`process_in_parallel` 的返回值中**可能包含 `None`**（当任务失败时）。调用者需要处理这种情况：

```python
# ✅ 推荐做法：过滤掉 None
results = indexer.process_in_parallel(items, process_func)
valid_results = [r for r in results if r is not None]

# ✅ 推荐做法：检查并记录失败项
for i, result in enumerate(results):
    if result is None:
        print(f"⚠️ 处理失败: {items[i]}")
    else:
        # 处理成功的结果
        process_result(result)

# ❌ 错误做法：假设所有结果都有效
for result in results:
    result.some_method()  # 如果 result 是 None，会抛出 AttributeError
```

## 测试建议

### 单元测试示例

```python
import unittest
from graphrag_agent.graph.core.base_indexer import BaseIndexer

class TestBaseIndexerParallel(unittest.TestCase):
    def test_order_preservation(self):
        """测试结果顺序与输入一致"""
        indexer = BaseIndexer(max_workers=4)

        def slow_process(x):
            import time
            # 模拟不同处理时间
            time.sleep(0.1 * (10 - x))  # x越小，处理越慢
            return x * 2

        items = list(range(10))
        results = indexer.process_in_parallel(items, slow_process)

        # 验证顺序
        expected = [x * 2 for x in items]
        self.assertEqual(results, expected)

    def test_exception_handling(self):
        """测试异常处理：失败位置为 None，长度保持一致"""
        indexer = BaseIndexer(max_workers=4)

        def failing_process(x):
            if x == 3 or x == 7:
                raise ValueError(f"Intentional error for {x}")
            return x * 2

        items = list(range(10))
        results = indexer.process_in_parallel(items, failing_process)

        # 验证长度
        self.assertEqual(len(results), len(items))

        # 验证失败位置
        self.assertIsNone(results[3])
        self.assertIsNone(results[7])

        # 验证成功位置
        self.assertEqual(results[0], 0)
        self.assertEqual(results[5], 10)

    def test_empty_input(self):
        """测试空输入"""
        indexer = BaseIndexer(max_workers=4)
        results = indexer.process_in_parallel([], lambda x: x)
        self.assertEqual(results, [])
```

### 集成测试场景

```python
# 测试向量化场景
def test_embedding_alignment():
    """测试向量与实体的对齐"""
    indexer = ChunkIndexer()

    chunks = ["文本A", "文本B", "文本C"]
    embeddings = indexer.process_in_parallel(chunks, embed_func)

    # 验证对齐
    for i, chunk in enumerate(chunks):
        if embeddings[i] is not None:
            assert embeddings[i].shape == (1536,)
            # 将向量与对应的文本块关联
            store_embedding(chunk, embeddings[i])
        else:
            print(f"⚠️ 向量化失败: {chunk}")
```

## 性能影响

### 时间复杂度

| 操作 | 原始代码 | 修复后代码 |
|------|---------|-----------|
| 列表初始化 | O(1) | O(n) - 预分配 |
| 结果收集 | O(1) - append | O(1) - 按索引赋值 |
| 总体 | O(n) | O(n) |

**结论**：预分配列表虽然增加了 O(n) 初始化开销，但整体仍为 O(n)，**无性能退化**。

### 空间复杂度

| 项目 | 原始代码 | 修复后代码 |
|------|---------|-----------|
| 结果列表 | O(n) | O(n) |
| 索引映射 | O(n) | O(n) |
| 总体 | O(n) | O(n) |

**结论**：空间复杂度保持不变。

### 实际性能测试

```python
import time
from graphrag_agent.graph.core.base_indexer import BaseIndexer

def benchmark():
    indexer = BaseIndexer(max_workers=8)
    items = list(range(1000))

    def process_func(x):
        import time
        time.sleep(0.001)  # 模拟 1ms 处理时间
        return x * 2

    start = time.time()
    results = indexer.process_in_parallel(items, process_func)
    end = time.time()

    print(f"处理 {len(items)} 项，耗时: {end - start:.2f}秒")
    print(f"成功: {sum(1 for r in results if r is not None)}/{len(results)}")

# 预期输出：
# 处理 1000 项，耗时: ~1.25秒 (1000ms / 8 workers)
# 成功: 1000/1000
```

## 最佳实践

### ✅ 推荐做法

1. **处理返回值时检查 None**：
   ```python
   results = indexer.process_in_parallel(items, process_func)
   for i, result in enumerate(results):
       if result is None:
           logger.warning(f"处理失败: {items[i]}")
       else:
           process_result(result)
   ```

2. **过滤有效结果**：
   ```python
   valid_results = [r for r in results if r is not None]
   ```

3. **记录失败统计**：
   ```python
   failed_count = sum(1 for r in results if r is None)
   success_rate = (len(results) - failed_count) / len(results) * 100
   print(f"成功率: {success_rate:.2f}%")
   ```

### ❌ 避免做法

1. **不检查 None 直接使用**：
   ```python
   # ❌ 错误：如果 result 是 None 会抛出 AttributeError
   for result in results:
       result.some_method()
   ```

2. **假设长度总是匹配**：
   ```python
   # ❌ 错误：虽然现在长度匹配了，但不检查 None 仍会出错
   for item, result in zip(items, results):
       store_mapping(item, result.embedding)  # result 可能是 None
   ```

3. **忽略失败项**：
   ```python
   # ❌ 不推荐：默默跳过失败项，没有记录日志
   results = [r for r in results if r is not None]
   # 应该记录失败项用于调试
   ```

## 相关文件

- **修复文件**：`graphrag_agent/graph/core/base_indexer.py`
- **子类文件**：
  - `graphrag_agent/graph/indexing/chunk_indexer.py`
  - `graphrag_agent/graph/indexing/entity_indexer.py`

## 参考资料

- [Python concurrent.futures 文档](https://docs.python.org/3/library/concurrent.futures.html)
- [ThreadPoolExecutor.as_completed 问题](https://stackoverflow.com/questions/32151776/how-to-get-futures-results-in-submitted-order)
- [Preserving Order in concurrent.futures](https://alexwlchan.net/2019/10/adventures-with-concurrent-futures/)
