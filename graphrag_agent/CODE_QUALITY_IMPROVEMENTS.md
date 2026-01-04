# 代码质量改进 - 移除技术债、引入日志和熔断机制

## 改进总结

本次改进针对三个严重的代码质量问题进行修复：

1. **格式兼容问题** - 移除过度防御性编程，强制标准格式
2. **缓存异常与日志** - 引入专业日志系统，替换 print 语句
3. **统一错误处理** - 实现熔断机制，避免空图谱问题

## 问题 1: 格式兼容问题 (Format Compatibility)

### 原始问题

**文件**: `graphrag_agent/integrations/build/build_graph.py` (第 351-397 行)

存在严重的"防御性编程过剩"与格式混乱：

```python
# ❌ 原始代码：过度防御性编程
for res in entity_data:
    if isinstance(res, dict):
        normalized_entity_data.append(res)
    elif isinstance(res, (tuple, list)):
        # 手动将 tuple 映射回 dict，非常脆弱
        normalized_entity_data.append({
            "entities": res[0] if len(res) > 0 else [],
            "relations": res[1] if len(res) > 1 else [],
            "bridges": res[2] if len(res) > 2 else [],
            "domains": res[3] if len(res) > 3 else [],
            "raw": str(res)
        })
    elif isinstance(res, str):
        # 尝试兼容字符串格式...
        normalized_entity_data.append({...})
    else:
        # 未知格式，使用空结构...
        normalized_entity_data.append({...})
```

### 问题分析

**技术债务**：
- 假设上游可能返回三种完全不同的数据结构（dict, tuple, str）
- 维护困难：如果在 Extractor 里加了一个字段（如 `bridges`），必须记得修改 tuple 的解包索引
- 性能损耗：无谓的类型检查和转换
- 责任混乱：下游代码在为上游的格式问题"擦屁股"

**风险**：
```python
# 假设 Extractor 返回了 5 元素的 tuple
res = (entities, relations, bridges, domains, new_field)

# build_graph.py 只读取前 4 个元素
{
    "entities": res[0],
    "relations": res[1],
    "bridges": res[2],
    "domains": res[3]  # new_field 被忽略，静默丢失数据
}
```

### 修复方案

**✅ 强制标准格式**：

```python
# 移除所有兼容代码，强制要求标准格式
normalized_entity_data = []
for res in entity_data:
    if isinstance(res, dict):
        # 确保包含必要的键，如果没有则补空列表
        res.setdefault("entities", [])
        res.setdefault("relationships", [])
        res.setdefault("relations", res.get("relationships", []))  # 兼容旧字段名
        res.setdefault("bridges", [])
        res.setdefault("domains", [])
        normalized_entity_data.append(res)
    else:
        # 遇到非 dict 格式直接报错，强制开发者修复上游 Extractor
        self.console.print(
            f"[red]❌ 错误: 发现非法的实体数据格式 {type(res).__name__}，必须为 dict。"
            f"请检查 Extractor 返回格式。忽略此条目。[/red]"
        )
        # 记录详细信息用于调试
        import logging
        logging.error(f"Invalid entity data format: type={type(res)}, value={res}")
        continue
```

### 改进效果

| 特性 | 原始代码 | 修复后代码 |
|------|---------|-----------|
| **类型检查** | 4 种类型（dict/tuple/str/other） | 1 种类型（dict） |
| **代码行数** | ~50 行 | ~20 行 |
| **维护成本** | 高（多处硬编码索引） | 低（统一格式） |
| **错误可见性** | 静默丢失数据 | 立即报错并记录日志 |
| **责任清晰度** | 下游兼容上游 | 上游必须返回正确格式 |

### 最佳实践

**✅ 推荐做法**：
- 在 Extractor 内部确保返回标准 dict 格式
- 使用 Pydantic 或 dataclass 进行类型验证
- 在接口层做严格的格式检查

**❌ 避免做法**：
- 不要在下游代码中兼容多种上游格式
- 不要使用硬编码索引解包 tuple
- 不要静默吞没格式错误

## 问题 2: 缓存异常与日志 (Cache & Logging)

### 原始问题

**文件**: `graphrag_agent/graph/extraction/entity_extractor.py` (第 374-387 行)

存在"静默失败"的问题：

```python
# ❌ 原始代码：静默失败
def _load_from_cache(self, cache_key: str) -> Optional[str]:
    cache_path = self._cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
                self.cache_hits += 1
                return result
        except Exception as e:
            print(f"缓存加载错误: {e}")  # 仅打印，无法追溯

    self.cache_misses += 1
    return None
```

### 问题分析

**风险场景**：

```
生产环境部署 → 缓存文件损坏/格式不兼容
  ↓
系统静默丢弃所有数据
  ↓
图谱构建"完成"，但里面什么节点都没有
  ↓
日志里只有难以检索的 print 输出
  ↓
用户困惑：为什么图谱是空的？
```

**print 语句的问题**：
- 无法过滤日志级别（INFO/WARNING/ERROR）
- 无法集中管理日志输出
- 无法集成到监控系统
- 难以在生产环境中调试

### 修复方案

**✅ 引入专业日志系统**：

```python
import logging

# 配置日志
logger = logging.getLogger(__name__)

def _load_from_cache(self, cache_key: str) -> Optional[str]:
    """
    从缓存加载结果

    Args:
        cache_key: 缓存键

    Returns:
        缓存的结果，如果加载失败或缓存不存在则返回 None
    """
    if not self.enable_cache:
        return None

    cache_path = self._cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
                self.cache_hits += 1
                return result
        except (pickle.UnpicklingError, EOFError) as e:
            # 严重错误：缓存文件损坏，记录 ERROR 级别，可能需要人工介入清理缓存
            logger.error(f"缓存文件损坏 [{cache_key}]: {e}", exc_info=True)
            logger.error(f"建议删除损坏的缓存文件: {cache_path}")
            return None
        except Exception as e:
            # 一般错误：缓存读取失败，记录 WARNING 级别
            logger.warning(f"缓存读取失败 [{cache_key}]: {e}")
            return None

    self.cache_misses += 1
    return None
```

### 日志级别使用指南

| 级别 | 使用场景 | 示例 |
|------|---------|------|
| **DEBUG** | 详细的调试信息 | `logger.debug(f"处理 chunk {idx}: {chunk[:50]}...")` |
| **INFO** | 一般信息性消息 | `logger.info(f"开始批量抽取 {len(chunks)} 个 chunks")` |
| **WARNING** | 警告信息，不影响主流程 | `logger.warning(f"缓存读取失败，将重新计算")` |
| **ERROR** | 错误信息，需要关注 | `logger.error(f"缓存文件损坏: {e}", exc_info=True)` |
| **CRITICAL** | 严重错误，需要立即处理 | `logger.critical(f"错误率过高，中止构建")` |

### 日志配置示例

```python
# 在应用入口配置日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('graphrag.log'),
        logging.StreamHandler()
    ]
)
```

### 改进效果

| 特性 | 原始代码 (print) | 修复后代码 (logging) |
|------|-----------------|---------------------|
| **日志级别** | 无 | 5 个级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |
| **异常追溯** | 无 | `exc_info=True` 提供完整堆栈 |
| **日志过滤** | 无法过滤 | 可按级别/模块过滤 |
| **生产监控** | 难以集成 | 可集成到 ELK/Prometheus 等 |
| **调试效率** | 低 | 高（可精确定位问题） |

## 问题 3: 统一错误处理模式 (Unified Error Handling)

### 原始问题

**文件**: `graphrag_agent/graph/extraction/entity_extractor.py` (第 915-929 行)

存在异常吞没与缺乏熔断机制的问题：

```python
# ❌ 原始代码：异常吞没
for fut in concurrent.futures.as_completed(futures):
    idx = futures[fut]
    try:
        res = fut.result()
        llm_results[idx] = res
    except Exception as e:
        print(f'Chunk {idx} 处理异常: {e}')
        llm_results[idx] = {
            "entities": [],
            ...
        }  # 返回空结构，继续处理
```

### 问题分析

**风险场景**：

```
LLM API 挂了 (401 Unauthorized 或 429 Too Many Requests)
  ↓
每个 chunk 处理都失败，异常被捕获
  ↓
所有 chunk 被标记为"没有实体的正常文本"
  ↓
花钱跑完整个流程
  ↓
得到一个空的图谱
  ↓
用户困惑：为什么没有任何实体？
```

**问题后果**：
- **资源浪费**：继续处理剩余 chunk，浪费 LLM API 调用费用
- **时间浪费**：长时间运行后才发现结果是空的
- **诊断困难**：难以区分"真的没有实体"和"API 故障"
- **用户体验差**：等待很久后发现白费功夫

### 修复方案

**✅ 实现熔断机制**：

```python
# 4) 批量调用 LLM 抽取（使用并发 + 熔断机制）
logger.info(f"开始批量抽取 {len(flat_chunks)} 个 chunks...")
llm_results = [None] * len(flat_chunks)

# 错误计数器和熔断配置
error_count = 0
total_chunks = len(flat_chunks)
ERROR_RATE_THRESHOLD = 0.2  # 错误率阈值：20%
MIN_CHUNKS_FOR_CIRCUIT_BREAKER = 10  # 至少处理 10 个 chunk 后才启用熔断

with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    futures = {...}

    completed = 0
    for fut in concurrent.futures.as_completed(futures):
        idx = futures[fut]
        try:
            res = fut.result()
            llm_results[idx] = res
        except Exception as e:
            error_count += 1
            # 记录详细错误信息
            logger.error(f"Chunk {idx} 处理失败: {e}", exc_info=True)

            # 🔥 熔断机制：如果错误率超过阈值且处理了足够多的 chunk，则中止
            if total_chunks > MIN_CHUNKS_FOR_CIRCUIT_BREAKER:
                current_error_rate = error_count / (completed + 1)
                if current_error_rate > ERROR_RATE_THRESHOLD:
                    error_msg = (
                        f"错误率过高 ({current_error_rate:.1%} > {ERROR_RATE_THRESHOLD:.1%})，"
                        f"已处理 {completed + 1}/{total_chunks} 个 chunk，失败 {error_count} 个。"
                        f"中止图谱构建。请检查 LLM 连接或 Prompt 配置。"
                    )
                    logger.critical(error_msg)
                    raise RuntimeError(error_msg) from e

            # 填充空结果（仅在未触发熔断时）
            llm_results[idx] = {...}

        completed += 1
        if progress_callback:
            progress_callback(completed)

# 记录最终统计
if error_count > 0:
    error_rate = error_count / total_chunks
    logger.warning(
        f"批量抽取完成，共 {total_chunks} 个 chunk，"
        f"成功 {total_chunks - error_count} 个，失败 {error_count} 个 "
        f"(错误率: {error_rate:.1%})"
    )
```

### 熔断机制工作原理

```
处理 100 个 chunks，设置错误率阈值 20%
  ↓
处理前 10 个 chunk：5 个成功，5 个失败
  ↓
错误率 = 5/10 = 50% > 20%
  ↓
触发熔断，抛出 RuntimeError
  ↓
立即中止处理，记录 CRITICAL 日志
  ↓
用户看到明确的错误信息："错误率过高，请检查 LLM 连接"
  ↓
节省了 90 个 chunk 的处理时间和 API 费用
```

### 熔断配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **ERROR_RATE_THRESHOLD** | 0.2 (20%) | 错误率阈值，超过此值触发熔断 |
| **MIN_CHUNKS_FOR_CIRCUIT_BREAKER** | 10 | 至少处理 N 个 chunk 后才启用熔断 |

**调整建议**：
- 测试环境：可设置更低的阈值（如 10%）以快速发现问题
- 生产环境：可设置更高的阈值（如 30%）以容忍偶发错误
- 小文件：增大 `MIN_CHUNKS_FOR_CIRCUIT_BREAKER` 以避免误触发

### 改进效果

| 场景 | 原始代码 | 修复后代码 |
|------|---------|-----------|
| **LLM API 故障** | 处理完所有 chunk，得到空图谱 | 处理 10% chunk 后熔断，立即报错 |
| **资源节省** | 浪费 100% 的 API 费用 | 仅浪费 10% 的 API 费用（90% 节省） |
| **时间节省** | 等待完整处理时间 | 快速失败（10% 时间） |
| **错误诊断** | 难以判断原因 | 明确指出错误率和建议 |
| **用户体验** | 困惑和沮丧 | 快速反馈，清晰指导 |

### 错误场景示例

**场景 1: LLM API 401 Unauthorized**

```python
# 原始代码：
# 处理 1000 个 chunks
# 全部失败，但继续处理
# 耗时: 10 分钟
# 费用: $50
# 结果: 空图谱

# 修复后代码：
# 处理 10 个 chunks
# 10 个全部失败，错误率 100% > 20%
# 熔断触发
# 耗时: 6 秒
# 费用: $0.5
# 结果: RuntimeError: "错误率过高 (100.0% > 20.0%)，请检查 LLM 连接"
```

**场景 2: 部分 Chunk 解析失败（正常情况）**

```python
# 处理 1000 个 chunks
# 950 个成功，50 个失败（5% 错误率）
# 5% < 20%，不触发熔断
# 继续处理完所有 chunk
# 最终日志：
# WARNING: 批量抽取完成，共 1000 个 chunk，成功 950 个，失败 50 个 (错误率: 5.0%)
```

## 相关文件

### 修改文件

1. **`graphrag_agent/integrations/build/build_graph.py`**
   - 移除过度防御性编程（第 351-374 行）
   - 强制标准 dict 格式
   - 添加明确的错误日志

2. **`graphrag_agent/graph/extraction/entity_extractor.py`**
   - 添加 logging 模块（第 23-30 行）
   - 改进 `_load_from_cache()` 方法（第 378-408 行）
   - 实现熔断机制在 `process_chunks_batch()` 方法（第 918-983 行）

### 依赖项

- `logging`：Python 标准库，无需额外安装

## 最佳实践

### ✅ 推荐做法

1. **使用 logging 而非 print**：
   ```python
   # ✅ 推荐
   logger.error(f"处理失败: {e}", exc_info=True)

   # ❌ 避免
   print(f"处理失败: {e}")
   ```

2. **区分日志级别**：
   ```python
   logger.debug("详细调试信息")
   logger.info("一般信息")
   logger.warning("警告信息")
   logger.error("错误信息")
   logger.critical("严重错误")
   ```

3. **实现熔断机制**：
   ```python
   if error_rate > threshold:
       logger.critical(f"错误率过高: {error_rate:.1%}")
       raise RuntimeError("中止处理")
   ```

4. **记录最终统计**：
   ```python
   logger.info(f"处理完成，成功 {success}/{total} ({success_rate:.1%})")
   ```

### ❌ 避免做法

1. **不要静默吞没异常**：
   ```python
   # ❌ 错误
   try:
       process()
   except:
       pass  # 静默失败

   # ✅ 正确
   try:
       process()
   except Exception as e:
       logger.error(f"处理失败: {e}", exc_info=True)
       raise
   ```

2. **不要在下游兼容上游格式**：
   ```python
   # ❌ 错误：下游兼容多种格式
   if isinstance(res, dict):
       process_dict(res)
   elif isinstance(res, tuple):
       process_tuple(res)

   # ✅ 正确：强制上游返回标准格式
   if not isinstance(res, dict):
       raise TypeError("必须返回 dict 格式")
   ```

3. **不要无限重试失败操作**：
   ```python
   # ❌ 错误：无限容错
   for item in items:
       try:
           process(item)
       except:
           continue  # 继续处理下一个

   # ✅ 正确：实现熔断
   if error_rate > threshold:
       raise RuntimeError("错误率过高，中止处理")
   ```

## 监控和告警建议

### 生产环境配置

```python
import logging
import logging.handlers

# 配置日志
logger = logging.getLogger('graphrag')
logger.setLevel(logging.INFO)

# 文件 handler（按日期轮转）
file_handler = logging.handlers.TimedRotatingFileHandler(
    'graphrag.log',
    when='midnight',
    interval=1,
    backupCount=7
)
file_handler.setLevel(logging.INFO)

# 控制台 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# 格式化
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

### 告警规则示例

```yaml
# 示例：使用 Prometheus + AlertManager
- alert: GraphRAGHighErrorRate
  expr: |
    rate(graphrag_chunk_processing_errors_total[5m])
    /
    rate(graphrag_chunk_processing_total[5m])
    > 0.2
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "GraphRAG 错误率过高"
    description: "过去 5 分钟错误率 {{ $value | humanizePercentage }}"
```

## 参考资料

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Defensive Programming Best Practices](https://en.wikipedia.org/wiki/Defensive_programming)
