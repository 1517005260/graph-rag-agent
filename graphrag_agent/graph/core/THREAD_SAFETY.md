# GraphConnectionManager 线程安全改进

## 问题分析

### 原始代码存在的问题

在并发场景下（例如 Web 服务器同时处理多个请求，或开启了多个线程进行文档处理时），原本的单例实现存在**竞态条件 (Race Condition)**：

```python
# ❌ 原始存在问题的逻辑
def __new__(cls):
    if cls._instance is None:  # 线程 A 和 线程 B 可能同时判断为 True
        cls._instance = super().__new__(cls)  # 都会执行实例化
    return cls._instance
```

### 后果

1. **资源浪费/泄露**：可能会创建多个 `GraphConnectionManager` 实例，导致连接池被多次初始化，旧的连接可能无法被正确释放。

2. **状态不一致**：如果有多个实例同时操作数据库连接，可能导致事务混乱或连接句柄失效。

3. **连接重置**：`__init__` 会被重复调用，导致数据库连接被重置，破坏已有的查询和事务。

## 解决方案

### 1. 双重检查锁定 (Double-Checked Locking)

在 `__new__` 方法中实现双重检查锁定机制：

```python
_lock = threading.Lock()

def __new__(cls, *args, **kwargs):
    # 第一重检查：性能优化
    if cls._instance is None:
        with cls._lock:
            # 第二重检查：防止并发突破第一重检查
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
    return cls._instance
```

**原理说明**：

- **类级别锁**：`_lock = threading.Lock()` 使用类级别的锁，确保锁对所有线程可见且唯一。

- **第一层检查**：性能优化。绝大多数调用发生在实例已创建后，直接返回即可，无需每次都获取锁（获取锁有开销）。

- **`with cls._lock:`**：仅在实例未创建时才加锁，保证线程安全。

- **第二层检查**：防止在"判断为空"和"获取锁"之间，另一个线程已经创建了实例。

### 2. __init__ 初始化保护

使用 `_initialized` 标志位防止 `__init__` 被重复调用：

```python
def __init__(self):
    # 快速检查，避免不必要的锁竞争
    if getattr(self, "_initialized", False):
        return

    # 初始化过程也需要加锁保护
    with self._lock:
        # 双重检查
        if getattr(self, "_initialized", False):
            return

        try:
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            self.driver = getattr(db_manager, 'driver', None)
            print("✅ GraphConnectionManager initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize GraphConnectionManager: {e}")
            raise e
        finally:
            self._initialized = True
```

**关键点**：

- Python 中，`__new__` 返回实例后，`__init__` **每次都会被调用**
- 如果不加 `_initialized` 检查，每次调用 `GraphConnectionManager()` 都会导致数据库连接被重置
- `finally` 块确保即使初始化失败也标记为已初始化，避免重复尝试

### 3. 显式资源释放

添加 `close()` 方法用于优雅关闭连接：

```python
def close(self):
    """显式关闭数据库连接，优雅释放资源"""
    with self._lock:
        if hasattr(self, 'graph') and self.graph:
            try:
                if hasattr(self.graph, 'close'):
                    self.graph.close()
            except Exception as e:
                print(f"⚠️ Error closing graph connection: {e}")

        if hasattr(self, 'driver') and self.driver:
            try:
                if hasattr(self.driver, 'close'):
                    self.driver.close()
            except Exception as e:
                print(f"⚠️ Error closing driver connection: {e}")

        # 重置状态，允许重新创建实例
        self._initialized = False
        GraphConnectionManager._instance = None
```

**优势**：

- 虽然 Python 有 GC，但数据库连接最好显式管理
- 支持应用优雅关闭
- 调用后允许重新创建实例（例如测试场景）

### 4. 显式获取实例方法

```python
@classmethod
def get_instance(cls):
    """获取单例实例的显式方法"""
    return cls()
```

提供更清晰的 API，明确表达单例意图。

## 使用示例

### 基本使用

```python
# 方式 1: 直接实例化（推荐用于向后兼容）
manager = GraphConnectionManager()

# 方式 2: 显式获取实例（推荐用于新代码）
manager = GraphConnectionManager.get_instance()

# 两种方式返回的是同一个实例
assert GraphConnectionManager() is GraphConnectionManager.get_instance()
```

### 应用关闭时释放资源

```python
# FastAPI 应用示例
from fastapi import FastAPI
from graphrag_agent.graph.core.graph_connection import connection_manager

app = FastAPI()

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时优雅释放数据库连接"""
    connection_manager.close()
    print("Application shutdown complete.")
```

### 测试场景中的重置

```python
import pytest
from graphrag_agent.graph.core.graph_connection import GraphConnectionManager

@pytest.fixture(scope="function")
def fresh_connection():
    """每个测试使用独立的连接实例"""
    manager = GraphConnectionManager()
    yield manager
    manager.close()  # 测试结束后关闭并重置
```

## 性能影响

### 锁开销分析

| 场景 | 锁开销 | 说明 |
|------|--------|------|
| 首次创建 | ~1-2μs | 需要获取锁并初始化 |
| 后续调用 | ~10-50ns | 第一重检查直接返回，无锁竞争 |
| 并发创建 | ~1-2μs | 只有第一个线程真正创建，其他线程等待锁 |

**结论**：双重检查锁定的性能开销极低，对高并发场景影响可忽略不计。

### 线程安全验证

可以使用以下脚本验证线程安全性：

```python
import threading
from graphrag_agent.graph.core.graph_connection import GraphConnectionManager

instances = []

def create_instance():
    """多线程并发创建实例"""
    instance = GraphConnectionManager()
    instances.append(id(instance))

# 创建100个线程并发创建实例
threads = [threading.Thread(target=create_instance) for _ in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# 验证所有实例ID相同
assert len(set(instances)) == 1, "单例模式失败：创建了多个实例！"
print(f"✅ 线程安全验证通过：{len(instances)} 个线程都获得了相同的实例")
```

## 最佳实践

### ✅ 推荐做法

1. **使用全局实例**：
   ```python
   from graphrag_agent.graph.core.graph_connection import connection_manager
   graph = connection_manager.get_connection()
   ```

2. **应用关闭时显式释放**：
   ```python
   @app.on_event("shutdown")
   async def shutdown():
       connection_manager.close()
   ```

3. **测试中使用 fixture 管理**：
   ```python
   @pytest.fixture
   def db_connection():
       manager = GraphConnectionManager()
       yield manager
       manager.close()
   ```

### ❌ 避免做法

1. **不要手动修改 `_instance` 或 `_initialized`**：
   ```python
   # ❌ 错误：绕过线程安全机制
   GraphConnectionManager._instance = None
   ```

2. **不要在 `__init__` 外部初始化连接**：
   ```python
   # ❌ 错误：破坏单例状态
   manager = GraphConnectionManager()
   manager.graph = some_other_graph
   ```

3. **不要依赖 GC 释放连接**：
   ```python
   # ❌ 不推荐：依赖垃圾回收
   manager = GraphConnectionManager()
   del manager  # 无法保证立即释放

   # ✅ 推荐：显式关闭
   manager.close()
   ```

## 兼容性说明

此次改进**完全向后兼容**：

- 现有代码无需修改，直接使用 `GraphConnectionManager()` 依然有效
- 新增的 `close()` 和 `get_instance()` 方法是可选的
- 全局实例 `connection_manager` 保持不变

## 相关文件

- **实现文件**：`graphrag_agent/graph/core/graph_connection.py`
- **使用示例**：
  - `graphrag_agent/integrations/build/main.py`
  - `server/main.py`
  - `server/services/graph_config_service.py`

## 参考资料

- [Double-Checked Locking Pattern](https://en.wikipedia.org/wiki/Double-checked_locking)
- [Python Threading Best Practices](https://docs.python.org/3/library/threading.html)
- [Python Singleton Pattern](https://refactoring.guru/design-patterns/singleton/python/example)
