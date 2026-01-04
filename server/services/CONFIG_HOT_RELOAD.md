# GraphConfig 热更新机制 (Configuration Hot Reload)

## 核心架构

实现"前端修改 JSON → 后端存储 → Extractor 运行时加载 → 逻辑即时生效"的完整闭环。

```
┌──────────┐      POST /admin/graph/config       ┌──────────────────┐
│ Frontend │ ──────────────────────────────────> │ GraphConfigService│
│  (JSON)  │                                      │   (单例缓存)      │
└──────────┘                                      └──────────────────┘
                                                           │
                                                           │ save() + 刷新缓存
                                                           ↓
                                                  ┌──────────────────┐
                                                  │ graph_config.json│
                                                  │   (持久化存储)   │
                                                  └──────────────────┘
                                                           │
                                                           │ load()
                                                           ↓
                                                  ┌──────────────────┐
┌────────────┐    _get_graph_config()            │ GraphConfigService│
│ Extractor  │ <────────────────────────────────│   (读取缓存)      │
│ (运行时)   │                                   └──────────────────┘
└────────────┘
```

## 三层配置优先级

在 `entity_extractor.py` 的 `_get_graph_config()` 方法中，按以下优先级获取配置：

### 优先级 1: 直接传入的 graph_config
```python
extractor = EntityRelationExtractor(
    llm=llm,
    entity_types=entity_types,
    relationship_types=relationship_types,
    graph_config=my_config  # ← 直接传入
)
```

**场景**: 测试、脚本化任务、临时配置

### 优先级 2: Factory 注入的 prompt_builder.config
```python
# extractor_factory.py 自动注入
extractor = create_entity_extractor(llm, ...)
# extractor.prompt_builder.config 已设置
```

**场景**: 使用 `extractor_factory` 创建实例时自动注入

### 优先级 3: GraphConfigService 动态读取（🔥 热更新）
```python
# Extractor 内部自动调用
config_service = get_config_service()
config = config_service.get_config()  # 从内存缓存读取
```

**场景**:
- 后台服务运行时
- 前端通过 API 修改配置后
- 无需重启服务即可生效

## API 使用示例

### 1. 保存配置（POST /admin/graph/config）

```python
import requests

# 准备配置数据
config_data = {
    "project_name": "学生管理系统",
    "version": "1.0",
    "industry": "教育",
    "description": "学生管理知识图谱",
    "domain_definitions": [
        {
            "domain_name": "student_policy",
            "trigger_condition": "contains('学生', '奖学金', '处分')",
            "schema": {
                "entities": ["机构", "部门", "政策", "当事人"],
                "relations": ["发布", "适用于", "需要"]
            }
        }
    ],
    "bridge_definitions": []
}

# 调用 API 保存配置
response = requests.post(
    "http://localhost:8000/admin/graph/config",
    json=config_data
)

print(response.json())
# 输出:
# {
#   "message": "配置保存成功，已自动刷新缓存",
#   "project_name": "学生管理系统",
#   "cache_status": {
#     "has_cache": True,
#     "project_name": "学生管理系统",
#     "last_reload_time": "2025-12-22T10:30:00",
#     "domain_count": 1,
#     "bridge_count": 0
#   }
# }
```

**关键点**:
- ✅ 配置立即写入文件
- ✅ 内存缓存自动刷新
- ✅ Extractor 下次调用时读取到新配置

### 2. 获取配置（GET /admin/graph/config）

```python
response = requests.get("http://localhost:8000/admin/graph/config")
config = response.json()

if config["exists"]:
    print(f"当前配置: {config['config']['project_name']}")
else:
    print("暂无配置")
```

### 3. 删除配置（DELETE /admin/graph/config）

```python
response = requests.delete("http://localhost:8000/admin/graph/config")
print(response.json())
# 输出:
# {
#   "message": "配置删除成功，缓存已清空",
#   "cache_status": {"has_cache": False, ...}
# }
```

## 运行时热更新流程

### 场景：用户在前端修改实体类型白名单

1. **前端操作**：用户将实体类型从 ["Organization", "Policy"] 修改为 ["机构", "部门", "政策"]

2. **API 调用**：
```javascript
// 前端代码
const updatedConfig = {
  ...currentConfig,
  domain_definitions: [
    {
      domain_name: "default",
      schema: {
        entities: ["机构", "部门", "政策"],  // 修改后
        relations: ["发布", "适用于"]
      }
    }
  ]
}

fetch('/admin/graph/config', {
  method: 'POST',
  body: JSON.stringify(updatedConfig)
})
```

3. **后端处理**（`admin.py:save_graph_config`）：
```python
@router.post("/graph/config")
async def save_graph_config(config: GraphConfig):
    config_service = get_config_service()
    saved_config = config_service.update_config_obj(config)
    # ✅ 文件已保存
    # ✅ 缓存已刷新
    return {"message": "配置保存成功，已自动刷新缓存"}
```

4. **Extractor 运行时读取**（`entity_extractor.py:process_chunks`）：
```python
# 用户触发构建任务
def process_chunks(self, file_contents, progress_callback=None):
    for filename, content in file_contents:
        # 🔥 运行时动态获取配置
        domain = self._route_domain(filename, content)
        ent_types, rel_types = self._schema_for_domain(domain)

        # 🔥 此时读取到的是最新配置
        # ent_types = {"机构", "部门", "政策"}  ← 已更新
```

5. **结果**：
- ✅ 无需重启服务
- ✅ 无需重新初始化 Extractor
- ✅ 配置即时生效

## GraphConfigService 核心 API

### get_config() - 获取配置
```python
from server.services.graph_config_service import get_config_service

config_service = get_config_service()
config = config_service.get_config()  # 优先读缓存，无缓存则读文件

if config:
    print(f"项目: {config.project_name}")
    print(f"领域数量: {len(config.domain_definitions)}")
```

### update_config(dict) - 更新配置
```python
new_config = {
    "project_name": "新项目",
    "version": "1.0",
    "domain_definitions": [...]
}

config_service = get_config_service()
saved_config = config_service.update_config(new_config)
# ✅ 验证 → 保存文件 → 刷新缓存
```

### reload() - 强制重载
```python
# 如果怀疑缓存过期，强制重新读取文件
config_service.reload()
```

### delete_config() - 删除配置
```python
config_service.delete_config()
# ✅ 删除文件 → 清空缓存
```

### get_cache_status() - 查看缓存状态
```python
status = config_service.get_cache_status()
print(status)
# {
#   "has_cache": True,
#   "project_name": "学生管理系统",
#   "last_reload_time": "2025-12-22T10:30:00.123456",
#   "domain_count": 3,
#   "bridge_count": 2
# }
```

## 线程安全设计

`GraphConfigService` 使用双重锁定确保线程安全：

```python
class GraphConfigService:
    _instance = None
    _lock = threading.Lock()  # 单例创建锁

    def __init__(self):
        self._cache_lock = threading.Lock()  # 缓存操作锁

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:  # 双重检查锁定
                if cls._instance is None:
                    cls._instance = GraphConfigService()
        return cls._instance

    def get_config(self):
        with self._cache_lock:  # 保证并发安全
            # ...
```

**支持场景**：
- 多线程并发读取配置
- 多个 Extractor 实例同时运行
- API 并发修改配置

## 调试技巧

### 1. 查看缓存状态
```python
# 在任意 Python 代码中
from server.services.graph_config_service import get_config_service
print(get_config_service().get_cache_status())
```

### 2. 查看 Extractor 读取的配置
```python
# entity_extractor.py 已内置日志
# 运行时会自动打印:
# [Extractor] 从 GraphConfigService 读取配置: 学生管理系统
```

### 3. API 响应包含缓存状态
```python
response = requests.post("/admin/graph/config", json=config)
print(response.json()["cache_status"])  # 查看缓存是否更新
```

## 常见问题

### Q: 修改配置后需要重启服务吗？
**A**: 不需要。配置保存后会立即刷新内存缓存，Extractor 运行时自动读取最新配置。

### Q: 如果配置文件被手动修改怎么办？
**A**: 调用 `config_service.reload()` 强制重新读取文件。

### Q: 多个 Extractor 实例会读到不同的配置吗？
**A**: 不会。所有实例共享同一个 `GraphConfigService` 单例，确保配置一致性。

### Q: 如何回滚到旧配置？
**A**:
1. 方法 1: 通过 API 重新提交旧配置
2. 方法 2: 手动恢复 `graph_config.json` 文件，然后调用 `reload()`

### Q: 配置缓存会永久存在吗？
**A**: 缓存存在于进程内存中，服务重启后会清空。重启后第一次调用 `get_config()` 会从文件重新加载。

## 性能优化

### 缓存命中率
- 首次读取：从文件加载（~10ms）
- 后续读取：从内存缓存（~0.01ms）
- **性能提升**: 1000x

### 避免频繁 I/O
```python
# ❌ 不推荐：每次都直接读文件
storage = get_storage()
config = storage.load()

# ✅ 推荐：使用缓存服务
config_service = get_config_service()
config = config_service.get_config()  # 优先读缓存
```

## 总结

配置驱动架构的核心优势：

1. ✅ **热更新**：无需重启服务
2. ✅ **即时生效**：前端修改 → 后端立即可用
3. ✅ **高性能**：内存缓存避免频繁 I/O
4. ✅ **线程安全**：支持并发访问
5. ✅ **灵活性**：支持三层配置优先级
6. ✅ **调试友好**：缓存状态透明可查

配置流转路径：
```
Frontend JSON → API → GraphConfigService → graph_config.json
                                    ↓
                            (内存缓存刷新)
                                    ↓
Extractor._get_graph_config() → GraphConfigService.get_config()
                                    ↓
                            (读取最新缓存)
```
