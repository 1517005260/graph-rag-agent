# 实体对齐 (Entity Resolution) 功能

## 概述

实体对齐模块用于识别和合并知识图谱中的重复实体，提升图谱质量和查询准确性。

## 功能特性

- ✅ **基于字符串相似度**：使用 Levenshtein 距离计算实体名称相似度
- ✅ **可配置阈值**：灵活调整相似度阈值（默认 0.9）
- ✅ **分块优化**：使用首字母分组降低计算复杂度
- ✅ **安全合并**：使用 Neo4j APOC 插件保留所有关系和属性
- ✅ **预览功能**：执行前预览可能重复的实体
- ✅ **批量处理**：支持大规模实体去重

## 依赖要求

### Python 库

```bash
# 推荐安装（性能更好）
pip install python-Levenshtein

# 备用方案（无需安装，但较慢）
# 使用内置 difflib.SequenceMatcher
```

### Neo4j 插件

**必须安装 APOC 插件**：

```bash
# 方法 1: Docker Compose（推荐）
# 在 docker-compose.yml 中添加：
environment:
  - NEO4J_PLUGINS=["apoc"]

# 方法 2: 手动安装
# 1. 下载 APOC JAR 文件
# https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases

# 2. 放置到 Neo4j plugins 目录
cp apoc-*.jar /path/to/neo4j/plugins/

# 3. 重启 Neo4j
```

验证 APOC 安装：

```cypher
CALL apoc.help('refactor.mergeNodes')
```

## 使用方法

### 方法 1: 通过 IncrementalUpdateManagerV2

```python
from graphrag_agent.integrations.build.incremental_update_v2 import (
    IncrementalUpdateManagerV2
)

# 创建管理器
manager = IncrementalUpdateManagerV2(files_dir="./files")

# 执行实体对齐
result = manager.resolve_entities(
    threshold=0.9,        # 相似度阈值
    entity_type=None      # None 表示所有类型
)

# 查看结果
print(f"合并了 {result['merged_count']} 组实体")
```

### 方法 2: 直接使用 EntityResolver

```python
from graphrag_agent.graph.processing import EntityResolver

# 创建对齐器
resolver = EntityResolver(threshold=0.9)

# 预览重复实体（不执行合并）
duplicates = resolver.preview_duplicates(limit=10)
for group in duplicates:
    print(f"将合并到: {group['merge_to']}")
    print(f"实体列表: {group['entities']}")

# 执行对齐
result = resolver.resolve_entities(entity_type=None)

# 获取统计信息
stats = resolver.get_statistics()
print(f"当前实体数: {stats['total_entities']}")
```

### 方法 3: 使用测试脚本

```bash
# 预览可能重复的实体
python test/test_entity_resolution.py --preview --threshold 0.9

# 执行实体对齐
python test/test_entity_resolution.py --resolve --threshold 0.9

# 指定实体类型
python test/test_entity_resolution.py --resolve --entity-type "学生类型"

# 运行所有测试
python test/test_entity_resolution.py --all-tests
```

## 算法说明

### V1: 规则版（Levenshtein 距离）

**相似度计算**：

```python
from Levenshtein import ratio

similarity = ratio("国家奖学金", "国家励志奖学金")
# 返回: 0.67
```

**分块策略**：

为了优化性能，按首字母分组：

```
- 块 "国": ["国家奖学金", "国家励志奖学金", "国家助学金"]
- 块 "优": ["优秀学生", "优秀学生干部"]
- 块 "学": ["学生会", "学生会主席"]
```

只在同一块内比较，降低复杂度从 O(N²) 到 O(k·N²/k) ≈ O(N²/k)

### 阈值选择建议

| 阈值 | 说明 | 适用场景 |
|------|------|----------|
| 0.95 | 非常严格 | 只合并几乎完全相同的实体 |
| **0.9** | **推荐** | **平衡精度和召回率** |
| 0.85 | 宽松 | 合并更多相似实体（可能误合并） |
| 0.8 | 非常宽松 | 不推荐（容易误合并） |

## 合并逻辑

使用 Neo4j APOC 的 `mergeNodes` 函数：

```cypher
MATCH (e:`__Entity__`)
WHERE e.id IN ["国家奖学金", "国家励志奖学金"]
WITH collect(e) as nodes
CALL apoc.refactor.mergeNodes(nodes, {
    properties: 'combine',  // 合并属性
    mergeRels: true         // 合并关系
})
YIELD node
RETURN node
```

**合并策略**：
- 保留第一个节点作为主节点
- 合并所有入边和出边
- 合并所有属性（数组合并）
- 删除重复的关系

## 性能优化

### 当前实现（适用于 < 100K 实体）

- 使用首字母分块
- 批处理查询
- 异步执行

### 大规模优化建议（> 100K 实体）

1. **使用 MinHash LSH**：
   ```python
   from datasketch import MinHash, MinHashLSH
   ```

2. **基于实体类型分块**：
   ```python
   resolver.resolve_entities(entity_type="学生类型")
   ```

3. **增量对齐**：
   只对新增实体执行对齐

4. **并行处理**：
   使用多进程处理不同的块

## 示例场景

### 场景 1: 学生管理系统

**问题**：
```
实体列表：
- "国家奖学金"
- "国家励志奖学金"
- "国家助学金"
- "优秀学生"
- "优秀学生干部"
```

**对齐后**（阈值 0.9）：
```
无合并（相似度都低于 0.9）
```

**对齐后**（阈值 0.7）：
```
合并：
- "国家奖学金" ← "国家励志奖学金" (0.67 相似度，低于 0.7，不合并)
- "优秀学生" ← "优秀学生干部" (0.67 相似度，低于 0.7，不合并)
```

### 场景 2: 拼写错误

**问题**：
```
- "计算机科学"
- "计算机科学"  (重复)
- "計算機科學"  (繁体)
```

**对齐后**（阈值 0.9）：
```
合并：
- "计算机科学" ← "计算机科学" (完全相同，1.0)
```

## 注意事项

⚠️ **重要提醒**：

1. **不可逆操作**：合并后无法自动恢复，建议先预览
2. **备份数据**：执行前备份 Neo4j 数据库
3. **测试阈值**：先在小数据集上测试合适的阈值
4. **检查结果**：合并后检查图谱一致性

## 集成到构建流程

在 `incremental_update_v2.py` 中的推荐位置：

```python
# L1 深度索引流程
def run_deep_indexing(self, file_paths=None):
    # 1. 实体提取
    self.updater.process_incremental_update()

    # 2. 实体对齐（在社区检测之前）
    self.resolve_entities(threshold=0.9)

    # 3. 社区检测
    self.detect_communities()

    # 4. 图谱验证
    self.verify_graph_consistency(repair=True)
```

## 故障排查

### 问题 1: APOC 未安装

**错误**：
```
Neo4j.ClientError.Procedure.ProcedureNotFound
```

**解决**：
安装 APOC 插件（见依赖要求）

### 问题 2: 性能慢

**原因**：
实体数量过多，O(N²) 复杂度高

**解决**：
- 降低阈值（减少合并组数）
- 按实体类型分批处理
- 使用分块策略（默认已启用）

### 问题 3: 误合并

**原因**：
阈值设置过低

**解决**：
- 提高阈值（0.9 → 0.95）
- 使用预览功能检查
- 针对特定实体类型调整

## 未来改进

- [ ] V2: 基于向量 Embedding 的语义相似度
- [ ] V3: 图神经网络 (GNN) 实体对齐
- [ ] 支持人工确认合并
- [ ] 提供撤销合并功能
- [ ] 性能监控和报告

## 参考资料

- [APOC User Guide](https://neo4j.com/labs/apoc/)
- [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Entity Resolution Survey](https://arxiv.org/abs/1905.06397)
