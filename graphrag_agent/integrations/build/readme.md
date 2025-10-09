# 知识图谱构建模块

## 目录结构

```
graphrag_agent/integrations/build/
├── __init__.py                           # 模块入口，导出类和函数
├── build_chunk_index.py                  # 文本块索引构建器
├── build_graph.py                        # 基础知识图谱构建器
├── build_index_and_community.py          # 实体索引和社区构建器
├── incremental/                          # 增量更新子模块
│   ├── __init__.py                       # 增量更新模块入口
│   ├── file_change_manager.py            # 文件变更管理器
│   ├── incremental_update_scheduler.py   # 增量更新调度器
│   └── manual_edit_manager.py            # 手动编辑同步管理器
├── incremental_graph_builder.py          # 增量图谱更新构建器
├── incremental_update.py                 # 增量更新管理程序
└── main.py                               # 主程序入口，整合完整流程
```

## 模块概述

`graphrag_agent/integrations/build` 模块是知识图谱系统的高层构建工具，封装了 `graphrag_agent/graph` 和 `graphrag_agent/community` 等底层模块，提供完整的图谱构建、索引创建和增量更新流程。本模块采用了模块化设计，将图谱构建过程分解为多个独立且可组合的步骤，既支持完整的一站式构建，也支持单独执行特定步骤。

## 核心实现思路

### 1. 分阶段构建流程

整个知识图谱构建过程被划分为三个主要阶段：
1. **基础图谱构建**：读取文档，分块，提取实体关系，创建基础图结构
2. **索引和社区构建**：创建实体索引，检测相似实体并合并，进行**实体质量提升**（消歧和对齐），执行社区检测
3. **文本块索引构建**：为文本块创建向量索引，支持后续检索

这种分阶段设计使得流程更加清晰，也便于调试和优化每个独立阶段。

### 2. 实体质量提升机制

在基础图谱构建和实体合并之后，模块通过**实体消歧（Disambiguation）**和**实体对齐（Alignment）**两个步骤进一步提升实体质量：

1. **实体消歧（EntityDisambiguator）**：
   - **字符串召回**：使用编辑距离快速找到相似实体候选
   - **向量重排**：利用语义相似度对候选进行重新排序
   - **NIL检测**：识别未登录实体（知识库中不存在的新实体）
   - **应用到图谱**：为WCC分组中的实体设置canonical_id，指向代表性最强的实体

2. **实体对齐（EntityAligner）**：
   - **按canonical_id分组**：找出所有指向同一canonical实体的实体
   - **冲突检测**：通过关系类型相似度检测语义冲突
   - **冲突解决**：使用LLM智能决策保留哪个实体
   - **实体合并**：将同组实体合并，保留关系和属性

### 3. 增量更新机制

为避免每次数据变更都需要重建整个图谱，模块实现了精细化的增量更新机制：
1. **文件变更检测**：追踪文件的添加、修改和删除
2. **增量图谱更新**：仅处理变更的部分，保留现有图谱结构
3. **智能调度系统**：根据不同组件的特性安排更新频率
4. **手动编辑保护**：确保用户手动添加或修改的内容不会被自动更新覆盖

### 4. 系统资源自适应

所有构建器都能根据系统资源动态调整处理参数：
1. **并行度调整**：根据CPU核心数调整并行线程数
2. **批处理大小优化**：根据可用内存动态计算最优批处理大小
3. **降级策略**：对于资源受限场景提供性能退化方案

## 核心类和功能

### KnowledgeGraphBuilder
处理基础图谱构建，包括文件读取、分块、实体关系提取和图结构创建：

```python
# 初始化并执行完整流程
builder = KnowledgeGraphBuilder()
processed_documents = builder.process()
```

核心方法：
- `build_base_graph()`: 构建基础知识图谱骨架
- `_initialize_components()`: 初始化所有必要组件并优化参数

### IndexCommunityBuilder
实体索引构建、实体质量提升和社区检测功能：

```python
# 构建实体索引、实体质量提升和社区检测
index_builder = IndexCommunityBuilder()
index_builder.process()
```

核心方法：
- `build_index_and_communities()`: 创建实体索引、执行实体消歧对齐、进行社区检测
- `update_entity_embeddings()`: 生成和更新实体的嵌入向量

内部整合了：
- `EntityQualityProcessor`: 实体质量提升处理器
- `EntityDisambiguator`: 实体消歧器
- `EntityAligner`: 实体对齐器

### ChunkIndexBuilder
构建文本块的向量索引，用于后续相似性检索：

```python
# 创建文本块索引
chunk_builder = ChunkIndexBuilder()
chunk_builder.process()
```

核心方法：
- `build_chunk_index()`: 为文本块创建嵌入向量索引

### IncrementalGraphUpdater
处理图谱的增量更新，仅更新变化的部分：

```python
# 初始化增量更新器
updater = IncrementalGraphUpdater(files_dir)
# 执行增量更新
updater.process_incremental_update()
```

核心方法：
- `detect_changes()`: 检测文件变更
- `process_new_files()`: 处理新添加的文件
- `update_changed_file_embeddings()`: 更新变更文件的嵌入向量
- `process_deleted_files()`: 处理已删除的文件

### IncrementalUpdateManager
管理整个增量更新流程，提供调度和后台运行支持：

```python
# 创建增量更新管理器
manager = IncrementalUpdateManager(files_dir)
# 单次执行
manager.run_once()
# 后台运行
manager.start_scheduler()
```

核心方法：
- `detect_file_changes()`: 检测文件变更并触发更新
- `verify_graph_consistency()`: 验证和修复图谱一致性
- `sync_manual_edits()`: 同步手动编辑，确保不被覆盖

## 特色功能

### 1. 三级性能优化策略

* **标准处理**: 正常资源条件下的处理流程
* **优化处理**: 针对大数据集的并行和批处理优化
* **降级处理**: 资源受限条件下的备用策略

### 2. 智能调度系统

增量更新调度器支持为不同组件设置不同的更新频率：
* 文件变更检测：高频（默认5分钟）
* 实体嵌入更新：中频（默认30分钟）
* 社区检测：低频（默认48小时）

### 3. 实体质量提升流程

`EntityQualityProcessor`整合实体消歧和对齐，确保图谱中实体的高质量：
1. **实体消歧**：将相似实体通过canonical_id链接到代表实体
2. **冲突检测**：识别同一canonical_id下的语义冲突
3. **实体对齐**：合并指向同一canonical实体的所有实体
4. **关系保留**：确保合并过程中不丢失关系信息

### 4. 手动编辑保护机制

`ManualEditManager`确保用户的手动编辑内容在自动更新中得到保护：
1. 检测数据库中的手动标记
2. 保护手动编辑的节点和关系
3. 智能解决冲突（可配置冲突解决策略）

### 5. 图谱一致性验证

`GraphConsistencyValidator`能够检测和修复常见的图谱问题：
1. 孤立节点检测
2. 断开的关系链接重建
3. 数据完整性修复

## 使用方式

### 完整构建流程

```python
from graphrag_agent.integrations.build.main import KnowledgeGraphProcessor

# 执行完整构建流程
processor = KnowledgeGraphProcessor()
processor.process_all()
```

### 增量更新

```python
from graphrag_agent.integrations.build.incremental_update import IncrementalUpdateManager

# 单次更新
manager = IncrementalUpdateManager("./data")
manager.run_once()

# 后台定时更新
manager.start_scheduler()
```

### 命令行运行

```bash
# 执行完整构建
python graphrag_agent/integrations/build/main.py

# 运行增量更新（单次）
python graphrag_agent/integrations/build/incremental_update.py --once

# 运行增量更新（守护进程模式）
python graphrag_agent/integrations/build/incremental_update.py --daemon --interval 300
```

## 性能和扩展性考量

1. **内存管理**: 所有组件都会根据系统内存动态调整批处理大小
2. **并行处理**: 利用多线程提高计算密集型任务的效率
3. **错误恢复**: 完善的错误处理和回退策略确保流程稳定
4. **模块化设计**: 可以轻松添加新的构建阶段或更新策略