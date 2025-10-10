# 配置模块

## 项目结构
```
graphrag_agent/config/    # 配置文件目录
├── __init__.py           # 包初始化文件
├── neo4jdb.py            # Neo4j数据库连接管理
├── prompt.py             # 各类提示模板定义
├── reasoning_prompts.py  # 推理提示模板
└── settings.py           # 全局配置参数
```

## 模块简介

配置模块是项目中负责数据库连接和查询的关键组件，主要基于Neo4j图数据库为上层应用提供数据存储和检索服务。该模块实现了数据库连接的管理、会话池优化以及各种提示模板的定义，为知识图谱构建和智能问答提供基础设施支持。

## 核心功能与实现思路

### 1. 数据库连接管理 (DBConnectionManager)

`DBConnectionManager` 类采用单例模式，确保全局只有一个数据库连接实例，有效管理Neo4j数据库连接资源：

- 基于环境变量配置连接参数，支持安全配置管理
- 实现会话池机制，优化会话资源利用
- 提供上下文管理器接口，确保资源的自动释放
- 集成Neo4j原生驱动和LangChain Neo4j图实例，支持多种查询方式

```python
# 使用示例
db_manager = get_db_manager()
with db_manager as manager:
    result = manager.execute_query("MATCH (n) RETURN count(n) as count")
```

### 2. 提示模板系统 (prompt.py)

`prompt.py` 定义了一系列用于知识图谱构建和查询的提示模板：

- 图谱构建模板 (system_template_build_graph)：引导模型从文本中识别实体和关系
- 实体索引模板 (system_template_build_index)：用于识别重复实体并进行合并
- 社区摘要模板 (community_template)：生成图社区的自然语言摘要
- 问答提示模板 (NAIVE_PROMPT, LC_SYSTEM_PROMPT)：支持基于图谱的智能问答

这些模板通过精心设计的格式化输出要求，确保模型生成的内容可以直接被系统解析和使用。

### 3. 推理提示系统 (reasoning_prompts.py)

`reasoning_prompts.py` 提供了一套专门用于复杂推理的提示模板：

- REASON_PROMPT：定义推理助手的基本行为和搜索方式
- RELEVANT_EXTRACTION_PROMPT：从搜索结果中提取相关信息
- SUB_QUERY_PROMPT：将复杂问题分解为子问题
- FOLLOWUP_QUERY_PROMPT：根据已有信息判断是否需要额外查询
- FINAL_ANSWER_PROMPT：基于推理过程生成最终答案

这些模板支持多轮搜索和推理，能够处理需要复杂逻辑推导的问题。

### 4. 全局配置 (settings.py)

`settings.py` 集中管理项目的全局配置参数：

- 知识库主题和图谱设置（实体类型、关系类型）
- 增量更新和冲突解决策略
- 文本分块参数设置
- 社区算法选择
- 回答方式配置
- 工具描述和示例问题
- **实体消歧和对齐配置**：
  - `DISAMBIG_STRING_THRESHOLD`: 字符串匹配阈值（默认0.7）
  - `DISAMBIG_VECTOR_THRESHOLD`: 向量相似度阈值（默认0.85）
  - `DISAMBIG_NIL_THRESHOLD`: NIL检测阈值（默认0.6）
  - `DISAMBIG_TOP_K`: 候选实体数量（默认5）
  - `ALIGNMENT_CONFLICT_THRESHOLD`: 冲突检测阈值（默认0.5）
  - `ALIGNMENT_MIN_GROUP_SIZE`: 最小分组大小（默认2）

## 核心函数

1. **`DBConnectionManager.execute_query()`**: 执行Cypher查询并返回结果数据框架。

2. **`DBConnectionManager.get_session()`**: 从会话池获取或创建Neo4j会话。

3. **`get_db_manager()`**: 获取全局唯一的数据库连接管理器实例。

4. **`DBConnectionManager.__enter__()` 和 `__exit__()`**: 支持上下文管理器接口，确保资源正确释放。