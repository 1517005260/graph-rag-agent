# 配置模块

## 项目结构
```
graphrag_agent/config/    # 配置文件目录
├── __init__.py           # 包初始化文件（空文件）
├── neo4jdb.py            # Neo4j数据库连接管理
├── prompts/              # 提示模板集合
└── settings.py           # 全局配置参数与环境变量管理
```

`prompts/` 子目录按功能拆分模板文件：

- `graph_prompts.py`：图谱构建、索引维护、社区摘要
- `qa_prompts.py`：各类检索问答（Naive/Local/Global/Reduce/上下文重写）
- `reasoning_prompts.py`：深度推理流程的提示与常量
- `planner_prompts.py`、`executor_prompts.py`、`reporter_prompts.py`：多智能体任务拆解、执行与报告模板
- `__init__.py`：统一导出，提供单一入口 `graphrag_agent.config.prompts`

## 模块简介

配置模块是GraphRAG Agent项目的核心基础设施层，负责三大职能：

1. **统一配置管理**：从.env文件加载环境变量，并提供类型安全的访问接口
2. **数据库连接管理**：基于Neo4j图数据库实现单例模式的连接池管理
3. **提示模板库**：为知识图谱构建、多种搜索策略和深度推理提供标准化的LLM提示模板

该模块确保整个系统的配置一致性、数据库资源的高效利用，以及LLM交互的规范化。

## 核心功能与实现思路

### 1. 全局配置管理 (settings.py)

`settings.py` 是整个系统的配置中枢，通过环境变量和默认值提供类型安全的配置访问：

**辅助函数**：
- `_get_env_int()`: 获取整型环境变量，包含错误检查
- `_get_env_float()`: 获取浮点型环境变量，包含错误检查
- `_get_env_bool()`: 获取布尔型环境变量，支持多种布尔表达（true/1/yes/y/on）
- `_get_env_choice()`: 从有限集合中获取字符串配置，自动验证合法性

**配置分类**（共15大类）：

1. **基础路径设置**：BASE_DIR、PROJECT_ROOT、FILES_DIR、FILE_REGISTRY_PATH
2. **知识库参数**：KB_NAME（知识库主题）、workers（FastAPI并发进程数）
3. **知识图谱配置**：theme、entity_types、relationship_types、conflict_strategy、community_algorithm
4. **文本处理配置**：CHUNK_SIZE、OVERLAP、MAX_TEXT_LENGTH、similarity_threshold
5. **回答生成配置**：response_type（默认"多个段落"）
6. **Agent工具描述**：lc_description、gl_description、naive_description、examples
7. **性能优化配置**：MAX_WORKERS、各类BATCH_SIZE（ENTITY/CHUNK/EMBEDDING/LLM/COMMUNITY）
8. **GDS配置**：GDS_MEMORY_LIMIT、GDS_CONCURRENCY、GDS_NODE_COUNT_LIMIT、GDS_TIMEOUT_SECONDS
9. **实体消歧与对齐配置**：
   - DISAMBIG_STRING_THRESHOLD（字符串匹配阈值，默认0.7）
   - DISAMBIG_VECTOR_THRESHOLD（向量相似度阈值，默认0.85）
   - DISAMBIG_NIL_THRESHOLD（NIL检测阈值，默认0.6）
   - DISAMBIG_TOP_K（候选实体数，默认5）
   - ALIGNMENT_CONFLICT_THRESHOLD（冲突检测阈值，默认0.5）
   - ALIGNMENT_MIN_GROUP_SIZE（最小分组大小，默认2）
10. **路径与缓存配置**：CACHE_DIR、MODEL_CACHE_DIR、TIKTOKEN_CACHE_DIR、CACHE_SETTINGS字典
11. **Neo4j连接配置**：NEO4J_CONFIG字典（uri、username、password、max_pool_size、refresh_schema）
12. **LLM与嵌入配置**：OPENAI_EMBEDDING_CONFIG、OPENAI_LLM_CONFIG字典
13. **相似实体检测配置**：SIMILAR_ENTITY_SETTINGS字典（word_edit_distance、batch_size等）
14. **搜索工具配置**：
    - BASE_SEARCH_CONFIG（通用搜索参数）
    - LOCAL_SEARCH_SETTINGS（本地搜索参数）
    - GLOBAL_SEARCH_SETTINGS（全局搜索参数）
    - NAIVE_SEARCH_TOP_K、HYBRID_SEARCH_SETTINGS
15. **Agent配置**：
    - AGENT_SETTINGS（递归限制、流式输出阈值等）
    - MULTI_AGENT系列配置（Plan-Execute-Report架构的20+配置项）

**设计理念**：
- 所有配置优先从环境变量读取，未设置时使用代码中的默认值
- 提供类型转换和验证，确保配置值的合法性
- 支持字典形式的配置组，便于模块化使用

### 2. 数据库连接管理 (neo4jdb.py)

`DBConnectionManager` 类采用**单例模式**管理Neo4j数据库连接：

**核心特性**：
- 单例模式实现：通过`__new__`和`_initialized`标志确保全局唯一实例
- 双驱动支持：
  - `self.driver`：Neo4j原生驱动（用于execute_query）
  - `self.graph`：LangChain Neo4j图实例（用于与LangChain集成）
- 会话池管理：
  - `session_pool`：维护可复用的会话对象
  - `get_session()`：从池中获取或创建新会话
  - `release_session()`：会话归还到池中（池满时关闭）
  - 池大小上限由`max_pool_size`控制
- 上下文管理器：支持`with`语句，自动关闭资源

**核心方法**：
- `execute_query(cypher, params)`: 执行Cypher查询，返回pandas DataFrame
- `get_driver()`: 获取Neo4j原生驱动实例
- `get_graph()`: 获取LangChain Neo4j图实例
- `close()`: 关闭所有会话和驱动

**使用示例**：
```python
# 方式1：直接使用全局实例
from graphrag_agent.config.neo4jdb import get_db_manager
db_manager = get_db_manager()
result = db_manager.execute_query("MATCH (n) RETURN count(n) as count")

# 方式2：上下文管理器（自动清理资源）
with get_db_manager() as manager:
    result = manager.execute_query("MATCH (n:Entity) RETURN n LIMIT 10")
```

### 3. 知识图谱构建与搜索提示模板 (prompts/graph_prompts.py & prompts/qa_prompts.py)

`prompts/graph_prompts.py` 收录了图谱构建与维护流程中的模板：

1. **system_template_build_graph**：实体关系抽取模板，定义实体、关系字段以及输出格式示例
2. **human_template_build_graph**：实体抽取的人类输入模板，占位符由上游流程填充
3. **system_template_build_index** / **user_template_build_index**：实体去重与列表格式化的双模板
4. **community_template**：社区摘要生成模板，用于社区级别的自然语言描述

`prompts/qa_prompts.py` 则集中管理问答阶段的模板：

1. **NAIVE_PROMPT**：NaiveRAG问答模板，强调依据检索片段作答和引用格式
2. **LC_SYSTEM_PROMPT**：本地检索问答模板，支持实体/关系/报告/原文的引用标注
3. **MAP_SYSTEM_PROMPT**：全局搜索Map阶段，输出结构化要点列表（含评分与引用）
4. **REDUCE_SYSTEM_PROMPT**：全局搜索Reduce阶段，整合要点列表生成最终回答
5. **contextualize_q_system_prompt**：对话上下文重写模板，使问题脱离对话历史依然可理解

所有问答模板均对引用数量、Markdown结构、回答长度等做了统一约束。

### 4. 深度推理与搜索提示模板 (prompts/reasoning_prompts.py)

`prompts/reasoning_prompts.py` 为 DeepResearch 相关工具提供推理循环所需的模板与常量：

- **BEGIN/END_SEARCH_QUERY、BEGIN/END_SEARCH_RESULT、MAX_SEARCH_LIMIT**：限定搜索交互的格式与次数
- **REASON_PROMPT**：主推理循环模板，示范如何迭代搜索并整理结论
- **RELEVANT_EXTRACTION_PROMPT**：从搜索结果中提取与当前查询相关的信息
- **SUB_QUERY_PROMPT**：将复杂问题分解为子查询
- **FOLLOWUP_QUERY_PROMPT**：判断是否需要追加搜索
- **FINAL_ANSWER_PROMPT**：基于思考过程和检索结果生成最终回答

这些模板支撑了多轮推理、假设生成、结论整合等高级能力，确保推理路径可追踪、响应可解释。

### Prompt 集中管理的例外说明

以下提示暂未迁入 `config/prompts/`，原因是它们在运行时需要根据上下文动态拼装大段内容，或与复杂流程紧密耦合，拆分会显著增加代码复杂度：

- `graphrag_agent/search/tool/deep_research_tool.py:1190`：答案修复 `fix_prompt` 直接插入实时 question/answer 详情，并在异步修复流程中内联调用。
- `graphrag_agent/search/tool/deeper_research_tool.py:844` 与 `1975`：`enhanced_prompt` 会在执行过程中不断串接检索片段、知识图谱摘要、社区洞察等多段文本，属流式构造。
- `graphrag_agent/search/tool/reasoning/community_enhance.py:278`、`chain_of_exploration.py:185/479`、`evidence.py:377`：这些推理工具按步骤动态拼接指南、路径和证据描述，难以抽象为单一模板。
- `graphrag_agent/evaluation/metrics/*`（`answer_metrics.py`, `deep_search_metrics.py`, `retrieval_metrics.py`, `graph_metrics.py`, `llm_metrics.py` 等）：评估流程会针对不同指标拼接问题、候选答案、引用片段，且存在多层 fallback 逻辑，暂维持本地定义以避免引入大量模板分支。

## 核心API接口

### 数据库管理API (neo4jdb.py)

**全局访问点**：
```python
from graphrag_agent.config.neo4jdb import get_db_manager

# 获取全局单例实例
db_manager = get_db_manager()
```

**DBConnectionManager类方法**：

1. **execute_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame**
   - 执行Cypher查询并返回pandas DataFrame
   - 参数：
     - `cypher`：Cypher查询语句
     - `params`：查询参数字典
   - 返回：查询结果DataFrame

2. **get_driver() -> neo4j.Driver**
   - 获取Neo4j原生驱动实例
   - 用于需要直接操作驱动的场景

3. **get_graph() -> Neo4jGraph**
   - 获取LangChain Neo4j图实例
   - 用于与LangChain工具链集成

4. **get_session() -> neo4j.Session**
   - 从会话池获取或创建Neo4j会话
   - 需手动调用`release_session()`归还

5. **release_session(session: neo4j.Session) -> None**
   - 释放会话回连接池
   - 池满时自动关闭会话

6. **close() -> None**
   - 关闭所有会话和驱动
   - 释放所有数据库资源

7. **__enter__() / __exit__() -> DBConnectionManager**
   - 上下文管理器接口
   - 自动管理资源生命周期

### 配置访问API (settings.py)

**直接导入配置常量**：
```python
from graphrag_agent.config.settings import (
    # 路径配置
    BASE_DIR, PROJECT_ROOT, FILES_DIR, FILE_REGISTRY_PATH,

    # 知识图谱配置
    theme, entity_types, relationship_types,
    conflict_strategy, community_algorithm,

    # 文本处理
    CHUNK_SIZE, OVERLAP, MAX_TEXT_LENGTH,

    # 性能配置
    MAX_WORKERS, BATCH_SIZE, ENTITY_BATCH_SIZE,

    # Neo4j连接配置
    NEO4J_CONFIG,  # 字典：uri, username, password, max_pool_size, refresh_schema

    # LLM配置
    OPENAI_EMBEDDING_CONFIG,  # 字典：model, api_key, base_url
    OPENAI_LLM_CONFIG,        # 字典：model, temperature, max_tokens, api_key, base_url

    # 搜索配置
    LOCAL_SEARCH_SETTINGS,    # 字典
    GLOBAL_SEARCH_SETTINGS,   # 字典
    HYBRID_SEARCH_SETTINGS,   # 字典

    # 缓存配置
    CACHE_SETTINGS,           # 字典

    # 多智能体配置
    MULTI_AGENT_PLANNER_MAX_TASKS,
    MULTI_AGENT_AUTO_GENERATE_REPORT,
    # ... 其他20+多智能体配置项
)
```

**辅助函数**（内部使用，不建议外部调用）：
- `_get_env_int(key, default)`：获取整型环境变量
- `_get_env_float(key, default)`：获取浮点型环境变量
- `_get_env_bool(key, default)`：获取布尔型环境变量
- `_get_env_choice(key, choices, default)`：获取枚举型环境变量

### 提示模板API (config.prompts)

**直接导入模板字符串**：
```python
from graphrag_agent.config.prompts import (
    # 图谱构建模板
    system_template_build_graph,  # 实体关系抽取
    human_template_build_graph,   # 用户输入模板
    system_template_build_index,  # 实体去重
    user_template_build_index,    # 用户输入模板
    community_template,           # 社区摘要

    # 问答模板
    NAIVE_PROMPT,                 # NaiveRAG
    LC_SYSTEM_PROMPT,             # Local Search
    MAP_SYSTEM_PROMPT,            # Global Search Map
    REDUCE_SYSTEM_PROMPT,         # Global Search Reduce
    contextualize_q_system_prompt,# 问题上下文化
)
```

**模板使用示例**：
```python
# 格式化图谱构建模板
formatted_prompt = system_template_build_graph.format(
    entity_types=str(entity_types),
    relationship_types=str(relationship_types),
    tuple_delimiter="|",
    record_delimiter="##",
    completion_delimiter="<COMPLETE>"
)

# 格式化问答模板
formatted_prompt = NAIVE_PROMPT.format(response_type="多个段落")
```

### 推理提示API (config.prompts)

**导入常量和模板**：
```python
from graphrag_agent.config.prompts import (
    # 搜索标记
    BEGIN_SEARCH_QUERY,     # "<|begin_search_query|>"
    END_SEARCH_QUERY,       # "<|end_search_query|>"
    BEGIN_SEARCH_RESULT,    # "<|begin_search_result|>"
    END_SEARCH_RESULT,      # "<|end_search_result|>"
    MAX_SEARCH_LIMIT,       # 5

    # 推理模板
    REASON_PROMPT,                 # 主推理循环
    RELEVANT_EXTRACTION_PROMPT,    # 信息提取
    SUB_QUERY_PROMPT,              # 子查询分解
    FOLLOWUP_QUERY_PROMPT,         # 后续查询判断
    FINAL_ANSWER_PROMPT,           # 最终答案
)
```

**模板使用示例**：
```python
# 格式化信息提取模板
extraction_prompt = RELEVANT_EXTRACTION_PROMPT.format(
    prev_reasoning="之前的推理内容...",
    search_query="当前搜索查询",
    document="搜索到的文档内容"
)

# 格式化最终答案模板
answer_prompt = FINAL_ANSWER_PROMPT.format(
    query="用户问题",
    retrieved_content="检索内容",
    thinking_process="思考过程"
)
```

## 配置最佳实践

### 1. 环境变量配置优先级

配置加载顺序：
1. 环境变量（`.env`文件或系统环境变量）
2. `settings.py`中的默认值

**推荐配置方式**：
```bash
# .env 文件示例
# Neo4j连接
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_MAX_POOL_SIZE=20

# LLM配置
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=http://localhost:13000/v1
OPENAI_LLM_MODEL=gpt-4o
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large

# 性能调优
MAX_WORKERS=8
ENTITY_BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=128

# 实体消歧阈值
DISAMBIG_VECTOR_THRESHOLD=0.90
DISAMBIG_NIL_THRESHOLD=0.65
```

### 2. 知识图谱Schema配置

**修改实体和关系类型**（在`settings.py`中）：
```python
entity_types = [
    "学生类型",
    "奖学金类型",
    "处分类型",
    "部门",
    "学生职责",
    "管理规定",
]

relationship_types = [
    "申请",
    "评选",
    "违纪",
    "资助",
    "申诉",
    "管理",
    "权利义务",
    "互斥",
]
```

### 3. 多智能体配置调优

**关键配置项**：
```bash
# 规划阶段
MA_PLANNER_MAX_TASKS=6                  # 最大任务数
MA_ALLOW_UNCLARIFIED_PLAN=true          # 允许未澄清的计划
MA_STRICT_PLAN_SIGNAL=true              # 严格信号检测

# 执行阶段
MA_WORKER_EXECUTION_MODE=parallel       # sequential / parallel
MA_WORKER_MAX_CONCURRENCY=4             # 并发度

# 报告生成阶段
MA_AUTO_GENERATE_REPORT=true            # 自动生成报告
MA_ENABLE_MAPREDUCE=true                # 启用Map-Reduce
MA_MAPREDUCE_THRESHOLD=20               # Map-Reduce阈值（sections数）
MA_SECTION_MAX_EVIDENCE=8               # 每节最大证据数
MA_SECTION_MAX_CONTEXT_CHARS=800        # 每节最大上下文字符数
```

### 4. 缓存配置优化

```bash
# 缓存后端
CACHE_MEMORY_ONLY=false                 # 是否仅使用内存
CACHE_MAX_MEMORY_SIZE=200               # 内存缓存大小(MB)
CACHE_MAX_DISK_SIZE=2000                # 磁盘缓存大小(MB)
CACHE_THREAD_SAFE=true                  # 线程安全

# 向量相似度缓存
CACHE_ENABLE_VECTOR_SIMILARITY=true     # 启用向量相似度匹配
CACHE_SIMILARITY_THRESHOLD=0.90         # 相似度阈值
CACHE_MAX_VECTORS=10000                 # 最大向量数
CACHE_EMBEDDING_PROVIDER=sentence_transformer  # openai / sentence_transformer
```

### 5. 数据库连接管理

**推荐使用上下文管理器**：
```python
# 推荐：自动管理资源
with get_db_manager() as db:
    result = db.execute_query("MATCH (n:Entity) RETURN n LIMIT 10")
    # 退出时自动关闭连接

# 不推荐：需手动管理
db = get_db_manager()
result = db.execute_query("MATCH (n:Entity) RETURN n")
db.close()  # 必须手动调用
```

## 与其他模块的交互

### 图谱构建模块 (graph/)
- 导入`entity_types`、`relationship_types`定义Schema
- 使用`system_template_build_graph`、`system_template_build_index`构建图谱
- 通过`get_db_manager()`执行Cypher查询

### 搜索模块 (search/)
- 导入`LOCAL_SEARCH_SETTINGS`、`GLOBAL_SEARCH_SETTINGS`、`HYBRID_SEARCH_SETTINGS`
- 使用`LC_SYSTEM_PROMPT`、`MAP_SYSTEM_PROMPT`、`REDUCE_SYSTEM_PROMPT`生成答案
- 使用`NAIVE_PROMPT`进行简单检索

### Agent模块 (agents/)
- 导入`AGENT_SETTINGS`配置递归限制和流式输出阈值
- 使用`REASON_PROMPT`系列模板实现DeepResearchAgent
- 导入`MULTI_AGENT_*`配置Plan-Execute-Report架构

### 缓存管理模块 (cache_manager/)
- 导入`CACHE_SETTINGS`配置缓存后端
- 导入`CACHE_EMBEDDING_PROVIDER`、`CACHE_SENTENCE_TRANSFORMER_MODEL`配置向量相似度

### 社区检测模块 (community/)
- 导入`community_algorithm`选择检测算法（leiden / sllpa）
- 使用`community_template`生成社区摘要
- 导入`GDS_*`配置Graph Data Science参数
