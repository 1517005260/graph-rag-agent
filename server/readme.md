# 知识图谱问答系统服务端

本模块是一个基于知识图谱的智能问答系统后端API实现。系统通过Neo4j数据库管理知识图谱数据，并使用FastAPI构建Web服务，支持聊天问答、知识推理查询、图谱可视化等功能。

## 项目结构

```
server/
├── main.py                   # FastAPI应用入口文件
├── models/                   # 数据模型定义
│   ├── __init__.py
│   └── schemas.py            # Pydantic模型定义，用于请求和响应校验
├── routers/                  # API路由定义
│   ├── __init__.py           # 总路由器，包含子路由
│   ├── chat.py               # 聊天相关API路由
│   ├── feedback.py           # 用户反馈API路由
│   ├── knowledge_graph.py    # 知识图谱操作API路由
│   ├── admin.py              # 🆕 管理API路由（文档/配置/构建管理，V2引擎集成）
│   └── source.py             # 源内容获取API路由
├── server_config/            # 服务器配置
│   ├── __init__.py
│   └── database.py           # 数据库连接管理
├── services/                 # 业务逻辑服务
│   ├── __init__.py
│   ├── agent_service.py      # Agent管理服务
│   ├── chat_service.py       # 聊天处理服务
│   └── kg_service.py         # 知识图谱服务
└── utils/                    # 工具函数
    ├── __init__.py
    ├── cache.py              # 缓存管理工具
    ├── concurrent.py         # 并发管理工具
    ├── keywords.py           # 关键词提取工具
    ├── neo4j_batch.py        # Neo4j批量处理工具
    └── performance.py        # 性能监控工具
```

## 核心实现思路

### 1. 基于FastAPI的路由结构

系统采用模块化设计，根据功能划分为不同的路由模块：
- `聊天`：处理用户的问答交互
- `反馈`：收集用户对回答的反馈信息
- `知识图谱`：提供图谱查询、操作和推理功能
- `源内容`：获取知识来源的原始内容
- `管理`：文档管理、配置管理、构建管理（集成V2增量更新引擎）

### 2. Agent系统设计

通过`AgentManager`类实现了多种智能Agent的统一管理：
- `graph_agent`：基于图谱的问答Agent
- `hybrid_agent`：混合模式Agent
- `naive_rag_agent`：基础检索增强生成Agent
- `deep_research_agent`：深度研究型Agent
- `fusion_agent`：融合图谱和RAG的Agent

系统可根据不同的问题类型选择合适的Agent进行处理。

### 3. 知识图谱服务

知识图谱服务(`kg_service.py`)提供了丰富的图谱操作功能：
- 图谱查询与可视化
- 从消息文本中提取知识图谱数据
- 实体间关系推理，包括最短路径、共同邻居、路径分析
- 实体影响范围分析
- 社区检测与分析

### 4. 并发与缓存管理

系统实现了高效的并发和缓存管理机制：
- `ConcurrentManager`：处理请求锁和超时清理
- `CacheManager`：管理查询结果缓存，提高响应速度
- 支持会话级和全局级缓存

### 5. 流式响应设计

支持通过Server-Sent Events (SSE)实现流式响应：
- `chat_stream`接口可发送增量更新的文本块
- 支持debug模式下的执行轨迹实时展示
- 提供思考过程和答案生成的分离展示

## 核心功能和函数

### 聊天处理

- `process_chat`：处理常规聊天请求，支持多种Agent类型和debug模式
- `process_chat_stream`：处理流式聊天请求，返回增量文本块
- `extract_iterations`：从深度研究过程中提取迭代轮次信息

### 知识图谱相关

- `extract_kg_from_message`：从问答结果中提取知识图谱
- `get_knowledge_graph`：获取图谱数据，支持查询和过滤
- `get_shortest_path`：查询实体间最短路径
- `get_entity_influence`：分析实体影响范围
- `get_simplified_community`：实现社区检测算法

### Agent管理

- `AgentManager.get_agent`：根据类型获取Agent实例
- `AgentManager.clear_history`：清除会话历史
- `format_execution_log`：格式化Agent执行日志

### 缓存和并发处理

- `CacheManager.set/get`：设置和获取缓存内容
- `ConcurrentManager.try_acquire_lock`：尝试获取锁，防止并发冲突
- `measure_performance`：性能测量装饰器，记录API执行时间

## 特色功能

1. **多Agent协同**：根据问题类型选择不同的Agent处理
2. **知识图谱推理**：支持多种关系推理算法和社区检测
3. **流式响应**：减少首次响应时间，提供更好的用户体验
4. **执行轨迹可视化**：在debug模式下展示详细的推理过程
5. **反馈学习**：通过用户反馈提升回答质量和缓存策略
6. **V2增量更新引擎集成**：Admin API集成了新一代增量更新引擎
   - L0/L1拆分架构：快速摄取 + 后台图谱构建
   - 实时进度推送：WebSocket广播构建进度和文件状态
   - 全量构建优化：清空数据库 + 重建图谱的完整流程
   - 增强的日志输出：便于调试和监控后台任务