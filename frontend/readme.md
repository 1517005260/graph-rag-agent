# 知识图谱问答系统前端

这个项目是一个基于知识图谱的智能问答系统的前端界面，使用Streamlit构建，提供了友好的用户交互体验。前端与服务端紧密配合，支持多种Agent类型的问答模式、知识图谱可视化、实时流式响应等功能。

## 项目结构

```
frontend/
├── app.py                          # 应用入口文件，初始化Streamlit应用
├── components/                     # UI组件模块
│   ├── __init__.py
│   ├── chat.py                     # 聊天界面组件
│   ├── debug.py                    # 调试面板组件
│   ├── knowledge_graph/            # 知识图谱相关组件
│   │   ├── __init__.py
│   │   ├── display.py              # 知识图谱展示组件
│   │   ├── interaction.py          # 知识图谱交互脚本
│   │   ├── kg_styles.py            # 知识图谱样式
│   │   ├── management.py           # 知识图谱管理组件
│   │   └── visualization.py        # 知识图谱可视化组件
│   ├── sidebar.py                  # 侧边栏组件
│   └── styles.py                   # 全局样式定义
├── frontend_config/                # 前端配置
│   ├── __init__.py
│   └── settings.py                 # 前端设置文件
└── utils/                          # 工具函数
    ├── __init__.py
    ├── api.py                      # API调用函数
    ├── helpers.py                  # 辅助函数
    ├── performance.py              # 性能监控工具
    └── state.py                    # 会话状态管理
```

## Demo 入口导航

| 目标 | 入口 | 说明 |
|---|---|---|
| 启动前端演示 | `streamlit run frontend/app.py` | 进入交互式问答与图谱可视化界面 |
| 启动后端 API | `uvicorn server.main:app --reload --port 8000` | 为前端提供聊天、图谱和反馈接口 |
| 查看流式终端输出 | `python test/search_with_stream.py` | 不经过网页，直接验证流式 Agent 输出 |
| 查看非流式问答结果 | `python test/search_without_stream.py` | 快速比较不同 Agent 的最终回答 |

建议先启动后端，再打开前端；如果前端页面无法加载回答，优先检查 `FRONTEND_API_URL` 或本地 API 地址是否与后端端口一致。

## 核心实现思路

### 1. 多模式的用户界面

系统实现了两种主要模式：
- **标准模式**：专注于聊天界面，提供流畅的问答体验
- **调试模式**：在聊天界面旁增加调试面板，显示执行轨迹、知识图谱和源内容

### 2. Agent类型选择

支持多种Agent类型，用户可以根据需求选择：
- `graph_agent`：使用图谱的局部与全局搜索
- `hybrid_agent`：混合搜索方式
- `naive_rag_agent`：传统向量检索增强生成
- `deep_research_agent`：深度研究型Agent，支持多轮推理
- `fusion_agent`：融合图谱和RAG的高级Agent

### 3. 流式响应设计

实现了流式响应功能，能够实时显示AI生成内容：
- 支持流式显示常规回答
- 支持流式显示思考过程（在deep_research_agent中）
- 通过SSE（Server-Sent Events）技术与后端通信

### 4. 知识图谱可视化

采用交互式知识图谱可视化：
- 使用pyvis实现Neo4j风格的图谱交互
- 支持双击节点聚焦、右键菜单操作
- 提供社区检测和关系推理功能

### 5. 会话状态管理

采用Streamlit的会话状态(session_state)管理应用状态：
- 维护用户会话ID、消息历史
- 处理调试模式和缓存
- 管理图谱显示设置和当前视图

## 核心功能和函数

### 聊天界面管理

- `display_chat_interface`：渲染主聊天界面，处理消息输入和显示
- `send_message`/`send_message_stream`：向后端发送消息并处理响应
- `send_feedback`：发送用户对回答的反馈

### 知识图谱功能

- `visualize_knowledge_graph`：将知识图谱数据可视化为交互式图表
- `display_knowledge_graph_tab`：展示知识图谱标签页，包括图谱显示和推理功能
- `get_kg_reasoning`：获取实体间的推理结果

### 调试功能

- `display_debug_panel`：显示包含多个标签页的调试面板
- `display_execution_trace_tab`：显示执行轨迹，针对不同Agent类型做专门处理
- `display_formatted_logs`：格式化显示深度研究Agent的迭代过程

### 会话状态管理

- `init_session_state`：初始化会话状态变量，设置默认值
- `clear_chat`：清除聊天历史和相关状态，重置会话

### 性能监控

- `monitor_performance`：性能监控装饰器，追踪API调用耗时
- `display_performance_stats`：显示性能统计信息和图表
- `PerformanceCollector`：收集和分析性能数据的类

## 特色功能

1. **多模式Agent选择**：支持不同类型的Agent，适应不同查询场景
2. **交互式知识图谱**：提供Neo4j风格的图谱交互，支持节点聚焦、社区检测
3. **深度研究模式**：支持思考过程可视化，展示AI的推理轨迹
4. **流式响应**：实时显示生成内容，提升用户体验
5. **反馈机制**：允许用户对回答提供反馈，提高系统质量
6. **知识图谱管理**：直接在UI中管理实体和关系
7. **性能监控**：提供详细的API性能监控和分析工具
