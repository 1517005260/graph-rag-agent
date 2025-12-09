# GraphRAG + DeepSearch 实现与问答系统（Agent）构建

本项目聚焦于结合 **GraphRAG** 与 **私域 Deep Search** 的方式，实现可解释、可推理的智能问答系统，同时结合多 Agent 协作与知识图谱增强，构建完整的 RAG 智能交互解决方案。

> 💡 灵感来源于检索增强推理与深度搜索场景，探索 RAG 与 Agent 在未来应用中的结合路径。

## 🏠 项目架构图

**注：本项目已被[deepwiki](https://deepwiki.com/1517005260/graph-rag-agent)官方收录，有助于理解整体的项目代码和核心的工作原理**，另外还有类似的中文网址[zreadai](https://zread.ai/1517005260/graph-rag-agent/1-overview)

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjxNTYgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNC45NjE1NiAxMC4zOTk5SDIuMjQxNTZDMS44ODgxIDEwLjM5OTkgMS42MDE1NiAxMC42ODY0IDEuNjAxNTYgMTEuMDM5OVYxMy43NTk5QzEuNjAxNTYgMTQuMTEzNCAxLjg4ODEgMTQuMzk5OSAyLjI0MTU2IDE0LjM5OTlINC45NjE1NkM1LjMxNTAyIDE0LjM5OTkgNS42MDE1NiAxNC4xMTM0IDUuNjAxNTYgMTMuNzU5OVYxMS4wMzk5QzUuNjAxNTYgMTAuNjg2NCA1LjMxNTAyIDEwLjM5OTkgNC45NjE1NiAxMC4zOTk5WiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNMTMuNzU4NCAxLjYwMDFIMTEuMDM4NEMxMC42ODUgMS42MDAxIDEwLjM5ODQgMS44ODY2NCAxMC4zOTg0IDIuMjQwMVY0Ljk2MDFDMTAuMzk4NCA1LjMxMzU2IDEwLjY4NSA1LjYwMDEgMTEuMDM4NCA1LjYwMDFIMTMuNzU4NEMxNC4xMTE5IDUuNjAwMSAxNC4zOTg0IDUuMzEzNTYgMTQuMzk4NCA0Ljk2MDFWMi4yNDAxQzE0LjM5ODQgMS44ODY2NCAxNC4xMTE5IDEuNjAwMSAxMy43NTg0IDEuNjAwMVoiIGZpbGw9IiNmZmYiLz4KPHBhdGggZD0iTTQgMTJMMTIgNEw0IDEyWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0IiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPC9zdmc+Cg%3D%3D&logoColor=ffffff)](https://zread.ai/1517005260/graph-rag-agent)

由Claude生成

![svg](./assets/structure.svg)

## 📂 项目结构

```
graph-rag-agent/
├── graphrag_agent/         # 🎯 核心包 - GraphRAG智能体系统
│   ├── agents/             # 🤖 Agent模块 - 智能体实现
│   │   ├── base.py         # Agent基类
│   │   ├── graph_agent.py  # 基于图结构的Agent
│   │   ├── hybrid_agent.py # 混合搜索Agent
│   │   ├── naive_rag_agent.py # 简单向量检索Agent
│   │   ├── deep_research_agent.py # 深度研究Agent
│   │   ├── fusion_agent.py # Fusion GraphRAG Agent
│   │   └── multi_agent/    # Plan-Execute-Report 多智能体编排栈
│   │       ├── planner/    # 计划生成模块（澄清、任务分解、计划审校）
│   │       ├── executor/   # 执行协调模块（检索、研究、反思执行器）
│   │       ├── reporter/   # 报告生成模块（纲要、章节、一致性检查）
│   │       ├── core/       # 核心数据模型（PlanSpec、State、ExecutionRecord）
│   │       ├── tools/      # 工具组件（证据追踪、检索适配器）
│   │       └── integration/ # 集成层（工厂类、兼容门面）
│   ├── ai_copilot/          # 🤖 AI Copilot - 智能配置向导
│   │   └── document_analyzer.py # 文档分析与推荐引擎
│   ├── cache_manager/      # 📦 缓存管理模块
│   │   ├── manager.py      # 统一缓存管理器
│   │   ├── backends/       # 存储后端
│   │   ├── models/         # 数据模型
│   │   └── strategies/     # 缓存键生成策略
│   ├── community/          # 🔍 社区检测与摘要模块
│   │   ├── detector/       # 社区检测算法
│   │   └── summary/        # 社区摘要生成
│   ├── config/             # ⚙️ 配置模块
│   │   ├── neo4jdb.py      # 数据库连接管理
│   │   ├── prompts/        # 提示模板集合
│   │   ├── settings.py     # 全局配置
│   │   ├── graph_config_model.py    # 🆕 通用图谱配置模型
│   │   └── graph_config_storage.py  # 🆕 配置持久化存储
│   ├── evaluation/         # 📊 评估系统
│   │   ├── core/           # 评估核心组件
│   │   ├── metrics/        # 评估指标实现
│   │   └── test/           # 评估测试脚本
│   ├── graph/              # 📈 图谱构建模块
│   │   ├── core/           # 核心组件
│   │   ├── extraction/     # 实体关系提取
│   │   │   ├── entity_extractor.py # 实体提取器
│   │   │   └── extractor_factory.py # 🆕 提取器工厂（支持动态配置）
│   │   ├── indexing/       # 索引管理
│   │   └── processing/     # 实体处理
│   ├── integrations/       # 🔌 集成模块
│   │   └── build/          # 🏗️ 知识图谱构建
│   │       ├── main.py     # 构建入口
│   │       ├── build_graph.py # 基础图谱构建（已集成动态配置）
│   │       ├── build_index_and_community.py # 索引和社区构建
│   │       ├── build_chunk_index.py # 文本块索引构建
│   │       ├── incremental/ # 增量更新子模块
│   │       └── incremental_update.py # 增量更新管理
│   ├── models/             # 🧩 模型管理
│   │   └── get_models.py   # 模型初始化
│   ├── pipelines/          # 🔄 数据管道
│   │   └── ingestion/      # 📄 文档摄取处理器
│   │       ├── document_processor.py # 文档处理核心
│   │       ├── file_reader.py # 多格式文件读取
│   │       └── text_chunker.py # 文本分块
│   ├── prompts/            # 🆕 💬 动态提示词生成
│   │   └── dynamic_prompt_builder.py # 基于配置的提示词构建器
│   └── search/             # 🔎 搜索模块
│       ├── local_search.py # 本地搜索
│       ├── global_search.py # 全局搜索
│       └── tool/           # 搜索工具集
│           ├── naive_search_tool.py # 简单搜索
│           ├── deep_research_tool.py # 深度研究工具
│           └── reasoning/  # 推理组件
├── server/                 # 🖧 后端服务（独立服务）
│   ├── main.py             # FastAPI应用入口
│   ├── models/             # 数据模型
│   ├── routers/            # API路由
│   │   └── admin.py        # 🆕 管理路由（新增 AI Copilot API）
│   └── services/           # 业务逻辑
├── frontend/               # 🖥️ 前端界面（独立服务）
│   ├── app.py              # 应用入口（新增 AI 向导导航）
│   ├── components/         # UI组件
│   ├── page_components/    # 🆕 管理页面组件
│   │   ├── document_manager.py    # 📚 文档管理
│   │   ├── config_manager.py      # ⚙️ 配置管理（重构）
│   │   ├── build_manager.py       # 🏗️ 构建管理
│   │   └── ai_config_wizard.py    # 🆕 🤖 AI 配置向导
│   └── utils/              # 前端工具
├── test/                  # 🧪 测试模块
│   ├── search_with_stream.py # 流式输出测试
│   └── search_without_stream.py # 标准输出测试
├── assets/                 # 🖼️ 静态资源
│   ├── deepsearch.svg      # RAG演进图
│   └── start.md            # 快速开始文档
└── files/                  # 📁 原始数据文件
```

**此外，每个模块下都有单独的readme来介绍模块的功能**



## 🚀 相关资源

- [大模型推理能力不断增强，RAG 和 Agent 何去何从](https://www.bilibili.com/video/BV1i6RNYpEwV)
- [企业级知识图谱交互问答系统方案](https://www.bilibili.com/video/BV1U599YrE26)
- [Jean - 用国产大模型 + LangChain + Neo4j 建图全过程](https://zhuanlan.zhihu.com/p/716089164)
- [GraphRAG vs DeepSearch？GraphRAG 提出者给你答案](https://mp.weixin.qq.com/s/FOT4pkEPHJR8xFvcVk1YFQ)

![svg](./assets/deepsearch.svg)

## ✨ 项目亮点

### 核心功能

- **从零开始复现 GraphRAG**：完整实现了 GraphRAG 的核心功能，将知识表示为图结构
- **DeepSearch 与 GraphRAG 创新融合**：现有 DeepSearch 框架主要基于向量数据库，本项目创新性地将其与知识图谱结合
- **多 Agent 协同架构**：实现不同类型 Agent 的协同工作，提升复杂问题处理能力
- **完整评估系统**：提供 20+ 种评估指标，全方位衡量系统性能
- **增量更新机制**：支持知识图谱的动态增量构建与智能去重
- **实体质量提升**：实体消歧和对齐机制，有效解决实体歧义和重复问题
- **思考过程可视化**：展示 AI 的推理轨迹，提高可解释性和透明度

### 🆕 最新特性（v2.0）

- **🌐 通用图谱构建器**：不再限定于特定领域，支持用户自定义任意行业的知识图谱
  - 用户定义**领域（Domain）**和**桥接点（Bridge）**代替硬编码实体/关系类型
  - 桥接点机制：跨领域的公共概念连接器（如"问题类型"、"风险等级"）
  - 领域隔离：每个领域拥有独立的 Schema（实体类型、关系类型）

- **🤖 AI Copilot 智能配置向导**：AI 辅助配置生成，零基础快速上手
  - 自动分析文档集合（TF-IDF + K-Means 聚类）
  - LLM 驱动的领域和桥接点推荐
  - 提供法务、电商、医疗三个开箱即用的行业模板

- **🎨 全新前端管理界面**：直观的可视化配置管理
  - 📚 文档管理：上传、删除、自动构建
  - ⚙️ 配置管理：Domain/Bridge 可视化编辑
  - 🏗️ 构建管理：实时查看图谱构建进度和统计
  - 🤖 AI 向导：4步智能配置流程

- **💡 动态提示词生成**：根据用户配置自动生成 LLM 提示词
  - 无需修改代码即可适配不同行业
  - 支持传统模式与动态模式自动切换
  - 完全向后兼容

## 🏁 快速开始

### 方式一：传统模式（预定义配置）

请参考：[快速开始文档](./assets/start.md)

### 方式二：AI 向导模式（推荐新用户）

**1. 环境准备**
```bash
# 克隆项目
git clone https://github.com/1517005260/graph-rag-agent.git
cd graph-rag-agent

# 创建 Python 环境
conda create -n graphrag python==3.10
conda activate graphrag

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

**2. 启动服务**
```bash
# 启动 Neo4j
docker compose up -d

# 启动后端
python server/main.py

# 启动前端（新开终端）
streamlit run frontend/app.py
```

**3. 使用 AI 向导创建配置**

访问 `http://localhost:8501`，点击「🤖 AI 配置向导」：

1. **步骤 1: 上传文档**
   - 在「📚 文档管理」上传你的业务文档（至少 2 个，推荐 10+ 个）

2. **步骤 2: AI 分析**
   - 输入行业提示（如"法务"、"电商"、"医疗"）
   - 点击「🚀 开始分析」
   - AI 将自动：
     - 对文档进行聚类分析
     - 提取关键概念
     - 推荐桥接点和领域

3. **步骤 3: 查看推荐**
   - 查看 AI 推荐的桥接点和领域
   - 理解推荐理由

4. **步骤 4: 应用配置**
   - 输入项目名称
   - 一键创建配置

**4. 构建知识图谱**

访问「🏗️ 构建管理」，点击「🚀 完整构建」

**5. 开始问答**

访问「💬 智能问答」，开始与你的私域知识对话！

### 方式三：使用行业模板

访问「⚙️ 配置管理」，选择一个行业模板：
- 📋 **法务/信访**：法规库、案件库、经验库
- 🛍️ **电商客服**：产品手册、工单记录、售后政策
- 🏥 **医疗问诊**：医学文献、电子病历、处方记录

一键加载，立即开始！

## 🧰 功能模块

### 通用图谱构建器（New！）

- **用户定义配置模型**：
  - `BridgeDefinition`：桥接点定义（跨领域公共概念）
  - `DomainDefinition`：领域定义（业务子图）
  - `GraphConfig`：完整的多领域配置容器

- **配置方式**：
  - 🤖 **AI 向导**：自动分析文档并推荐配置
  - 📋 **行业模板**：法务、电商、医疗预置模板
  - 🆕 **手动创建**：完全自定义配置

- **配置管理**：
  - 可视化编辑桥接点和领域
  - 实时预览配置效果
  - 导出/导入 JSON 配置
  - 配置版本管理

### AI Copilot 智能向导（New！）

- **文档智能分析**：
  - TF-IDF 向量化 + K-Means 自动聚类
  - 关键概念提取（支持 1-3 词短语）
  - 代表性词汇识别

- **AI 推荐引擎**：
  - LLM 驱动的桥接点推荐
  - 基于聚类的领域推荐
  - Schema 自动生成（实体类型、关系类型）
  - 推荐理由解释

- **交互式优化**：
  - 支持用户反馈迭代优化
  - 配置一致性检查
  - 推荐质量评分

### 图谱构建与管理

- **多格式文档处理**：支持 TXT、PDF、MD、DOCX、DOC、CSV、JSON、YAML/YML 等格式
- **动态实体关系提取**：根据用户配置动态生成提示词进行提取
- **增量更新机制**：支持已有图谱上的动态更新，智能处理冲突
- **实体质量提升**：通过实体消歧和对齐提升实体准确性
  - **实体消歧（Entity Disambiguation）**：使用字符串召回、向量重排和NIL检测将mention映射到规范实体
  - **实体对齐（Entity Alignment）**：智能检测和解决同一canonical实体下的冲突，保留所有关系信息
- **社区检测与摘要**：自动识别知识社区并生成摘要，支持 Leiden 和 SLLPA 算法
- **一致性验证**：内置图谱一致性检查与修复机制

### GraphRAG 实现

- **多级检索策略**：支持本地搜索、全局搜索、混合搜索等多种模式
- **图谱增强上下文**：利用图结构丰富检索内容，提供更全面的知识背景
- **Chain of Exploration**：实现在知识图谱上的多步探索能力
- **社区感知检索**：根据知识社区结构优化搜索结果

### DeepSearch 融合

- **多步骤思考-搜索-推理**：支持复杂问题的分解与深入挖掘
- **证据链追踪**：记录每个推理步骤的证据来源，提高可解释性
- **思考过程可视化**：实时展示 AI 的推理轨迹
- **多路径并行搜索**：同时执行多种搜索策略，综合利用不同知识来源

### 多种 Agent 实现

- **NaiveRagAgent**：基础向量检索型 Agent，适合简单问题
- **GraphAgent**：基于图结构的 Agent，支持关系推理
- **HybridAgent**：混合多种检索方式的 Agent
- **DeepResearchAgent**：深度研究型 Agent，支持复杂问题多步推理
- **FusionGraphRAGAgent**：最先进的 Agent，采用 Plan-Execute-Report 多智能体协作架构，支持智能任务规划、并行执行和长文档生成

### 多智能体协作系统

基于 **Plan-Execute-Report** 模式的新一代多智能体架构（`agents/multi_agent/`）：

- **Planner（规划器）**：通过 Clarifier（澄清）、TaskDecomposer（任务分解）、PlanReviewer（计划审校）三个子组件生成结构化的 `PlanSpec`
- **WorkerCoordinator（执行协调器）**：根据计划信号调度不同类型的执行器（检索、研究、反思），并记录执行证据
- **Reporter（报告生成器）**：采用 Map-Reduce 模式，通过 OutlineBuilder（纲要生成）、SectionWriter（章节写作）、ConsistencyChecker（一致性检查）组装长文档报告
- **Legacy Facade（兼容层）**：提供与旧版协调器相同的 `process_query` 接口，实现平滑迁移

### 系统评估与监控

- **多维度评估**：包括答案质量、检索性能、图评估和深度研究评估
- **性能监控**：跟踪 API 调用耗时，优化系统性能
- **用户反馈机制**：收集用户对回答的评价，持续改进系统

### 前后端实现

- **Web 管理界面**（New！）：
  - 📚 **文档管理**：拖拽上传、批量删除、二次确认
  - ⚙️ **配置管理**：可视化编辑 Domain/Bridge、模板切换、配置导出
  - 🏗️ **构建管理**：实时进度、图谱统计、日志查看
  - 🤖 **AI 向导**：4步智能配置流程

- **流式响应**：支持 AI 生成内容的实时流式显示
- **交互式知识图谱**：提供 Neo4j 风格的图谱交互界面
- **调试模式**：开发者可查看执行轨迹和搜索过程
- **RESTful API**：完善的后端 API 设计，支持扩展开发
  - 新增 AI Copilot API 端点（文档分析、配置推荐、应用推荐）

## 🖥️ 简单演示

### 新版前端界面

#### 1. AI 配置向导

![AI Wizard](./assets/ai-wizard.png)

**步骤式引导流程**：
- 上传文档 → AI 分析 → 查看推荐 → 应用配置

#### 2. 配置管理

![Config Manager](./assets/config-manager.png)

**可视化配置编辑**：
- 桥接点管理
- 领域Schema定义
- 模板快速切换

#### 3. 文档管理

![Document Manager](./assets/document-manager.png)

**直观的文档操作**：
- 拖拽上传
- 自动构建
- 二次确认删除

#### 4. 构建管理

![Build Manager](./assets/build-manager.png)

**实时监控**：
- 构建进度可视化
- 图谱统计展示
- 详细日志输出

### 网页端演示（原有功能）

非调试模式下的问答：

![no-debug](./assets/web-nodebug.png)

调试模式下的问答（包含轨迹追踪（langgraph节点）、命中的知识图谱与文档源内容，知识图谱推理问答等）：

![debug1](./assets/web-debug1.png)

![debug2](./assets/web-debug2.png)

![debug3](./assets/web-debug3.png)

### 终端测试输出：

```bash
cd test/
python search_with_stream.py

# 本例为测试MultiAgent的输出，其他Agent可以在测试脚本中删除注释自行测试
# 额外配置：.env中MA_REFLECTION_ALLOW_RETRY = true
开始测试: 2025-10-24 14:22:13

===== 开始非流式Agent测试 =====


===== 测试查询: 优秀学生的申请条件是什么？ =====

[测试] FusionGraphRAGAgent - 查询: '优秀学生的申请条件是什么？'
[PlanSpec] 规划结果:
{
  "plan_id": "27737b8d-ae8b-460e-9e35-4fbc0c814ae5",
  "version": 1,
  "status": "draft",
  "tasks": [
    {
      "task_id": "task_001",
      "description": "检索优秀学生的定义和常见标准",
      "tool": "global_search",
      "parameters": {},
      "priority": 1,
      "depends_on": []
    },
    ...
  ]
}
...
```

可以看到，由于嵌入的相似性原因，LLM有概率会把"优秀学生"（学校的荣誉称号）近似为"国家奖学金"（称号≠奖学金），这个问题需要后续的微调embedding来解决。

## 🔄 代码更新与维护

### 日常启动流程

**首次部署后，每次使用只需：**

#### 方式 1: 快速启动脚本（推荐）

在项目根目录创建 `start.sh`：

```bash
#!/bin/bash

echo "🚀 启动 Graph-RAG-Agent..."

# 1. 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "📦 启动 Docker Desktop..."
    open -a Docker  # macOS 用户
    # Linux 用户使用: sudo systemctl start docker
    echo "⏳ 等待 Docker 启动 (30秒)..."
    sleep 30
fi

# 2. 启动 Neo4j
echo "🗄️  启动 Neo4j 数据库..."
docker compose up -d

# 3. 激活 Python 环境并启动后端
echo "🔧 启动后端服务..."
conda activate graphrag
python server/main.py &

# 4. 等待后端启动
sleep 5

# 5. 启动前端
echo "🎨 启动前端界面..."
streamlit run frontend/app.py
```

赋予执行权限并运行：
```bash
chmod +x start.sh
./start.sh
```

#### 方式 2: 手动启动

```bash
# 1. 确保 Docker Desktop 运行（查看菜单栏是否有 🐋 图标）
open -a Docker  # macOS
# sudo systemctl start docker  # Linux

# 2. 启动 Neo4j
docker compose up -d

# 3. 激活环境并启动后端（终端1）
conda activate graphrag
python server/main.py

# 4. 启动前端（新开终端2）
conda activate graphrag
streamlit run frontend/app.py

# 访问 http://localhost:8501
```

### 代码更新流程

执行 `git pull` 后，根据更新内容执行相应操作：

#### 自动检查更新影响

创建 `check_update.sh` 脚本：

```bash
#!/bin/bash

echo "🔍 检查代码更新影响..."

# 检查依赖变化
if git diff HEAD~1 HEAD --name-only | grep -q "requirements.txt"; then
    echo "⚠️  依赖文件已更新，需要执行："
    echo "   pip install -r requirements.txt --upgrade"
fi

# 检查配置变化
if git diff HEAD~1 HEAD --name-only | grep -q ".env.example\|settings.py\|graph_config"; then
    echo "⚠️  配置文件已更新，需要检查："
    echo "   diff .env .env.example"
    echo "   检查 graphrag_agent/config/ 目录下的新文件"
fi

# 检查图谱构建逻辑变化
if git diff HEAD~1 HEAD --name-only | grep -q "graphrag_agent/graph/\|graphrag_agent/community/\|graphrag_agent/prompts/"; then
    echo "⚠️  图谱相关代码已更新，建议："
    echo "   python graphrag_agent/integrations/build/main.py"
fi

# 检查 Docker 配置变化
if git diff HEAD~1 HEAD --name-only | grep -q "docker-compose.yaml"; then
    echo "⚠️  Docker 配置已更新，需要执行："
    echo "   docker compose down && docker compose up -d"
fi

# 显示具体变动
echo ""
echo "📝 具体变动文件："
git diff HEAD~1 HEAD --name-status
```

**使用方法：**
```bash
chmod +x check_update.sh
git pull
./check_update.sh
```

#### 快速更新流程（推荐）

适用于大多数情况的更新步骤：

```bash
# 1. 拉取最新代码
git pull

# 2. 更新 Python 依赖
conda activate graphrag
pip install -r requirements.txt --upgrade
pip install -e . --upgrade

# 3. 检查配置文件（手动对比 .env 和 .env.example）
diff .env .env.example
# 如有新增配置项，手动添加到 .env

# 4. 重启服务
docker compose restart
# 然后重启 Python 后端和前端服务

# 5. 测试验证
cd test/
python search_without_stream.py
```

#### 根据更新类型的详细操作

| 更新内容 | 判断方法 | 需要的操作 | 耗时 |
|---------|---------|-----------|------|
| **Python 依赖** | `git diff HEAD~1 HEAD requirements.txt` | `pip install -r requirements.txt --upgrade` | ~2分钟 |
| **配置模型** | `graph_config_model.py` 变化 | 检查现有配置兼容性，可能需要迁移 | ~10分钟 |
| **配置文件** | `.env.example` 或 `settings.py` 变化 | 手动对比更新 `.env` | ~5分钟 |
| **Docker 配置** | `docker-compose.yaml` 变化 | `docker compose down && docker compose up -d` | ~2分钟 |
| **搜索/Agent 逻辑** | `graphrag_agent/search/` 或 `agents/` 变化 | 仅需重启服务 | ~1分钟 |
| **动态提示词** | `prompts/dynamic_prompt_builder.py` 变化 | 重新构建图谱以应用新提示词 | ~数小时 |
| **实体抽取逻辑** | `graphrag_agent/graph/extraction/` 变化 | 增量更新图谱 | ~10-30分钟 |
| **配置方式变更** | 从传统模式切换到动态模式 | 使用 AI 向导或模板创建新配置 | ~5-15分钟 |

#### 何时需要重建知识图谱？

**需要完全重建：**
- ✅ 切换配置模式（传统 → 动态 或 反之）
- ✅ 修改领域或桥接点定义
- ✅ 更换行业模板
- ✅ 实体/关系抽取 Prompt 大幅修改
- ✅ 实体类型定义变化（`entity_types` 修改）
- ✅ 关系类型定义变化（`relationship_types` 修改）

**只需增量更新：**
- ⚠️ 实体抽取算法优化
- ⚠️ 新增文档到 `files/` 目录
- ⚠️ 动态提示词生成逻辑优化（不改变配置）

```bash
# 增量更新（推荐，更快）
python graphrag_agent/integrations/build/incremental_update.py --once

# 完全重建（仅在必要时）
python graphrag_agent/integrations/build/main.py
```

**无需重建：**
- ❌ 只改了搜索逻辑
- ❌ 只改了 Agent 推理逻辑
- ❌ 只改了前端 UI
- ❌ 只改了性能优化参数
- ❌ 只改了 AI Copilot 分析逻辑

### 配置文件管理

```bash
# 备份您的自定义配置
cp .env .env.backup
cp graph_config.json graph_config.json.backup  # 新增：图谱配置备份

# git pull 后如果配置冲突，可以还原
```

### 关闭系统

```bash
# 方式 1: 优雅关闭
# 在运行服务的终端按 Ctrl+C

# 方式 2: 停止 Neo4j 容器
docker compose down

# 方式 3: 停止 Docker Desktop（释放所有资源）
# macOS: 菜单栏图标 -> Quit Docker Desktop
# Linux: sudo systemctl stop docker
```

### Docker 自动启动（可选）

**macOS 用户：**
1. 打开 Docker Desktop
2. 进入 **Settings** → **General**
3. 勾选 **Start Docker Desktop when you log in**

**Linux 用户：**
```bash
sudo systemctl enable docker
```

## 🔮 未来规划

1. **自动化数据获取**：
   - 加入定时爬虫功能，替代当前的手动文档更新方式
   - 实现资源自动发现与增量爬取

2. **图谱构建优化**：
   - 采用 GRPO 训练小模型支持图谱抽取
   - 降低当前 DeepResearch 进行图谱抽取/Chain of Exploration的成本与延迟

3. **领域特化嵌入**：
   - 解决语义相近但概念不同的术语区分问题
   - 优化如"优秀学生"vs"国家奖学金"、"过失杀人"vs"故意杀人"等的嵌入区分

4. **AI Copilot 增强**：
   - 多轮对话式配置优化
   - 语义聚类代替 TF-IDF
   - 配置质量自动评分
   - 配置可视化（2D/3D 聚类展示）

5. 引入多模态rag等功能

## 🙏 参考与致谢

- [GraphRAG](https://github.com/microsoft/graphrag) – 微软开源的知识图谱增强 RAG 框架
- [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) – Neo4j 官方 LLM 建图工具
- [LightRAG](https://github.com/HKUDS/LightRAG) – 轻量级知识增强生成方案
- [deep-searcher](https://github.com/zilliztech/deep-searcher) – Zilliz团队开源的私域语义搜索框架
- [ragflow](https://github.com/infiniflow/ragflow) – 企业级 RAG 系统

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1517005260/graph-rag-agent&type=Date)](https://www.star-history.com/#1517005260/graph-rag-agent&Date)
