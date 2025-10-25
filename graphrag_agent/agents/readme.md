# Agent 模块

`agents` 模块是项目的核心交互层，负责整合各种搜索工具并提供用户交互接口。本模块实现了多种智能 Agent，支持从简单的向量检索到复杂的多智能体协同工作，为用户提供灵活、高效的知识检索和推理服务。

## 目录结构

```
graphrag_agent/agents/
├── __init__.py                # 模块初始化文件
├── base.py                    # Agent 基类，提供通用功能和接口
├── graph_agent.py             # 基于图结构的 Agent 实现
├── hybrid_agent.py            # 使用混合搜索的 Agent 实现
├── naive_rag_agent.py         # 使用简单向量检索的 Naive RAG Agent
├── deep_research_agent.py     # 使用深度研究工具的 Agent，支持多步推理
├── fusion_agent.py            # Fusion GraphRAG Agent，基于多智能体协作架构
└── multi_agent/               # Plan-Execute-Report 多智能体编排栈
    ├── planner/               # 规划器模块
    │   ├── base_planner.py    # 规划器基类
    │   ├── clarifier.py       # 澄清器：识别和澄清需求
    │   ├── task_decomposer.py # 任务分解器：将查询分解为子任务
    │   └── plan_reviewer.py   # 计划审校器：审核和优化计划
    ├── executor/              # 执行器模块
    │   ├── base_executor.py   # 执行器基类
    │   ├── retrieval_executor.py # 检索执行器：执行各类搜索任务
    │   ├── research_executor.py  # 研究执行器：执行深度研究任务
    │   ├── reflector.py       # 反思器：质量控制和改进建议
    │   └── worker_coordinator.py # 工作协调器：调度和管理执行器
    ├── reporter/              # 报告生成器模块
    │   ├── base_reporter.py   # 报告器基类
    │   ├── outline_builder.py # 纲要生成器
    │   ├── section_writer.py  # 章节写作器
    │   ├── consistency_checker.py # 一致性检查器
    │   ├── formatter.py       # 引用格式化器
    │   └── mapreduce/         # Map-Reduce报告生成
    │       ├── evidence_mapper.py   # 证据映射器
    │       ├── section_reducer.py   # 章节归约器
    │       └── report_assembler.py  # 报告组装器
    ├── core/                  # 核心数据模型
    │   ├── plan_spec.py       # PlanSpec：任务计划规范
    │   ├── state.py           # State：状态管理
    │   ├── execution_record.py # ExecutionRecord：执行记录
    │   └── retrieval_result.py # RetrievalResult：检索结果
    ├── tools/                 # 工具组件
    │   ├── evidence_tracker.py # 证据追踪器
    │   ├── retrieval_adapter.py # 检索适配器
    │   └── json_parser.py     # JSON解析器
    ├── integration/           # 集成层
    │   ├── multi_agent_factory.py # 多智能体工厂
    │   └── legacy_facade.py   # 兼容门面
    └── orchestrator.py        # 编排器：协调Planner-Executor-Reporter
```

## 实现思路

本模块基于 LangGraph 框架构建，采用状态图的方式组织 Agent 的工作流程，使用基类-子类的设计模式实现不同功能的 Agent。

### 基类设计 (BaseAgent)

`BaseAgent` 类提供了所有 Agent 共享的基础功能：

1. **缓存管理**：实现会话内和全局缓存，提高响应速度
2. **工作流定义**：基于 StateGraph 构建标准化工作流
3. **流式处理**：支持流式生成和响应
4. **性能监控**：跟踪工作流各节点的执行时间和资源消耗
5. **质量控制**：提供答案质量验证和反馈机制

```python
def _setup_graph(self):
    """设置工作流图 - 基础结构，子类可以通过_add_retrieval_edges自定义"""
    # 定义状态类型
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # 创建工作流图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", self._agent_node)
    workflow.add_node("retrieve", ToolNode(self.tools))
    workflow.add_node("generate", self._generate_node)
    
    # 添加从开始到Agent的边
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    
    # 添加从检索到生成的边 - 这个逻辑由子类实现
    self._add_retrieval_edges(workflow)
    
    # 从生成到结束
    workflow.add_edge("generate", END)
    
    # 编译图
    self.graph = workflow.compile(checkpointer=self.memory)
```

### 多样化 Agent 实现

模块提供了多种 Agent 实现，满足不同场景的需求：

1. **GraphAgent**：基于图结构的 Agent，利用图数据库执行本地和全局搜索，支持 reduce 操作
   
2. **HybridAgent**：使用混合搜索的 Agent，结合低级实体详情和高级主题概念
   
3. **NaiveRagAgent**：最简单的实现，仅使用向量检索的轻量级 Agent
   
4. **DeepResearchAgent**：使用深度研究工具实现多步骤思考-搜索-推理的 Agent，支持显示思考过程
   
5. **FusionGraphRAGAgent**：最复杂的实现，基于多 Agent 协作架构，集成多种搜索策略和知识融合方法

### 多智能体协作系统 (FusionGraphRAG)

`FusionGraphRAGAgent` 通过 `MultiAgentFacade` 将查询交给新的 Plan-Execute-Report 栈执行。该栈由 `agents.multi_agent` 模块提供，核心组件包括：

#### 1. Planner（规划器） - `planner/`

通过三个子组件生成结构化的 `PlanSpec`：

- **Clarifier（澄清器）** - `clarifier.py`：识别查询中的歧义和未明确需求，生成澄清问题
- **TaskDecomposer（任务分解器）** - `task_decomposer.py`：将复杂查询分解为多个子任务，构建任务依赖图（TaskGraph）
- **PlanReviewer（计划审校器）** - `plan_reviewer.py`：审核任务计划的合理性，优化任务顺序和参数

**输出**：`PlanSpec`，包含问题陈述（ProblemStatement）、任务图（TaskGraph）、验收标准（AcceptanceCriteria）

#### 2. Executor（执行器） - `executor/`

根据计划信号调度不同类型的执行器：

- **WorkerCoordinator（工作协调器）** - `worker_coordinator.py`：调度和管理各类执行器
- **RetrievalExecutor（检索执行器）** - `retrieval_executor.py`：执行 local_search、global_search、hybrid_search、chain_exploration 等检索任务
- **ResearchExecutor（研究执行器）** - `research_executor.py`：执行 deep_research、deeper_research 等深度研究任务
- **Reflector（反思器）** - `reflector.py`：对执行结果进行质量评估，提供改进建议

**特性**：
- 支持串行（sequential）、并行（parallel）、自适应（adaptive）三种执行模式
- 记录每个任务的执行证据（ExecutionRecord）和元数据
- 通过 EvidenceTracker 统一管理所有检索结果
- 支持反思与自动重试机制

#### 3. Reporter（报告生成器） - `reporter/`

采用 Map-Reduce 模式生成结构化长文档：

- **OutlineBuilder（纲要生成器）** - `outline_builder.py`：基于任务计划和证据生成报告大纲
- **SectionWriter（章节写作器）** - `section_writer.py`：为每个章节编写内容，支持传统模式和 Map-Reduce 模式
- **EvidenceMapper（证据映射器）** - `mapreduce/evidence_mapper.py`：将大量证据分批映射为摘要
- **SectionReducer（章节归约器）** - `mapreduce/section_reducer.py`：将证据摘要归约为连贯的章节文本，支持 collapse、refine、tree 三种归约策略
- **ReportAssembler（报告组装器）** - `mapreduce/report_assembler.py`：组装最终报告，添加引言和结论
- **ConsistencyChecker（一致性检查器）** - `consistency_checker.py`：检查报告内容与证据的一致性
- **CitationFormatter（引用格式化器）** - `formatter.py`：生成标准化的引用列表

**特性**：
- 支持报告级缓存和章节级缓存，减少重复计算
- 证据数量超过阈值（默认 20 条）时自动启用 Map-Reduce 模式
- 支持并行 Map 操作，提高生成速度
- 章节分批写作，处理超长内容

#### 4. Integration（集成层） - `integration/`

- **MultiAgentFactory（多智能体工厂）** - `multi_agent_factory.py`：提供便捷的组件创建和配置，返回 OrchestratorBundle
- **MultiAgentFacade（兼容门面）** - `legacy_facade.py`：封装 Plan-Execute-Report 流程，提供与旧版协调器相同的 `process_query` 接口，方便逐步迁移

`FusionGraphRAGAgent` 在常规查询场景下调用 `MultiAgentFacade`，同时保留深度研究工具等扩展能力用于高复杂度任务。

#### 5. Core（核心数据模型） - `core/`

- **PlanSpec（计划规范）** - `plan_spec.py`：定义任务计划、TaskGraph、PlanExecutionSignal 等
- **ExecutionRecord（执行记录）** - `execution_record.py`：记录任务执行过程、工具调用和证据
- **RetrievalResult（检索结果）** - `retrieval_result.py`：标准化的检索结果格式，包含粒度、证据、元数据、来源和相关度分数
- **State（状态管理）** - `state.py`：PlanExecuteState、PlanContext、ExecutionContext、ReportContext 等状态对象

#### 6. Tools（工具组件） - `tools/`

- **EvidenceTracker（证据追踪器）** - `evidence_tracker.py`：统一管理检索结果，按来源与粒度去重，优先保留高分证据
- **RetrievalAdapter（检索适配器）** - `retrieval_adapter.py`：适配不同搜索工具接口，标准化搜索结果格式
- **JSONParser（JSON 解析器）** - `json_parser.py`：安全解析 LLM 输出的 JSON 内容

详细的多智能体系统文档请参考 `multi_agent/readme.md`。

### 流式处理支持

所有 Agent 都实现了异步流式处理，提供更好的用户体验：

```python
async def ask_stream(self, query: str, thread_id: str = "default", 
                     recursion_limit: int = 5, show_thinking: bool = False) -> AsyncGenerator[str, None]:
    """
    向Agent提问，返回流式响应
    """
    # 检查缓存
    fast_result = self.check_fast_cache(query, thread_id)
    if fast_result:
        # 缓存命中，分块返回
        # ...
        return
            
    # 根据是否显示思考过程决定调用哪个流式方法
    if show_thinking:
        # 使用工具的流式思考接口
        async for chunk in self.research_tool.thinking_stream(query):
            # 处理思考过程或最终答案
            # ...
    else:
        # 普通搜索，仅返回最终答案
        async for chunk in self.research_tool.search_stream(query):
            yield chunk
```

## GraphRAG 与 Fusion GraphRAG

本模块实现了从基础 GraphRAG 到 Fusion GraphRAG 的演进：

### 1. 基础 GraphRAG (GraphAgent)

最初的 GraphRAG 实现使用图数据库存储和检索知识，支持本地和全局搜索：

```python
def _add_retrieval_edges(self, workflow):
    """添加从检索到生成的边"""
    # 添加 reduce 节点
    workflow.add_node("reduce", self._reduce_node)
    
    # 添加条件边，根据文档评分决定路由
    workflow.add_conditional_edges(
        "retrieve",
        self._grade_documents,
        {
            "generate": "generate", 
            "reduce": "reduce"
        }
    )
```

### 2. Fusion GraphRAG (FusionGraphRAGAgent)

Fusion GraphRAG 扩展了基础 GraphRAG，通过多 Agent 协作架构实现更强大的功能：

1. **Plan-Execute-Report 架构**：采用三阶段流程，实现任务规划、并行执行和结构化报告生成
2. **社区感知**：利用社区检测算法（Leiden/SLLPA），识别知识的聚类结构
3. **Chain of Exploration**：从起始实体出发，自主探索图谱发现关联知识
4. **多路径搜索**：同时执行多种搜索策略（local/global/hybrid/chain），全面覆盖知识空间
5. **证据链跟踪**：通过 EvidenceTracker 跟踪每个推理步骤使用的证据，提高可解释性
6. **反思与重试**：基于 AnswerValidationTool 自动质量评估，支持失败任务回滚重试
7. **Map-Reduce 写作**：处理大量证据时采用并行 Map 和多策略 Reduce，生成结构化长文档

```python
class FusionGraphRAGAgent(BaseAgent):
    """
    Fusion GraphRAG Agent

    基于多 Agent 协作架构的增强型 GraphRAG Agent，集成了多种搜索策略和知识融合方法。
    提供图谱感知、社区结构、Chain of Exploration 等高级功能，实现更深度的知识检索和推理。

    核心特性：
    - 通过 MultiAgentFacade 调用 Plan-Execute-Report 多智能体栈
    - 支持任务依赖管理、证据追踪与去重、反思与重试
    - 支持传统模式和 Map-Reduce 模式的报告生成
    - 支持报告级和章节级缓存
    """

    def __init__(self):
        # 设置缓存目录
        self.cache_dir = "./cache/fusion_graphrag"

        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)

        # 创建多智能体编排入口
        self.multi_agent = MultiAgentFacade(cache_manager=self.cache_manager)
```

## 使用场景

不同的 Agent 适用于不同的使用场景：

1. **NaiveRagAgent**：适用于简单查询，资源受限环境
2. **GraphAgent**：适用于需要结构化知识的查询，如关系查询
3. **HybridAgent**：平衡低级细节与高级概念，适用于一般性问答
4. **DeepResearchAgent**：适用于复杂问题，需要多步推理
5. **FusionGraphRAGAgent**：适用于最复杂的问题，需要多角度分析和深度探索
