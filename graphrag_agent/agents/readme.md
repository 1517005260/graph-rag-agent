# Agent 模块

`agent` 模块是项目的核心交互层，负责整合各种搜索工具并提供用户交互接口。本模块实现了多种智能 Agent，支持从简单的向量检索到复杂的多 Agent 协同工作，为用户提供灵活、高效的知识检索和推理服务。

## 目录结构

```
agent/
├── __init__.py                # 模块初始化文件
├── base.py                    # Agent 基类，提供通用功能和接口
├── graph_agent.py             # 基于图结构的 Agent 实现
├── hybrid_agent.py            # 使用混合搜索的 Agent 实现
├── naive_rag_agent.py         # 使用简单向量检索的 Naive RAG Agent
├── deep_research_agent.py     # 使用深度研究工具的 Agent，支持多步推理
├── fusion_agent.py            # Fusion GraphRAG Agent，基于多 Agent 协作架构
└── agent_coordinator.py       # 多 Agent 协作系统协调器
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

### 多 Agent 协作系统 (FusionGraphRAG)

`FusionGraphRAGAgent` 和 `GraphRAGAgentCoordinator` 实现了多 Agent 协作的高级架构。这种方法将搜索、推理、合成等任务分配给专门的 Agent 处理，大大提高了系统的灵活性和性能。

#### 1. 协调器设计

协调器负责管理多个专用 Agent，协调它们的工作流程：

```python
def __init__(self, llm=None):
    # 初始化语言模型
    self.llm = llm or get_llm_model()
    self.stream_llm = get_stream_llm_model()
    self.embeddings = get_embeddings_model()
    
    # 创建专用Agent
    self.retrieval_planner = self._create_retrieval_planner()
    self.local_searcher = self._create_local_searcher()
    self.global_searcher = self._create_global_searcher()
    self.explorer = self._create_explorer()
    self.chain_explorer = self._create_chain_explorer()
    self.synthesizer = self._create_synthesizer()
    self.thinking_engine = self._create_thinking_engine()
```

#### 2. 检索计划生成器

该组件分析查询，创建最佳检索计划，决定使用哪些搜索策略以及它们的优先级：

```python
def plan(self, query: str) -> Dict[str, Any]:
    """分析查询并生成检索计划"""
    prompt = f"""
    分析以下查询，创建一个全面的检索计划以获取所需信息。
    
    查询: "{query}"
    
    请考虑:
    1. 查询的复杂度和所需的检索深度
    2. 可能涉及的知识领域和关键实体
    3. 是否需要全局概览或具体细节
    4. 需要进行的探索步骤
    5. 是否需要追踪实体间的关系路径
    6. 查询是否涉及时间信息
    ...
    """
```

#### 3. 多路径执行

协调器根据检索计划并行执行多种搜索策略，包括本地搜索、全局搜索、深度探索和链式探索：

```python
# 按优先级排序任务
tasks = sorted(retrieval_plan.get("tasks", []), 
             key=lambda x: x.get("priority", 3), 
             reverse=True)

for task in tasks:
    task_type = task.get("type", "")
    task_query = task.get("query", query)
    
    # 根据任务类型执行不同的搜索
    if task_type == "local_search":
        # 执行本地搜索...
    elif task_type == "global_search":
        # 执行全局搜索...
    elif task_type == "exploration":
        # 执行深度探索...
    elif task_type == "chain_exploration":
        # 执行Chain of Exploration...
```

#### 4. 结果合成

最后，协调器使用合成器 Agent 整合所有检索结果，生成最终答案：

```python
def synthesize(self, query: str, results: Dict[str, List], plan: Dict[str, Any],
             thinking_process: str = None) -> str:
    """整合结果并生成答案"""
    # 构建提示
    prompt = f"""
    基于以下检索结果，回答用户的问题。
    
    用户问题: "{query}"
    
    ## 检索计划
    {json.dumps(plan, ensure_ascii=False, indent=2)}
    """
    
    # 添加各类检索结果...
    
    # 生成最终答案
    response = self.llm.invoke(prompt)
    return response.content
```

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

1. **社区感知**：利用社区检测算法，识别知识的聚类结构
2. **Chain of Exploration**：从起始实体出发，自主探索图谱发现关联知识
3. **多路径搜索**：同时执行多种搜索策略，全面覆盖知识空间
4. **证据链跟踪**：跟踪每个推理步骤使用的证据，提高可解释性

```python
class FusionGraphRAGAgent(BaseAgent):
    """
    Fusion GraphRAG Agent
    
    基于多Agent协作架构的增强型GraphRAGAgent，集成了多种搜索策略和知识融合方法。
    提供图谱感知、社区结构、Chain of Exploration等高级功能，实现更深度的知识检索和推理。
    """
    
    def __init__(self):
        # 设置缓存目录
        self.cache_dir = "./cache/fusion_graphrag"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)
        
        # 创建协调器
        self.coordinator = GraphRAGAgentCoordinator(self.llm)
```

## 使用场景

不同的 Agent 适用于不同的使用场景：

1. **NaiveRagAgent**：适用于简单查询，资源受限环境
2. **GraphAgent**：适用于需要结构化知识的查询，如关系查询
3. **HybridAgent**：平衡低级细节与高级概念，适用于一般性问答
4. **DeepResearchAgent**：适用于复杂问题，需要多步推理
5. **FusionGraphRAGAgent**：适用于最复杂的问题，需要多角度分析和深度探索