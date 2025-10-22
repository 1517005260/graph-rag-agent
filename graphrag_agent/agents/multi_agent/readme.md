# Multi-Agent Plan-Execute-Report 架构

多智能体编排栈是新一代智能体协作架构，采用 Plan-Execute-Report 模式，实现复杂查询的智能化任务规划、并行执行和结构化报告生成。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                      FusionGraphRAGAgent                     │
│                  (通过 MultiAgentFacade 调用)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                       Orchestrator                           │
│            (协调 Planner → Executor → Reporter)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌────────┐    ┌──────────┐   ┌─────────┐
   │Planner │    │ Executor │   │Reporter │
   └────────┘    └──────────┘   └─────────┘
        │              │              │
        │              │              │
        ▼              ▼              ▼
    PlanSpec    ExecutionRecords  ReportResult
```

## 核心组件

### 1. Planner（规划器）

负责将用户查询转换为结构化的执行计划。

#### 子组件

- **Clarifier（澄清器）** - `planner/clarifier.py`
  - 识别查询中的歧义和不明确需求
  - 生成澄清问题列表
  - 基于用户反馈优化查询

- **TaskDecomposer（任务分解器）** - `planner/task_decomposer.py`
  - 将复杂查询分解为多个子任务
  - 构建任务依赖图（TaskGraph）
  - 支持的任务类型：
    - `local_search`: 本地搜索
    - `global_search`: 全局搜索
    - `hybrid_search`: 混合搜索
    - `naive_search`: 简单向量搜索
    - `deep_research`: 深度研究
    - `deeper_research`: 更深度研究
    - `chain_exploration`: 链式探索
    - `reflection`: 反思
    - `custom`: 自定义任务

- **PlanReviewer（计划审校器）** - `planner/plan_reviewer.py`
  - 审核任务计划的合理性
  - 优化任务顺序和参数
  - 生成最终的 PlanSpec

#### 输出数据模型

**PlanSpec**（计划规范）- `core/plan_spec.py`

```python
class PlanSpec:
    plan_id: str                      # 计划唯一标识
    version: int                      # 版本号
    problem_statement: ProblemStatement  # 问题陈述
    assumptions: List[str]            # 假设前提
    task_graph: TaskGraph             # 任务依赖图
    acceptance_criteria: AcceptanceCriteria  # 验收标准
    status: str                       # 计划状态
```

**TaskGraph**（任务图）

```python
class TaskGraph:
    nodes: List[TaskNode]             # 任务节点列表
    execution_mode: str               # sequential/parallel/adaptive

    def validate_dependencies() -> bool  # 验证依赖关系
    def get_ready_tasks() -> List[TaskNode]  # 获取可执行任务
    def topological_sort() -> List[TaskNode]  # 拓扑排序
```

### 2. Executor（执行器）

负责执行任务计划中的各个任务。

#### 执行器类型

- **RetrievalExecutor（检索执行器）** - `executor/retrieval_executor.py`
  - 执行各类搜索任务（local/global/hybrid/naive）
  - 使用 RetrievalAdapter 适配不同的搜索工具
  - 将搜索结果标准化为 RetrievalResult

- **ResearchExecutor（研究执行器）** - `executor/research_executor.py`
  - 执行深度研究任务（deep_research/chain_exploration）
  - 支持多步推理和证据链追踪
  - 记录思考过程和中间结果

- **Reflector（反思器）** - `executor/reflector.py`
  - 对执行结果进行质量评估
  - 识别问题和不足
  - 提供改进建议

- **WorkerCoordinator（工作协调器）** - `executor/worker_coordinator.py`
  - 调度和管理各类执行器
  - 支持三种执行模式：
    - `sequential`: 串行执行，严格按依赖顺序
    - `parallel`: 并行执行，最大化并发
    - `adaptive`: 自适应，根据任务特性动态调整
  - 记录执行证据和元数据

#### 核心工具

- **EvidenceTracker（证据追踪器）** - `tools/evidence_tracker.py`
  - 统一管理所有检索结果
  - 支持按类型、来源、粒度查询证据
  - 自动去重和排序

- **RetrievalAdapter（检索适配器）** - `tools/retrieval_adapter.py`
  - 适配不同的搜索工具接口
  - 标准化搜索结果格式
  - 支持 LocalSearch、GlobalSearch、HybridSearch 等

#### 输出数据模型

**ExecutionRecord**（执行记录）- `core/execution_record.py`

```python
class ExecutionRecord:
    task_id: str                      # 任务ID
    task_type: str                    # 任务类型
    status: str                       # 执行状态
    start_time: datetime              # 开始时间
    end_time: datetime                # 结束时间
    evidence: List[RetrievalResult]   # 检索到的证据
    metadata: Dict[str, Any]          # 元数据
    error_message: Optional[str]      # 错误信息
```

**RetrievalResult**（检索结果）- `core/retrieval_result.py`

```python
class RetrievalResult:
    result_id: str                    # 结果唯一ID
    evidence: Union[str, Dict]        # 证据内容
    granularity: str                  # 粒度 (chunk/entity/community/document)
    source: str                       # 来源
    score: float                      # 相关度分数
    metadata: RetrievalMetadata       # 元数据
```

### 3. Reporter（报告生成器）

负责将执行结果整合为结构化的长文档报告。

#### 报告生成流程

1. **OutlineBuilder（纲要生成）** - `reporter/outline_builder.py`
   - 基于任务计划和证据生成报告大纲
   - 确定章节结构和层次关系
   - 为每个章节分配相关证据

2. **章节写作** - `reporter/section_writer.py`
   - 传统模式：直接基于证据生成章节内容
   - Map-Reduce 模式：大量证据时启用
     - **EvidenceMapper** - `reporter/mapreduce/evidence_mapper.py`
       - 将证据分批映射为摘要
       - 支持并行 Map 操作
     - **SectionReducer** - `reporter/mapreduce/section_reducer.py`
       - 将证据摘要归约为连贯文本
       - 支持线性归约和树形归约策略
     - **ReportAssembler** - `reporter/mapreduce/report_assembler.py`
       - 组装最终报告
       - 添加引言和结论

3. **质量保障**
   - **ConsistencyChecker（一致性检查）** - `reporter/consistency_checker.py`
     - 检查报告内容与证据的一致性
     - 识别矛盾和不支持的陈述
     - 评估整体质量
   - **CitationFormatter（引用格式化）** - `reporter/formatter.py`
     - 生成标准化的引用列表
     - 支持多种引用格式

#### 报告缓存机制

Reporter 实现了两级缓存：

1. **报告级缓存**
   - 基于 `plan_id:version:report_type` 生成 `report_id`
   - 使用证据指纹（evidence_fingerprint）检测证据是否变化
   - 证据未变化时直接返回缓存的完整报告

2. **章节级缓存**
   - 缓存每个章节的内容和使用的证据ID
   - 章节标题、摘要和证据指纹均未变化时可复用
   - 支持部分章节复用，其他章节重新生成

#### 输出数据模型

**ReportResult**（报告结果）- `reporter/base_reporter.py`

```python
class ReportResult:
    outline: ReportOutline            # 报告纲要
    sections: List[SectionContent]    # 章节内容列表
    final_report: str                 # 最终报告（Markdown）
    references: Optional[str]         # 引用列表
    consistency_check: Optional[ConsistencyCheckResult]  # 一致性检查结果
```

**ReportOutline**（报告纲要）- `reporter/outline_builder.py`

```python
class ReportOutline:
    title: str                        # 报告标题
    abstract: str                     # 摘要
    sections: List[SectionOutline]    # 章节大纲
    report_type: str                  # 报告类型
```

### 4. Core（核心数据模型）

#### State（状态管理）- `core/state.py`

**PlanExecuteState**（计划执行状态）

```python
class PlanExecuteState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    input: str                        # 用户输入
    plan: Optional[PlanSpec]          # 任务计划
    plan_context: Optional[PlanContext]  # 计划上下文
    execution_records: List[ExecutionRecord]  # 执行记录
    execution_context: Optional[ExecutionContext]  # 执行上下文
    report_context: Optional[ReportContext]  # 报告上下文
    response: str                     # 最终响应
```

**PlanContext**（计划上下文）

```python
class PlanContext(BaseModel):
    original_query: str               # 原始查询
    refined_query: Optional[str]      # 优化后的查询
    clarifications: List[str]         # 澄清问题
    user_confirmations: Dict[str, str]  # 用户确认
    domain_context: Optional[str]     # 领域上下文
    user_preferences: Dict[str, Any]  # 用户偏好
```

**ExecutionContext**（执行上下文）

```python
class ExecutionContext(BaseModel):
    completed_task_ids: List[str]     # 已完成任务ID
    failed_task_ids: List[str]        # 失败任务ID
    task_results: Dict[str, Any]      # 任务结果
    evidence_registry: Dict[str, Any]  # 证据注册表
```

**ReportContext**（报告上下文）

```python
class ReportContext(BaseModel):
    report_id: str                    # 报告ID
    report_type: str                  # 报告类型
    outline: Dict[str, Any]           # 大纲
    section_drafts: Dict[str, str]    # 章节草稿
    citations: List[Dict[str, Any]]   # 引用
    consistency_check_results: Optional[Dict[str, Any]]  # 一致性检查结果
    cache_hit: bool                   # 是否命中缓存
```

### 5. Integration（集成层）

#### MultiAgentFactory（工厂类）- `integration/multi_agent_factory.py`

提供便捷的组件创建和配置：

```python
class MultiAgentFactory:
    @staticmethod
    def create_default_bundle(cache_manager=None) -> OrchestratorBundle:
        """创建默认的编排器组件包"""
        # 返回包含 Planner, Executor, Reporter, Orchestrator 的完整组件包
```

**OrchestratorBundle**

```python
class OrchestratorBundle:
    planner: BasePlanner              # 规划器
    worker_coordinator: WorkerCoordinator  # 执行协调器
    reporter: BaseReporter            # 报告生成器
    orchestrator: Orchestrator        # 编排器
```

#### Legacy Facade（兼容层）- `integration/legacy_facade.py`

**MultiAgentFacade**

提供与旧版协调器兼容的接口：

```python
class MultiAgentFacade:
    def process_query(
        self,
        query: str,
        *,
        assumptions: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
        extra_messages: Optional[Iterable[HumanMessage]] = None,
    ) -> Dict[str, Any]:
        """
        执行多智能体流程，返回结构化结果

        返回格式:
        {
            "status": "completed" | "failed",
            "response": "最终报告文本",
            "planner": {...},           # 规划器输出
            "execution_records": [...], # 执行记录
            "report": {...},            # 报告详情
            "report_context": {...},    # 报告上下文
            "errors": [...],            # 错误信息
            "metrics": {...}            # 性能指标
        }
        """
```

### 6. Orchestrator（编排器）

**Orchestrator** - `orchestrator.py`

协调 Planner → Executor → Reporter 的完整流程：

```python
class Orchestrator:
    def run(
        self,
        state: PlanExecuteState,
        *,
        assumptions: Optional[List[str]] = None,
        report_type: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        执行完整的 Plan-Execute-Report 流程

        流程:
        1. 调用 Planner 生成 PlanSpec
        2. 调用 WorkerCoordinator 执行任务
        3. 调用 Reporter 生成报告
        4. 收集指标和错误信息
        """
```

**OrchestratorResult**

```python
class OrchestratorResult(BaseModel):
    status: str                       # completed | failed
    planner: Optional[PlanSpec]       # 规划结果
    execution_records: List[ExecutionRecord]  # 执行记录
    report: Optional[ReportResult]    # 报告结果
    errors: List[str]                 # 错误列表
    metrics: OrchestratorMetrics      # 性能指标
```

## 配置选项

### PlannerConfig（规划器配置）

```python
class PlannerConfig(BaseModel):
    max_tasks: int = 6                # 最大任务数
    allow_unclarified_plan: bool = True  # 是否允许未完全澄清的计划
    default_domain: str = "通用"       # 默认领域
```

### ReporterConfig（报告器配置）

```python
class ReporterConfig(BaseModel):
    default_report_type: str = "long_document"  # 默认报告类型
    citation_style: str = "default"   # 引用格式
    max_evidence_summary: int = 30    # 纲要生成时展示的最大证据条数
    enable_consistency_check: bool = True  # 是否启用一致性检查
    enable_mapreduce: bool = True     # 是否启用 Map-Reduce 模式
    reduce_strategy: str = "tree"     # 归约策略 (linear/tree)
    max_tokens_per_reduce: int = 4000 # Reduce 阶段最大 token 数
    enable_parallel_map: bool = True  # 是否启用并行 Map
    mapreduce_evidence_threshold: int = 20  # 触发 Map-Reduce 的证据数阈值
```

### SectionWriterConfig（章节写作配置）

```python
class SectionWriterConfig(BaseModel):
    max_evidence_per_call: int = 15   # 每次调用 LLM 的最大证据数
    enable_evidence_ranking: bool = True  # 是否启用证据排序
    min_section_length: int = 100     # 章节最小长度
    max_section_length: int = 2000    # 章节最大长度
```

## 使用示例

### 基础使用

```python
from graphrag_agent.agents.fusion_agent import FusionGraphRAGAgent

# 创建 Agent（内部自动创建 MultiAgentFacade）
agent = FusionGraphRAGAgent()

# 执行查询
result = agent.ask("复杂查询问题")
print(result)
```

### 高级配置

```python
from graphrag_agent.agents.multi_agent.integration import (
    MultiAgentFactory,
    MultiAgentFacade,
)
from graphrag_agent.agents.multi_agent.planner import PlannerConfig
from graphrag_agent.agents.multi_agent.reporter import ReporterConfig
from graphrag_agent.cache_manager import CacheManager

# 创建缓存管理器
cache_manager = CacheManager()

# 自定义配置
planner_config = PlannerConfig(
    max_tasks=8,
    allow_unclarified_plan=False,
)

reporter_config = ReporterConfig(
    enable_mapreduce=True,
    mapreduce_evidence_threshold=15,
    enable_parallel_map=True,
)

# 创建组件包
bundle = MultiAgentFactory.create_default_bundle(
    cache_manager=cache_manager,
)

# 创建 Facade
facade = MultiAgentFacade(
    bundle=bundle,
    cache_manager=cache_manager,
)

# 执行查询
result = facade.process_query(
    "复杂查询问题",
    assumptions=["假设1", "假设2"],
    report_type="long_document",
)

print(result["response"])          # 最终报告
print(result["planner"])           # 规划详情
print(result["execution_records"]) # 执行记录
print(result["report"])            # 报告详情
print(result["metrics"])           # 性能指标
```

### 访问详细信息

```python
# 查看任务计划
plan_spec = result["planner"]
for task in plan_spec["task_graph"]["nodes"]:
    print(f"任务: {task['task_id']}")
    print(f"类型: {task['task_type']}")
    print(f"描述: {task['description']}")
    print(f"状态: {task['status']}")

# 查看执行记录
for record in result["execution_records"]:
    print(f"任务: {record['task_id']}")
    print(f"状态: {record['status']}")
    print(f"证据数量: {len(record['evidence'])}")
    print(f"执行时间: {record['end_time'] - record['start_time']}")

# 查看报告结构
report = result["report"]
outline = report["outline"]
print(f"报告标题: {outline['title']}")
print(f"章节数量: {len(outline['sections'])}")
for section in outline["sections"]:
    print(f"- {section['title']}: {section['summary']}")

# 查看性能指标
metrics = result["metrics"]
print(f"总耗时: {metrics['total_duration']}秒")
print(f"规划耗时: {metrics['planner_duration']}秒")
print(f"执行耗时: {metrics['executor_duration']}秒")
print(f"报告耗时: {metrics['reporter_duration']}秒")
```