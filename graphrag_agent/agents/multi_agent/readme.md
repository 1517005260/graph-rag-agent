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
│                   MultiAgentOrchestrator                     │
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

**Clarifier（澄清器）** - `planner/clarifier.py`
- 识别查询中的歧义和不明确需求
- 生成澄清问题列表
- 基于用户反馈优化查询

**TaskDecomposer（任务分解器）** - `planner/task_decomposer.py`
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

**PlanReviewer（计划审校器）** - `planner/plan_reviewer.py`
- 审核任务计划的合理性
- 优化任务顺序和参数
- 生成最终的 PlanSpec

#### 核心数据模型

**PlanSpec（计划规范）** - `core/plan_spec.py`

```python
class PlanSpec:
    plan_id: str                           # 计划唯一标识
    version: int                           # 版本号
    problem_statement: ProblemStatement    # 问题陈述
    assumptions: List[str]                 # 假设前提
    task_graph: TaskGraph                  # 任务依赖图
    acceptance_criteria: AcceptanceCriteria  # 验收标准
    status: str                            # 计划状态
```

**TaskGraph（任务图）**

```python
class TaskGraph:
    nodes: List[TaskNode]                  # 任务节点列表
    execution_mode: str                    # sequential/parallel/adaptive
    
    def validate_dependencies() -> bool    # 验证依赖关系
    def get_ready_tasks() -> List[TaskNode]  # 获取可执行任务
    def topological_sort() -> List[TaskNode]  # 拓扑排序
```

**PlanExecutionSignal（执行信号）**

```python
class PlanExecutionSignal:
    plan_id: str                           # 计划标识
    version: int                           # 版本号
    execution_mode: str                    # 执行模式（sequential/parallel/adaptive，adaptive会降级为sequential）
    tasks: List[Dict[str, Any]]            # 任务详情列表
    execution_sequence: List[str]          # 拓扑排序后的执行序列
    assumptions: List[str]                 # 假设条件
    acceptance_criteria: Dict[str, Any]    # 验收标准
```

### 2. Executor（执行器）

负责执行任务计划中的各个任务。

#### 执行器类型

**RetrievalExecutor（检索执行器）** - `executor/retrieval_executor.py`
- 执行各类搜索任务（local/global/hybrid/naive/chain_exploration）
- 调用 TOOL_REGISTRY 中注册的搜索工具
- 将搜索结果标准化为 RetrievalResult

**ResearchExecutor（研究执行器）** - `executor/research_executor.py`
- 执行深度研究任务（deep_research/deeper_research）
- 支持多步推理和证据链追踪
- 记录思考过程和中间结果

**ReflectionExecutor（反思执行器）** - `executor/reflector.py`
- 基于 AnswerValidationTool 对执行结果进行质量评估
- 提取参考关键词并验证答案覆盖度
- 支持自动重试机制

**WorkerCoordinator（工作协调器）** - `executor/worker_coordinator.py`
- 调度和管理各类执行器
- 支持两种执行模式（通过环境变量或构造参数配置）：
  - `sequential`: 串行执行，严格按依赖顺序
  - `parallel`: 并行执行，使用线程池并发执行独立任务
- 注：若 Planner 请求 `adaptive` 模式，会自动降级为 `sequential`
- 验证任务依赖关系（串行模式检查前置依赖，并行模式动态调度）
- 处理反思重试逻辑
- 可配置最大并发数（`max_parallel_workers`）

#### 核心工具

**EvidenceTracker（证据追踪器）** - `tools/evidence_tracker.py`
- 统一管理所有检索结果
- 按来源与粒度去重
- 优先保留高分证据
- 支持证据查询和溯源

**RetrievalAdapter（检索适配器）** - `tools/retrieval_adapter.py`
- 适配不同的搜索工具接口
- 标准化搜索结果格式
- 支持 LocalSearch、GlobalSearch、HybridSearch 等

#### 核心数据模型

**ExecutionRecord（执行记录）** - `core/execution_record.py`

```python
class ExecutionRecord:
    record_id: str                         # 记录唯一标识
    task_id: str                           # 关联任务ID
    session_id: str                        # 会话ID
    worker_type: str                       # Worker类型
    inputs: Dict[str, Any]                 # 输入参数
    tool_calls: List[ToolCall]             # 工具调用列表
    evidence: List[RetrievalResult]        # 检索到的证据
    reflection: Optional[ReflectionResult] # 反思结果
    metadata: ExecutionMetadata            # 执行元数据
```

**RetrievalResult（检索结果）** - `core/retrieval_result.py`

```python
class RetrievalResult:
    result_id: str                         # 结果唯一ID
    granularity: str                       # 粒度（DO/L2-DO/Chunk/AtomicKnowledge/KGSubgraph）
    evidence: Union[str, Dict[str, Any]]   # 证据内容
    metadata: RetrievalMetadata            # 元数据
    source: str                            # 检索来源
    score: float                           # 相关度分数（0.0-1.0）
```

### 3. Reporter（报告生成器）

负责将执行结果整合为结构化的长文档报告。

#### 报告生成流程

**OutlineBuilder（纲要生成）** - `reporter/outline_builder.py`
- 基于任务计划和证据生成报告大纲
- 确定章节结构和层次关系
- 为每个章节分配相关证据

**SectionWriter（章节写作）** - `reporter/section_writer.py`
- 传统模式：直接基于证据生成章节内容
- 支持分批写作，处理超长章节
- 自动去除重复标题

**Map-Reduce 写作模式** - `reporter/mapreduce/`
- **EvidenceMapper（证据映射器）** - `evidence_mapper.py`
  - 将证据分批映射为摘要
  - 提取关键论点和实体
  - 支持并行 Map 操作
- **SectionReducer（章节归约器）** - `section_reducer.py`
  - 将证据摘要归约为连贯文本
  - 支持三种策略：collapse（合并）、refine（精炼）、tree（树形）
  - 控制每次 LLM 调用的 token 上限
- **ReportAssembler（报告组装器）** - `report_assembler.py`
  - 组装最终报告
  - 生成引言和结论
  - 提取术语表

**质量保障**
- **ConsistencyChecker（一致性检查）** - `consistency_checker.py`
  - 检查报告内容与证据的一致性
  - 识别矛盾和不支持的陈述
  - 评估整体质量
- **CitationFormatter（引用格式化）** - `formatter.py`
  - 生成标准化的引用列表
  - 支持多种引用格式

#### 报告缓存机制

Reporter 实现了两级缓存：

**报告级缓存**
- 基于 `plan_id:version:report_type` 生成 `report_id`
- 使用证据指纹（evidence_fingerprint）检测证据是否变化
- 证据未变化时直接返回缓存的完整报告

**章节级缓存**
- 缓存每个章节的内容和使用的证据ID
- 章节标题、摘要和证据指纹均未变化时可复用
- 支持部分章节复用，其他章节重新生成

#### 核心数据模型

**ReportResult（报告结果）**

```python
class ReportResult:
    outline: ReportOutline                 # 报告纲要
    sections: List[SectionContent]         # 章节内容列表
    final_report: str                      # 最终报告（Markdown）
    references: Optional[str]              # 引用列表
    consistency_check: Optional[ConsistencyCheckResult]  # 一致性检查结果
```

**ReportOutline（报告纲要）**

```python
class ReportOutline:
    report_type: str                       # 报告类型
    title: str                             # 报告标题
    abstract: Optional[str]                # 摘要
    sections: List[SectionOutline]         # 章节大纲
    total_estimated_words: Optional[int]   # 预估总字数
```

### 4. Core（核心数据模型）

#### State（状态管理）- `core/state.py`

**PlanExecuteState（计划执行状态）**

```python
class PlanExecuteState(BaseModel):
    session_id: str                        # 会话唯一标识
    messages: List[BaseMessage]            # 对话消息历史
    input: str                             # 用户输入
    plan_context: Optional[PlanContext]    # 计划上下文
    plan: Optional[PlanSpec]               # 任务计划
    execution_context: Optional[ExecutionContext]  # 执行上下文
    execution_records: List[ExecutionRecord]       # 执行记录
    report_context: Optional[ReportContext]        # 报告上下文
    response: Optional[str]                # 最终响应
```

**PlanContext（计划上下文）**

```python
class PlanContext(BaseModel):
    original_query: str                    # 原始查询
    refined_query: Optional[str]           # 优化后的查询
    clarification_history: List[Dict[str, str]]  # 澄清历史
    user_preferences: Dict[str, Any]       # 用户偏好
    domain_context: Optional[str]          # 领域上下文
```

**ExecutionContext（执行上下文）**

```python
class ExecutionContext(BaseModel):
    current_task_id: Optional[str]         # 当前任务ID
    completed_task_ids: List[str]          # 已完成任务ID
    retrieval_cache: Dict[str, Any]        # 检索结果缓存
    tool_call_history: List[Dict[str, Any]]  # 工具调用历史
    intermediate_results: Dict[str, Any]   # 中间结果
    reflection_retry_counts: Dict[str, int]  # 反思重试计数
    errors: List[Dict[str, Any]]           # 错误记录
    evidence_registry: Dict[str, Dict[str, Any]]  # 证据追踪注册表
```

**ReportContext（报告上下文）**

```python
class ReportContext(BaseModel):
    outline: Optional[Dict[str, Any]]      # 报告纲要
    section_drafts: Dict[str, str]         # 章节草稿
    citations: List[Dict[str, Any]]        # 引用
    consistency_check_results: Optional[Dict[str, Any]]  # 一致性检查结果
    report_type: str                       # 报告类型
    report_id: Optional[str]               # 报告ID
    cache_hit: bool                        # 是否命中缓存
```

### 5. Integration（集成层）

#### MultiAgentFactory（工厂类）- `integration/multi_agent_factory.py`

提供便捷的组件创建和配置：

```python
class MultiAgentFactory:
    @staticmethod
    def create_default_bundle(
        cache_manager: Optional[CacheManager] = None
    ) -> OrchestratorBundle:
        """创建默认的编排器组件包"""
```

**OrchestratorBundle（编排器组件包）**

```python
@dataclass
class OrchestratorBundle:
    planner: BasePlanner                   # 规划器
    worker: WorkerCoordinator              # 执行协调器
    reporter: BaseReporter                 # 报告生成器
    orchestrator: MultiAgentOrchestrator   # 编排器
```

#### MultiAgentFacade（兼容层）- `integration/legacy_facade.py`

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
            "status": "completed" | "needs_clarification" | "failed" | "partial",
            "response": "最终报告文本",
            "planner": {...},              # 规划器输出
            "execution_records": [...],    # 执行记录
            "report": {...},               # 报告详情
            "report_context": {...},       # 报告上下文
            "errors": [...],               # 错误信息
            "metrics": {...}               # 性能指标
        }
        """
```

### 6. Orchestrator（编排器）

**MultiAgentOrchestrator** - `orchestrator.py`

协调 Planner → Executor → Reporter 的完整流程：

```python
class MultiAgentOrchestrator:
    def run(
        self,
        state: PlanExecuteState,
        *,
        assumptions: Optional[Sequence[str]] = None,
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

**OrchestratorResult（编排结果）**

```python
class OrchestratorResult(BaseModel):
    status: str                            # completed/needs_clarification/failed/partial
    planner: Optional[PlannerResult]       # 规划结果
    execution_records: List[ExecutionRecord]  # 执行记录
    report: Optional[ReportResult]         # 报告结果
    errors: List[str]                      # 错误列表
    metrics: OrchestratorMetrics           # 性能指标
```

## 配置选项

### 全局配置

通过环境变量或 `settings.py` 中的配置项进行配置：

```python
# 规划器配置
MULTI_AGENT_PLANNER_MAX_TASKS = 6                  # 最大任务数（MA_PLANNER_MAX_TASKS）
MULTI_AGENT_ALLOW_UNCLARIFIED_PLAN = True          # 是否允许未完全澄清的计划（MA_ALLOW_UNCLARIFIED_PLAN）
MULTI_AGENT_DEFAULT_DOMAIN = "通用"                # 默认领域（MA_DEFAULT_DOMAIN）

# 编排器配置
MULTI_AGENT_AUTO_GENERATE_REPORT = True            # 是否自动生成报告（MA_AUTO_GENERATE_REPORT）
MULTI_AGENT_STOP_ON_CLARIFICATION = True           # 需要澄清时是否停止（MA_STOP_ON_CLARIFICATION）
MULTI_AGENT_STRICT_PLAN_SIGNAL = True              # 严格要求执行信号（MA_STRICT_PLAN_SIGNAL）

# 执行器配置
MULTI_AGENT_WORKER_EXECUTION_MODE = "sequential"   # 执行模式：sequential/parallel（MA_WORKER_EXECUTION_MODE）
MULTI_AGENT_WORKER_MAX_CONCURRENCY = 4             # 最大并发数（MA_WORKER_MAX_CONCURRENCY）
MULTI_AGENT_REFLECTION_ALLOW_RETRY = False         # 是否允许反思重试（MA_REFLECTION_ALLOW_RETRY）
MULTI_AGENT_REFLECTION_MAX_RETRIES = 1             # 反思重试最大次数（MA_REFLECTION_MAX_RETRIES）

# 报告器配置
MULTI_AGENT_DEFAULT_REPORT_TYPE = "long_document"  # 默认报告类型（MA_DEFAULT_REPORT_TYPE）
MULTI_AGENT_ENABLE_CONSISTENCY_CHECK = True        # 是否启用一致性检查（MA_ENABLE_CONSISTENCY_CHECK）
MULTI_AGENT_ENABLE_MAPREDUCE = True                # 是否启用 Map-Reduce 模式（MA_ENABLE_MAPREDUCE）
MULTI_AGENT_MAPREDUCE_THRESHOLD = 20               # 触发 Map-Reduce 的证据数阈值（MA_MAPREDUCE_THRESHOLD）
MULTI_AGENT_MAX_TOKENS_PER_REDUCE = 4000           # Reduce 阶段最大 token 数（MA_MAX_TOKENS_PER_REDUCE）
MULTI_AGENT_ENABLE_PARALLEL_MAP = True             # 是否启用并行 Map（MA_ENABLE_PARALLEL_MAP）
MULTI_AGENT_SECTION_MAX_EVIDENCE = 8               # 章节写作每次调用的最大证据数（MA_SECTION_MAX_EVIDENCE）
MULTI_AGENT_SECTION_MAX_CONTEXT_CHARS = 800        # 多批写作时保留的前文字符数（MA_SECTION_MAX_CONTEXT_CHARS）
```

### PlannerConfig（规划器配置）

```python
class PlannerConfig(BaseModel):
    max_tasks: int = 6                     # 最大任务数
    allow_unclarified_plan: bool = True    # 是否允许未完全澄清的计划
    default_domain: str = "通用"            # 默认领域
```

### ReporterConfig（报告器配置）

```python
class ReporterConfig(BaseModel):
    default_report_type: str = "long_document"     # 默认报告类型
    citation_style: str = "default"                # 引用格式
    max_evidence_summary: int = 30                 # 纲要生成时展示的最大证据条数
    section_writer: SectionWriterConfig            # 章节写作配置
    enable_consistency_check: bool = True          # 是否启用一致性检查
    enable_mapreduce: bool = True                  # 是否启用 Map-Reduce 模式
    reduce_strategy: str = "tree"                  # 归约策略（collapse/refine/tree）
    max_tokens_per_reduce: int = 4000              # Reduce 阶段最大 token 数
    enable_parallel_map: bool = True               # 是否启用并行 Map
    mapreduce_evidence_threshold: int = 20         # 触发 Map-Reduce 的证据数阈值
```

### SectionWriterConfig（章节写作配置）

```python
class SectionWriterConfig(BaseModel):
    max_evidence_per_call: int = 8         # 每次调用 LLM 的最大证据数
    max_previous_context_chars: int = 800  # 多批写作时保留的前文字符数
    enable_multi_pass: bool = True         # 是否启用多批写作
```

### ExecutorConfig（执行器配置）

```python
class ExecutorConfig(BaseModel):
    max_retries: int = 1                   # 单个任务的最大重试次数
    retry_delay_seconds: float = 0.0       # 重试延迟
    enable_reflection: bool = False        # 是否启用反思节点
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
plan_spec = result["planner"]["plan_spec"]
for task in plan_spec["task_graph"]["nodes"]:
    print(f"任务: {task['task_id']}")
    print(f"类型: {task['task_type']}")
    print(f"描述: {task['description']}")
    print(f"状态: {task['status']}")

# 查看执行记录
for record in result["execution_records"]:
    print(f"任务: {record['task_id']}")
    print(f"Worker: {record['worker_type']}")
    print(f"证据数量: {len(record['evidence'])}")
    print(f"执行时间: {record['metadata']['latency_seconds']}秒")

# 查看报告结构
report = result["report"]
outline = report["outline"]
print(f"报告标题: {outline['title']}")
print(f"章节数量: {len(outline['sections'])}")
for section in outline["sections"]:
    print(f"- {section['title']}: {section['summary']}")

# 查看性能指标
metrics = result["metrics"]
print(f"规划耗时: {metrics['planning_seconds']}秒")
print(f"执行耗时: {metrics['execution_seconds']}秒")
print(f"报告耗时: {metrics['reporting_seconds']}秒")
```

## 关键特性

### 1. 任务依赖管理
- 自动验证任务依赖的合法性
- 检测循环依赖
- 拓扑排序生成执行序列
- 动态获取可执行任务

### 2. 证据追踪与去重
- 统一的证据追踪器管理所有检索结果
- 按来源与粒度自动去重
- 优先保留高分证据
- 支持证据溯源和引用

### 3. 反思与重试机制
- 基于 AnswerValidationTool 自动质量评估
- 提取参考关键词验证答案覆盖度
- 配置化的重试策略
- 回滚失败任务并重新执行

### 4. 分级缓存
- 报告级缓存：基于证据指纹判断是否复用
- 章节级缓存：部分章节可独立复用
- 减少重复计算，提升响应速度

### 5. Map-Reduce 写作
- 证据映射：批量压缩原始证据
- 章节归约：支持多种归约策略
- 并行处理：支持并行 Map 操作
- 适应超长文档和大量证据场景

### 6. 多模式执行
- Sequential（串行）：严格按依赖顺序执行，依赖未完成则阻塞
- Parallel（并行）：使用线程池并发执行，动态调度满足依赖的任务
- Adaptive（自适应）：当前会降级为串行模式（未来规划）

## 扩展指南

### 添加新的任务类型

1. 在 `core/plan_spec.py` 中扩展 `TASK_TYPE_CHOICES`
2. 实现对应的 Executor 或复用现有 Executor
3. 在 WorkerCoordinator 中注册新的 Executor

### 添加新的报告格式

1. 在 Reporter 配置中添加新的 `report_type`
2. 在 OutlineBuilder 中处理新类型的大纲生成
3. 在 ReportAssembler 中处理新类型的组装逻辑

### 自定义归约策略

1. 在 `reporter/mapreduce/section_reducer.py` 中实现新策略
2. 添加到 `ReduceStrategy` 枚举
3. 在 `reduce` 方法中添加分支处理