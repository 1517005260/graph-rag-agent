"""
计划规范模块

定义PlanSpec及相关数据模型，规范化任务计划
"""

from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, field_validator
from collections import deque, defaultdict

TASK_TYPE_CHOICES: Tuple[str, ...] = (
    "local_search",
    "global_search",
    "hybrid_search",
    "naive_search",
    "deep_research",
    "deeper_research",
    "chain_exploration",
    "reflection",
    "custom",
)

TaskTypeLiteral = Literal[
    "local_search",
    "global_search",
    "hybrid_search",
    "naive_search",
    "deep_research",
    "deeper_research",
    "chain_exploration",
    "reflection",
    "custom",
]


class ProblemStatement(BaseModel):
    """
    问题陈述

    描述用户要解决的问题和背景信息
    """
    # 原始查询
    original_query: str = Field(description="用户的原始查询")

    # 重写后的查询（更明确）
    refined_query: Optional[str] = Field(
        default=None,
        description="经过澄清和重写后的查询"
    )

    # 背景信息
    background_info: Optional[str] = Field(
        default=None,
        description="相关的背景信息和上下文"
    )

    # 用户意图
    user_intent: Optional[str] = Field(
        default=None,
        description="分析出的用户意图"
    )


class TaskNode(BaseModel):
    """
    任务节点

    TaskGraph中的单个任务定义
    """
    # 任务唯一标识
    task_id: str = Field(
        default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}",
        description="任务唯一标识符"
    )

    # 任务类型
    task_type: TaskTypeLiteral = Field(description="任务类型")

    # 任务描述
    description: str = Field(description="任务的详细描述")

    # 优先级（1=高, 2=中, 3=低）
    priority: Literal[1, 2, 3] = Field(default=2, description="任务优先级")

    # 预估token消耗
    estimated_tokens: int = Field(default=500, description="预估的token消耗")

    # 依赖的任务ID列表
    depends_on: List[str] = Field(
        default_factory=list,
        description="依赖的任务ID列表，空列表表示可以立即执行"
    )

    # 任务参数（具体执行时使用）
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="任务执行所需的参数"
    )

    # 相关实体（用于chain_exploration等类型）
    entities: List[str] = Field(
        default_factory=list,
        description="任务相关的实体列表"
    )

    # 状态
    status: Literal["pending", "running", "completed", "failed"] = Field(
        default="pending",
        description="任务执行状态"
    )


class TaskGraph(BaseModel):
    """
    任务依赖图

    使用轻量级dict结构存储任务DAG，避免引入networkx依赖
    """
    # 任务节点列表
    nodes: List[TaskNode] = Field(description="任务节点列表")

    # 执行模式
    execution_mode: Literal["sequential", "parallel", "adaptive"] = Field(
        default="sequential",
        description="任务执行模式：sequential(串行), parallel(并行), adaptive(自适应)"
    )

    @field_validator("nodes")
    @classmethod
    def validate_unique_task_ids(cls, nodes: List[TaskNode]) -> List[TaskNode]:
        """验证任务ID唯一性"""
        task_ids = [node.task_id for node in nodes]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("任务ID必须唯一")
        return nodes

    def validate_dependencies(self) -> bool:
        """
        验证任务依赖的合法性

        检查：
        1. 依赖的任务ID是否存在
        2. 是否存在循环依赖
        """
        task_id_set = {node.task_id for node in self.nodes}

        # 检查依赖的任务是否存在
        for node in self.nodes:
            for dep_id in node.depends_on:
                if dep_id not in task_id_set:
                    raise ValueError(f"任务 {node.task_id} 依赖的任务 {dep_id} 不存在")

        # 检查循环依赖（拓扑排序）
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            # 找到当前任务节点
            current_node = next((n for n in self.nodes if n.task_id == task_id), None)
            if not current_node:
                return False

            for dep_id in current_node.depends_on:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(task_id)
            return False

        for node in self.nodes:
            if node.task_id not in visited:
                if has_cycle(node.task_id):
                    raise ValueError("任务图中存在循环依赖")

        return True

    def get_ready_tasks(self, completed_task_ids: List[str]) -> List[TaskNode]:
        """
        获取可以执行的任务

        参数:
            completed_task_ids: 已完成的任务ID列表

        返回:
            可以执行的任务节点列表（依赖已满足且状态为pending）
        """
        completed_set = set(completed_task_ids)
        ready_tasks = []

        for node in self.nodes:
            if node.status != "pending":
                continue

            # 检查依赖是否全部完成
            if all(dep_id in completed_set for dep_id in node.depends_on):
                ready_tasks.append(node)

        # 按优先级排序（优先级数字越小越高）
        ready_tasks.sort(key=lambda x: x.priority)
        return ready_tasks

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            "nodes": [node.model_dump() for node in self.nodes],
            "execution_mode": self.execution_mode
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """从字典格式创建"""
        nodes = [TaskNode(**node_data) for node_data in data.get("nodes", [])]
        return cls(
            nodes=nodes,
            execution_mode=data.get("execution_mode", "sequential")
        )

    def topological_sort(self) -> List[TaskNode]:
        """
        获取任务的拓扑排序

        返回:
            List[TaskNode]: 按依赖顺序排列的任务节点列表
        """
        task_map = {node.task_id: node for node in self.nodes}
        in_degree: Dict[str, int] = {node.task_id: 0 for node in self.nodes}
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for node in self.nodes:
            for dep_id in node.depends_on:
                adjacency[dep_id].append(node.task_id)
                in_degree[node.task_id] += 1

        queue = deque(sorted(
            (task_map[task_id] for task_id, degree in in_degree.items() if degree == 0),
            key=lambda x: (x.priority, x.task_id),
        ))

        ordered_nodes: List[TaskNode] = []
        while queue:
            current = queue.popleft()
            ordered_nodes.append(current)

            for neighbor_id in adjacency.get(current.task_id, []):
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(task_map[neighbor_id])
            queue = deque(sorted(list(queue), key=lambda x: (x.priority, x.task_id)))

        if len(ordered_nodes) != len(self.nodes):
            raise ValueError("拓扑排序失败，任务图可能存在循环依赖")

        return ordered_nodes


class AcceptanceCriteria(BaseModel):
    """
    验收标准

    定义任务完成的标准和质量要求
    """
    # 完成条件列表
    completion_conditions: List[str] = Field(
        default_factory=list,
        description="任务完成必须满足的条件"
    )

    # 质量要求
    quality_requirements: List[str] = Field(
        default_factory=list,
        description="输出质量要求"
    )

    # 最小证据数量
    min_evidence_count: int = Field(
        default=1,
        description="至少需要的证据数量"
    )

    # 最小置信度
    min_confidence: float = Field(
        default=0.7,
        description="最低置信度阈值 (0.0-1.0)"
    )


class PlanSpec(BaseModel):
    """
    计划规范

    完整的任务执行计划，包含问题陈述、任务图和验收标准
    """
    # 计划ID
    plan_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="计划唯一标识符"
    )

    # 版本号
    version: int = Field(default=1, description="计划版本号")

    # 问题陈述
    problem_statement: ProblemStatement = Field(description="问题陈述和背景")

    # 假设列表（用户确认的前提条件）
    assumptions: List[str] = Field(
        default_factory=list,
        description="计划基于的假设和前提条件"
    )

    # 任务依赖图
    task_graph: TaskGraph = Field(description="任务依赖图")

    # 验收标准
    acceptance_criteria: AcceptanceCriteria = Field(
        default_factory=AcceptanceCriteria,
        description="任务验收标准"
    )

    # 计划状态
    status: Literal["draft", "approved", "executing", "completed", "failed"] = Field(
        default="draft",
        description="计划执行状态"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)

    # 更新时间
    updated_at: datetime = Field(default_factory=datetime.now)

    def validate(self) -> bool:
        """
        验证计划的合法性

        检查任务图依赖关系等
        """
        try:
            self.task_graph.validate_dependencies()
            return True
        except ValueError as e:
            raise ValueError(f"计划验证失败: {str(e)}")

    def get_ready_tasks(self, completed_task_ids: List[str]) -> List[TaskNode]:
        """
        获取可执行的任务

        参数:
            completed_task_ids: 已完成的任务ID列表

        返回:
            可以执行的任务节点列表
        """
        return self.task_graph.get_ready_tasks(completed_task_ids)

    def update_task_status(self, task_id: str, status: str):
        """
        更新任务状态

        参数:
            task_id: 任务ID
            status: 新状态
        """
        for node in self.task_graph.nodes:
            if node.task_id == task_id:
                node.status = status  # type: ignore
                self.updated_at = datetime.now()
                break

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "problem_statement": self.problem_statement.model_dump(),
            "assumptions": self.assumptions,
            "task_graph": self.task_graph.to_dict(),
            "acceptance_criteria": self.acceptance_criteria.model_dump(),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_execution_signal(self) -> "PlanExecutionSignal":
        """
        转换为执行层可消费的信号结构
        """
        ordered_nodes = self.task_graph.topological_sort()
        return PlanExecutionSignal(
            plan_id=self.plan_id,
            version=self.version,
            execution_mode=self.task_graph.execution_mode,
            tasks=[node.model_dump() for node in self.task_graph.nodes],
            execution_sequence=[node.task_id for node in ordered_nodes],
            assumptions=self.assumptions,
            acceptance_criteria=self.acceptance_criteria.model_dump(),
        )


class PlanExecutionSignal(BaseModel):
    """
    Planner输出给Executor的标准信号
    """
    plan_id: str = Field(description="计划唯一标识")
    version: int = Field(description="计划版本号")
    execution_mode: Literal["sequential", "parallel", "adaptive"] = Field(description="建议执行模式")
    tasks: List[Dict[str, Any]] = Field(description="任务节点详细信息列表")
    execution_sequence: List[str] = Field(description="拓扑排序后的任务执行顺序")
    assumptions: List[str] = Field(description="计划假设条件")
    acceptance_criteria: Dict[str, Any] = Field(description="验收标准")
