"""
执行层基础抽象

为不同类型的Worker（检索、研究、反思）提供统一的配置、输入规范与结果包装。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import copy

from pydantic import BaseModel, Field

from graphrag_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
from graphrag_agent.agents.multi_agent.core.execution_record import ExecutionRecord


class ExecutorConfig(BaseModel):
    """
    Worker通用配置
    """

    max_retries: int = Field(default=1, ge=0, description="单个任务的最大重试次数")
    retry_delay_seconds: float = Field(default=0.0, ge=0.0, description="重试之间的延迟")
    enable_reflection: bool = Field(default=False, description="是否启用反思节点")


class TaskExecutionResult(BaseModel):
    """
    任务执行结果包装
    """

    record: ExecutionRecord = Field(description="任务执行记录")
    success: bool = Field(default=True, description="任务是否执行成功")
    error: Optional[str] = Field(default=None, description="失败时的错误信息")


class BaseExecutor(ABC):
    """
    Worker基类

    负责根据PlanExecutionSignal中的任务节点执行具体操作并产出ExecutionRecord。
    """

    worker_type: str = "base_executor"

    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        self.config = config or ExecutorConfig()

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """
        判断当前Executor是否能够处理指定类型任务
        """

    @abstractmethod
    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        """
        执行单个任务
        """

    def build_default_inputs(self, task: TaskNode) -> Dict[str, Any]:
        """
        构造标准化的工具输入

        默认策略：
            - query: 参数中的query，否则使用任务描述
            - entities: 任务节点中的实体列表
            - parameters: 任务自带参数（深拷贝，避免原数据被修改）
        """
        parameters = copy.deepcopy(task.parameters or {})
        query = parameters.get("query") or task.description
        payload: Dict[str, Any] = {"query": query}

        if task.entities:
            payload.setdefault("entities", task.entities)
            payload.setdefault("start_entities", task.entities)

        # 合并参数，保持参数优先级
        payload.update(parameters)
        return payload
