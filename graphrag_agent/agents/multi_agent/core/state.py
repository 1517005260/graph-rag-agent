"""
状态管理模块

定义Plan-Execute-Report流程中的各种状态模型
使用pydantic.BaseModel实现自动序列化和字段验证
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class PlanContext(BaseModel):
    """
    Planner专用上下文

    用于Planner阶段保存查询信息、澄清历史和用户偏好
    """
    # 原始查询
    original_query: str = Field(description="用户原始输入的查询")

    # 重写后的查询（澄清后）
    refined_query: Optional[str] = Field(default=None, description="经过澄清和重写后的查询")

    # 澄清历史记录（问题-回答对）
    clarification_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="澄清问答历史，格式: [{'question': '...', 'answer': '...'}]"
    )

    # 用户偏好和约束
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="用户偏好设置，如输出格式、详细程度等"
    )

    # 背景信息和领域知识
    domain_context: Optional[str] = Field(
        default=None,
        description="领域背景信息"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)


class ExecutionContext(BaseModel):
    """
    Executor专用上下文

    用于Executor阶段跟踪任务执行状态和工具调用历史
    """
    # 当前正在执行的任务ID
    current_task_id: Optional[str] = Field(default=None, description="当前执行的任务ID")

    # 已完成的任务ID列表
    completed_task_ids: List[str] = Field(default_factory=list, description="已完成任务ID")

    # 检索结果缓存（task_id -> 结果）
    retrieval_cache: Dict[str, Any] = Field(
        default_factory=dict,
        description="检索结果缓存，避免重复检索"
    )

    # 工具调用历史
    tool_call_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="工具调用历史记录"
    )

    # 中间结果存储（用于任务间依赖传递）
    intermediate_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="任务间依赖传递的中间结果，格式: {task_id: result}"
    )

    # 执行错误记录
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="执行过程中的错误记录"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)

    # 更新时间
    updated_at: datetime = Field(default_factory=datetime.now)


class ReportContext(BaseModel):
    """
    Reporter专用上下文

    用于Reporter阶段管理报告生成的纲要、草稿和引用
    """
    # 报告纲要（章节结构）
    outline: Optional[Dict[str, Any]] = Field(
        default=None,
        description="报告纲要结构，包含章节标题和摘要"
    )

    # 段落草稿（section_id -> 内容）
    section_drafts: Dict[str, str] = Field(
        default_factory=dict,
        description="各章节的草稿内容"
    )

    # 引用列表
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="引用列表，每个引用包含source_id, content, metadata"
    )

    # 一致性检查结果
    consistency_check_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="一致性校验的结果，包含问题和修正建议"
    )

    # 报告类型（短回答或长文档）
    report_type: str = Field(
        default="short_answer",
        description="报告类型: short_answer(短回答) | long_document(长文档)"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)

    # 更新时间
    updated_at: datetime = Field(default_factory=datetime.now)


class PlanExecuteState(BaseModel):
    """
    完整的Plan-Execute-Report状态

    这是LangGraph的主状态，包含整个流程的所有信息
    继承自BaseModel，与现有BaseAgent.AgentState兼容
    """
    # 会话ID（全局唯一标识）
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="会话唯一标识符"
    )

    # 兼容现有AgentState：消息列表
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="对话消息历史，兼容LangChain的BaseMessage"
    )

    # 用户输入
    input: str = Field(default="", description="用户输入的原始查询")

    # Planner阶段的上下文
    plan_context: Optional[PlanContext] = Field(
        default=None,
        description="Planner阶段的上下文信息"
    )

    # 生成的计划（PlanSpec）
    plan: Optional[Any] = Field(  # 类型为PlanSpec，但为了避免循环依赖，这里用Any
        default=None,
        description="Planner生成的执行计划"
    )

    # Executor阶段的上下文
    execution_context: Optional[ExecutionContext] = Field(
        default=None,
        description="Executor阶段的上下文信息"
    )

    # 执行记录列表（ExecutionRecord）
    execution_records: List[Any] = Field(  # 类型为List[ExecutionRecord]
        default_factory=list,
        description="所有任务的执行记录"
    )

    # Reporter阶段的上下文
    report_context: Optional[ReportContext] = Field(
        default=None,
        description="Reporter阶段的上下文信息"
    )

    # 最终响应
    response: Optional[str] = Field(
        default=None,
        description="最终生成的响应或报告"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)

    # 更新时间
    updated_at: datetime = Field(default_factory=datetime.now)

    # 配置选项
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型（如BaseMessage）

    def model_post_init(self, __context):
        """模型初始化后的钩子，初始化默认上下文"""
        if self.plan_context is None:
            self.plan_context = PlanContext(original_query=self.input)
        if self.execution_context is None:
            self.execution_context = ExecutionContext()
        if self.report_context is None:
            self.report_context = ReportContext()

    def to_legacy_state(self) -> Dict[str, Any]:
        """
        转换为现有BaseAgent.AgentState格式
        用于与旧系统兼容
        """
        return {
            "messages": self.messages,
            # 可以添加其他需要的字段
        }

    @classmethod
    def from_legacy_state(cls, state: Dict[str, Any], input_query: str = "") -> "PlanExecuteState":
        """
        从现有BaseAgent.AgentState格式创建
        用于与旧系统兼容
        """
        messages = state.get("messages", [])
        return cls(
            messages=messages,
            input=input_query or (messages[-1].content if messages else "")
        )

    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()
