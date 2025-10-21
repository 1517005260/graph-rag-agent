"""
执行记录模块

定义ExecutionRecord及相关模型，记录任务执行过程
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """
    工具调用记录

    记录单次工具调用的详细信息
    """
    # 工具名称
    tool_name: str = Field(description="调用的工具名称")

    # 工具参数
    args: Dict[str, Any] = Field(description="工具调用参数")

    # 工具返回结果
    result: Optional[Any] = Field(default=None, description="工具返回的结果")

    # 执行状态
    status: str = Field(default="success", description="执行状态: success | failed")

    # 错误信息（如果失败）
    error: Optional[str] = Field(default=None, description="错误信息")

    # 执行延迟（毫秒）
    latency_ms: Optional[float] = Field(default=None, description="执行延迟（毫秒）")

    # 调用时间
    timestamp: datetime = Field(default_factory=datetime.now)


class ReflectionResult(BaseModel):
    """
    反思结果

    对执行结果进行反思和评估
    """
    # 执行是否成功
    success: bool = Field(description="任务执行是否成功")

    # 置信度评分 (0.0-1.0)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="结果置信度 (0.0-1.0)"
    )

    # 改进建议
    suggestions: List[str] = Field(
        default_factory=list,
        description="改进建议列表"
    )

    # 是否需要重试
    needs_retry: bool = Field(default=False, description="是否需要重试")

    # 反思理由
    reasoning: Optional[str] = Field(default=None, description="反思的详细理由")

    # 反思时间
    timestamp: datetime = Field(default_factory=datetime.now)


class ExecutionMetadata(BaseModel):
    """
    执行元数据

    记录执行过程的性能指标和环境信息
    """
    # Worker类型
    worker_type: str = Field(description="执行此任务的Worker类型")

    # 执行延迟（秒）
    latency_seconds: float = Field(default=0.0, description="任务执行总延迟（秒）")

    # Token消耗
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token使用情况，如 {'prompt': 100, 'completion': 200}"
    )

    # 调用的工具数量
    tool_calls_count: int = Field(default=0, description="调用的工具总数")

    # 检索到的证据数量
    evidence_count: int = Field(default=0, description="检索到的证据数量")

    # 执行环境信息
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="执行环境信息"
    )

    # 创建时间
    timestamp: datetime = Field(default_factory=datetime.now)


class ExecutionRecord(BaseModel):
    """
    执行记录

    记录单个任务的完整执行过程，包括输入、工具调用、证据和反思
    """
    # 记录唯一标识
    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="执行记录唯一标识符"
    )

    # 关联的任务ID
    task_id: str = Field(description="关联到PlanSpec中的TaskGraph节点")

    # 会话ID
    session_id: str = Field(description="关联到PlanExecuteState的session_id")

    # Worker类型
    worker_type: str = Field(
        description="执行此任务的Worker类型: retrieval_executor | research_executor | reflector"
    )

    # 输入参数
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="任务输入参数"
    )

    # 工具调用列表
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="调用的工具列表"
    )

    # 检索到的证据（RetrievalResult列表）
    evidence: List[Any] = Field(  # 类型为List[RetrievalResult]，避免循环依赖用Any
        default_factory=list,
        description="检索到的证据列表，元素为RetrievalResult对象"
    )

    # 反思结果
    reflection: Optional[ReflectionResult] = Field(
        default=None,
        description="对执行结果的反思评估"
    )

    # 执行元数据
    metadata: ExecutionMetadata = Field(
        default_factory=lambda: ExecutionMetadata(worker_type="unknown"),
        description="执行过程的元数据和性能指标"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)

    # 更新时间
    updated_at: datetime = Field(default_factory=datetime.now)

    # 配置
    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context):
        """同步元数据中的worker类型"""
        if self.metadata.worker_type in ("", "unknown"):
            self.metadata.worker_type = self.worker_type

    def to_cache_entry(self) -> Dict[str, Any]:
        """
        转换为CacheManager兼容的缓存条目

        用于缓存执行记录，支持增量写入
        """
        return {
            "key": f"{self.session_id}:{self.task_id}:{self.record_id}",
            "value": self.model_dump(mode='json'),
            "metadata": {
                "record_id": self.record_id,
                "worker_type": self.worker_type,
                "created_at": self.created_at.isoformat()
            }
        }

    @classmethod
    def from_cache_entry(cls, entry: Dict[str, Any]) -> "ExecutionRecord":
        """
        从缓存条目恢复ExecutionRecord对象
        """
        return cls.model_validate(entry["value"])

    def to_legacy_log(self) -> Dict[str, Any]:
        """
        转换为现有BaseAgent.execution_log格式

        兼容旧的日志系统
        """
        return {
            "task_id": self.task_id,
            "record_id": self.record_id,
            "worker_type": self.worker_type,
            "latency": self.metadata.latency_seconds,
            "tool_calls": len(self.tool_calls),
            "evidence_count": len(self.evidence),
            "success": self.reflection.success if self.reflection else True,
            "timestamp": self.created_at.isoformat()
        }

    def append_tool_call(self, tool_call: ToolCall):
        """添加工具调用记录"""
        self.tool_calls.append(tool_call)
        self.metadata.tool_calls_count += 1
        self.updated_at = datetime.now()

    def append_evidence(self, evidence: Any):
        """添加证据"""
        self.evidence.append(evidence)
        self.metadata.evidence_count += 1
        self.updated_at = datetime.now()

    def set_reflection(self, reflection: ReflectionResult):
        """设置反思结果"""
        self.reflection = reflection
        self.updated_at = datetime.now()

    def update_metadata(self, **kwargs):
        """更新元数据字段"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        self.updated_at = datetime.now()
