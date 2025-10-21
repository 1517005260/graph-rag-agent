"""
检索结果模块

定义RetrievalResult统一接口，对齐IoD风格的多粒度检索
"""

from typing import Union, Dict, Any, Optional, Literal
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class RetrievalMetadata(BaseModel):
    """
    检索元数据

    记录检索结果的来源、置信度和层级信息
    """
    # 数据源ID（DO ID或Chunk ID，可追溯到原始数据源）
    source_id: str = Field(description="数据源唯一标识符")

    # 数据源类型
    source_type: Literal[
        "document",
        "chunk",
        "entity",
        "relationship",
        "community",
        "subgraph"
    ] = Field(description="数据源类型")

    # 置信度 (0.0-1.0)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="检索结果的置信度"
    )

    # 数据时间戳（用于过滤过期数据）
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="数据的时间戳"
    )

    # DO层级（Digital Object层级）
    do_level: Optional[Literal["L0-DO", "L1-DO", "L2-DO"]] = Field(
        default=None,
        description="Digital Object层级，仅DO粒度时有效"
    )

    # 社区ID（LocalSearch时有效）
    community_id: Optional[str] = Field(
        default=None,
        description="所属社区ID"
    )

    # 图遍历跳数（GraphSearch时有效）
    hop_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="图遍历的跳数"
    )

    # 额外元数据（扩展用）
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外的自定义元数据"
    )


class RetrievalResult(BaseModel):
    """
    检索结果统一接口

    适配IoD风格的多粒度检索（DO/Chunk/AtomicKnowledge/KGSubgraph）
    """
    # 结果唯一标识
    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="检索结果唯一标识符"
    )

    # 检索粒度
    granularity: Literal[
        "DO",           # Digital Object（文档级）
        "L2-DO",        # Level 2 Digital Object（段落级）
        "Chunk",        # 文本块
        "AtomicKnowledge",  # 原子知识（三元组等）
        "KGSubgraph"    # 知识图谱子图
    ] = Field(description="检索粒度")

    # 检索内容（可以是文本或结构化数据）
    evidence: Union[str, Dict[str, Any]] = Field(
        description="检索到的内容，可以是文本或结构化数据"
    )

    # 元数据
    metadata: RetrievalMetadata = Field(description="检索结果的元数据")

    # 检索来源
    source: Literal[
        "local_search",
        "global_search",
        "graph_search",
        "deep_research",
        "chain_exploration",
        "hybrid"
    ] = Field(description="检索来源工具")

    # 相似度/相关性分数 (0.0-1.0)
    score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="相似度或相关性分数"
    )

    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)

    def get_citation(self, format_type: str = "default") -> str:
        """
        生成引用格式

        参数:
            format_type: 引用格式类型（default | apa | mla）

        返回:
            格式化的引用字符串
        """
        if format_type == "apa":
            # APA格式示例
            return f"[{self.result_id[:8]}] ({self.metadata.timestamp.year}). {self.metadata.source_type}. Retrieved from {self.metadata.source_id}"
        elif format_type == "mla":
            # MLA格式示例
            return f'[{self.result_id[:8]}] "{self.metadata.source_type}." {self.metadata.source_id}, {self.metadata.timestamp.year}.'
        else:
            # 默认格式
            source_desc = f"{self.metadata.source_type}:{self.metadata.source_id}"
            if self.metadata.community_id:
                source_desc += f" (社区:{self.metadata.community_id})"
            return f"[{self.result_id[:8]}] 来源: {source_desc} (置信度:{self.metadata.confidence:.2f})"

    @classmethod
    def merge(cls, results: list["RetrievalResult"]) -> "RetrievalResult":
        """
        合并多个相同source_id的检索结果

        选择score最高的结果作为合并后的结果

        参数:
            results: 要合并的结果列表

        返回:
            合并后的单个结果
        """
        if not results:
            raise ValueError("无法合并空的结果列表")

        # 按score排序，取最高分
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        best_result = sorted_results[0]

        # 可以选择性地合并evidence（这里保持最高分的evidence）
        return best_result

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "result_id": self.result_id,
            "granularity": self.granularity,
            "evidence": self.evidence,
            "metadata": self.metadata.model_dump(),
            "source": self.source,
            "score": self.score,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """从字典格式创建"""
        metadata_data = data.get("metadata", {})
        metadata = RetrievalMetadata(**metadata_data)

        created_at_value = data.get("created_at")
        if created_at_value:
            # 兼容ISO格式中的Z结尾及偏移量缺失情况
            cleaned_value = created_at_value.replace("Z", "+00:00")
            try:
                created_at = datetime.fromisoformat(cleaned_value)
            except ValueError:
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        return cls(
            result_id=data.get("result_id", str(uuid.uuid4())),
            granularity=data["granularity"],
            evidence=data["evidence"],
            metadata=metadata,
            source=data["source"],
            score=data.get("score", 0.5),
            created_at=created_at
        )
