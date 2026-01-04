"""
统一的搜索返回结构定义（工程级实践）

解决问题：
1. Tool/Retriever/Agent 返回格式不统一
2. Frontend 不应该解析 JSON（LLM 输出的 JSON 不可靠）
3. 明确分离：检索结果 vs 生成回答

架构原则：
- Tool: 返回 dict（标准化结构）
- Agent: dict → prompt → LLM → answer
- Frontend: 直接展示 answer（不解析 JSON）
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class ReferenceData(BaseModel):
    """引用数据结构"""
    chunks: List[str] = Field(default_factory=list, description="引用的文本块 ID 列表")
    entities: List[str] = Field(default_factory=list, description="引用的实体 ID 列表")
    communities: List[str] = Field(default_factory=list, description="引用的社区 ID 列表")
    relationships: List[str] = Field(default_factory=list, description="引用的关系 ID 列表")


class MetaData(BaseModel):
    """元数据结构"""
    retriever: str = Field(..., description="使用的检索器类型: naive | graph | hybrid | deep_research")
    search_time: Optional[float] = Field(None, description="搜索耗时（秒）")
    llm_time: Optional[float] = Field(None, description="LLM 生成耗时（秒）")
    total_time: Optional[float] = Field(None, description="总耗时（秒）")
    scores: Optional[List[float]] = Field(None, description="相似度分数列表")
    top_k: Optional[int] = Field(None, description="检索的文档数量")
    cache_hit: bool = Field(False, description="是否命中缓存")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")


class SearchResponse(BaseModel):
    """
    统一的搜索返回结构

    使用场景：
    1. Tool._run() 返回值：返回 dict（response.model_dump()）
    2. Agent 处理：dict → 构建 prompt context → LLM 生成
    3. Frontend 展示：直接展示 answer 字段

    示例：
    >>> response = SearchResponse(
    ...     answer="根据检索结果，学生旷课达到20学时将被退学。",
    ...     references=ReferenceData(chunks=["chunk_1", "chunk_2"]),
    ...     meta=MetaData(retriever="naive", search_time=0.5)
    ... )
    >>> response.model_dump()
    """
    answer: str = Field(..., description="生成的回答文本")
    references: ReferenceData = Field(default_factory=ReferenceData, description="引用的资源")
    meta: MetaData = Field(..., description="元数据信息")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "根据检索结果，学生旷课达到20学时将被退学。",
                "references": {
                    "chunks": ["chunk_001", "chunk_045"],
                    "entities": ["entity_student_001"],
                    "communities": [],
                    "relationships": []
                },
                "meta": {
                    "retriever": "naive",
                    "search_time": 0.25,
                    "llm_time": 1.5,
                    "total_time": 1.75,
                    "scores": [0.89, 0.76],
                    "top_k": 10,
                    "cache_hit": False,
                    "timestamp": "2025-12-17T10:30:00"
                }
            }
        }


class ResponseBuilder:
    """
    响应构建器（工程级实践）

    功能：
    1. 标准化构建 SearchResponse
    2. 自动记录性能指标
    3. 统一错误处理

    使用示例：
    >>> builder = ResponseBuilder(retriever_name="naive")
    >>> builder.add_chunks(["chunk_1", "chunk_2"])
    >>> builder.add_scores([0.9, 0.8])
    >>> builder.set_timing(search_time=0.5, llm_time=1.2)
    >>> response = builder.build(answer="这是回答")
    """

    def __init__(self, retriever_name: str):
        """
        初始化响应构建器

        Args:
            retriever_name: 检索器名称 (naive | graph | hybrid | deep_research)
        """
        self.retriever_name = retriever_name
        self.references = ReferenceData()
        self.meta_data = {
            "retriever": retriever_name,
            "search_time": None,
            "llm_time": None,
            "total_time": None,
            "scores": None,
            "top_k": None,
            "cache_hit": False
        }

    def add_chunks(self, chunk_ids: List[str]) -> "ResponseBuilder":
        """添加引用的文本块"""
        self.references.chunks.extend(chunk_ids)
        return self

    def add_entities(self, entity_ids: List[str]) -> "ResponseBuilder":
        """添加引用的实体"""
        self.references.entities.extend(entity_ids)
        return self

    def add_communities(self, community_ids: List[str]) -> "ResponseBuilder":
        """添加引用的社区"""
        self.references.communities.extend(community_ids)
        return self

    def add_relationships(self, relationship_ids: List[str]) -> "ResponseBuilder":
        """添加引用的关系"""
        self.references.relationships.extend(relationship_ids)
        return self

    def add_scores(self, scores: List[float]) -> "ResponseBuilder":
        """添加相似度分数"""
        self.meta_data["scores"] = scores
        return self

    def set_top_k(self, top_k: int) -> "ResponseBuilder":
        """设置检索数量"""
        self.meta_data["top_k"] = top_k
        return self

    def set_timing(self,
                   search_time: Optional[float] = None,
                   llm_time: Optional[float] = None,
                   total_time: Optional[float] = None) -> "ResponseBuilder":
        """
        设置性能指标

        Args:
            search_time: 搜索耗时（秒）
            llm_time: LLM 生成耗时（秒）
            total_time: 总耗时（秒）
        """
        if search_time is not None:
            self.meta_data["search_time"] = search_time
        if llm_time is not None:
            self.meta_data["llm_time"] = llm_time
        if total_time is not None:
            self.meta_data["total_time"] = total_time
        return self

    def set_cache_hit(self, cache_hit: bool) -> "ResponseBuilder":
        """设置缓存命中状态"""
        self.meta_data["cache_hit"] = cache_hit
        return self

    def build(self, answer: str) -> SearchResponse:
        """
        构建最终的 SearchResponse

        Args:
            answer: 生成的回答文本

        Returns:
            SearchResponse: 标准化的响应对象
        """
        meta = MetaData(**self.meta_data)
        return SearchResponse(
            answer=answer,
            references=self.references,
            meta=meta
        )

    def build_dict(self, answer: str) -> Dict[str, Any]:
        """
        构建字典格式（用于 Tool._run() 返回值）

        Args:
            answer: 生成的回答文本

        Returns:
            Dict: 符合标准的字典结构
        """
        response = self.build(answer)
        return response.model_dump()


def create_error_response(
    retriever_name: str,
    error_message: str,
    error_type: str = "search_error"
) -> Dict[str, Any]:
    """
    创建错误响应（工程级实践）

    Args:
        retriever_name: 检索器名称
        error_message: 错误信息
        error_type: 错误类型

    Returns:
        Dict: 标准化的错误响应
    """
    builder = ResponseBuilder(retriever_name)
    error_answer = f"搜索过程中出现错误 ({error_type}): {error_message}"

    return builder.build_dict(answer=error_answer)
