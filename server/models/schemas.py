from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from graphrag_agent.config.settings import community_algorithm


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    session_id: str
    debug: bool = False
    agent_type: str = "naive_rag_agent"
    use_deeper_tool: Optional[bool] = True
    show_thinking: Optional[bool] = False


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    execution_log: Optional[List[Dict]] = None
    kg_data: Optional[Dict] = None
    reference: Optional[Dict] = None
    iterations: Optional[List[Dict]] = None


class SourceRequest(BaseModel):
    """源内容请求模型"""
    source_id: str


class SourceResponse(BaseModel):
    """源内容响应模型"""
    content: str


class SourceInfoResponse(BaseModel):
    """源文件信息响应模型"""
    file_name: str


class ClearRequest(BaseModel):
    """清除聊天历史请求模型"""
    session_id: str


class ClearResponse(BaseModel):
    """清除聊天历史响应模型"""
    status: str
    remaining_messages: Optional[str] = None


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    message_id: str
    query: str
    is_positive: bool
    thread_id: str
    agent_type: Optional[str] = "naive_rag_agent"


class FeedbackResponse(BaseModel):
    """反馈响应模型"""
    status: str
    action: str

class SourceInfoBatchRequest(BaseModel):
    source_ids: List[str]

class ContentBatchRequest(BaseModel):
    chunk_ids: List[str]

class ReasoningRequest(BaseModel):
    reasoning_type: str
    entity_a: str
    entity_b: Optional[str] = None
    max_depth: Optional[int] = 3
    algorithm: Optional[str] = community_algorithm

class EntityData(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = ""
    properties: Optional[Dict[str, Any]] = {}

class EntityUpdateData(BaseModel):
    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class EntitySearchFilter(BaseModel):
    term: Optional[str] = None
    type: Optional[str] = None
    limit: Optional[int] = 100

class RelationData(BaseModel):
    source: str
    type: str
    target: str
    description: Optional[str] = ""
    weight: Optional[float] = 0.5
    properties: Optional[Dict[str, Any]] = {}

class RelationUpdateData(BaseModel):
    source: str
    original_type: str
    target: str
    new_type: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None

class RelationSearchFilter(BaseModel):
    source: Optional[str] = None
    target: Optional[str] = None
    type: Optional[str] = None
    limit: Optional[int] = 100

class EntityDeleteData(BaseModel):
    id: str

class RelationDeleteData(BaseModel):
    source: str
    type: str
    target: str