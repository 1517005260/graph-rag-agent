import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

# 统一加载环境变量，确保配置来源一致
load_dotenv()

# MinerU 默认使用 ModelScope 模型源，除非用户显式指定
os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")


def _get_env_int(key: str, default: Optional[int]) -> Optional[int]:
    """获取整型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要整数值，但当前为 {raw}") from exc


def _get_env_float(key: str, default: Optional[float]) -> Optional[float]:
    """获取浮点型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要浮点值，但当前为 {raw}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    """获取布尔型环境变量，支持 true/false/1/0 表达式"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_choice(key: str, choices: set[str], default: str) -> str:
    """获取有限集合中的字符串配置，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(
            f"环境变量 {key} 必须为 {', '.join(sorted(choices))} 之一，但当前为 {raw}"
        )
    return value


def _get_env_int_pair(key: str, default: Tuple[int, int]) -> Tuple[int, int]:
    """获取由两个整数构成的配置，使用逗号分隔"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default

    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"环境变量 {key} 需要形如 'width,height' 的两个整数，但当前为 {raw}"
        )

    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"环境变量 {key} 需要形如 'width,height' 的两个整数，但当前为 {raw}"
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"环境变量 {key} 中的宽高必须为正整数，但当前为 {raw}")

    return width, height


def _resolve_project_path(raw_value: Optional[str], default: str) -> Path:
    """
    将环境变量解析为相对于项目根目录的路径

    Args:
        raw_value: 环境变量值
        default: 默认路径（可相对可绝对）

    Returns:
        Path: 解析后的绝对路径
    """
    base = PROJECT_ROOT
    value = raw_value.strip() if raw_value and raw_value.strip() else default
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


# ===== 基础路径设置 =====

BASE_DIR = Path(__file__).resolve().parent.parent  # graphrag_agent包目录
PROJECT_ROOT = BASE_DIR.parent  # 项目根目录
FILES_DIR = PROJECT_ROOT / "files"
FILE_REGISTRY_PATH = PROJECT_ROOT / "file_registry.json"  # 文件注册表路径

# ===== MinerU & 文档处理配置 =====

DOCUMENT_PROCESSOR_MODE = os.getenv("DOCUMENT_PROCESSOR_MODE", "legacy").strip().lower()

MINERU_API_URL = os.getenv("MINERU_API_URL", "http://localhost:8899").rstrip("/")
MINERU_API_TIMEOUT = _get_env_int("MINERU_API_TIMEOUT", 300) or 300

MINERU_HOST = os.getenv("MINERU_HOST", "0.0.0.0")
MINERU_PORT = _get_env_int("MINERU_PORT", 8899) or 8899
MINERU_WORKERS = _get_env_int("MINERU_WORKERS", 1) or 1

MINERU_HOME = _resolve_project_path(os.getenv("MINERU_HOME"), "../MinerU")
MINERU_OUTPUT_DIR = _resolve_project_path(os.getenv("MINERU_OUTPUT_DIR"), "./mineru_outputs")
MINERU_TEMP_DIR = _resolve_project_path(os.getenv("MINERU_TEMP_DIR"), "./.tmp/mineru_temp")
MINERU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MINERU_TEMP_DIR.mkdir(parents=True, exist_ok=True)
MINERU_ASSET_BASE_URL = os.getenv("MINERU_ASSET_BASE_URL")
if not MINERU_ASSET_BASE_URL:
    MINERU_ASSET_BASE_URL = f"{MINERU_API_URL}/download"

MINERU_CACHE_REGISTRY_PATH = _resolve_project_path(
    os.getenv("MINERU_CACHE_REGISTRY"), "./cache/mineru_registry.json"
)
MINERU_CACHE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
MINERU_CACHE_DATA_DIR = (MINERU_CACHE_REGISTRY_PATH.parent / "mineru_cache").resolve()
MINERU_CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

MINERU_DEFAULT_BACKEND = os.getenv("MINERU_DEFAULT_BACKEND", "pipeline")
MINERU_DEFAULT_PARSE_METHOD = os.getenv("MINERU_DEFAULT_PARSE_METHOD", "auto")
MINERU_DEFAULT_LANG = os.getenv("MINERU_DEFAULT_LANG", "ch")
MINERU_FORMULA_ENABLE = _get_env_bool("MINERU_FORMULA_ENABLE", True)
MINERU_TABLE_ENABLE = _get_env_bool("MINERU_TABLE_ENABLE", True)

MINERU_DUMP_MD = _get_env_bool("MINERU_DUMP_MD", True)
MINERU_DUMP_CONTENT_LIST = _get_env_bool("MINERU_DUMP_CONTENT_LIST", True)
MINERU_DUMP_MIDDLE_JSON = _get_env_bool("MINERU_DUMP_MIDDLE_JSON", False)
MINERU_DUMP_MODEL_OUTPUT = _get_env_bool("MINERU_DUMP_MODEL_OUTPUT", False)

MINERU_DEVICE_MODE = os.getenv("MINERU_DEVICE_MODE", "auto")
MINERU_OUTPUT_RETENTION_HOURS = _get_env_int("MINERU_OUTPUT_RETENTION_HOURS", 2400) or 2400

# ===== 知识库与系统参数 =====

KB_NAME = "华东理工大学"  # 知识库主题，用于deepsearch
workers = _get_env_int("FASTAPI_WORKERS", 2) or 2  # FastAPI 并发进程数

# ===== 知识图谱配置 =====

theme = "华东理工大学学生管理"  # 知识图谱主题

entity_types = [
    "学生类型",
    "奖学金类型",
    "处分类型",
    "部门",
    "学生职责",
    "管理规定",
]  # 知识图谱实体类型

relationship_types = [
    "申请",
    "评选",
    "违纪",
    "资助",
    "申诉",
    "管理",
    "权利义务",
    "互斥",
]  # 知识图谱关系类型

# 冲突解决策略：manual_first / auto_first / merge
conflict_strategy = os.getenv("GRAPH_CONFLICT_STRATEGY", "manual_first")

# 社区检测算法：leiden / sllpa
community_algorithm = os.getenv("GRAPH_COMMUNITY_ALGORITHM", "leiden")

# ===== 文本处理配置 =====

CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 500) or 500  # 文本分块大小
OVERLAP = _get_env_int("CHUNK_OVERLAP", 100) or 100  # 分块重叠长度
MAX_TEXT_LENGTH = _get_env_int("MAX_TEXT_LENGTH", 500000) or 500000  # 最大文本长度
similarity_threshold = _get_env_float("SIMILARITY_THRESHOLD", 0.9) or 0.9  # 向量相似度阈值

# ===== 回答生成配置 =====

response_type = os.getenv("RESPONSE_TYPE", "多个段落")  # 默认回答形式

# ===== Agent 工具描述 =====

lc_description = (
    "用于需要具体细节的查询。检索华东理工大学学生管理文件中的具体规定、条款、流程等详细内容。"
    "适用于'某个具体规定是什么'、'处理流程如何'等问题。"
)
gl_description = (
    "用于需要总结归纳的查询。分析华东理工大学学生管理体系的整体框架、管理原则、学生权利义务等宏观内容。"
    "适用于'学校的学生管理总体思路'、'学生权益保护机制'等需要系统性分析的问题。"
)
naive_description = (
    "基础检索工具，直接查找与问题最相关的文本片段，不做复杂分析。快速获取华东理工大学相关政策，返回最匹配的原文段落。"
)

examples = [
    "旷课多少学时会被退学？",
    "国家奖学金和国家励志奖学金互斥吗？",
    "优秀学生要怎么申请？",
    "那上海市奖学金呢？",
]  # 前端示例问题

# ===== 性能优化配置 =====

MAX_WORKERS = _get_env_int("MAX_WORKERS", 4) or 4  # 并行工作线程数
BATCH_SIZE = _get_env_int("BATCH_SIZE", 100) or 100  # 批处理大小
ENTITY_BATCH_SIZE = _get_env_int("ENTITY_BATCH_SIZE", 50) or 50  # 实体批次大小
CHUNK_BATCH_SIZE = _get_env_int("CHUNK_BATCH_SIZE", 100) or 100  # 文本块批次
EMBEDDING_BATCH_SIZE = _get_env_int("EMBEDDING_BATCH_SIZE", 64) or 64  # 向量批次
LLM_BATCH_SIZE = _get_env_int("LLM_BATCH_SIZE", 5) or 5  # LLM 批次
COMMUNITY_BATCH_SIZE = _get_env_int("COMMUNITY_BATCH_SIZE", 50) or 50  # 社区批次大小

# ===== GDS 相关配置 =====

GDS_MEMORY_LIMIT = _get_env_int("GDS_MEMORY_LIMIT", 6) or 6  # GDS 内存限制(GB)
GDS_CONCURRENCY = _get_env_int("GDS_CONCURRENCY", 4) or 4  # GDS 并发度
GDS_NODE_COUNT_LIMIT = _get_env_int("GDS_NODE_COUNT_LIMIT", 50000) or 50000  # 节点上限
GDS_TIMEOUT_SECONDS = _get_env_int("GDS_TIMEOUT_SECONDS", 300) or 300  # 超时时长

# ===== 实体消歧与对齐配置 =====

DISAMBIG_STRING_THRESHOLD = _get_env_float("DISAMBIG_STRING_THRESHOLD", 0.7) or 0.7
DISAMBIG_VECTOR_THRESHOLD = _get_env_float("DISAMBIG_VECTOR_THRESHOLD", 0.85) or 0.85
DISAMBIG_NIL_THRESHOLD = _get_env_float("DISAMBIG_NIL_THRESHOLD", 0.6) or 0.6
DISAMBIG_TOP_K = _get_env_int("DISAMBIG_TOP_K", 5) or 5

ALIGNMENT_CONFLICT_THRESHOLD = (
    _get_env_float("ALIGNMENT_CONFLICT_THRESHOLD", 0.5) or 0.5
)
ALIGNMENT_MIN_GROUP_SIZE = _get_env_int("ALIGNMENT_MIN_GROUP_SIZE", 2) or 2

# ===== 路径与缓存配置 =====

DEFAULT_CACHE_ROOT = Path(
    os.getenv("CACHE_ROOT", PROJECT_ROOT / "cache")
).expanduser()
MODEL_CACHE_ROOT = Path(
    os.getenv("MODEL_CACHE_ROOT", DEFAULT_CACHE_ROOT)
).expanduser()
MODEL_CACHE_DIR = MODEL_CACHE_ROOT / "model"
CACHE_DIR = Path(os.getenv("CACHE_DIR", DEFAULT_CACHE_ROOT)).expanduser()
TIKTOKEN_CACHE_DIR = Path(
    os.getenv("TIKTOKEN_CACHE_DIR", DEFAULT_CACHE_ROOT / "tiktoken")
).expanduser()
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

SENTENCE_TRANSFORMER_MODELS = [
    item.strip()
    for item in os.getenv("SENTENCE_TRANSFORMER_MODELS", "").split(",")
    if item.strip()
]  # 预加载的本地模型列表
CACHE_EMBEDDING_PROVIDER = os.getenv(
    "CACHE_EMBEDDING_PROVIDER", "openai"
).lower()
CACHE_SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "CACHE_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
)

CACHE_SETTINGS = {
    "dir": CACHE_DIR,
    "memory_only": _get_env_bool("CACHE_MEMORY_ONLY", False),
    "max_memory_size": _get_env_int("CACHE_MAX_MEMORY_SIZE", 100) or 100,
    "max_disk_size": _get_env_int("CACHE_MAX_DISK_SIZE", 1000) or 1000,
    "thread_safe": _get_env_bool("CACHE_THREAD_SAFE", True),
    "enable_vector_similarity": _get_env_bool(
        "CACHE_ENABLE_VECTOR_SIMILARITY", True
    ),
    "similarity_threshold": _get_env_float(
        "CACHE_SIMILARITY_THRESHOLD", similarity_threshold
    )
    or similarity_threshold,
    "max_vectors": _get_env_int("CACHE_MAX_VECTORS", 10000) or 10000,
}

# ===== Neo4j 连接配置 =====

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_MAX_POOL_SIZE = _get_env_int("NEO4J_MAX_POOL_SIZE", 10) or 10
NEO4J_REFRESH_SCHEMA = _get_env_bool("NEO4J_REFRESH_SCHEMA", False)

NEO4J_CONFIG = {
    "uri": NEO4J_URI,
    "username": NEO4J_USERNAME,
    "password": NEO4J_PASSWORD,
    "max_pool_size": NEO4J_MAX_POOL_SIZE,
    "refresh_schema": NEO4J_REFRESH_SCHEMA,
}

# ===== LLM 与嵌入模型配置 =====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL") or None
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL") or None
LLM_TEMPERATURE = _get_env_float("TEMPERATURE", None)
LLM_MAX_TOKENS = _get_env_int("MAX_TOKENS", None)

OPENAI_EMBEDDING_CONFIG = {
    "model": OPENAI_EMBEDDINGS_MODEL,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

OPENAI_LLM_CONFIG = {
    "model": OPENAI_LLM_MODEL,
    "temperature": LLM_TEMPERATURE,
    "max_tokens": LLM_MAX_TOKENS,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_VISION_CONFIG = {
    "model": OPENAI_VISION_MODEL,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

VISION_MAX_IMAGE_COUNT = _get_env_int("VISION_MAX_IMAGE_COUNT", 3) or 3
VISION_IMAGE_MAX_SIZE = _get_env_int_pair("VISION_IMAGE_MAX_SIZE", (1024, 1024))
VISION_IMAGE_QUALITY = _get_env_int("VISION_IMAGE_QUALITY", 85) or 85
VISION_IMAGE_DETAIL = _get_env_choice(
    "VISION_IMAGE_DETAIL",
    {"low", "high", "auto"},
    "high",
)

VISION_IMAGE_SETTINGS = {
    "max_count": VISION_MAX_IMAGE_COUNT,
    "max_size": VISION_IMAGE_MAX_SIZE,
    "quality": VISION_IMAGE_QUALITY,
    "detail": VISION_IMAGE_DETAIL,
}

# ===== 相似实体检测参数 =====

SIMILAR_ENTITY_SETTINGS = {
    "word_edit_distance": _get_env_int("SIMILAR_ENTITY_WORD_EDIT_DISTANCE", 3) or 3,
    "batch_size": _get_env_int("SIMILAR_ENTITY_BATCH_SIZE", 500) or 500,
    "memory_limit": _get_env_int(
        "SIMILAR_ENTITY_MEMORY_LIMIT", GDS_MEMORY_LIMIT
    )
    or GDS_MEMORY_LIMIT,
    "top_k": _get_env_int("SIMILAR_ENTITY_TOP_K", 10) or 10,
}

# ===== 搜索工具配置 =====

BASE_SEARCH_CONFIG = {
    "cache_max_size": _get_env_int("SEARCH_CACHE_MEMORY_SIZE", 200) or 200,
    "vector_limit": _get_env_int("SEARCH_VECTOR_LIMIT", 5) or 5,
    "text_limit": _get_env_int("SEARCH_TEXT_LIMIT", 5) or 5,
    "semantic_top_k": _get_env_int("SEARCH_SEMANTIC_TOP_K", 5) or 5,
    "relevance_top_k": _get_env_int("SEARCH_RELEVANCE_TOP_K", 5) or 5,
}

LOCAL_SEARCH_SETTINGS = {
    "top_chunks": _get_env_int("LOCAL_SEARCH_TOP_CHUNKS", 3) or 3,
    "top_communities": _get_env_int("LOCAL_SEARCH_TOP_COMMUNITIES", 3) or 3,
    "top_outside_relationships": _get_env_int(
        "LOCAL_SEARCH_TOP_OUTSIDE_RELS", 10
    )
    or 10,
    "top_inside_relationships": _get_env_int(
        "LOCAL_SEARCH_TOP_INSIDE_RELS", 10
    )
    or 10,
    "top_entities": _get_env_int("LOCAL_SEARCH_TOP_ENTITIES", 10) or 10,
    "index_name": os.getenv("LOCAL_SEARCH_INDEX_NAME", "vector"),
}

GLOBAL_SEARCH_SETTINGS = {
    "default_level": _get_env_int("GLOBAL_SEARCH_LEVEL", 0) or 0,
    "community_batch_size": _get_env_int("GLOBAL_SEARCH_BATCH_SIZE", 5) or 5,
    "community_chunk_limit": _get_env_int("GLOBAL_SEARCH_CHUNK_LIMIT", 5) or 5,
}

NAIVE_SEARCH_TOP_K = _get_env_int("NAIVE_SEARCH_TOP_K", 3) or 3
_raw_naive_candidate_limit = _get_env_int("NAIVE_SEARCH_CANDIDATE_LIMIT", 2000)
NAIVE_SEARCH_CANDIDATE_LIMIT = (
    None if _raw_naive_candidate_limit is None or _raw_naive_candidate_limit <= 0 else _raw_naive_candidate_limit
)

HYBRID_SEARCH_SETTINGS = {
    "entity_limit": _get_env_int("HYBRID_SEARCH_ENTITY_LIMIT", 15) or 15,
    "max_hop_distance": _get_env_int("HYBRID_SEARCH_MAX_HOP", 2) or 2,
    "top_communities": _get_env_int("HYBRID_SEARCH_TOP_COMMUNITIES", 3) or 3,
    "batch_size": _get_env_int("HYBRID_SEARCH_BATCH_SIZE", 10) or 10,
    "community_level": _get_env_int("HYBRID_SEARCH_COMMUNITY_LEVEL", 0) or 0,
}

# ===== Agent 配置 =====

AGENT_SETTINGS = {
    "default_recursion_limit": _get_env_int("AGENT_RECURSION_LIMIT", 5) or 5,
    "chunk_size": _get_env_int("AGENT_CHUNK_SIZE", 4) or 4,
    "stream_flush_threshold": _get_env_int("AGENT_STREAM_FLUSH_THRESHOLD", 40)
    or 40,
    "deep_stream_flush_threshold": _get_env_int(
        "DEEP_AGENT_STREAM_FLUSH_THRESHOLD", 80
    )
    or 80,
    "fusion_stream_flush_threshold": _get_env_int(
        "FUSION_AGENT_STREAM_FLUSH_THRESHOLD", 60
    )
    or 60,
}

# ===== 多智能体（Plan-Execute-Report）配置 =====

MULTI_AGENT_PLANNER_MAX_TASKS = _get_env_int("MA_PLANNER_MAX_TASKS", 6) or 6
MULTI_AGENT_ALLOW_UNCLARIFIED_PLAN = _get_env_bool("MA_ALLOW_UNCLARIFIED_PLAN", True)
MULTI_AGENT_DEFAULT_DOMAIN = os.getenv("MA_DEFAULT_DOMAIN", "通用")

MULTI_AGENT_AUTO_GENERATE_REPORT = _get_env_bool("MA_AUTO_GENERATE_REPORT", True)
MULTI_AGENT_STOP_ON_CLARIFICATION = _get_env_bool("MA_STOP_ON_CLARIFICATION", True)
MULTI_AGENT_STRICT_PLAN_SIGNAL = _get_env_bool("MA_STRICT_PLAN_SIGNAL", True)

MULTI_AGENT_DEFAULT_REPORT_TYPE = os.getenv("MA_DEFAULT_REPORT_TYPE", "long_document")
MULTI_AGENT_ENABLE_CONSISTENCY_CHECK = _get_env_bool(
    "MA_ENABLE_CONSISTENCY_CHECK", True
)
MULTI_AGENT_ENABLE_MAPREDUCE = _get_env_bool("MA_ENABLE_MAPREDUCE", True)
MULTI_AGENT_MAPREDUCE_THRESHOLD = _get_env_int("MA_MAPREDUCE_THRESHOLD", 20) or 20
MULTI_AGENT_MAX_TOKENS_PER_REDUCE = (
    _get_env_int("MA_MAX_TOKENS_PER_REDUCE", 4000) or 4000
)
MULTI_AGENT_ENABLE_PARALLEL_MAP = _get_env_bool("MA_ENABLE_PARALLEL_MAP", True)

MULTI_AGENT_SECTION_MAX_EVIDENCE = (
    _get_env_int("MA_SECTION_MAX_EVIDENCE", 8) or 8
)
MULTI_AGENT_SECTION_MAX_CONTEXT_CHARS = (
    _get_env_int("MA_SECTION_MAX_CONTEXT_CHARS", 800) or 800
)
MULTI_AGENT_REFLECTION_ALLOW_RETRY = _get_env_bool(
    "MA_REFLECTION_ALLOW_RETRY", False
)
MULTI_AGENT_REFLECTION_MAX_RETRIES = (
    _get_env_int("MA_REFLECTION_MAX_RETRIES", 1) or 1
)
MULTI_AGENT_WORKER_EXECUTION_MODE = _get_env_choice(
    "MA_WORKER_EXECUTION_MODE",
    {"sequential", "parallel"},
    "sequential",
)
MULTI_AGENT_WORKER_MAX_CONCURRENCY = (
    _get_env_int("MA_WORKER_MAX_CONCURRENCY", MAX_WORKERS) or MAX_WORKERS
)
if MULTI_AGENT_WORKER_MAX_CONCURRENCY < 1:
    raise ValueError("MA_WORKER_MAX_CONCURRENCY 必须大于等于 1")
