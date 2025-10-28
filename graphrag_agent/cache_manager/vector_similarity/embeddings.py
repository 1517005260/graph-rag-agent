import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import threading
from pathlib import Path

try:  # 允许在缺失 sentence_transformers/torch 时降级
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # 捕获 ImportError 及其依赖错误
    SentenceTransformer = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

from graphrag_agent.config.settings import (
    MODEL_CACHE_DIR,
    CACHE_EMBEDDING_PROVIDER,
    CACHE_SENTENCE_TRANSFORMER_MODEL,
)


class EmbeddingProvider(ABC):
    """嵌入向量提供者抽象基类"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本编码为向量"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于OpenAI API的嵌入向量提供者，复用RAG的向量模型"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式，避免重复创建"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 导入并复用现有的embedding模型
        try:
            from graphrag_agent.models.get_models import get_embeddings_model
            self.model = get_embeddings_model()
            self._dimension = None
            self._initialized = True
        except ImportError as e:
            raise ImportError(f"无法导入embedding模型: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        # 使用OpenAI embedding模型
        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # 归一化向量
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self._dimension is None:
            # 使用一个简单文本获取维度
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension


class SentenceTransformerEmbedding(EmbeddingProvider):
    """基于SentenceTransformer的嵌入向量提供者，支持模型缓存"""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        """单例模式，避免重复加载模型"""
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = super().__new__(cls)
                cls._instances[model_name]._initialized = False
            return cls._instances[model_name]

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.model_name = model_name

        if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            raise ImportError(
                "需要安装 sentence-transformers/torch 才能使用 SentenceTransformerEmbedding。"
                " 可执行 `pip install sentence-transformers torch`，"
                " 或在环境变量/配置中将 CACHE_EMBEDDING_PROVIDER 设置为 'openai'。"
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

        # 设置模型缓存目录
        if cache_dir is None:
            cache_dir = MODEL_CACHE_DIR

        # 确保缓存目录存在
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 加载模型，指定缓存目录
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_path))
        self._dimension = None
        self._initialized = True

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self._dimension is None:
            # 使用一个简单文本获取维度
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension


def get_cache_embedding_provider() -> EmbeddingProvider:
    """根据配置获取缓存向量提供者"""
    provider_type = CACHE_EMBEDDING_PROVIDER

    if provider_type == 'openai':
        return OpenAIEmbeddingProvider()
    else:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            warnings.warn(
                "未检测到 sentence-transformers/torch，自动回退到 OpenAI embedding。"
                " 若需使用本地 SentenceTransformer，请安装依赖或在配置中显式设置。",
                RuntimeWarning,
            )
            return OpenAIEmbeddingProvider()
        # 使用sentence transformer
        model_name = CACHE_SENTENCE_TRANSFORMER_MODEL
        return SentenceTransformerEmbedding(model_name=model_name, cache_dir=MODEL_CACHE_DIR)
