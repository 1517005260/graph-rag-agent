import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
# from sentence_transformers import SentenceTransformer
import threading
from pathlib import Path
import os

ENABLE_SENTENCE_TRANSFORMERS = os.getenv("ENABLE_SENTENCE_TRANSFORMERS", "0") == "1"

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

        # 设置模型缓存目录
        if cache_dir is None:
            cache_dir = MODEL_CACHE_DIR

        # 确保缓存目录存在
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 加载模型，指定缓存目录
        if not ENABLE_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "SentenceTransformerEmbedding 被调用，但当前默认禁用了 sentence_transformers。"
                "如确实需要本地向量模型，请先执行：\n"
                "  export ENABLE_SENTENCE_TRANSFORMERS=1\n"
                "并确保已正确安装 sentence-transformers/torch。\n"
                "如果你只是跑 OpenAI embeddings（推荐），请设置：\n"
                "  export CACHE_EMBEDDING_PROVIDER=openai\n"
            )

        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433 (local import)
        except Exception as e:
            raise RuntimeError(
                f"无法导入 sentence_transformers（通常会引入 torch）。错误: {e}\n"
                "如果你不需要本地 embedding，请保持 CACHE_EMBEDDING_PROVIDER=openai。"
            ) from e
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
    provider_type = (CACHE_EMBEDDING_PROVIDER or "").lower()

    if provider_type in ("openai", ""):
        return OpenAIEmbeddingProvider()

    if provider_type in ("sentence_transformer", "sentence-transformer", "st"):
        if not ENABLE_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "CACHE_EMBEDDING_PROVIDER 配置为 sentence_transformer，但 ENABLE_SENTENCE_TRANSFORMERS 未开启。\n"
                "请执行：export ENABLE_SENTENCE_TRANSFORMERS=1\n"
                "或改回：export CACHE_EMBEDDING_PROVIDER=openai"
            )
        model_name = CACHE_SENTENCE_TRANSFORMER_MODEL
        return SentenceTransformerEmbedding(model_name=model_name, cache_dir=MODEL_CACHE_DIR)

    raise ValueError(
        f"不支持的 CACHE_EMBEDDING_PROVIDER={provider_type}，可选：openai / sentence_transformer"
    )

