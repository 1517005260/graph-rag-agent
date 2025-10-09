import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Union
from sentence_transformers import SentenceTransformer
import threading
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


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
            from model.get_models import get_embeddings_model
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
            cache_root = os.getenv('MODEL_CACHE_ROOT', './cache')
            cache_dir = os.path.join(cache_root, 'model')

        # 确保缓存目录存在
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # 加载模型，指定缓存目录
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
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
    provider_type = os.getenv('CACHE_EMBEDDING_PROVIDER', 'sentence_transformer').lower()

    if provider_type == 'openai':
        return OpenAIEmbeddingProvider()
    else:
        # 使用sentence transformer
        model_name = os.getenv('CACHE_SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        cache_root = os.getenv('MODEL_CACHE_ROOT', './cache')
        cache_dir = os.path.join(cache_root, 'model')
        return SentenceTransformerEmbedding(model_name=model_name, cache_dir=cache_dir)