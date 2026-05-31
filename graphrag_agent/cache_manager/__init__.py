from .strategies import (
    CacheKeyStrategy,
    SimpleCacheKeyStrategy,
    ContextAwareCacheKeyStrategy,
    ContextAndKeywordAwareCacheKeyStrategy
)

from .backends import (
    CacheStorageBackend,
    MemoryCacheBackend,
    DiskCacheBackend,
    HybridCacheBackend,
    ThreadSafeCacheBackend
)

from .models import CacheItem
from .manager import CacheManager
from .vector_similarity import VectorSimilarityMatcher
from .model_cache import initialize_model_cache, ensure_model_cache_dir

__all__ = [
    # Key strategies
    'CacheKeyStrategy',
    'SimpleCacheKeyStrategy',
    'ContextAwareCacheKeyStrategy',
    'ContextAndKeywordAwareCacheKeyStrategy',

    # Storage backends
    'CacheStorageBackend',
    'MemoryCacheBackend',
    'DiskCacheBackend',
    'HybridCacheBackend',
    'ThreadSafeCacheBackend',

    # Models
    'CacheItem',

    # Main manager
    'CacheManager',

    # Vector similarity
    'VectorSimilarityMatcher',

    # Model cache
    'initialize_model_cache',
    'ensure_model_cache_dir'
]