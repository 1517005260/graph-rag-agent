"""
模型缓存管理模块，用于预加载和管理模型缓存
"""

from typing import List, Optional

from graphrag_agent.config.settings import (
    MODEL_CACHE_DIR,
    SENTENCE_TRANSFORMER_MODELS,
    CACHE_EMBEDDING_PROVIDER,
    CACHE_SENTENCE_TRANSFORMER_MODEL,
)


def ensure_model_cache_dir() -> str:
    """确保模型缓存目录存在，并返回路径"""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODEL_CACHE_DIR)


def preload_sentence_transformer_models(models: Optional[List[str]] = None) -> None:
    """预加载SentenceTransformer模型到缓存目录"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # 获取要预加载的模型列表
        if models is None:
            models = list(SENTENCE_TRANSFORMER_MODELS)
        
        if not models:
            return
            
        # 获取缓存目录
        cache_dir = ensure_model_cache_dir()

        # 加载每个模型
        for model_name in models:
            try:
                # 加载模型，指定缓存目录
                _ = SentenceTransformer(model_name, cache_folder=cache_dir)
                print(f"模型 {model_name} 加载成功")
            except Exception as e:
                print(f"加载模型 {model_name} 失败: {e}")

    except ImportError as e:
        print(f"未安装sentence_transformers，跳过预加载: {e}")


def preload_cache_embedding_model() -> None:
    """预加载缓存使用的嵌入模型"""
    provider_type = CACHE_EMBEDDING_PROVIDER
    
    if provider_type == 'openai':
        # OpenAI模型不需要预加载
        return
    
    # 预加载SentenceTransformer模型
    model_name = CACHE_SENTENCE_TRANSFORMER_MODEL
    preload_sentence_transformer_models([model_name])


def initialize_model_cache() -> None:
    """初始化模型缓存，预加载配置的模型"""
    # 确保缓存目录存在
    cache_dir = ensure_model_cache_dir()

    # 预加载缓存使用的嵌入模型
    preload_cache_embedding_model()


if __name__ == "__main__":
    # 直接运行此脚本可以预加载模型
    initialize_model_cache()
