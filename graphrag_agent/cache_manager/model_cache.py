"""
模型缓存管理模块，用于预加载和管理模型缓存
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelCache")

# 加载环境变量
load_dotenv()


def ensure_model_cache_dir() -> str:
    """确保模型缓存目录存在，并返回路径"""
    cache_root = os.getenv('MODEL_CACHE_ROOT', './cache')
    model_cache_dir = os.path.join(cache_root, 'model')
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    return model_cache_dir


def preload_sentence_transformer_models(models: Optional[List[str]] = None) -> None:
    """预加载SentenceTransformer模型到缓存目录"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # 获取要预加载的模型列表
        if models is None:
            models_str = os.getenv('SENTENCE_TRANSFORMER_MODELS', '')
            if not models_str:
                return
            models = [m.strip() for m in models_str.split(',') if m.strip()]
        
        if not models:
            return
            
        # 获取缓存目录
        cache_dir = ensure_model_cache_dir()
        logger.info(f"预加载SentenceTransformer模型到 {cache_dir}")
        
        # 加载每个模型
        for model_name in models:
            try:
                logger.info(f"加载模型: {model_name}")
                # 加载模型，指定缓存目录
                _ = SentenceTransformer(model_name, cache_folder=cache_dir)
                logger.info(f"模型 {model_name} 加载成功")
            except Exception as e:
                logger.error(f"加载模型 {model_name} 失败: {e}")
                
    except ImportError:
        logger.warning("未安装sentence_transformers，跳过预加载")


def preload_cache_embedding_model() -> None:
    """预加载缓存使用的嵌入模型"""
    provider_type = os.getenv('CACHE_EMBEDDING_PROVIDER', 'sentence_transformer').lower()
    
    if provider_type == 'openai':
        # OpenAI模型不需要预加载
        logger.info("使用OpenAI作为缓存嵌入提供者，无需预加载模型")
        return
    
    # 预加载SentenceTransformer模型
    model_name = os.getenv('CACHE_SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    preload_sentence_transformer_models([model_name])


def initialize_model_cache() -> None:
    """初始化模型缓存，预加载配置的模型"""
    logger.info("初始化模型缓存...")
    
    # 确保缓存目录存在
    cache_dir = ensure_model_cache_dir()
    logger.info(f"模型缓存目录: {cache_dir}")
    
    # 预加载缓存使用的嵌入模型
    preload_cache_embedding_model()
    
    logger.info("模型缓存初始化完成")


if __name__ == "__main__":
    # 直接运行此脚本可以预加载模型
    initialize_model_cache()
