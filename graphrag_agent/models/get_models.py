import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

from graphrag_agent.config.settings import (
    TIKTOKEN_CACHE_DIR,
    OPENAI_EMBEDDING_CONFIG,
    OPENAI_LLM_CONFIG,
    EMBEDDING_DIM,
)


# 设置 tiktoken 缓存目录，避免每次联网拉取
def setup_cache():
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)


setup_cache()


def get_embeddings_model():
    _validate_embedding_dimension(OPENAI_EMBEDDING_CONFIG.get("model"))

    config = {k: v for k, v in OPENAI_EMBEDDING_CONFIG.items() if v}
    configured_dim = config.get("dimensions")
    if configured_dim is not None and int(configured_dim) != int(EMBEDDING_DIM):
        raise ValueError(
            f"OPENAI_EMBEDDINGS_MODEL 的维度 {configured_dim} 与 EMBEDDING_DIM="
            f"{EMBEDDING_DIM} 不一致，请更新配置或 .env。"
        )
    return OpenAIEmbeddings(**config)


def get_llm_model():
    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    return ChatOpenAI(**config)


def get_stream_llm_model():
    callback_handler = AsyncIteratorCallbackHandler()
    # 将回调handler放进AsyncCallbackManager中
    manager = AsyncCallbackManager(handlers=[callback_handler])

    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    config.update({"streaming": True, "callbacks": manager})
    return ChatOpenAI(**config)


def count_tokens(text):
    """简单通用的token计数"""
    if not text:
        return 0
    
    model_name = (OPENAI_LLM_CONFIG.get("model") or "").lower()
    
    # 如果是deepseek，使用transformers
    if 'deepseek' in model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
            return len(tokenizer.encode(text))
        except:
            pass
    
    # 如果是gpt，使用tiktoken
    if 'gpt' in model_name:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            pass
    
    # 备用方案：简单计算
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4


_EXPECTED_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 1536,
    "text-embedding-3-small": 512,
    "text-embedding-ada-002": 1536,
}


def _validate_embedding_dimension(model_name: Optional[str]) -> None:
    """
    Ensure EMBEDDING_DIM matches the selected embedding model.

    Args:
        model_name: Embedding model configured for OpenAI embeddings.
    """
    if not model_name:
        return

    expected_dimension = _EXPECTED_EMBEDDING_DIMENSIONS.get(model_name)
    if expected_dimension is None:
        return

    if int(expected_dimension) != int(EMBEDDING_DIM):
        raise ValueError(
            f"配置的 EMBEDDING_DIM={EMBEDDING_DIM} 与模型 {model_name} 的"
            f" 维度 {expected_dimension} 不一致，请同步更新配置或 .env。"
        )


if __name__ == '__main__':
    # 测试llm
    llm = get_llm_model()
    print(llm.invoke("你好"))

    # 由于langchain版本问题，这个目前测试会报错
    # llm_stream = get_stream_llm_model()
    # print(llm_stream.invoke("你好"))

    # 测试embedding
    test_text = "你好，这是一个测试。"
    embeddings = get_embeddings_model()
    print(embeddings.embed_query(test_text))

    # 测试计数
    test_text = "Hello 你好世界"
    tokens = count_tokens(test_text)
    print(f"Token计数: '{test_text}' = {tokens} tokens")
