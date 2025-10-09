from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager


import os
from pathlib import Path
from dotenv import load_dotenv

# 设置tiktoken缓存避免网络问题
def setup_cache():
    cache_dir = Path.home() / "cache" / "tiktoken"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)

setup_cache()

load_dotenv()

def get_embeddings_model():
    model = OpenAIEmbeddings(
        model=os.getenv('OPENAI_EMBEDDINGS_MODEL'),
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
    return model


def get_llm_model():
    model = ChatOpenAI(
        model=os.getenv('OPENAI_LLM_MODEL'),
        temperature=os.getenv('TEMPERATURE'),
        max_tokens=os.getenv('MAX_TOKENS'),
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
    return model

def get_stream_llm_model():
    callback_handler = AsyncIteratorCallbackHandler()
    # 将回调handler放进AsyncCallbackManager中
    manager = AsyncCallbackManager(handlers=[callback_handler])

    model = ChatOpenAI(
        model=os.getenv('OPENAI_LLM_MODEL'),
        temperature=os.getenv('TEMPERATURE'),
        max_tokens=os.getenv('MAX_TOKENS'),
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
        streaming=True,
        callbacks=manager,
    )
    return model

def count_tokens(text):
    """简单通用的token计数"""
    if not text:
        return 0
    
    model_name = os.getenv('OPENAI_LLM_MODEL', '').lower()
    
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
