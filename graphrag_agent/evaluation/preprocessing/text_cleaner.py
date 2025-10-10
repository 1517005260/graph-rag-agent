import re

def clean_references(answer: str) -> str:
    """
    清理AI回答中的引用数据部分
    
    参数:
        answer: AI生成的回答
        
    返回:
        清理后的回答
    """
    # 移除引用数据部分
    cleaned = re.sub(r'###\s*引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', '', answer)
    
    # 如果没有引用数据部分，尝试其他格式
    if cleaned == answer:
        cleaned = re.sub(r'#### 引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', '', answer)
    
    # 移除任何尾部空行
    cleaned = cleaned.rstrip()
    
    return cleaned

def clean_thinking_process(answer: str) -> str:
    """
    清理deep agent回答中的思考过程
    
    参数:
        answer: AI生成的回答
        
    返回:
        清理后的回答，没有思考过程
    """
    # 移除思考过程部分
    cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', answer)
    
    # 移除任何多余的空行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned