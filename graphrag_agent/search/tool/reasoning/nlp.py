import re
from typing import List, Optional

def extract_between(text: str, start_marker: str, end_marker: str) -> List[str]:
    """
    提取起始和结束标记之间的内容
    
    参数:
        text: 要搜索的文本
        start_marker: 起始标记
        end_marker: 结束标记
        
    返回:
        List[str]: 提取的内容字符串列表
    """
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    return re.findall(pattern, text, flags=re.DOTALL)

def extract_from_templates(text: str, templates: List[str], regex: bool = False) -> List[str]:
    """
    基于带占位符的模板提取内容
    
    参数:
        text: 要搜索的文本
        templates: 带{}占位符的模板字符串列表
        regex: 是否将模板作为正则表达式处理
        
    返回:
        List[str]: 提取的内容字符串列表
    """
    results = []
    
    for template in templates:
        if regex:
            # 直接使用模板作为正则表达式
            matches = re.findall(template, text, re.DOTALL)
            results.extend(matches)
        else:
            # 将模板转换为正则表达式（通过转义和替换占位符）
            pattern = template.replace("{}", "(.*?)")
            pattern = re.escape(pattern).replace("\\(\\*\\*\\?\\)", "(.*?)")
            matches = re.findall(pattern, text, re.DOTALL)
            results.extend(matches)
    
    return results

def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """
    从文本中提取句子
    
    参数:
        text: 要提取句子的文本
        max_sentences: 最大提取句子数
        
    返回:
        List[str]: 句子列表
    """
    if not text:
        return []
    
    # 简单的句子分割（可以使用NLP库进行改进）
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # 移除空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if max_sentences:
        return sentences[:max_sentences]
    return sentences