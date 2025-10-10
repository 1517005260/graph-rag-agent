import os
import json
from typing import List, Dict, Any, Optional, Union

def save_json(data: Any, file_path: str, ensure_ascii: bool = False, indent: int = 2):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        ensure_ascii: 是否确保ASCII编码
        indent: 缩进空格数
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)

def load_json(file_path: str) -> Any:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_questions_from_data(data: Union[List, Dict], field_name: str = "question") -> List[str]:
    """
    从数据中提取问题
    
    Args:
        data: 数据源
        field_name: 问题字段名称
        
    Returns:
        List[str]: 问题列表
    """
    questions = []
    
    # 如果是列表
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and field_name in item:
                questions.append(item[field_name])
            elif isinstance(item, str):
                questions.append(item)
    # 如果是字典
    elif isinstance(data, dict):
        if field_name in data:
            questions.append(data[field_name])
        else:
            # 尝试寻找可能的问题字段
            possible_fields = ["question", "q", "query", "text", "content"]
            for field in possible_fields:
                if field in data:
                    questions.append(data[field])
                    break
    
    return questions

def extract_answers_from_data(data: Union[List, Dict], field_name: str = "answer") -> List[str]:
    """
    从数据中提取答案
    
    Args:
        data: 数据源
        field_name: 答案字段名称
        
    Returns:
        List[str]: 答案列表
    """
    answers = []
    
    # 如果是列表
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and field_name in item:
                answers.append(item[field_name])
            elif isinstance(item, str):
                answers.append(item)
    # 如果是字典
    elif isinstance(data, dict):
        if field_name in data:
            answers.append(data[field_name])
        else:
            # 尝试寻找可能的答案字段
            possible_fields = ["answer", "a", "response", "text", "content"]
            for field in possible_fields:
                if field in data:
                    answers.append(data[field])
                    break
    
    return answers