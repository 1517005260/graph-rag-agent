import re
import string
from typing import List, Dict

def normalize_answer(s: str) -> str:
    """
    标准化答案文本，移除冠词、标点符号，转为小写，修复空格
    
    Args:
        s (str): 原始文本
        
    Returns:
        str: 标准化后的文本
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|一个|一种|这个|那个)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation + "，。！？《》【】""''：；（）、")
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_precision_recall_f1(pred: List[str], truth: List[str]) -> Dict[str, float]:
    """
    计算精确率、召回率、F1分数
    
    Args:
        pred (List[str]): 预测列表
        truth (List[str]): 真实列表
        
    Returns:
        Dict[str, float]: 包含precision, recall, f1的字典
    """
    if not pred or not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 标准化处理
    pred_norm = [normalize_answer(p) for p in pred]
    truth_norm = [normalize_answer(t) for t in truth]
    
    # 计算交集大小
    tp = len(set(pred_norm).intersection(set(truth_norm)))
    
    # 计算精确率和召回率
    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall = tp / len(truth_norm) if truth_norm else 0.0
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }