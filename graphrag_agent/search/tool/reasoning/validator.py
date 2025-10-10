from typing import Dict, List

class AnswerValidator:
    """
    答案验证器：评估生成答案的质量，确保满足基本要求
    """
    
    def __init__(self, keyword_extractor=None):
        """
        初始化验证器
        
        参数:
            keyword_extractor: 用于提取关键词的函数或对象
        """
        self.keyword_extractor = keyword_extractor
        self.error_patterns = [
            "抱歉，处理您的问题时遇到了错误",
            "技术原因:",
            "无法获取",
            "无法回答这个问题",
            "没有找到相关信息",
            "对不起，我不能"
        ]
    
    def validate(self, query: str, answer: str) -> Dict[str, bool]:
        """
        验证生成答案的质量
        
        参数:
            query: 原始查询
            answer: 生成的答案
            
        返回:
            Dict[str, bool]: 各项验证的结果
        """
        results = {}
        
        # 检查最小长度
        results["length"] = len(answer) >= 50
        if not results["length"]:
            print(f"[验证] 答案太短: {len(answer)}字符")
        
        # 检查是否包含错误模式
        results["no_error_patterns"] = not any(pattern in answer for pattern in self.error_patterns)
        if not results["no_error_patterns"]:
            for pattern in self.error_patterns:
                if pattern in answer:
                    print(f"[验证] 答案包含错误模式: {pattern}")
                    break
        
        # 关键词相关性检查
        results["keyword_relevance"] = self._check_keyword_relevance(query, answer)
        
        # 总体通过验证
        results["passed"] = all(results.values())
        
        return results
    
    def _check_keyword_relevance(self, query: str, answer: str) -> bool:
        """
        检查答案是否包含查询的关键词
        
        参数:
            query: 查询字符串
            answer: 生成的答案
            
        返回:
            bool: 是否满足关键词相关性要求
        """
        # 如果没有关键词提取器，则默认通过
        if not self.keyword_extractor:
            return True
            
        # 提取关键词
        keywords = self.keyword_extractor(query)
        if not keywords:
            return True
            
        high_level_keywords = keywords.get("high_level", [])
        low_level_keywords = keywords.get("low_level", [])
        
        # 至少有一个高级关键词应该在答案中出现
        if high_level_keywords:
            keyword_found = any(keyword.lower() in answer.lower() for keyword in high_level_keywords)
            if not keyword_found:
                print(f"[验证] 答案未包含任何高级关键词: {high_level_keywords}")
                return False
                
        # 至少有一半的低级关键词应该在答案中出现
        if low_level_keywords and len(low_level_keywords) > 1:
            matches = sum(1 for keyword in low_level_keywords if keyword.lower() in answer.lower())
            if matches < len(low_level_keywords) / 2:
                print(f"[验证] 答案未包含足够的低级关键词: {matches}/{len(low_level_keywords)}")
                return False
        
        print("[验证] 答案通过关键词相关性检查")
        return True

def complexity_estimate(query: str) -> float:
    """
    估计查询复杂度
    
    Args:
        query: 查询字符串
        
    Returns:
        float: 复杂度评分(0.0-1.0)
    """
    # 添加None检查和类型验证
    if query is None:
        print(f'complexity_estimate: 返回0，因为query:{query}为空\n')
        return 0.0
    
    # 确保query是字符串
    if not isinstance(query, str):
        query = str(query) if query is not None else ""
    
    # 如果查询为空，返回0
    if not query.strip():
        print(f'complexity_estimate: 返回0，因为query:{query}为空\n')
        return 0.0
    
    try:
        # 基于查询长度、问号数量和关键词数量的简单启发式方法
        length_factor = min(1.0, len(query) / 100)
        question_marks = query.count("?") + query.count("？")
        question_factor = min(1.0, question_marks * 0.2)
        
        # 识别复杂问题的关键词
        complexity_indicators = [
            "为什么", "如何", "机制", "原因", "关系", "比较", "区别",
            "影响", "分析", "评估", "预测", "如果", "假设", "还是",
            "多少", "怎样", "多大", "是否", "哪些", "优缺点"
        ]
        
        # 检查关键词
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query)
        indicator_factor = min(1.0, indicator_count * 0.15)
        
        # 综合评分
        if all(factor is not None for factor in [length_factor, question_factor, indicator_factor]):
            complexity = (length_factor * 0.3 + question_factor * 0.3 + indicator_factor * 0.4)
            return min(1.0, max(0.0, complexity))  # 确保在0-1范围内
        else:
            return 0.5  # 默认中等复杂度
            
    except Exception as e:
        print(f"计算查询复杂度时出错: {e}")
        return 0.5  # 出错时返回默认值