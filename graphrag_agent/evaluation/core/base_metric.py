from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseMetric(ABC):
    """所有评估指标的基类"""
    
    # 指标名称，子类必须重写
    metric_name = "base"
    
    def __init__(self, config):
        """
        初始化评估指标基类
        
        Args:
            config: 评估配置
        """
        # 支持字典或EvaluatorConfig对象
        if isinstance(config, dict):
            from evaluator.evaluator_config.evaluatorConfig import EvaluatorConfig
            self.config = EvaluatorConfig(config)
        else:
            self.config = config
            
        self.dataset_name = self.config.get('dataset_name', 'default')
        self.debug = self.config.get('debug', False)
        # 获取LLM模型，用于回退评估
        self.llm = self.config.get('llm', None)
    
    @abstractmethod
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List]:
        """
        计算评估指标
        
        Args:
            data: 评估数据对象
            
        Returns:
            Tuple[Dict, List]: 评估结果和每个样本的评分
        """
        return {}, []
    
    def log(self, message, *args, **kwargs):
        """
        输出调试日志
        
        Args:
            message: 日志消息
            *args, **kwargs: 额外参数
        """
        from evaluator import debug_print
        if self.debug:
            debug_print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)
            
    def get_llm_fallback_score(self, prompt: str, default_score: float = 0.5) -> float:
        """
        使用LLM进行回退评分
        
        Args:
            prompt: 提示文本
            default_score: 默认分数，当LLM评分失败时返回
            
        Returns:
            float: LLM评分结果或默认分数
        """
        # 如果没有LLM，直接返回默认分数
        if not self.llm:
            self.log(f"  LLM不可用，使用默认分数: {default_score:.4f}")
            return default_score
            
        try:
            self.log("  正在使用LLM进行回退评分...")
            response = self.llm.invoke(prompt)
            score_text = response.content if hasattr(response, 'content') else response
            
            self.log(f"  LLM响应: {score_text}")
            
            # 提取数字
            import re
            score_match = re.search(r'(\d+(\.\d+)?)', score_text)
            if score_match:
                extracted_score = float(score_match.group(1))
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, extracted_score))
                self.log(f"  LLM评分结果: {score:.4f}")
                return score
            else:
                self.log(f"  无法从LLM响应中提取分数，使用默认分数: {default_score:.4f}")
                return default_score
        except Exception as e:
            self.log(f"  LLM评分出错: {e}")
            return default_score