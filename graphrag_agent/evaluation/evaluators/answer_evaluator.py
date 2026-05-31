import os
from typing import Dict

from graphrag_agent.evaluation.core.base_evaluator import BaseEvaluator
from graphrag_agent.evaluation.core.evaluation_data import AnswerEvaluationData
class AnswerEvaluator(BaseEvaluator):
    """答案评估器，用于评估系统回答的质量"""
    
    def __init__(self, config):
        """
        初始化答案评估器
        
        Args:
            config: 评估配置
        """
        super().__init__(config)
    
    def evaluate(self, data: AnswerEvaluationData) -> Dict[str, float]:
        """
        执行评估 - 修复版本，解决 LLM 评估器的字典类型问题
        """
        self.log("\n======== 开始评估答案质量 ========")
        self.log(f"样本总数: {len(data.samples)}")
        self.log(f"使用的评估指标: {', '.join(self.metrics)}")
        
        result_dict = {}
        
        for metric_name in self.metrics:
            try:
                self.log(f"\n开始计算指标: {metric_name}")
                metric_class_name = self.metric_class[metric_name].__class__.__name__
                self.log(f"使用评估类: {metric_class_name}")
                
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                result_dict.update(metric_result)
                
                # 统计基本信息 - 处理不同类型的评分
                if metric_scores and not isinstance(metric_scores[0], dict):
                    min_score = min(metric_scores)
                    max_score = max(metric_scores)
                    avg_score = sum(metric_scores) / len(metric_scores)
                    self.log(f"指标统计: 最小值={min_score:.4f}, 最大值={max_score:.4f}, 平均值={avg_score:.4f}")
                
                # 更新每个样本的评分
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)
                    
                self.log(f"完成指标 {metric_name} 计算，总体得分: {list(metric_result.values())[0]:.4f}")
            except Exception as e:
                import traceback
                self.log(f'评估 {metric_name} 时出错: {e}')
                self.log(traceback.format_exc())
                continue
        
        self.log("\n所有指标计算结果:")
        for metric, score in result_dict.items():
            self.log(f"  {metric}: {score:.4f}")
        
        self.log("======== 答案质量评估结束 ========\n")
        
        # 保存评估结果
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
            self.log(f"评估结果已保存至: {os.path.join(self.save_dir, 'metric_score.txt')}")
        
        # 保存评估数据
        if self.save_data_flag:
            self.save_data(data)
            self.log(f"评估中间数据已保存至: {os.path.join(self.save_dir, 'intermediate_data.json')}")
        
        return result_dict