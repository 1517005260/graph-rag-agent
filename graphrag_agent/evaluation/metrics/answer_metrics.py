import re
from typing import Dict, List, Tuple
from graphrag_agent.evaluation.core.base_metric import BaseMetric
from graphrag_agent.evaluation.core.evaluation_data import AnswerEvaluationData
from graphrag_agent.evaluation.utils.text_utils import normalize_answer

class ExactMatch(BaseMetric):
    """精确匹配评估指标"""
    
    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """
        计算单个预测的精确匹配得分
        
        Args:
            prediction: 预测答案
            golden_answer: 标准答案
            
        Returns:
            float: 得分（1.0表示匹配，0.0表示不匹配）
        """
        if not prediction or not golden_answer:
            return 0.0
            
        normalized_prediction = normalize_answer(prediction)
        normalized_golden = normalize_answer(golden_answer)
        
        # 完全匹配
        if normalized_prediction == normalized_golden:
            return 1.0
        return 0.0
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算精确匹配指标 - 使用规则匹配和LLM回退混合评分
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("======== ExactMatch 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        metric_score_list = []
        
        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()
            
            # 标准化答案
            normalized_pred = normalize_answer(cleaned_pred)
            normalized_golden = normalize_answer(golden)
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  系统答案(前30字符): {pred[:30]}...")
            self.log(f"  清理后的系统答案(前30字符): {cleaned_pred[:30]}...")
            self.log(f"  标准化后的标准答案(前30字符): {normalized_golden[:30]}...")
            self.log(f"  标准化后的系统答案(前30字符): {normalized_pred[:30]}...")
            
            # 完全匹配
            if normalized_pred == normalized_golden:
                score = 1.0
                self.log(f"  完全匹配 ✓")
            else:
                # 规则匹配失败，尝试内容相似性评估
                similarity_score = self._calculate_content_similarity(cleaned_pred, golden)
                self.log(f"  基本内容相似度: {similarity_score:.4f}")
                
                # 如果内容相似度较高，给予一定分数
                if similarity_score >= 0.7:
                    score = 0.7 + (similarity_score - 0.7) * 3/3  # 0.7-1.0 映射到 0.7-1.0
                    self.log(f"  内容高度相似，给予分数: {score:.4f}")
                # 如果内容相似度一般，回退到LLM评分
                elif self.llm:
                    self.log(f"  内容相似度一般，回退到LLM评分")
                    
                    # 构建LLM评估提示
                    prompt = f"""
                    请比较下面两个答案，评估它们在内容上的等价性，给出0到1之间的分数。
                    0表示完全不同，1表示内容上完全等价。
                    请只考虑实质内容，忽略格式、表达方式和顺序的差异。
                    
                    标准答案:
                    {golden}
                    
                    系统答案:
                    {cleaned_pred}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    # 使用基类的LLM回退评分方法
                    score = self.get_llm_fallback_score(prompt, default_score=similarity_score)
                    self.log(f"  LLM评估的匹配度分数: {score:.4f}")
                else:
                    # 没有LLM，使用内容相似度作为分数
                    score = similarity_score
                    self.log(f"  使用内容相似度作为分数: {score:.4f}")
            
            metric_score_list.append(score)
        
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        self.log(f"\n样本总数: {len(metric_score_list)}")
        self.log(f"匹配样本数: {sum(1 for s in metric_score_list if s > 0.8)}")
        self.log(f"精确匹配平均得分: {em_score:.4f}")
        self.log("======== ExactMatch 计算结束 ========\n")
        
        return {"em": em_score}, metric_score_list
    
    def _calculate_content_similarity(self, pred: str, golden: str) -> float:
        """
        计算两个文本的内容相似度
        
        Args:
            pred: 预测答案
            golden: 标准答案
            
        Returns:
            float: 内容相似度分数 (0-1)
        """
        # 标准化处理
        pred_norm = normalize_answer(pred).split()
        golden_norm = normalize_answer(golden).split()
        
        if not pred_norm or not golden_norm:
            return 0.0
            
        # 计算共有词的数量
        common_words = set(pred_norm) & set(golden_norm)
        
        # 计算Jaccard相似度
        union_words = set(pred_norm) | set(golden_norm)
        if union_words:
            jaccard = len(common_words) / len(union_words)
        else:
            jaccard = 0.0
            
        # 计算词覆盖率
        pred_coverage = len(common_words) / len(set(pred_norm)) if pred_norm else 0
        golden_coverage = len(common_words) / len(set(golden_norm)) if golden_norm else 0
        
        # 综合得分 - Jaccard占40%，两个覆盖率各占30%
        similarity = 0.4 * jaccard + 0.3 * pred_coverage + 0.3 * golden_coverage
        
        return similarity

class F1Score(BaseMetric):
    """F1分数评估指标"""
    
    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算F1分数 - 使用规则匹配和LLM回退混合评分
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== F1Score 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        f1_scores = []
        
        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()
            
            # 将文本标准化
            pred_text = normalize_answer(cleaned_pred)
            golden_text = normalize_answer(golden)
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  系统答案(前30字符): {pred[:30]}...")
            
            # 尝试使用传统F1计算
            try:
                # 进行中文分词
                import jieba
                pred_tokens = list(jieba.cut(pred_text))
                golden_tokens = list(jieba.cut(golden_text))
                
                # 过滤停用词和过短的词
                stopwords = {'的', '了', '和', '在', '是', '为', '以', '与', '或', '且'}
                pred_tokens = [token for token in pred_tokens if len(token) > 1 and token not in stopwords]
                golden_tokens = [token for token in golden_tokens if len(token) > 1 and token not in stopwords]
                
                self.log(f"  标准答案分词数: {len(golden_tokens)}")
                self.log(f"  系统答案分词数: {len(pred_tokens)}")
                
                if not pred_tokens or not golden_tokens:
                    # 空文本处理
                    if not pred_tokens and not golden_tokens:
                        rule_f1 = 1.0  # 两者都为空，视为完全匹配
                        self.log(f"  两者都为空，视为完全匹配，F1=1.0")
                    else:
                        rule_f1 = 0.0  # 一个为空一个不为空
                        self.log(f"  一个为空一个不为空，规则F1=0.0")
                else:
                    # 计算标准F1
                    common_tokens = set(pred_tokens) & set(golden_tokens)
                    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                    recall = len(common_tokens) / len(golden_tokens) if golden_tokens else 0
                    
                    if precision + recall > 0:
                        rule_f1 = 2 * precision * recall / (precision + recall)
                    else:
                        rule_f1 = 0.0
                    
                    self.log(f"  共有词汇: {len(common_tokens)}/{len(set(pred_tokens) | set(golden_tokens))}")
                    self.log(f"  精确率: {precision:.4f}")
                    self.log(f"  召回率: {recall:.4f}")
                    self.log(f"  规则F1分数: {rule_f1:.4f}")
            except Exception as e:
                self.log(f"  规则F1计算出错: {e}")
                rule_f1 = 0.0
            
            # 无论规则F1分数如何，如果有LLM都尝试使用LLM评估
            if self.llm:
                self.log(f"  尝试使用LLM评估内容相似度")
                
                # 构建内容相似度评估提示
                prompt = f"""
                请比较下面两个答案的内容相似度，评估它们包含的信息重叠程度，并给出0到1之间的分数。
                0表示完全不同信息，1表示信息完全重叠。
                请考虑实质内容的相似性，而不仅是表面文字的匹配。在评估时，请特别关注关键信息点是否一致。
                
                标准答案:
                {golden}
                
                系统答案:
                {cleaned_pred}
                
                只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                """
                
                # 使用基类的LLM回退评分方法
                llm_f1 = self.get_llm_fallback_score(prompt, default_score=0.5)
                self.log(f"  LLM评估的F1分数: {llm_f1:.4f}")
                
                # 如果LLM分数更高，使用LLM分数；否则使用规则F1分数
                if llm_f1 > rule_f1:
                    self.log(f"  LLM分数更高，采用LLM评估")
                    f1 = llm_f1
                else:
                    self.log(f"  规则F1分数更高，保留规则评估")
                    f1 = rule_f1
            else:
                # 没有LLM可用，使用规则F1分数
                f1 = rule_f1
            
            f1_scores.append(f1)
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        self.log(f"\n样本总数: {len(f1_scores)}")
        self.log(f"F1得分大于0.5的样本数: {sum(1 for s in f1_scores if s > 0.5)}")
        self.log(f"F1平均得分: {avg_f1:.4f}")
        self.log("======== F1Score 计算结束 ========\n")
        
        return {"f1": avg_f1}, f1_scores