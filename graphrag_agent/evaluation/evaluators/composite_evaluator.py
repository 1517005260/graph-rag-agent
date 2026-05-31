import os
import json
import time
from typing import Dict, List, Any

from graphrag_agent.evaluation.core.evaluation_data import (
    AnswerEvaluationData, AnswerEvaluationSample,
    RetrievalEvaluationData, RetrievalEvaluationSample
)
from graphrag_agent.evaluation.evaluators.answer_evaluator import AnswerEvaluator
from graphrag_agent.evaluation.evaluators.retrieval_evaluator import GraphRAGRetrievalEvaluator
from graphrag_agent.evaluation.preprocessing.text_cleaner import clean_references, clean_thinking_process
from graphrag_agent.evaluation.preprocessing.reference_extractor import extract_references_from_answer
from graphrag_agent.evaluation.evaluator_config.evaluatorConfig import EvaluatorConfig

class CompositeGraphRAGEvaluator:
    """
    组合评估器，同时评估答案质量和检索性能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化组合评估器
        
        Args:
            config: 评估配置
        """
        # 支持字典或EvaluatorConfig对象
        if isinstance(config, dict):
            self.config = EvaluatorConfig(config)
        else:
            self.config = config or EvaluatorConfig()
            
        self.save_dir = self.config.get('save_dir', './evaluation_results')
        self.debug = self.config.get('debug', True)
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建答案评估器
        answer_config = self.config.to_dict().copy()
        answer_config['save_dir'] = os.path.join(self.save_dir, 'answer_evaluation')
        
        # 默认的答案质量评估指标
        default_answer_metrics = [
            'em', 'f1', 'response_coherence', 
            'answer_comprehensiveness', 'factual_consistency'
        ]
        
        # 如果配置中有'llm'参数，添加LLM评估指标
        if self.config.get('llm'):
            default_answer_metrics.append('llm_evaluation')
        
        # 先使用传入的所有指标，过滤出答案质量相关的
        passed_metrics = [m for m in self.config.get_metrics() 
                          if not m.startswith('retrieval_') and 
                          not m in ['entity_coverage', 'graph_coverage', 'relationship_utilization',
                                   'community_relevance', 'subgraph_quality']]
        
        # 如果没有传入答案指标，使用默认指标
        if not passed_metrics:
            passed_metrics = default_answer_metrics
            
        answer_config['metrics'] = passed_metrics
        self.answer_evaluator = AnswerEvaluator(answer_config)
        
        # 创建检索评估器
        retrieval_config = self.config.to_dict().copy()
        retrieval_config['save_dir'] = os.path.join(self.save_dir, 'retrieval_evaluation')
        
        # 默认的检索评估指标
        default_retrieval_metrics = [
            'retrieval_latency', 'retrieval_precision', 
            'retrieval_utilization', 'entity_coverage',
            'relationship_utilization', 'graph_coverage',
            'community_relevance'
        ]
        
        # 如果支持subgraph_quality指标，添加它
        try:
            from graphrag_agent.evaluation.metrics.graph_metrics import SubgraphQualityMetric
            default_retrieval_metrics.append('subgraph_quality')
        except ImportError:
            pass
        
        # 先使用传入的所有指标，过滤出检索相关的
        passed_retrieval_metrics = [m for m in self.config.get_metrics() 
                                  if m.startswith('retrieval_') or 
                                  m in ['entity_coverage', 'graph_coverage', 
                                       'relationship_utilization', 'community_relevance', 
                                       'subgraph_quality']]
                                  
        # 如果没有传入检索指标，使用默认指标
        if not passed_retrieval_metrics:
            passed_retrieval_metrics = default_retrieval_metrics
            
        retrieval_config['metrics'] = passed_retrieval_metrics
        self.retrieval_evaluator = GraphRAGRetrievalEvaluator(retrieval_config)
        
        # Agent实例
        self.agents = {
            "naive": self.config.get_agent("naive"),
            "hybrid": self.config.get_agent("hybrid"),
            "graph": self.config.get_agent("graph"),
            "deep": self.config.get_agent("deep"),
            "fusion" : self.config.get_agent("fusion"),
        }
        
        # 创建存放回答的目录
        self.answers_dir = os.path.join(self.save_dir, "answers")
        os.makedirs(self.answers_dir, exist_ok=True)
    
    def log(self, message, *args, **kwargs):
        """
        输出调试日志
        
        Args:
            message: 日志消息
            *args, **kwargs: 额外参数
        """
        from graphrag_agent.evaluation import debug_print
        if self.debug:
            debug_print(f"[CompositeEvaluator] {message}", *args, **kwargs)
            
    def evaluate_with_golden_answers(self, agent_name: str, questions: List[str], golden_answers: List[str]) -> Dict[str, float]:
        """
        使用标准答案评估特定Agent
        
        Args:
            agent_name: Agent名称
            questions: 问题列表
            golden_answers: 标准答案列表
                
        Returns:
            Dict[str, float]: 评估结果
        """
        if len(questions) != len(golden_answers):
            raise ValueError("问题和标准答案数量不匹配")
                
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"未找到Agent: {agent_name}")
        
        # 创建答案评估数据集
        answer_data = AnswerEvaluationData()
        
        # 创建检索评估数据集
        retrieval_data = RetrievalEvaluationData()
        
        # 准备存储回答的结构
        answers = []
        
        # 处理每个问题
        for i, (question, golden_answer) in enumerate(zip(questions, golden_answers)):
            self.log(f"处理问题 {i+1}/{len(questions)}: {question[:30]}...")
            
            # 创建答案评估样本
            answer_sample = AnswerEvaluationSample(
                question=question,
                golden_answer=golden_answer
            )
            
            # 创建检索评估样本
            retrieval_sample = RetrievalEvaluationSample(
                question=question
            )

            # 记录开始时间
            start_time = time.time()
            
            try:
                # 获取回答
                answer = agent.ask(question)
                
                # 计算检索时间
                retrieval_time = time.time() - start_time
                
                # 用于展示的清理版本
                clean_answer = answer
                if agent_name == "deep":
                    clean_answer = clean_thinking_process(clean_answer)
                clean_answer = clean_references(clean_answer)
                
                # 立即将答案添加到列表，避免后续重复调用
                answers.append({
                    "question": question,
                    "answer": clean_answer
                })
                
                # 更新样本 - 提取引用信息
                answer_sample.update_system_answer(answer, agent_name)
                retrieval_sample.update_system_answer(answer, agent_name)
                retrieval_sample.retrieval_time = retrieval_time
                
                # 从答案中提取引用实体和关系
                refs = extract_references_from_answer(answer)
                entities = refs.get("entities", [])
                relationships = refs.get("relationships", [])
                
                # 同时将引用信息添加到答案评估样本中
                answer_sample.retrieved_entities = entities
                answer_sample.retrieved_relationships = relationships
                
                # 尝试使用Neo4j获取相关图数据（如果可用）
                neo4j_client = self.config.get('neo4j_client')
                if neo4j_client:
                    try:
                        entities, relationships = self.retrieval_evaluator._get_relevant_graph_data(question)
                        retrieval_sample.update_retrieval_data(entities, relationships)
                    except Exception as e:
                        self.log(f"  获取相关图数据时出错: {e}")
                    
            except Exception as e:
                self.log(f"  获取{agent_name}对问题的回答时出错: {e}")
                retrieval_time = time.time() - start_time
                
                # 保存错误信息
                error_message = f"获取回答时出错: {str(e)}"
                answer_sample.update_system_answer(error_message, agent_name)
                retrieval_sample.update_system_answer(error_message, agent_name)
                retrieval_sample.retrieval_time = retrieval_time
                
                answers.append({
                    "question": question,
                    "answer": error_message
                })
            
            # 添加到评估数据
            answer_data.append(answer_sample)
            retrieval_data.append(retrieval_sample)
        
        # 保存所有回答 - 使用已收集的数据，不再重复调用
        self._save_agent_answers(agent_name, answers)
        
        # 执行答案评估
        answer_results = {}
        if hasattr(self.answer_evaluator, 'evaluate') and callable(self.answer_evaluator.evaluate):
            try:
                answer_results = self.answer_evaluator.evaluate(answer_data)
            except Exception as e:
                self.log(f"答案评估出错: {e}")
                import traceback
                self.log(traceback.format_exc())
                answer_results = {"em": 0.0, "f1": 0.0}
        
        # 执行检索评估
        retrieval_results = {}
        if hasattr(self.retrieval_evaluator, 'evaluate') and callable(self.retrieval_evaluator.evaluate):
            try:
                retrieval_results = self.retrieval_evaluator.evaluate(retrieval_data)
            except Exception as e:
                self.log(f"检索评估出错: {e}")
        
        # 合并结果
        results = {**answer_results, **retrieval_results}
        
        # 保存合并结果
        results_path = os.path.join(self.save_dir, f"{agent_name}_evaluation.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def compare_agents_with_golden_answers(self, questions: List[str], golden_answers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        使用标准答案比较所有Agent
        
        Args:
            questions: 问题列表
            golden_answers: 标准答案列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个Agent的评估结果
        """
        if len(questions) != len(golden_answers):
            raise ValueError("问题和标准答案数量不匹配")
        
        results = {}
        
        for agent_name, agent in self.agents.items():
            if agent:
                self.log(f"评估Agent: {agent_name}")
                agent_results = self.evaluate_with_golden_answers(agent_name, questions, golden_answers)
                results[agent_name] = agent_results
                
                # 打印结果
                self.log(f"{agent_name} 评估结果:")
                for metric, score in agent_results.items():
                    self.log(f"  {metric}: {score:.4f}")
                self.log("")
        
        # 保存比较结果
        results_path = os.path.join(self.save_dir, "agent_comparison_with_golden.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def evaluate_retrieval_only(self, agent_name: str, questions: List[str]) -> Dict[str, float]:
        """
        仅评估检索性能
        
        Args:
            agent_name: Agent名称
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 评估结果
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"未找到Agent: {agent_name}")
        
        # 准备存储回答的结构
        answers = []
        
        # 创建检索评估数据集
        retrieval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for i, question in enumerate(questions):
            self.log(f"处理问题 {i+1}/{len(questions)}: {question[:30]}...")
            
            # 创建检索评估样本
            retrieval_sample = RetrievalEvaluationSample(
                question=question
            )
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 获取回答 - 直接询问Agent一次
                answer = agent.ask(question)
                
                # 计算检索时间
                retrieval_time = time.time() - start_time
                
                # 用于展示的清理版本
                clean_answer = answer
                if agent_name == "deep":
                    clean_answer = clean_thinking_process(clean_answer)
                clean_answer = clean_references(clean_answer)
                
                # 立即将答案添加到列表，避免后续重复调用
                answers.append({
                    "question": question,
                    "answer": clean_answer
                })
                
                # 更新样本
                retrieval_sample.update_system_answer(answer, agent_name)
                retrieval_sample.retrieval_time = retrieval_time
                
                # 尝试使用Neo4j获取相关图数据（如果可用）
                neo4j_client = self.config.get('neo4j_client')
                if neo4j_client:
                    try:
                        entities, relationships = self.retrieval_evaluator._get_relevant_graph_data(question)
                        retrieval_sample.update_retrieval_data(entities, relationships)
                    except Exception as e:
                        self.log(f"  获取相关图数据时出错: {e}")
            
            except Exception as e:
                self.log(f"  获取{agent_name}对问题的回答时出错: {e}")
                retrieval_time = time.time() - start_time
                
                # 保存错误信息
                error_message = f"获取回答时出错: {str(e)}"
                retrieval_sample.update_system_answer(error_message, agent_name)
                retrieval_sample.retrieval_time = retrieval_time
                
                answers.append({
                    "question": question,
                    "answer": error_message
                })
            
            # 添加到评估数据
            retrieval_data.append(retrieval_sample)
        
        # 保存所有回答 - 使用已收集的数据，不再重复调用
        self._save_agent_answers(agent_name, answers)
        
        # 执行检索评估
        try:
            results = self.retrieval_evaluator.evaluate(retrieval_data)
        except Exception as e:
            self.log(f"检索评估出错: {e}")
            results = {}
        
        # 保存评估结果
        results_path = os.path.join(self.save_dir, f"{agent_name}_retrieval_evaluation.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def compare_retrieval_only(self, questions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        仅比较检索性能
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个Agent的评估结果
        """
        results = {}
        
        for agent_name, agent in self.agents.items():
            if agent:
                self.log(f"评估Agent: {agent_name}")
                agent_results = self.evaluate_retrieval_only(agent_name, questions)
                results[agent_name] = agent_results
                
                # 打印结果
                self.log(f"{agent_name} 检索评估结果:")
                for metric, score in agent_results.items():
                    self.log(f"  {metric}: {score:.4f}")
                self.log("")
        
        # 保存比较结果
        results_path = os.path.join(self.save_dir, "retrieval_comparison.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _save_agent_answers(self, agent_name: str, answers: List[Dict[str, str]]):
        """
        保存Agent回答
        
        Args:
            agent_name: Agent名称
            answers: 回答列表
        """
        # 确保答案目录存在
        os.makedirs(self.answers_dir, exist_ok=True)
        
        # 保存JSON格式
        answers_path = os.path.join(self.answers_dir, f"{agent_name}_answers.json")
        with open(answers_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        
        # 生成更易读的markdown格式
        markdown_path = os.path.join(self.answers_dir, f"{agent_name}_answers.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(f"# {agent_name.capitalize()} Agent的回答\n\n")
            
            for i, qa in enumerate(answers):
                f.write(f"## 问题 {i+1}: {qa['question']}\n\n")
                f.write(f"{qa['answer']}\n\n")
                f.write("---\n\n")
        
        self.log(f"  已保存{agent_name}的回答到 {answers_path}")
    
    def format_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        将比较结果格式化为表格
        
        Args:
            results: 比较结果
            
        Returns:
            str: 表格字符串
        """
        # 获取所有指标
        all_metrics = set()
        for agent_results in results.values():
            all_metrics.update(agent_results.keys())
        
        # 区分答案和检索指标，并添加LLM评估指标作为单独类别
        answer_metrics = sorted([m for m in all_metrics 
                              if not m.startswith('retrieval_') 
                              and not m.startswith('llm_')
                              and not m in ['entity_coverage', 'graph_coverage', 
                                          'relationship_utilization', 'community_relevance', 
                                          'subgraph_quality']])
        
        llm_metrics = sorted([m for m in all_metrics if m.startswith('llm_')])
        
        retrieval_metrics = sorted([m for m in all_metrics 
                                 if m.startswith('retrieval_') 
                                 or m in ['entity_coverage', 'graph_coverage', 
                                         'relationship_utilization', 'community_relevance', 
                                         'subgraph_quality']])
        
        # 构建表头
        header = "| 指标 | " + " | ".join(results.keys()) + " |"
        separator = "| --- | " + " | ".join(["---" for _ in results]) + " |"
        
        # 构建行
        rows = []
        
        # 添加答案指标
        if answer_metrics:
            rows.append("| **答案质量指标** | " + " | ".join(["" for _ in results]) + " |")
            
            for metric in answer_metrics:
                row = f"| {metric} |"
                for agent in results:
                    score = results[agent].get(metric, "N/A")
                    if isinstance(score, float):
                        score_str = f"{score:.4f}"
                    else:
                        score_str = str(score)
                    row += f" {score_str} |"
                rows.append(row)
        
        # 添加LLM评估指标
        if llm_metrics:
            rows.append("| **LLM评估指标** | " + " | ".join(["" for _ in results]) + " |")
            
            for metric in llm_metrics:
                # 美化指标名称显示
                display_name = metric.replace('llm_', '').replace('_', ' ').capitalize()
                row = f"| {display_name} |"
                for agent in results:
                    score = results[agent].get(metric, "N/A")
                    if isinstance(score, float):
                        score_str = f"{score:.4f}"
                    else:
                        score_str = str(score)
                    row += f" {score_str} |"
                rows.append(row)
        
        # 添加检索指标
        if retrieval_metrics:
            rows.append("| **检索性能指标** | " + " | ".join(["" for _ in results]) + " |")
            
            for metric in retrieval_metrics:
                row = f"| {metric} |"
                for agent in results:
                    score = results[agent].get(metric, "N/A")
                    if isinstance(score, float):
                        score_str = f"{score:.4f}"
                    else:
                        score_str = str(score)
                    row += f" {score_str} |"
                rows.append(row)
        
        # 拼接表格
        table = "\n".join([header, separator] + rows)
        return table
    
    def save_agent_answers(self, questions: List[str], output_dir: str = None):
        """
        保存Agent回答
        
        Args:
            questions: 问题列表
            output_dir: 输出目录，默认为self.save_dir/answers
        """
        if output_dir is None:
            output_dir = self.answers_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否已经有答案文件
        self.log("检查是否需要获取额外回答...")
        
        for agent_name, agent in self.agents.items():
            if not agent:
                continue
            
            answers_path = os.path.join(output_dir, f"{agent_name}_answers.json")
            
            # 如果答案文件已存在，跳过
            if os.path.exists(answers_path):
                self.log(f"  {agent_name}Agent的回答文件已存在，跳过")
                continue
                
            self.log(f"获取{agent_name}Agent的回答...")
            answers = []
            
            # 使用缓存记录已处理的问题
            processed_questions = {}
            
            for i, question in enumerate(questions):
                self.log(f"  问题 {i+1}/{len(questions)}")
                
                # 先检查缓存
                cache_key = f"{agent_name}:{question}"
                if cache_key in processed_questions:
                    answers.append({
                        "question": question,
                        "answer": processed_questions[cache_key]
                    })
                    self.log(f"    使用缓存的回答")
                    continue
                
                try:
                    # 获取回答
                    answer = agent.ask(question)
                    
                    # 清理回答
                    if agent_name == "deep":
                        answer = clean_thinking_process(answer)
                    answer = clean_references(answer)
                    
                    # 保存到缓存
                    processed_questions[cache_key] = answer
                    
                    answers.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    self.log(f"  获取{agent_name}回答时出错: {e}")
                    answers.append({
                        "question": question,
                        "answer": f"获取回答时出错: {str(e)}"
                    })
            
            # 使用抽取的方法保存回答
            self._save_agent_answers(agent_name, answers)

    def load_questions_from_file(self, file_path: str) -> List[str]:
        """
        从文件加载问题列表，支持多种格式
        
        Args:
            file_path: 问题文件路径（JSON格式）
            
        Returns:
            List[str]: 问题列表
        """
        with open(file_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
        
        # 提取问题列表
        if isinstance(questions_data, list):
            if len(questions_data) > 0:
                if isinstance(questions_data[0], dict):
                    # 如果是字典列表，尝试提取问题字段
                    if "question" in questions_data[0]:
                        questions = [item["question"] for item in questions_data]
                    elif "text" in questions_data[0]:
                        questions = [item["text"] for item in questions_data]
                    else:
                        # 如果没有找到标准字段，尝试找到任何可能的问题字段
                        possible_fields = ["q", "query", "content", "input"]
                        for field in possible_fields:
                            if field in questions_data[0]:
                                questions = [item[field] for item in questions_data]
                                break
                        else:
                            # 如果还是找不到，转为字符串
                            questions = [str(item) for item in questions_data]
                else:
                    # 如果不是字典列表，直接使用
                    questions = questions_data
        else:
            # 如果不是列表，包装为列表
            questions = [questions_data]
        
        # 确保所有问题都是字符串
        questions = [str(q).strip() for q in questions]
        
        return questions

    def load_answers_from_file(self, file_path: str) -> List[str]:
        """
        从文件加载答案列表，支持多种格式
        
        Args:
            file_path: 答案文件路径（JSON格式）
            
        Returns:
            List[str]: 答案列表
        """
        with open(file_path, "r", encoding="utf-8") as f:
            answers_data = json.load(f)
        
        # 提取答案列表
        if isinstance(answers_data, list):
            if len(answers_data) > 0:
                if isinstance(answers_data[0], dict):
                    # 如果是字典列表，尝试提取答案字段
                    if "answer" in answers_data[0]:
                        answers = [item["answer"] for item in answers_data]
                    elif "text" in answers_data[0]:
                        answers = [item["text"] for item in answers_data]
                    else:
                        # 如果没有找到标准字段，尝试找到任何可能的答案字段
                        possible_fields = ["a", "response", "content", "output"]
                        for field in possible_fields:
                            if field in answers_data[0]:
                                answers = [item[field] for item in answers_data]
                                break
                        else:
                            # 如果还是找不到，转为字符串
                            answers = [str(item) for item in answers_data]
                else:
                    # 如果不是字典列表，直接使用
                    answers = answers_data
        else:
            # 如果不是列表，包装为列表
            answers = [answers_data]
        
        # 确保所有答案都是字符串
        answers = [str(a).strip() for a in answers]
        
        return answers