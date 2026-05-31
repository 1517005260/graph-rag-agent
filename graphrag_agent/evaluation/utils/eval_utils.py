import os
import json
import time
from typing import Dict, List, Optional, Tuple

from graphrag_agent.evaluation import set_debug_mode
from graphrag_agent.evaluation.utils.logging_utils import setup_logger
from graphrag_agent.evaluation.evaluator_config.evaluatorConfig import EvaluatorConfig
from graphrag_agent.evaluation.evaluators.composite_evaluator import CompositeGraphRAGEvaluator
from graphrag_agent.evaluation.evaluator_config.agent_evaluation_config import get_agent_metrics

def load_agent(agent_type: str):
    """加载指定类型的Agent"""
    try:
        if agent_type == "graph":
            from graphrag_agent.agents.graph_agent import GraphAgent
            return GraphAgent()
        elif agent_type == "hybrid":
            from graphrag_agent.agents.hybrid_agent import HybridAgent
            return HybridAgent()
        elif agent_type == "naive":
            from graphrag_agent.agents.naive_rag_agent import NaiveRagAgent
            return NaiveRagAgent()
        elif agent_type == "fusion":
            from graphrag_agent.agents.fusion_agent import FusionGraphRAGAgent
            return FusionGraphRAGAgent()
        elif agent_type == "deep":
            # 根据配置加载标准DeepSearchAgent或增强版
            try:
                # 尝试加载增强版
                from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
                return DeepResearchAgent(use_deeper_tool=True)
            except:
                # 加载标准版
                from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
                return DeepResearchAgent(use_deeper_tool=False)
        else:
            raise ValueError(f"不支持的Agent类型: {agent_type}")
    except ImportError as e:
        raise ImportError(f"无法导入{agent_type}Agent: {e}")

def load_dependencies():
    """加载Neo4j和LLM依赖"""
    neo4j = None
    llm = None

    try:
        from graphrag_agent.config.neo4jdb import get_db_manager
        neo4j = get_db_manager().get_driver()
        print("成功连接Neo4j数据库")
    except ImportError:
        print("无法连接Neo4j数据库，部分指标可能无法计算")

    try:
        from graphrag_agent.models.get_models import get_llm_model
        llm = get_llm_model()
        print("成功加载LLM模型")
    except ImportError:
        print("无法加载LLM模型，部分指标可能无法计算")
    
    return neo4j, llm

def evaluate_agent(
    agent_type: str,
    questions: List[str],
    golden_answers: Optional[List[str]] = None,
    save_dir: str = "./evaluation_results",
    metrics: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    评估指定类型的Agent
    
    Args:
        agent_type: Agent类型
        questions: 问题列表
        golden_answers: 标准答案列表（可选）
        save_dir: 保存目录
        metrics: 评估指标（如果为None，则使用默认指标）
        verbose: 是否打印详细日志
        
    Returns:
        Dict[str, float]: 评估结果
    """
    # 设置保存目录
    agent_save_dir = os.path.join(save_dir, agent_type)
    os.makedirs(agent_save_dir, exist_ok=True)
    
    # 设置日志记录
    logger = setup_logger("evaluation", os.path.join(agent_save_dir, "evaluation.log"))
    logger.info(f"开始评估{agent_type}Agent")
    
    # 设置全局调试模式
    set_debug_mode(verbose)
    
    # 加载Agent
    try:
        agent = load_agent(agent_type)
        logger.info(f"成功加载{agent_type}Agent")
    except Exception as e:
        logger.error(f"加载Agent失败: {e}")
        return {}
        
    # 加载依赖
    neo4j, llm = load_dependencies()
    
    # 默认使用Agent类型对应的全部指标
    if not metrics:
        metrics = get_agent_metrics(agent_type)
    
    logger.info(f"使用的评估指标: {', '.join(metrics)}")
    
    # 配置评估器
    config = {
        f"{agent_type}_agent": agent,
        "neo4j_client": neo4j,
        "llm": llm,
        "save_dir": agent_save_dir,
        "debug": verbose,
        "metrics": metrics
    }
    
    evaluator_config = EvaluatorConfig(config)
    evaluator = CompositeGraphRAGEvaluator(evaluator_config)
    
    start_time = time.time()
    results = {}
    
    try:
        # 有无标准答案决定评估方式
        if golden_answers:
            logger.info("使用标准答案进行评估...")
            if len(questions) != len(golden_answers):
                logger.warning(f"问题数量({len(questions)})与标准答案数量({len(golden_answers)})不匹配")
                min_len = min(len(questions), len(golden_answers))
                questions = questions[:min_len]
                golden_answers = golden_answers[:min_len]
                logger.info(f"截取了问题和答案，现在有{min_len}对问答")
            
            results = evaluator.evaluate_with_golden_answers(agent_type, questions, golden_answers)
        else:
            logger.info("仅评估检索性能...")
            results = evaluator.evaluate_retrieval_only(agent_type, questions)
        
        # 保存Agent回答
        logger.info("保存Agent回答以供人工检查...")
        evaluator.save_agent_answers(questions, output_dir=os.path.join(agent_save_dir, "answers"))
        
    except Exception as e:
        logger.error(f"评估时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    end_time = time.time()
    logger.info(f"评估完成，耗时: {end_time - start_time:.2f}秒")
    
    # 打印结果
    logger.info("\n评估结果:")
    for metric, score in results.items():
        logger.info(f"  {metric}: {score:.4f}")
    
    # 保存结果
    results_path = os.path.join(agent_save_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存至: {results_path}")
    
    # 关闭资源
    if hasattr(agent, 'close') and callable(agent.close):
        try:
            agent.close()
            logger.info("已关闭Agent")
        except Exception as e:
            logger.warning(f"关闭Agent时出错: {e}")
    
    return results

def load_questions_and_answers(questions_file: str, golden_answers_file: Optional[str] = None) -> Tuple[List[str], Optional[List[str]]]:
    """
    加载问题和答案文件
    
    Args:
        questions_file: 问题文件路径
        golden_answers_file: 标准答案文件路径（可选）
        
    Returns:
        Tuple[List[str], Optional[List[str]]]: 问题列表和答案列表（如果提供）
    """
    evaluator = CompositeGraphRAGEvaluator()
    
    # 加载问题
    questions = evaluator.load_questions_from_file(questions_file)
    print(f"已加载{len(questions)}个问题")
    
    # 加载标准答案（如果提供）
    golden_answers = None
    if golden_answers_file:
        golden_answers = evaluator.load_answers_from_file(golden_answers_file)
        print(f"已加载{len(golden_answers)}个标准答案")
        
        # 确保问题和答案数量匹配
        if len(questions) != len(golden_answers):
            print(f"问题数量({len(questions)})与标准答案数量({len(golden_answers)})不匹配")
            min_len = min(len(questions), len(golden_answers))
            questions = questions[:min_len]
            golden_answers = golden_answers[:min_len]
            print(f"截取了问题和答案，现在有{min_len}对问答")
    
    return questions, golden_answers