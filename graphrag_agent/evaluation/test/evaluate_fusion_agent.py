import os
import sys
import argparse

# 添加父目录到路径，使得可以导入evaluator模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from graphrag_agent.evaluation import set_debug_mode
from graphrag_agent.evaluation.utils.logging_utils import setup_logger
from graphrag_agent.evaluation.evaluator_config.agent_evaluation_config import get_agent_metrics
from graphrag_agent.evaluation.utils.eval_utils import evaluate_agent, load_questions_and_answers

def parse_args():
    parser = argparse.ArgumentParser(description="评估Fusion GraphRAG Agent性能")
    parser.add_argument("--save_dir", type=str, default="./evaluation_results/fusion_agent",
                        help="评估结果保存目录")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="要评估的问题文件（JSON格式）")
    parser.add_argument("--golden_answers_file", type=str, default=None,
                        help="标准答案文件（JSON格式，可选）")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细评估过程")
    parser.add_argument("--metrics", type=str, default="",
                        help="要评估的指标，用逗号分隔，留空则使用默认指标")
    parser.add_argument("--eval_type", type=str, default="all",
                        choices=["all", "answer", "retrieval", "reasoning"],
                        help="评估类型: all(全面评估), answer(仅答案质量), retrieval(仅检索性能), reasoning(仅推理能力)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置日志记录
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("fusion_evaluation", os.path.join(args.save_dir, "evaluation.log"))
    logger.info("开始评估Fusion GraphRAG Agent")
    
    # 设置全局调试模式
    set_debug_mode(args.verbose)
    
    # 确定使用哪些指标
    metrics = []
    if args.metrics:
        # 使用用户指定的指标
        metrics = args.metrics.split(',')
        logger.info(f"使用用户指定的评估指标: {args.metrics}")
    else:
        # 根据评估类型使用默认指标
        if args.eval_type == "answer":
            metrics = get_agent_metrics("fusion", "answer")
            logger.info(f"使用答案评估指标: {', '.join(metrics)}")
        elif args.eval_type == "retrieval":
            metrics = get_agent_metrics("fusion", "retrieval")
            logger.info(f"使用检索评估指标: {', '.join(metrics)}")
        elif args.eval_type == "reasoning":
            metrics = get_agent_metrics("fusion", "reasoning")
            logger.info(f"使用推理评估指标: {', '.join(metrics)}")
        else:
            metrics = get_agent_metrics("fusion")
            logger.info(f"使用全部评估指标: {', '.join(metrics)}")
    
    try:
        # 加载问题和答案
        questions, golden_answers = load_questions_and_answers(
            args.questions_file, 
            args.golden_answers_file
        )
        
        # 执行评估
        evaluate_agent(
            agent_type="fusion",
            questions=questions,
            golden_answers=golden_answers,
            save_dir=args.save_dir,
            metrics=metrics,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()