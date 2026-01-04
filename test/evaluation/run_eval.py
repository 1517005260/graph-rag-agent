#!/usr/bin/env python3
"""
RAGAS 评估流水线

使用 RAGAS 框架评估 GraphRAG 系统的性能。

支持的评估指标：
- Faithfulness: 答案的忠实度（是否基于提供的上下文）
- Answer Relevancy: 答案与问题的相关性
- Context Precision: 上下文的精确度
- Context Recall: 上下文的召回率
- Answer Correctness: 答案的正确性（与 ground truth 比较）

使用方式：
    python test/evaluation/run_eval.py
    python test/evaluation/run_eval.py --agent hybrid_agent --metrics all
    python test/evaluation/run_eval.py --test-file custom_test.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

# 检查必要的依赖
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
except ImportError as e:
    console.print("[bold red]错误：缺少必要的依赖包[/bold red]")
    console.print("\n请安装以下依赖：")
    console.print("  pip install ragas datasets")
    console.print(f"\n详细错误：{e}")
    sys.exit(1)

# 导入 GraphRAG 组件
try:
    from graphrag_agent.agents.hybrid_agent import HybridAgent
    from graphrag_agent.agents.graph_agent import GraphAgent
    from graphrag_agent.agents.naive_rag_agent import NaiveRagAgent
    from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
    from graphrag_agent.search.global_search import GlobalSearch
except ImportError as e:
    console.print(f"[bold red]错误：无法导入 GraphRAG 组件: {e}[/bold red]")
    sys.exit(1)


class RAGASEvaluator:
    """RAGAS 评估器"""

    def __init__(
        self,
        agent_type: str = "hybrid_agent",
        metrics: List[str] = None,
        output_dir: str = "./test/evaluation/results"
    ):
        """
        初始化评估器

        Args:
            agent_type: Agent 类型 (naive_rag, graph_agent, hybrid_agent, deep_research)
            metrics: 要评估的指标列表
            output_dir: 输出目录
        """
        self.agent_type = agent_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置评估指标
        self.available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness
        }

        if metrics is None or "all" in metrics:
            self.metrics = list(self.available_metrics.values())
            self.metric_names = list(self.available_metrics.keys())
        else:
            self.metrics = [self.available_metrics[m] for m in metrics if m in self.available_metrics]
            self.metric_names = [m for m in metrics if m in self.available_metrics]

        console.print(f"[cyan]使用的评估指标: {', '.join(self.metric_names)}[/cyan]")

        # 初始化 Agent
        self.agent = self._init_agent()

    def _init_agent(self):
        """初始化 Agent"""
        console.print(f"[cyan]初始化 Agent: {self.agent_type}[/cyan]")

        agent_map = {
            "naive_rag": NaiveRagAgent,
            "graph_agent": GraphAgent,
            "hybrid_agent": HybridAgent,
            "deep_research": DeepResearchAgent
        }

        if self.agent_type not in agent_map:
            raise ValueError(f"不支持的 Agent 类型: {self.agent_type}")

        # 创建 Agent 实例（使用唯一的 session_id）
        session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        agent_class = agent_map[self.agent_type]
        agent = agent_class(session_id=session_id)

        console.print(f"[green]✅ {self.agent_type} 初始化完成[/green]")
        return agent

    def load_test_cases(self, test_file: str) -> List[Dict[str, Any]]:
        """
        加载测试用例

        Args:
            test_file: 测试文件路径

        Returns:
            测试用例列表
        """
        console.print(f"\n[cyan]加载测试用例: {test_file}[/cyan]")

        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")

        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        console.print(f"[green]✅ 加载了 {len(test_cases)} 个测试用例[/green]")

        # 显示测试用例分布
        metadata = data.get("metadata", {})
        if metadata:
            console.print("\n[bold]测试用例分布:[/bold]")
            console.print(f"  总数: {metadata.get('total_questions', len(test_cases))}")
            console.print(f"  类别: {metadata.get('categories', {})}")
            console.print(f"  难度: {metadata.get('difficulty_distribution', {})}")

        return test_cases

    def run_system(self, test_cases: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        批量运行系统获取答案和上下文

        Args:
            test_cases: 测试用例列表

        Returns:
            包含 questions, answers, contexts, ground_truths 的字典
        """
        console.print("\n[bold cyan]开始批量生成回答...[/bold cyan]\n")

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, case in enumerate(track(test_cases, description="生成答案")):
            question = case["question"]
            ground_truth = case["ground_truth"]

            try:
                # 调用 Agent 生成答案
                # 对于需要详细信息的 Agent，使用 ask_with_thinking
                if hasattr(self.agent, 'ask_with_thinking'):
                    result = self.agent.ask_with_thinking(
                        question,
                        thread_id=self.agent.session_id
                    )
                    answer = result.get("answer", "")
                    retrieved_info = result.get("retrieved_info", [])

                    # 提取上下文
                    context_list = []
                    if isinstance(retrieved_info, list):
                        for item in retrieved_info:
                            if isinstance(item, str):
                                context_list.append(item)
                            elif isinstance(item, dict):
                                context_list.append(item.get("content", str(item)))
                    elif isinstance(retrieved_info, str):
                        context_list = [retrieved_info]

                    # 如果没有上下文，使用答案作为上下文（RAGAS 需要）
                    if not context_list:
                        context_list = [answer]

                else:
                    # 简单 Agent，只获取答案
                    answer = self.agent.ask(
                        question,
                        thread_id=self.agent.session_id
                    )
                    context_list = [answer]  # 使用答案作为上下文

                questions.append(question)
                answers.append(answer)
                contexts.append(context_list)
                ground_truths.append(ground_truth)

                console.print(f"[green]✅ Q{i+1}/{len(test_cases)}: {case['id']}[/green]")

            except Exception as e:
                console.print(f"[red]❌ Q{i+1} 失败: {str(e)}[/red]")
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

                # 添加空结果以保持一致性
                questions.append(question)
                answers.append(f"ERROR: {str(e)}")
                contexts.append(["Error generating answer"])
                ground_truths.append(ground_truth)

        console.print(f"\n[green]✅ 成功生成 {len(answers)} 个回答[/green]")

        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

    def evaluate(self, dataset_dict: Dict[str, List]) -> pd.DataFrame:
        """
        运行 RAGAS 评估

        Args:
            dataset_dict: 包含 question, answer, contexts, ground_truth 的字典

        Returns:
            评估结果 DataFrame
        """
        console.print("\n[bold cyan]开始 RAGAS 评估...[/bold cyan]\n")

        # 构造 RAGAS Dataset
        dataset = Dataset.from_dict(dataset_dict)

        console.print(f"数据集大小: {len(dataset)} 条")
        console.print(f"评估指标: {', '.join(self.metric_names)}")

        try:
            # 运行评估
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics
            )

            # 转换为 DataFrame
            df = results.to_pandas()

            console.print(f"\n[green]✅ 评估完成[/green]")
            return df

        except Exception as e:
            console.print(f"[red]❌ 评估失败: {str(e)}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise

    def generate_report(
        self,
        results_df: pd.DataFrame,
        test_cases: List[Dict],
        dataset_dict: Dict[str, List]
    ):
        """
        生成评估报告

        Args:
            results_df: 评估结果 DataFrame
            test_cases: 原始测试用例
            dataset_dict: 数据集字典
        """
        console.print("\n[bold cyan]生成评估报告...[/bold cyan]\n")

        # 1. 保存详细结果 CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"eval_results_{self.agent_type}_{timestamp}.csv"

        # 添加测试用例信息
        results_with_info = results_df.copy()
        results_with_info.insert(0, "id", [case["id"] for case in test_cases])
        results_with_info.insert(1, "category", [case["category"] for case in test_cases])
        results_with_info.insert(2, "difficulty", [case["difficulty"] for case in test_cases])

        results_with_info.to_csv(csv_path, index=False, encoding="utf-8")
        console.print(f"[green]✅ 详细结果已保存: {csv_path}[/green]")

        # 2. 计算汇总统计
        summary = {}
        for metric_name in self.metric_names:
            if metric_name in results_df.columns:
                metric_values = results_df[metric_name]
                summary[metric_name] = {
                    "mean": metric_values.mean(),
                    "std": metric_values.std(),
                    "min": metric_values.min(),
                    "max": metric_values.max(),
                    "median": metric_values.median()
                }

        # 3. 显示汇总表格
        table = Table(title=f"RAGAS 评估结果汇总 ({self.agent_type})")
        table.add_column("指标", style="cyan", no_wrap=True)
        table.add_column("平均值", style="magenta")
        table.add_column("标准差", style="yellow")
        table.add_column("最小值", style="red")
        table.add_column("最大值", style="green")
        table.add_column("中位数", style="blue")

        for metric_name, stats in summary.items():
            table.add_row(
                metric_name,
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['median']:.4f}"
            )

        console.print(table)

        # 4. 按类别和难度分析
        if "category" in results_with_info.columns:
            console.print("\n[bold]按类别统计:[/bold]")
            for category in results_with_info["category"].unique():
                category_df = results_with_info[results_with_info["category"] == category]
                console.print(f"\n  {category}:")
                for metric_name in self.metric_names:
                    if metric_name in category_df.columns:
                        avg = category_df[metric_name].mean()
                        console.print(f"    {metric_name}: {avg:.4f}")

        if "difficulty" in results_with_info.columns:
            console.print("\n[bold]按难度统计:[/bold]")
            for difficulty in ["easy", "medium", "hard"]:
                difficulty_df = results_with_info[results_with_info["difficulty"] == difficulty]
                if len(difficulty_df) > 0:
                    console.print(f"\n  {difficulty}:")
                    for metric_name in self.metric_names:
                        if metric_name in difficulty_df.columns:
                            avg = difficulty_df[metric_name].mean()
                            console.print(f"    {metric_name}: {avg:.4f}")

        # 5. 保存 JSON 报告
        report = {
            "timestamp": timestamp,
            "agent_type": self.agent_type,
            "metrics": self.metric_names,
            "summary": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
            "total_questions": len(test_cases),
            "passed_threshold": self._check_thresholds(summary)
        }

        json_path = self.output_dir / f"eval_report_{self.agent_type}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]✅ JSON 报告已保存: {json_path}[/green]")

        return report

    def _check_thresholds(self, summary: Dict) -> Dict[str, bool]:
        """
        检查是否达到阈值

        Args:
            summary: 汇总统计

        Returns:
            各指标是否通过的字典
        """
        thresholds = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.7,
            "answer_correctness": 0.6
        }

        results = {}
        for metric, threshold in thresholds.items():
            if metric in summary:
                passed = summary[metric]["mean"] >= threshold
                results[metric] = passed

                status = "✅ PASS" if passed else "❌ FAIL"
                console.print(
                    f"  {metric}: {summary[metric]['mean']:.4f} (threshold: {threshold}) {status}"
                )

        return results

    def run_pipeline(self, test_file: str) -> Dict:
        """
        运行完整的评估流水线

        Args:
            test_file: 测试文件路径

        Returns:
            评估报告
        """
        console.print(Panel.fit(
            "[bold cyan]RAGAS 评估流水线[/bold cyan]\n\n"
            f"Agent: {self.agent_type}\n"
            f"指标: {', '.join(self.metric_names)}",
            border_style="cyan"
        ))

        # 1. 加载测试用例
        test_cases = self.load_test_cases(test_file)

        # 2. 批量运行系统
        dataset_dict = self.run_system(test_cases)

        # 3. 运行评估
        results_df = self.evaluate(dataset_dict)

        # 4. 生成报告
        report = self.generate_report(results_df, test_cases, dataset_dict)

        # 5. 检查是否通过
        console.print("\n" + "="*60)
        console.print("[bold]评估验收结果:[/bold]\n")

        all_passed = all(report["passed_threshold"].values())

        if all_passed:
            console.print("[bold green]✅ 评估通过！所有指标达到阈值。[/bold green]")
            exit_code = 0
        else:
            console.print("[bold red]❌ 评估失败！部分指标未达到阈值。[/bold red]")
            exit_code = 1

        console.print("="*60 + "\n")

        return report, exit_code


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAGAS 评估流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="hybrid_agent",
        choices=["naive_rag", "graph_agent", "hybrid_agent", "deep_research"],
        help="要评估的 Agent 类型"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
        help="要使用的评估指标"
    )

    parser.add_argument(
        "--test-file",
        type=str,
        default="test/evaluation/golden_set.json",
        help="测试用例文件路径"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test/evaluation/results",
        help="输出目录"
    )

    args = parser.parse_args()

    try:
        # 创建评估器
        evaluator = RAGASEvaluator(
            agent_type=args.agent,
            metrics=args.metrics,
            output_dir=args.output_dir
        )

        # 运行评估流水线
        report, exit_code = evaluator.run_pipeline(args.test_file)

        sys.exit(exit_code)

    except Exception as e:
        console.print(f"\n[bold red]❌ 评估流水线失败: {str(e)}[/bold red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
