#!/usr/bin/env python
"""
HotpotQA 多智能体评测脚本。

使用说明:
    1. 确保已下载 HotpotQA distractor 数据集，默认路径为 datasets/hotpot_dev_distractor_v1.json。
    2. 配置所需的大模型环境变量 (OPENAI_API_KEY 等)，以便多智能体编排链路正常调用。
    3. 在项目根目录执行:
           python test/hotpot_multi_agent_eval.py --limit 20 --retrieval-top-k 5 \
               --report-type short_answer --predictions outputs/hotpot_predictions.json
       其中各参数含义如下:
           --dataset           HotpotQA 数据集 JSON 文件路径。
           --limit             仅评测前 N 条样本，缺省为全部样本。
           --retrieval-top-k   每次检索返回的段落数量，控制 TF-IDF 召回。
           --report-type       多智能体 Reporter 输出类型 (short_answer / long_document)。
           --predictions       可选，保存预测结果及指标的输出文件。
           --quiet             若指定则屏蔽逐题日志，仅输出最终指标。

脚本功能:
    - 默认会注入一个 Neo4j 空实现（若需要真实图数据库，可设置环境变量 GRAPHRAG_FORCE_NEO4J=1）。
    - 使用 TF-IDF 方式替换默认的 Neo4j 检索工具，使多智能体流程无需 Neo4j 也可运行。
    - 针对每个问题调用 MultiAgentFacade.process_query，并从最终报告中提取答案。
    - 复用了官方 HotpotQA 评测指标 (EM / F1 / Precision / Recall)，并支持记录失败用例。
    - 可选 `--llm-eval` 启用大模型判定准确率（使用 graphrag_agent 内置模型访问接口）。
"""
import argparse
import json
import os
import re
import string
import sys
import types
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# 运行时禁用 Neo4j 连接：在加载 graphrag_agent 前注入一个轻量级的空实现
# ---------------------------------------------------------------------------

if os.environ.get("GRAPHRAG_FORCE_NEO4J", "0") != "1":
    class _DummyGraph:
        """提供 query / refresh_schema 等空操作，避免触发真实 Neo4j 连接。"""

        def refresh_schema(self) -> None:
            return None

        def query(self, _query: str, _params=None):
            return []

    class _DummySession:
        def close(self) -> None:
            return None

    class _DummyDriver:
        """模仿 Neo4j driver 的最小接口."""

        def execute_query(self, _cypher: str, parameters_=None, result_transformer_=None):
            if callable(result_transformer_):
                return pd.DataFrame()
            return pd.DataFrame()

        def session(self):
            return _DummySession()

        def close(self) -> None:
            return None

    class _DummyDBManager:
        """替代 DBConnectionManager，避免脚本导入阶段尝试连接数据库。"""

        def __init__(self):
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            self.driver = _DummyDriver()
            self.graph = _DummyGraph()
            self.session_pool = []
            self.max_pool_size = 0

        def get_driver(self):
            return self.driver

        def get_graph(self):
            return self.graph

        def execute_query(self, _cypher: str, params=None):
            return pd.DataFrame()

        def get_session(self):
            return _DummySession()

        def release_session(self, _session) -> None:
            return None

        def close(self) -> None:
            return None

    neo4jdb_stub = types.ModuleType("graphrag_agent.config.neo4jdb")

    def _get_db_manager():
        return neo4jdb_stub.db_manager

    neo4jdb_stub.DBConnectionManager = _DummyDBManager
    neo4jdb_stub.db_manager = _DummyDBManager()
    neo4jdb_stub.get_db_manager = _get_db_manager

    sys.modules["graphrag_agent.config.neo4jdb"] = neo4jdb_stub

from graphrag_agent.agents.multi_agent.integration import MultiAgentFacade
from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RETRIEVAL_SOURCE_CHOICES,
)
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.search import tool_registry


# --- HotpotQA 官方评测工具函数（按需裁剪与重写） ------------------------------------

def normalize_answer(text: str) -> str:
    """对答案进行大小写、标点、冠词与空白统一化处理。"""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """计算单个样本的 F1、精确率与召回率。"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero = (0.0, 0.0, 0.0)

    if (
        normalized_prediction in {"yes", "no", "noanswer"}
        and normalized_prediction != normalized_ground_truth
    ):
        return zero
    if (
        normalized_ground_truth in {"yes", "no", "noanswer"}
        and normalized_prediction != normalized_ground_truth
    ):
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return zero

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """HotpotQA 标准 EM 指标。"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer_metrics(
    metrics: Dict[str, float],
    prediction: str,
    gold: str,
) -> Tuple[float, float, float]:
    """累计 EM / F1 / Precision / Recall 指标。"""
    em = exact_match_score(prediction, gold)
    f1, precision, recall = f1_score(prediction, gold)
    metrics["em"] += em
    metrics["f1"] += f1
    metrics["precision"] += precision
    metrics["recall"] += recall
    return em, precision, recall


def llm_grade_answer(
    llm,
    *,
    question: str,
    prediction: str,
    reference: str,
) -> Tuple[float, str]:
    """
    调用 LLM 进行答案判定，返回 (分数, 解释)。

    LLM 要求输出 JSON：{"score": 0/1, "reason": "..."}。
    """
    prompt = (
        "你是HotpotQA数据集的自动评估器，需要判断模型回答是否正确。"
        "请仅返回JSON，格式为："
        '{"score": 0或1, "reason": "简要说明判断依据"}。\n\n'
        f"问题: {question}\n"
        f"模型答案: {prediction or '（空）'}\n"
        f"参考答案: {reference or '（空）'}\n"
        "请严格按照HotpotQA官方答案进行核对，若答案明显匹配或等价，则返回1，否则返回0。"
    )

    response = llm.invoke(prompt)
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        content = str(response)

    match = re.search(r"\{.*\}", content, re.S)
    if match is None:
        raise ValueError(f"LLM 返回内容无法解析为 JSON: {content}")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:  # noqa: F841
        raise ValueError(f"解析 JSON 失败: {content}") from exc

    score = float(payload.get("score", 0))
    score = max(0.0, min(score, 1.0))
    reason = str(payload.get("reason", "")).strip() or "LLM 未给出解释"
    return score, reason


# --- 数据集驱动的检索逻辑 ----------------------------------------------------------


@dataclass(frozen=True)
class CorpusDocument:
    """HotpotQA 段落文档的轻量封装。"""

    doc_id: str
    entry_id: str
    title: str
    text: str


class HotpotCorpus:
    """基于 HotpotQA 段落构建的简易 TF-IDF 索引。"""

    def __init__(self, examples: Sequence[Mapping[str, Any]]) -> None:
        documents: List[CorpusDocument] = []
        for example in examples:
            entry_id = example["_id"]
            for title, sentences in example.get("context", []):
                text = " ".join(sentence.strip() for sentence in sentences if sentence)
                if not text:
                    continue
                doc_id = f"{entry_id}::{title}"
                documents.append(
                    CorpusDocument(
                        doc_id=doc_id,
                        entry_id=entry_id,
                        title=title,
                        text=text,
                    )
                )

        if not documents:
            raise ValueError("Unable to build corpus: dataset did not contain paragraphs.")

        self._documents = documents
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(doc.text for doc in self._documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """返回与查询相似度排序后的段落。"""
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        query_vector = self._vectorizer.transform([clean_query])
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        if not np.any(scores):
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_indices[0]]) if top_indices.size else 0.0

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue
            normalized = score / max_score if max_score > 0 else 0.0
            document = self._documents[int(idx)]
            results.append(
                {
                    "id": document.doc_id,
                    "entry_id": document.entry_id,
                    "title": document.title,
                    "text": document.text,
                    "score": max(min(normalized, 1.0), 0.0),
                    "rank": rank,
                }
            )
        return results


class HotpotDatasetTool:
    """基于 TF-IDF 的检索工具，为多智能体流程提供段落证据。"""

    def __init__(self, corpus: HotpotCorpus, tool_name: str, *, top_k: int = 5) -> None:
        self._corpus = corpus
        self._tool_name = tool_name
        self._top_k = top_k

        valid_sources = set(RETRIEVAL_SOURCE_CHOICES)
        self._source_label = tool_name if tool_name in valid_sources else "custom"

    def _parse_query(self, query_input: Any) -> str:
        if isinstance(query_input, dict):
            return str(
                query_input.get("query")
                or query_input.get("input")
                or query_input.get("text")
                or ""
            )
        return str(query_input or "")

    def structured_search(self, query_input: Any) -> Dict[str, Any]:
        query = self._parse_query(query_input)
        hits = self._corpus.search(query, top_k=self._top_k)
        timestamp = datetime.utcnow().isoformat()

        retrieval_results: List[Dict[str, Any]] = []
        for hit in hits:
            retrieval_results.append(
                {
                    "result_id": f"{hit['id']}::{hit['rank']}",
                    "granularity": "Chunk",
                    "evidence": hit["text"],
                    "metadata": {
                        "source_id": hit["id"],
                        "source_type": "chunk",
                        "confidence": round(hit["score"], 4),
                        "timestamp": timestamp,
                        "extra": {
                            "title": hit["title"],
                            "entry_id": hit["entry_id"],
                            "retrieval_rank": hit["rank"],
                        },
                    },
                    "source": self._source_label,
                    "score": round(hit["score"], 4),
                    "created_at": timestamp,
                }
            )

        top_answer = hits[0]["text"] if hits else ""
        return {
            "answer": top_answer,
            "retrieval_results": retrieval_results,
        }

    def search(self, query_input: Any) -> str:
        return self.structured_search(query_input).get("answer", "")


class ToolRegistryPatcher:
    """用于在上下文中暂时替换全局工具注册表的辅助类。"""

    DATASET_TOOLS = [
        "local_search",
        "global_search",
        "hybrid_search",
        "naive_search",
        "deep_research",
        "deeper_research",
    ]

    def __init__(self, corpus: HotpotCorpus, *, top_k: int) -> None:
        self._corpus = corpus
        self._top_k = top_k
        self._original_registry: Optional[Dict[str, Any]] = None
        self._original_extra: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "ToolRegistryPatcher":
        self._original_registry = dict(tool_registry.TOOL_REGISTRY)
        self._original_extra = dict(tool_registry.EXTRA_TOOL_FACTORIES)

        def factory_for(tool_name: str):
            def factory() -> HotpotDatasetTool:
                return HotpotDatasetTool(self._corpus, tool_name, top_k=self._top_k)

            return factory

        for name in self.DATASET_TOOLS:
            tool_registry.TOOL_REGISTRY[name] = factory_for(name)

        tool_registry.EXTRA_TOOL_FACTORIES["chain_exploration"] = factory_for("chain_exploration")
        tool_registry.EXTRA_TOOL_FACTORIES["hypothesis_generator"] = factory_for("hypothesis_generator")
        tool_registry.EXTRA_TOOL_FACTORIES["answer_validator"] = factory_for("answer_validator")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._original_registry is not None:
            tool_registry.TOOL_REGISTRY.clear()
            tool_registry.TOOL_REGISTRY.update(self._original_registry)
        if self._original_extra is not None:
            tool_registry.EXTRA_TOOL_FACTORIES.clear()
            tool_registry.EXTRA_TOOL_FACTORIES.update(self._original_extra)


# --- 评测主流程 --------------------------------------------------------------------


def extract_answer(agent_payload: Mapping[str, Any]) -> str:
    """从多智能体输出中提取最终答案文本。"""
    response = agent_payload.get("response") if isinstance(agent_payload, Mapping) else ""
    if not isinstance(response, str):
        return ""
    split_tokens = ["#### 引用", "#### References", "#### Citation", "\n## References"]
    for token in split_tokens:
        if token in response:
            response = response.split(token, 1)[0]
            break
    return response.strip()


def run_evaluation(
    examples: Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int],
    retrieval_top_k: int,
    report_type: str,
    verbose: bool,
    enable_llm_eval: bool,
) -> Tuple[
    Dict[str, float],
    Dict[str, str],
    List[Tuple[str, str]],
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
    List[Tuple[str, str]],
]:
    """运行多智能体评测流程并累计指标。"""
    total_examples = len(examples) if limit is None else min(limit, len(examples))
    if total_examples == 0:
        raise ValueError("No examples selected for evaluation. Check --limit or dataset path.")

    corpus = HotpotCorpus(examples)

    metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    predictions: Dict[str, str] = {}
    failures: List[Tuple[str, str]] = []
    llm_summary = {"total": 0, "accept": 0, "score": 0.0}
    llm_details: Dict[str, Dict[str, Any]] = {}
    llm_failures: List[Tuple[str, str]] = []

    with ToolRegistryPatcher(corpus, top_k=retrieval_top_k):
        agent = MultiAgentFacade()
        llm = None

        for index, example in enumerate(examples[:total_examples], start=1):
            question_id = example["_id"]
            question = example["question"]

            if verbose:
                print(f"[{index}/{total_examples}] QID={question_id} :: {question}")

            try:
                result = agent.process_query(question, report_type=report_type)
                answer = extract_answer(result)
            except Exception as exc:  # noqa: BLE001
                answer = ""
                failures.append((question_id, str(exc)))
                if verbose:
                    print(f"    ✗ failed: {exc}")
            else:
                if verbose:
                    preview = answer[:120] + ("…" if len(answer) > 120 else "")
                    print(f"    ✓ answer: {preview}")

            predictions[question_id] = answer
            update_answer_metrics(metrics, answer, example["answer"])

            if not enable_llm_eval:
                continue

            if llm is None:
                try:
                    llm = get_llm_model()
                except Exception as exc:  # noqa: BLE001
                    llm_failures.append((question_id, f"初始化LLM失败: {exc}"))
                    if verbose:
                        print(f"    ⚠ LLM初始化失败: {exc}")
                    enable_llm_eval = False
                    continue

            try:
                score, reason = llm_grade_answer(
                    llm,
                    question=question,
                    prediction=answer,
                    reference=example["answer"],
                )
            except Exception as exc:  # noqa: BLE001
                llm_failures.append((question_id, str(exc)))
                if verbose:
                    print(f"    ⚠ LLM判定失败: {exc}")
            else:
                llm_summary["total"] += 1
                llm_summary["score"] += score
                if score >= 0.5:
                    llm_summary["accept"] += 1
                llm_details[question_id] = {"score": score, "reason": reason}
                if verbose:
                    print(f"    ↳ LLM评估: score={score:.2f}, reason={reason}")

    if total_examples > 0:
        for key in metrics:
            metrics[key] /= total_examples

    if llm_summary["total"] > 0:
        llm_summary["average_score"] = llm_summary["score"] / llm_summary["total"]
        llm_summary["accept_rate"] = llm_summary["accept"] / llm_summary["total"]
    else:
        llm_summary["average_score"] = 0.0
        llm_summary["accept_rate"] = 0.0

    return metrics, predictions, failures, llm_summary, llm_details, llm_failures


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in HotpotQA file, got {type(data)}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-agent pipeline on HotpotQA without Neo4j.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/hotpot_dev_distractor_v1.json"),
        help="Path to the HotpotQA distractor JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Number of TF-IDF paragraphs to return per retrieval call.",
    )
    parser.add_argument(
        "--report-type",
        type=str,
        default="short_answer",
        help="Reporter output type passed to the orchestrator.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional path to write model predictions as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-example logging.",
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="启用大模型判定答案正确性（需要已配置OPENAI等环境变量）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    limit = args.limit if args.limit and args.limit > 0 else None
    selected = dataset if limit is None else dataset[:limit]

    (
        metrics,
        predictions,
        failures,
        llm_summary,
        llm_details,
        llm_failures,
    ) = run_evaluation(
        selected,
        limit=limit,
        retrieval_top_k=args.retrieval_top_k,
        report_type=args.report_type,
        verbose=not args.quiet,
        enable_llm_eval=args.llm_eval,
    )

    print("\n=== HotpotQA Answer Metrics ===")
    print(f"Exact Match: {metrics['em']:.4f}")
    print(f"F1:          {metrics['f1']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Failures:    {len(failures)}")

    if args.llm_eval:
        print("\n=== LLM 评估结果 ===")
        print(f"平均得分:   {llm_summary.get('average_score', 0.0):.4f}")
        print(f"通过率:     {llm_summary.get('accept_rate', 0.0):.4f}")
        print(f"LLM失败数:  {len(llm_failures)}")
        if llm_failures:
            print("LLM失败示例（最多5条）:")
            for question_id, error in llm_failures[:5]:
                print(f"- {question_id}: {error}")

    if failures:
        print("\nFirst few failures:")
        for question_id, error in failures[:5]:
            print(f"- {question_id}: {error}")

    if args.predictions:
        payload = {
            "predictions": predictions,
            "metrics": metrics,
            "failures": failures,
        }
        if args.llm_eval:
            payload["llm_summary"] = llm_summary
            payload["llm_details"] = llm_details
            payload["llm_failures"] = llm_failures
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nSaved predictions to {args.predictions}")


if __name__ == "__main__":
    main()
