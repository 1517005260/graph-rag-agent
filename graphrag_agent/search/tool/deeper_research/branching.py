from __future__ import annotations

from typing import Dict, List, Optional, Any


def create_multiple_reasoning_branches(tool, query_id, hypotheses: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    基于生成的假设创建多个推理分支。
    """
    branch_results: Dict[str, Dict[str, Any]] = {}

    if hypotheses is None:
        original_query = None
        for query, current_id in tool.current_query_context.items():
            if current_id == query_id:
                original_query = query
                break

        if original_query is None:
            if hasattr(tool.thinking_engine, "query"):
                original_query = tool.thinking_engine.query
            else:
                tool._log("\n[分支推理] 无法找到原始查询，无法生成假设")
                return {}

        if not hasattr(tool, "_hypotheses_cache"):
            tool._hypotheses_cache = {}

        if query_id in tool._hypotheses_cache:
            hypotheses = tool._hypotheses_cache[query_id]
        else:
            hypotheses = tool.query_generator.generate_multiple_hypotheses(original_query, tool.llm)
            tool._hypotheses_cache[query_id] = hypotheses

    for i, hypothesis in enumerate((hypotheses or [])[:3]):
        branch_name = f"branch_{i+1}"
        tool.thinking_engine.branch_reasoning(branch_name)
        tool._log(f"\n[分支推理] 创建分支 {branch_name}: {hypothesis}")

        step_id = tool.evidence_tracker.add_reasoning_step(
            query_id,
            f"branch_{branch_name}",
            f"基于假设: {hypothesis} 创建推理分支",
        )

        tool.explored_branches[branch_name] = {
            "hypothesis": hypothesis,
            "step_id": step_id,
            "evidence": [],
        }

        tool.thinking_engine.add_reasoning_step(f"探索假设: {hypothesis}")

        if i == 0:
            counter_cache_key = f"counter:{query_id}:{hypothesis}"
            if hasattr(tool, "_counter_cache") and counter_cache_key in tool._counter_cache:
                counter_analysis = tool._counter_cache[counter_cache_key]
            else:
                counter_analysis = tool.thinking_engine.counter_factual_analysis(f"假设 {hypothesis} 不成立")
                if not hasattr(tool, "_counter_cache"):
                    tool._counter_cache = {}
                tool._counter_cache[counter_cache_key] = counter_analysis

            tool.evidence_tracker.add_evidence(
                step_id,
                f"counter_analysis_{i}",
                counter_analysis,
                "counter_factual_analysis",
            )

            branch_results[branch_name] = {
                "hypothesis": hypothesis,
                "counter_analysis": counter_analysis,
            }
        else:
            branch_results[branch_name] = {"hypothesis": hypothesis}

    tool.thinking_engine.switch_branch("main")
    return branch_results


def detect_and_resolve_contradictions(tool, query_id):
    """
    对已收集证据进行矛盾检测，并记录分析结果。
    """
    cache_key = f"contradiction:{query_id}"
    if hasattr(tool, "_contradiction_detailed_cache") and cache_key in tool._contradiction_detailed_cache:
        return tool._contradiction_detailed_cache[cache_key]

    all_evidence: List[str] = []
    reasoning_chain = tool.evidence_tracker.get_reasoning_chain(query_id)

    for step in reasoning_chain.get("steps", []):
        evidence_ids = step.get("evidence_ids", [])
        if evidence_ids:
            all_evidence.extend(evidence_ids)

    contradictions = tool.evidence_tracker.detect_contradictions(all_evidence)

    if contradictions:
        tool._log(f"\n[矛盾检测] 发现 {len(contradictions)} 个矛盾")

        contradiction_step_id = tool.evidence_tracker.add_reasoning_step(
            query_id,
            "contradiction_analysis",
            f"分析 {len(contradictions)} 个信息矛盾",
        )

        for i, contradiction in enumerate(contradictions):
            contradiction_type = contradiction.get("type", "unknown")
            if contradiction_type == "numerical":
                analysis = (
                    f"数值矛盾: 在 '{contradiction.get('context', '')}' 中, "
                    f"发现值 {contradiction.get('value1')} 和 {contradiction.get('value2')}"
                )
            elif contradiction_type == "semantic":
                analysis = f"语义矛盾: {contradiction.get('analysis', '')}"
            else:
                analysis = contradiction.get("analysis", "")

            tool.evidence_tracker.add_evidence(
                contradiction_step_id,
                f"contradiction_{i}",
                analysis,
                "contradiction_evidence",
            )

            tool._log(f"\n[矛盾分析] {analysis}")

        result = {"contradictions": contradictions, "step_id": contradiction_step_id}
    else:
        result = {"contradictions": [], "step_id": None}

    if not hasattr(tool, "_contradiction_detailed_cache"):
        tool._contradiction_detailed_cache = {}
    tool._contradiction_detailed_cache[cache_key] = result

    return result


def generate_citations(tool, answer, query_id):
    """
    使用证据链跟踪器为答案生成引用标记。
    """
    citation_result = tool.evidence_tracker.generate_citations(answer)
    cited_answer = citation_result.get("cited_answer", answer)
    tool._log(f"\n[引用生成] 添加了 {len(citation_result.get('citations', []))} 个引用")
    return cited_answer


def merge_reasoning_branches(tool, query_id):
    """
    汇总多个推理分支的要点，并返回 Markdown 片段。
    """
    merged_reasoning = "## 多分支推理结果\n\n"
    branch_names = list(tool.explored_branches.keys())

    if not branch_names:
        return ""

    for branch_name in branch_names:
        branch_info = tool.explored_branches[branch_name]
        evidence = tool.evidence_tracker.get_step_evidence(branch_info["step_id"])

        merged_reasoning += f"### 分支: {branch_name}\n"
        merged_reasoning += f"假设: {branch_info['hypothesis']}\n\n"

        if evidence:
            merged_reasoning += "主要发现:\n"
            for ev in evidence[:3]:
                content = ev.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                merged_reasoning += f"- {content}\n"

        if "counter_analysis" in branch_info:
            counter_analysis = branch_info["counter_analysis"]
            if len(counter_analysis) > 200:
                counter_analysis = counter_analysis[:200] + "..."
            merged_reasoning += f"\n反事实分析: {counter_analysis}\n\n"

        merged_reasoning += "\n"

    for branch_name in branch_names:
        tool.deep_research.thinking_engine.switch_branch(branch_name)
        tool.deep_research.thinking_engine.merge_branches(branch_name, "main")

    tool.deep_research.thinking_engine.switch_branch("main")
    return merged_reasoning
