"""Utility helpers to safely invoke tools and normalize outputs."""

from typing import Any

from graphrag_agent.utils.retrieval_normalize import normalize_retrieval_output


def _resolve_tool_runner(tool: Any) -> Any:
    """
    Resolve a callable tool runner that exposes _run.

    Supports:
    - BaseTool instances (have _run)
    - Tool containers exposing get_tool()
    """
    if hasattr(tool, "_run"):
        return tool
    if hasattr(tool, "get_tool"):
        inner = tool.get_tool()
        if inner is not tool:
            return _resolve_tool_runner(inner)
    return None


def invoke_tool_str(tool: Any, payload: Any) -> str:
    """
    Invoke a tool and guarantee a string output.

    Preference order:
    1) _run (tool or inner tool)
    2) search (legacy fallback)
    """
    runner = _resolve_tool_runner(tool)
    if runner and hasattr(runner, "_run"):
        result = runner._run(payload)
    elif hasattr(tool, "search"):
        result = tool.search(payload)
    else:
        raise ValueError("Tool does not provide _run or search")

    return normalize_retrieval_output(result)


def invoke_tool_structured(tool: Any, payload: Any) -> dict:
    """
    Invoke a tool expecting structured output; fallback to string answer.
    """
    if hasattr(tool, "structured_search"):
        return tool.structured_search(payload)

    answer = invoke_tool_str(tool, payload)
    return {"answer": answer, "retrieval_results": []}
