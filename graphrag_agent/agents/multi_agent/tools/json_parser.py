"""用于从 LLM 输出中提取 JSON 片段的工具函数。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_CANDIDATE_RE = re.compile(r"{.*}", re.DOTALL)


def extract_json_text(text: str) -> str:
    """从带格式的模型输出中提取 JSON 子串。"""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("空响应，无法提取JSON")

    fenced = _CODE_BLOCK_RE.search(cleaned)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未找到JSON结构")
    return cleaned[start : end + 1]


def parse_json_text(text: str) -> Dict[str, Any]:
    """将模型输出解析为 JSON 对象，解析失败时抛出 ValueError。"""
    candidate = extract_json_text(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("无法解析JSON结构") from exc


__all__ = ["extract_json_text", "parse_json_text"]
