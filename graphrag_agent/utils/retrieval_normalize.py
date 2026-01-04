"""Utility helpers to normalize retrieval outputs into strings."""

import json
from typing import Any


def normalize_retrieval_output(value: Any) -> str:
    """
    Normalize retrieval outputs to a string to avoid type errors in prompt construction.

    This prevents generator steps from failing when they receive dicts, lists, or
    LangChain message objects instead of raw strings.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, list):
        return "\n".join(normalize_retrieval_output(item) for item in value)

    if hasattr(value, "content"):
        return normalize_retrieval_output(getattr(value, "content"))

    return str(value)
