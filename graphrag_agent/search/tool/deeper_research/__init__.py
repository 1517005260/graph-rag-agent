"""
增强版深度研究工具的辅助模块集合。

该包存放从 `deeper_research_tool.py` 中拆分出的功能模块，
用于保持主文件精简且便于维护。
"""

from .enhancer import enhance_search_with_coe
from .branching import (
    create_multiple_reasoning_branches,
    detect_and_resolve_contradictions,
    generate_citations,
    merge_reasoning_branches,
)

__all__ = [
    "enhance_search_with_coe",
    "create_multiple_reasoning_branches",
    "detect_and_resolve_contradictions",
    "generate_citations",
    "merge_reasoning_branches",
]
