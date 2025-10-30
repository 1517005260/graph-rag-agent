"""
MinerU 解析结果增强工具。

负责为图片段落生成视觉描述，并在需要时写回段落结构。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphrag_agent.config.prompts import get_prompt_by_name
from graphrag_agent.config.settings import (
    MINERU_VISION_SUMMARY_ENABLE,
    MINERU_VISION_PROMPT_NAME,
)
from graphrag_agent.agents.image_utils import load_image_as_base64
from graphrag_agent.search.modal_renderer import ModalAssetProcessor


def load_image_summary_prompt() -> Optional[str]:
    """获取视觉描述提示词。"""
    if not MINERU_VISION_SUMMARY_ENABLE:
        return None
    return get_prompt_by_name(
        MINERU_VISION_PROMPT_NAME,
    default="请详细描述这张图片的重要内容和文字信息。",
)


def augment_image_segments_with_vision(
    segments: List[Dict[str, Any]],
    method_dir: Optional[Path],
    prompt_text: Optional[str],
) -> None:
    """为 MinerU 的图片段落生成视觉摘要。"""
    if not prompt_text:
        return

    if not segments:
        return

    base_dir: Optional[Path] = None
    if isinstance(method_dir, Path):
        base_dir = method_dir
    elif method_dir:
        try:
            base_dir = Path(method_dir)
        except Exception:  # pragma: no cover - 防御性
            base_dir = None

    cache: Dict[str, Optional[str]] = {}

    for segment in segments:
        if (segment.get("type") or "").lower() != "image":
            continue
        if segment.get("vision_summary"):
            continue

        image_path = _resolve_image_path(segment, base_dir)
        if image_path is None:
            continue

        cache_key = str(image_path)
        if cache_key in cache:
            summary = cache[cache_key]
        else:
            summary = _call_vision_model(image_path, prompt_text)
            cache[cache_key] = summary

        if not summary:
            continue

        segment["vision_summary"] = summary
        _merge_summary_into_segment(segment, summary)


def _resolve_image_path(
    segment: Dict[str, Any],
    base_dir: Optional[Path],
) -> Optional[Path]:
    raw_path = segment.get("image_path")
    if raw_path:
        try:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate
        except Exception:  # pragma: no cover - 防御性
            pass

    rel_path = segment.get("image_relative_path")
    if base_dir and rel_path:
        try:
            candidate = (base_dir / rel_path).resolve()
            if candidate.exists():
                return candidate
        except Exception:  # pragma: no cover
            pass
    return None


def _call_vision_model(image_path: Path, prompt_text: str) -> Optional[str]:
    base64_data = load_image_as_base64(image_path)
    if not base64_data:
        return None
    summary = ModalAssetProcessor._invoke_vision_model(  # pylint: disable=protected-access
        question=prompt_text,
        context="",
        base64_images=[(str(image_path), base64_data)],
    )
    if isinstance(summary, str):
        summary = summary.strip()
    return summary or None


def _merge_summary_into_segment(segment: Dict[str, Any], summary: str) -> None:
    existing = segment.get("text") or ""
    if summary in existing:
        return
    if existing:
        combined = f"{existing}\n视觉描述：{summary}"
    else:
        combined = f"视觉描述：{summary}"
    segment["text"] = combined.strip()
