"""
MinerU 解析结果增强工具。

负责为图片段落生成视觉描述，并在需要时写回段落结构。
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphrag_agent.config.prompts import get_prompt_by_name
from graphrag_agent.config.settings import (
    MINERU_VISION_SUMMARY_ENABLE,
    MINERU_VISION_PROMPT_NAME,
    MINERU_VISION_MAX_WORKERS,
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
) -> Tuple[int, int, float]:
    """为 MinerU 的图片段落生成视觉摘要，返回(总图片数, 新增描述数, 耗时)。"""
    start_time = time.time()

    if not segments:
        return 0, 0, 0.0

    total_images = 0
    newly_generated = 0

    base_dir: Optional[Path] = None
    if isinstance(method_dir, Path):
        base_dir = method_dir
    elif method_dir:
        try:
            base_dir = Path(method_dir)
        except Exception:  # pragma: no cover - 防御性
            base_dir = None

    pending: Dict[Path, List[int]] = {}

    for index, segment in enumerate(segments):
        if (segment.get("type") or "").lower() != "image":
            continue
        total_images += 1

        if segment.get("vision_summary"):
            continue

        image_path = _resolve_image_path(segment, base_dir)
        if image_path is None:
            continue

        pending.setdefault(image_path, []).append(index)

    if not prompt_text or not pending:
        # 即便没有生成新描述，也返回已有统计
        elapsed = time.time() - start_time
        return total_images, 0, elapsed

    max_workers = max(1, MINERU_VISION_MAX_WORKERS)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path in pending:
            futures[executor.submit(_call_vision_model, path, prompt_text)] = path

        for future in as_completed(futures):
            path = futures[future]
            try:
                summary = future.result()
            except Exception:  # pragma: no cover - 防御性
                summary = None

            if not summary:
                continue

            for index in pending[path]:
                segment = segments[index]
                segment["vision_summary"] = summary
                _merge_summary_into_segment(segment, summary)
                newly_generated += 1

    elapsed = time.time() - start_time
    return total_images, newly_generated, elapsed


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
