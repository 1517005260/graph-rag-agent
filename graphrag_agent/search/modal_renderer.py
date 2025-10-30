from __future__ import annotations

"""多模态渲染辅助模块。

负责将检索阶段返回的多模态段落（特别是图片）转换为便于前端/LLM消费的结构。
功能包括：
- 过滤图片段落并提取基础元数据
- 尝试从 MinerU 产物中加载本地图片，供视觉模型分析
- 调用视觉模型生成图片解读摘要（若配置了视觉模型）
- 输出标准化的 Markdown 片段与结构化描述
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from graphrag_agent.agents.image_utils import (
    build_vision_message_content,
    load_multiple_images,
)
from graphrag_agent.config.settings import MINERU_OUTPUT_DIR
from graphrag_agent.models.get_models import get_vision_model
from graphrag_agent.search.modal_enricher import ModalSummary


@dataclass
class ImageDetail:
    """图片渲染所需信息。"""

    segment_id: Optional[str]
    url: str
    relative_path: Optional[str] = None
    caption: Optional[str] = None
    description: Optional[str] = None
    footnotes: Sequence[str] = field(default_factory=list)
    source: Optional[str] = None
    page_index: Optional[int] = None
    vision_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化结构。"""
        return {
            "segment_id": self.segment_id,
            "url": self.url,
            "relative_path": self.relative_path,
            "caption": self.caption,
            "description": self.description,
            "footnotes": list(self.footnotes),
            "source": self.source,
            "page_index": self.page_index,
            "vision_summary": self.vision_summary,
        }


@dataclass
class ModalEnhancement:
    """多模态增强输出。"""

    markdown: str = ""
    vision_analysis: Optional[str] = None
    image_details: List[ImageDetail] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "markdown": self.markdown,
            "vision_analysis": self.vision_analysis,
            "image_details": [detail.to_dict() for detail in self.image_details],
        }

    def apply_to_answer(self, answer: str) -> str:
        """将多模态Markdown附加到原始答案后。"""
        if not self.markdown:
            return answer
        answer = answer.strip()
        if answer:
            return f"{answer}\n\n{self.markdown}"
        return self.markdown

    def has_images(self) -> bool:
        return bool(self.image_details)


class ModalAssetProcessor:
    """处理多模态（图片）检索结果，生成Markdown与结构化描述。"""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_images: int = 3,
    ) -> None:
        self.base_dir = Path(base_dir or MINERU_OUTPUT_DIR).resolve()
        self.max_images = max(1, max_images)

    @staticmethod
    def _collect_image_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """筛选带图片URL的段落。"""
        image_segments: List[Dict[str, Any]] = []
        for segment in segments:
            if not segment:
                continue
            if (segment.get("type") or "").lower() != "image":
                continue
            image_url = segment.get("image_url")
            if not image_url:
                continue
            image_segments.append(segment)
        return image_segments

    def _build_details(self, segments: Sequence[Dict[str, Any]]) -> List[ImageDetail]:
        """从段落构建基础的图片描述。"""
        details: List[ImageDetail] = []
        for segment in segments[: self.max_images]:
            caption_parts: List[str] = []
            if segment.get("image_caption"):
                caption_parts.extend(
                    str(item).strip()
                    for item in segment["image_caption"]
                    if isinstance(item, str) and item.strip()
                )
            caption_text = "；".join(caption_parts) if caption_parts else None
            description = segment.get("text") or caption_text
            details.append(
                ImageDetail(
                    segment_id=segment.get("segment_id"),
                    url=segment.get("image_url"),
                    relative_path=segment.get("image_relative_path"),
                    caption=caption_text,
                    description=description,
                    footnotes=tuple(segment.get("image_footnote") or []),
                    source=segment.get("source"),
                    page_index=segment.get("page_index"),
                )
            )
        return details

    def _load_images_for_vision(self, image_urls: Sequence[str]) -> List[Tuple[str, str]]:
        """尝试从 MinerU 输出目录加载图片并编码为Base64。"""
        if not image_urls:
            return []
        if not self.base_dir.exists():
            return []
        try:
            return load_multiple_images(
                list(dict.fromkeys(image_urls)),
                self.base_dir,
                max_count=self.max_images,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"加载图片资源失败: {exc}")
            return []

    @staticmethod
    def _invoke_vision_model(
        question: str,
        context: str,
        base64_images: Sequence[Tuple[str, str]],
    ) -> Optional[str]:
        """调用视觉模型生成图片摘要。"""
        if not base64_images:
            return None
        try:
            client, model_name = get_vision_model()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"初始化视觉模型失败: {exc}")
            return None
        if not model_name:
            return None

        try:
            content = build_vision_message_content(
                question=question,
                image_base64_list=list(base64_images),
                context=context,
            )
            # 使用 OpenAI Chat Completions API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=1000,
            )
            # 解析 OpenAI 响应
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message.content and isinstance(message.content, str):
                    return message.content.strip()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"视觉模型解析失败: {exc}")
        return None

    @staticmethod
    def _compose_markdown(
        details: Sequence[ImageDetail],
        vision_analysis: Optional[str],
    ) -> str:
        """根据图片详情与视觉摘要生成Markdown片段。"""
        if not details:
            return ""

        lines: List[str] = ["### 相关图片参考"]
        if vision_analysis:
            lines.append(vision_analysis.strip())

        for index, detail in enumerate(details, start=1):
            caption = detail.caption or detail.description or f"图片 {index}"
            lines.append(f"{index}. {caption}")
            if detail.description and detail.description.strip() != caption.strip():
                lines.append(f"   描述：{detail.description.strip()}")
            if detail.vision_summary and detail.vision_summary.strip():
                lines.append(f"   模型解读：{detail.vision_summary.strip()}")
            if detail.page_index is not None:
                lines.append(f"   页码：{detail.page_index}")
            lines.append(f"![{caption}]({detail.url})")
        return "\n".join(lines)

    def enhance_answer(
        self,
        *,
        question: str,
        answer: str,
        modal_summary: Optional[ModalSummary],
        context: Optional[str] = None,
    ) -> ModalEnhancement:
        """根据多模态摘要生成Markdown与结构化描述。"""

        if modal_summary is None or not modal_summary.segments:
            return ModalEnhancement()

        image_segments = self._collect_image_segments(modal_summary.segments)
        if not image_segments:
            return ModalEnhancement()

        details = self._build_details(image_segments)
        image_urls = [detail.url for detail in details if detail.url]

        base64_images = self._load_images_for_vision(image_urls)
        vision_analysis = self._invoke_vision_model(
            question=question,
            context=context or answer or "",
            base64_images=base64_images,
        )

        # 将整体视觉摘要分配给各个图片详情
        if vision_analysis:
            for detail in details:
                detail.vision_summary = vision_analysis

        markdown = self._compose_markdown(details, vision_analysis)
        return ModalEnhancement(
            markdown=markdown,
            vision_analysis=vision_analysis,
            image_details=details,
        )
