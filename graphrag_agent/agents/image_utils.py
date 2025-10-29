"""
图片处理工具模块

提供多模态RAG所需的图片处理功能，包括：
- 本地图片文件的加载与Base64编码
- 批量图片处理与尺寸优化
- OpenAI Vision API消息格式构造
- Token消耗估算

主要用于支持Vision Model对图片内容的理解与分析。
"""

import base64
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import io


def load_image_as_base64(
    image_path: Path,
    max_size: Optional[Tuple[int, int]] = None,
    quality: int = 85
) -> Optional[str]:
    """
    从本地路径加载图片并转换为Base64编码

    该函数负责读取本地图片文件，进行必要的格式转换和尺寸优化，
    最终返回符合OpenAI Vision API要求的Base64编码字符串。

    参数:
        image_path: 图片文件的绝对路径
        max_size: 可选的最大尺寸限制 (width, height)，用于压缩大尺寸图片
        quality: JPEG压缩质量参数 (1-100)，默认值为85，值越大质量越高

    返回:
        Base64编码的图片字符串。如果加载失败则返回None。

    异常:
        加载失败时记录错误日志但不抛出异常，确保系统稳定性。
    """
    try:
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return None

        # 读取图片
        with Image.open(image_path) as img:
            # 转换为RGB（某些图片可能是RGBA或其他模式）
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # 如果指定了最大尺寸，进行resize
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 保存到内存中的字节流
            buffer = io.BytesIO()
            img_format = 'JPEG' if image_path.suffix.lower() in ['.jpg', '.jpeg'] else 'PNG'

            if img_format == 'JPEG':
                img.save(buffer, format=img_format, quality=quality, optimize=True)
            else:
                img.save(buffer, format=img_format, optimize=True)

            # 转换为base64
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode('utf-8')

            print(f"Loaded image {image_path.name}: {len(encoded)} bytes (base64)")
            return encoded

    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None


def load_multiple_images(
    image_urls: List[str],
    base_dir: Path,
    max_count: int = 3,
    max_size: Tuple[int, int] = (1024, 1024),
    quality: int = 85
) -> List[Tuple[str, str]]:
    """
    批量加载多张图片并编码为Base64

    该函数从MinerU输出的URL列表中解析图片路径，批量加载本地文件，
    并进行尺寸优化和Base64编码。支持数量限制以控制Token消耗。

    参数:
        image_urls: 图片URL列表，格式为: http://host/download/{task_id}/{filename}
        base_dir: MinerU输出目录的根路径（通常为mineru_outputs/）
        max_count: 单次最多加载的图片数量，用于控制成本
        max_size: 每张图片的最大尺寸限制（宽度, 高度）
        quality: JPEG压缩质量参数 (1-100)

    返回:
        包含(url, base64_data)元组的列表，其中url为原始URL，
        base64_data为编码后的图片数据。加载失败的图片会被跳过。

    说明:
        函数会在mineru_outputs/{task_id}/目录下递归查找对应的图片文件。
    """
    results = []

    for url in image_urls[:max_count]:
        try:
            # 从URL提取task_id和filename
            # URL格式: http://localhost:8899/download/task_id/filename
            parts = url.split('/download/')
            if len(parts) < 2:
                print(f"Invalid URL format: {url}")
                continue

            path_parts = parts[1].split('/', 1)
            if len(path_parts) < 2:
                print(f"Invalid URL path: {url}")
                continue

            task_id, filename = path_parts[0], path_parts[1]

            # 构建本地路径：mineru_outputs/task_id/下递归查找filename
            task_dir = base_dir / task_id
            if not task_dir.exists():
                print(f"Task directory not found: {task_dir}")
                continue

            # 递归查找文件
            image_path = None
            for found_path in task_dir.rglob(filename):
                image_path = found_path
                break

            if not image_path:
                print(f"Image file not found: {filename} in {task_dir}")
                continue

            # 加载并编码
            base64_data = load_image_as_base64(image_path, max_size, quality)
            if base64_data:
                results.append((url, base64_data))
                print(f"Loaded image: {image_path.name}")

        except Exception as e:
            print(f"Failed to process image URL {url}: {e}")
            continue

    return results


def get_image_mime_type(image_path: Path) -> str:
    """
    根据文件扩展名获取MIME类型

    Args:
        image_path: 图片路径

    Returns:
        MIME类型字符串，如 "image/jpeg"
    """
    suffix = image_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return mime_types.get(suffix, 'image/jpeg')


def build_vision_message_content(
    question: str,
    image_base64_list: List[Tuple[str, str]],
    context: Optional[str] = None
) -> List[dict]:
    """
    构建符合OpenAI Vision API规范的消息内容

    该函数将用户问题、文本上下文和图片数据组合成OpenAI Vision API
    所需的消息格式，支持多图片的批量处理。

    参数:
        question: 用户提出的问题文本
        image_base64_list: 图片数据列表，每个元素为(url, base64_data)元组
        context: 可选的检索上下文文本，用于提供额外信息

    返回:
        符合OpenAI Vision API规范的content列表，包含：
        - 文本部分：整合了问题、上下文和图片提示
        - 图片部分：每张图片的Base64数据URL

    格式示例:
        [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}}
        ]
    """
    content = []

    # 添加问题和上下文
    text_parts = []
    if context:
        text_parts.append(f"参考上下文：\n{context}\n")
    text_parts.append(f"用户问题：{question}\n")

    if image_base64_list:
        text_parts.append(f"\n以下是{len(image_base64_list)}张相关图片，请结合图片内容回答问题：")

    content.append({
        "type": "text",
        "text": "\n".join(text_parts)
    })

    # 添加图片
    for i, (url, base64_data) in enumerate(image_base64_list, 1):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}",
                "detail": "high"  # 可选: "low", "high", "auto"
            }
        })

    return content


def estimate_vision_tokens(
    text_length: int,
    num_images: int,
    image_detail: str = "high"
) -> int:
    """
    估算vision API的token消耗

    根据OpenAI文档：
    - low detail: 85 tokens per image
    - high detail: 170-765 tokens per image (取决于图片尺寸)

    Args:
        text_length: 文本字符数
        num_images: 图片数量
        image_detail: "low" or "high"

    Returns:
        估算的总token数
    """
    # 文本token估算（中文约1.5字符/token，英文约4字符/token）
    text_tokens = text_length / 2  # 保守估计

    # 图片token
    if image_detail == "low":
        image_tokens = num_images * 85
    else:  # high or auto
        image_tokens = num_images * 450  # 取中间值

    return int(text_tokens + image_tokens)
