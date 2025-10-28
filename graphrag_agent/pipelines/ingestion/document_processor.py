import json
import os
import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from graphrag_agent.config.settings import (
    CHUNK_SIZE,
    DOCUMENT_PROCESSOR_MODE,
    FILES_DIR,
    MINERU_API_TIMEOUT,
    MINERU_API_URL,
    MINERU_DEFAULT_BACKEND,
    MINERU_DEFAULT_LANG,
    MINERU_DEFAULT_PARSE_METHOD,
    MINERU_FORMULA_ENABLE,
    MINERU_OUTPUT_DIR,
    MINERU_TABLE_ENABLE,
    OVERLAP,
)
from graphrag_agent.pipelines.ingestion.file_reader import FileReader
from graphrag_agent.pipelines.ingestion.text_chunker import ChineseTextChunker
from graphrag_agent.pipelines.mineru_client import MinerUClient, ParseOptions, ParseResult


class DocumentProcessor:
    """
    文档处理器：支持传统文本分块与 MinerU 多模态解析
    """

    MINERU_SUPPORTED_SUFFIXES = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
        ".jp2",
    }

    def __init__(
        self,
        directory_path: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
        mode: Optional[str] = None,
    ):
        self.directory_path = directory_path
        self.file_reader = FileReader(directory_path)
        self.chunker = ChineseTextChunker(chunk_size, overlap)

        self.mode = (mode or DOCUMENT_PROCESSOR_MODE or "legacy").lower()
        if self.mode not in {"legacy", "mineru"}:
            print(f"警告: 文档处理模式 {self.mode} 不受支持，退回使用 legacy。")
            self.mode = "legacy"

        self.mineru_client: Optional[MinerUClient] = None
        self._mineru_options: Optional[ParseOptions] = None
        if self.mode == "mineru":
            self._init_mineru()

    def _init_mineru(self) -> None:
        """初始化 MinerU 客户端"""
        try:
            self.mineru_client = MinerUClient(MINERU_API_URL, timeout=MINERU_API_TIMEOUT)
            self._mineru_options = ParseOptions(
                backend=MINERU_DEFAULT_BACKEND,
                parse_method=MINERU_DEFAULT_PARSE_METHOD,
                lang=MINERU_DEFAULT_LANG,
                formula_enable=MINERU_FORMULA_ENABLE,
                table_enable=MINERU_TABLE_ENABLE,
                output_format="mm_md",
            )
        except Exception as exc:
            print(f"初始化 MinerU 客户端失败: {exc}，自动回退至 legacy 模式。")
            self.mode = "legacy"
            self.mineru_client = None
            self._mineru_options = None

    # ====== 公共接口 ======
    def process_directory(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        处理目录中的文件，按配置选择解析策略
        """
        selected_files = self._collect_files(file_extensions, recursive)
        print(f"DocumentProcessor找到的文件数量: {len(selected_files)}")
        if selected_files:
            print(f"文件类型: {[Path(path).suffix for path in selected_files]}")

        results: List[Dict[str, Any]] = []
        for relative_path in selected_files:
            extension = Path(relative_path).suffix.lower()
            try:
                if self._should_use_mineru(extension):
                    result = self._process_file_with_mineru(relative_path, extension)
                else:
                    result = self._process_file_legacy(relative_path, extension)
            except Exception as exc:
                result = {
                    "filepath": relative_path,
                    "filename": Path(relative_path).name,
                    "extension": extension,
                    "processing_mode": "error",
                    "error": str(exc),
                }
                print(f"处理文件 {relative_path} 失败: {exc}")

            results.append(result)

        return results

    def get_file_stats(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """统计文件目录信息"""
        selected_files = self._collect_files(file_extensions, recursive)

        extension_counts: Dict[str, int] = {}
        total_content_length = 0
        directories = set()

        for relative_path in selected_files:
            ext = Path(relative_path).suffix.lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

            dirpath = os.path.dirname(relative_path)
            if dirpath:
                directories.add(dirpath)

            if ext in self.file_reader.supported_extensions:
                try:
                    _, content = self.file_reader.read_file(relative_path)
                    total_content_length += len(content)
                except Exception:
                    pass

        total_files = len(selected_files)
        avg_length = total_content_length / total_files if total_files else 0

        return {
            "total_files": total_files,
            "extension_counts": extension_counts,
            "total_content_length": total_content_length,
            "average_file_length": avg_length,
            "directories": list(directories),
            "directory_count": len(directories),
        }

    def get_extension_type(self, extension: str) -> str:
        """获取扩展名对应的文档类型"""
        extension_types = {
            ".txt": "文本文件",
            ".pdf": "PDF文档",
            ".md": "Markdown文档",
            ".doc": "Word文档",
            ".docx": "Word文档",
            ".csv": "CSV数据文件",
            ".json": "JSON数据文件",
            ".yaml": "YAML配置文件",
            ".yml": "YAML配置文件",
            ".png": "图片",
            ".jpg": "图片",
            ".jpeg": "图片",
            ".gif": "图片",
            ".bmp": "图片",
            ".tiff": "图片",
            ".tif": "图片",
            ".webp": "图片",
            ".jp2": "图片",
        }
        return extension_types.get(extension.lower(), "未知类型")

    # ====== 内部逻辑 ======
    def _collect_files(self, file_extensions: Optional[List[str]], recursive: bool) -> List[str]:
        files = self.file_reader.list_all_files(recursive=recursive)
        selected: List[str] = []

        if file_extensions:
            target_exts = {ext.lower() for ext in file_extensions}
        else:
            target_exts = None

        for relative_path in files:
            ext = Path(relative_path).suffix.lower()
            if target_exts is not None and ext not in target_exts:
                continue

            if ext in self.file_reader.supported_extensions:
                selected.append(relative_path)
            elif self._should_use_mineru(ext):
                selected.append(relative_path)

        selected.sort()
        return selected

    def _should_use_mineru(self, extension: str) -> bool:
        return (
            self.mode == "mineru"
            and extension in self.MINERU_SUPPORTED_SUFFIXES
            and self.mineru_client is not None
            and self._mineru_options is not None
        )

    def _process_file_legacy(self, relative_path: str, extension: str) -> Dict[str, Any]:
        """使用传统读取 + 分块"""
        result: Dict[str, Any] = {
            "filepath": relative_path,
            "filename": Path(relative_path).name,
            "extension": extension,
            "processing_mode": "legacy",
            "content": "",
            "content_length": 0,
            "chunks": None,
        }

        try:
            _, content = self.file_reader.read_file(relative_path)
            content = content or ""
            result["content"] = content
            result["content_length"] = len(content)

            chunks = self.chunker.chunk_text(content) if content else []
            result["chunks"] = chunks
            result["chunk_count"] = len(chunks)

            chunk_texts = ["".join(chunk) for chunk in chunks]
            chunk_lengths = [len(text) for text in chunk_texts]
            result["chunk_lengths"] = chunk_lengths
            result["average_chunk_length"] = (
                sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            )

            self._finalize_modal_structure(
                result,
                content,
                segments=None,
                chunk_texts=chunk_texts,
                source="legacy",
            )
        except Exception as exc:
            result["chunk_error"] = str(exc)
            print(f"分块错误 ({relative_path}): {exc}")

        return result

    def _process_file_with_mineru(
        self,
        relative_path: str,
        extension: str,
    ) -> Dict[str, Any]:
        """使用 MinerU 进行高级解析"""
        assert self.mineru_client is not None and self._mineru_options is not None

        full_path = Path(self.directory_path) / relative_path
        result: Dict[str, Any] = {
            "filepath": relative_path,
            "filename": Path(relative_path).name,
            "extension": extension,
            "processing_mode": "mineru",
        }

        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")

        try:
            parse_result = self.mineru_client.parse_file(
                full_path,
                backend=self._mineru_options.backend,
                parse_method=self._mineru_options.parse_method,
                lang=self._mineru_options.lang,
                formula_enable=self._mineru_options.formula_enable,
                table_enable=self._mineru_options.table_enable,
                start_page=self._mineru_options.start_page,
                end_page=self._mineru_options.end_page,
                output_format=self._mineru_options.output_format,
            )
        except Exception as exc:
            print(f"MinerU 解析失败 ({relative_path}): {exc}，退回 legacy。")
            fallback = self._process_file_legacy(relative_path, extension)
            fallback["processing_mode"] = "legacy_fallback"
            fallback["mineru_error"] = str(exc)
            return fallback

        if not parse_result.is_success():
            message = parse_result.error or "MinerU 返回失败"
            print(f"MinerU 解析失败 ({relative_path}): {message}，退回 legacy。")
            fallback = self._process_file_legacy(relative_path, extension)
            fallback["processing_mode"] = "legacy_fallback"
            fallback["mineru_error"] = message
            fallback["mineru_result"] = asdict(parse_result)
            return fallback

        rich_text, segments = self._build_modal_segments(parse_result)
        rich_text = rich_text or ""
        result["content"] = rich_text
        result["content_length"] = len(rich_text)

        chunks = self.chunker.chunk_text(rich_text) if rich_text else []
        result["chunks"] = chunks
        result["chunk_count"] = len(chunks)
        chunk_texts = ["".join(chunk) for chunk in chunks]
        chunk_lengths = [len(text) for text in chunk_texts]
        result["chunk_lengths"] = chunk_lengths
        result["average_chunk_length"] = (
            sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        )

        self._finalize_modal_structure(
            result,
            rich_text,
            segments,
            chunk_texts,
            source="mineru",
        )

        result["mineru_result"] = asdict(parse_result)
        result["original_filename"] = parse_result.original_filename or parse_result.filename
        result["mineru_task_id"] = parse_result.task_id
        result["mineru_output_dir"] = parse_result.output_dir
        result["markdown_path"] = parse_result.markdown_path
        result["content_list_path"] = parse_result.content_list_path

        return result

    def _build_modal_segments(self, parse_result: ParseResult) -> Tuple[str, List[Dict[str, Any]]]:
        """根据 MinerU 解析结果生成富文本段落"""
        segments: List[Dict[str, Any]] = []
        method_dir = Path(parse_result.output_dir) if parse_result.output_dir else None

        if parse_result.content_list_path and Path(parse_result.content_list_path).exists():
            try:
                with open(parse_result.content_list_path, "r", encoding="utf-8") as fp:
                    content_items = json.load(fp)
                for item in content_items:
                    segment = self._normalize_mineru_item(item, method_dir)
                    if segment.get("text"):
                        segments.append(segment)
            except Exception as exc:
                print(f"解析 content_list 失败 ({parse_result.content_list_path}): {exc}")

        if not segments and parse_result.markdown_path and Path(parse_result.markdown_path).exists():
            try:
                markdown_text = Path(parse_result.markdown_path).read_text(encoding="utf-8")
                return markdown_text, [{"type": "markdown", "text": markdown_text}]
            except Exception as exc:
                print(f"读取 MinerU Markdown 失败 ({parse_result.markdown_path}): {exc}")

        rich_text = "\n\n".join(
            segment["text"].strip() for segment in segments if segment.get("text")
        )
        return rich_text, segments

    def _normalize_mineru_item(
        self,
        item: Dict[str, Any],
        method_dir: Optional[Path],
    ) -> Dict[str, Any]:
        """将 MinerU content_list 的元素转为标准段落"""
        segment: Dict[str, Any] = {
            "type": item.get("type", "unknown"),
            "raw": item,
        }
        segment["source"] = "mineru"
        segment["page_idx"] = item.get("page_idx")
        segment["bbox"] = item.get("bbox")

        item_type = segment["type"].lower()
        if item_type == "text":
            raw_text = item.get("text") or ""
            segment["text"] = raw_text.strip()
            segment["text_raw"] = raw_text
        elif item_type == "equation":
            latex = item.get("text") or ""
            segment["text"] = f"[公式]\n{latex.strip()}"
            segment["latex"] = latex.strip()
            if item.get("text_format"):
                segment["text_format"] = item.get("text_format")
        elif item_type == "table":
            caption = " ".join(item.get("table_caption", [])).strip()
            table_body = item.get("table_body") or ""
            parts = ["[表格]"]
            if caption:
                parts.append(caption)
            if table_body:
                parts.append(table_body)
            segment["text"] = "\n".join(parts).strip()
            segment["table_html"] = table_body
            segment["table_caption"] = caption
            segment["table_footnote"] = item.get("table_footnote") or []
        elif item_type == "image":
            placeholder = "[图片]"
            caption = " ".join(item.get("image_caption", [])).strip()
            image_rel_path: Optional[str] = None
            image_abs_path: Optional[str] = None

            img_path = item.get("img_path")
            if method_dir and img_path:
                candidate = (method_dir / img_path).resolve()
                if candidate.exists():
                    image_abs_path = candidate
                    try:
                        rel_path = candidate.relative_to(MINERU_OUTPUT_DIR)
                        image_rel_path = rel_path.as_posix()
                    except ValueError:
                        image_rel_path = candidate.as_posix()

            if image_rel_path:
                placeholder = f"[图片:{image_rel_path}]"
                segment["image_relative_path"] = image_rel_path
            if image_abs_path:
                segment["image_path"] = image_abs_path.as_posix()

            parts = [placeholder]
            if caption:
                parts.append(caption)
            segment["text"] = " ".join(parts).strip()
            segment["image_caption"] = item.get("image_caption") or []
            segment["image_footnote"] = item.get("image_footnote") or []
        else:
            text = item.get("text") or item.get("raw_text") or ""
            if text:
                segment["text"] = text.strip()
            else:
                segment["text"] = ""

        return segment

    def _finalize_modal_structure(
        self,
        result: Dict[str, Any],
        rich_text: Optional[str],
        segments: Optional[List[Dict[str, Any]]],
        chunk_texts: Optional[List[str]],
        source: str,
    ) -> None:
        """统一整理分段与分块的多模态元数据"""
        rich_text = rich_text or ""
        chunk_texts = chunk_texts or []

        segment_candidates = segments or []
        if not segment_candidates and chunk_texts:
            segment_candidates = self._build_segments_from_chunks(chunk_texts, source)

        prepared_segments = self._prepare_segments(
            result.get("filepath") or result.get("filename") or "",
            segment_candidates,
            rich_text,
            default_source=source,
        )
        segment_map = {segment["segment_id"]: segment for segment in prepared_segments}
        chunk_annotations = self._build_chunk_annotations(chunk_texts, prepared_segments, segment_map)

        result["modal_segments"] = prepared_segments
        result["segment_count"] = len(prepared_segments)
        result["chunk_texts"] = chunk_texts
        result["chunk_annotations"] = chunk_annotations
        result["chunk_modal_types"] = [item.get("segment_types", []) for item in chunk_annotations]
        result["chunk_modal_segment_ids"] = [item.get("segment_ids", []) for item in chunk_annotations]

        result["image_assets"] = [
            segment.get("image_relative_path")
            for segment in prepared_segments
            if segment.get("image_relative_path")
        ]
        result["image_segment_count"] = len(result["image_assets"])
        result["equation_segment_count"] = sum(
            1 for segment in prepared_segments if segment.get("type") == "equation"
        )
        result["table_segment_count"] = sum(
            1 for segment in prepared_segments if segment.get("type") == "table"
        )

    def _prepare_segments(
        self,
        filename: str,
        segments: List[Dict[str, Any]],
        rich_text: str,
        default_source: str,
    ) -> List[Dict[str, Any]]:
        """为段落补全ID、顺序、字符位置信息"""
        prepared: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        doc_key = filename or "document"

        for index, raw_segment in enumerate(segments):
            segment = dict(raw_segment)
            segment_type = (segment.get("type") or "text").lower()
            text_value = segment.get("text") or ""

            segment["type"] = segment_type
            segment["text"] = text_value
            segment["segment_index"] = index
            segment["source"] = segment.get("source") or default_source
            if segment.get("page_index") is not None and segment.get("page_idx") is None:
                segment["page_idx"] = segment["page_index"]

            segment_id = segment.get("segment_id")
            if not segment_id:
                seed = f"{doc_key}|{index}|{segment_type}|{text_value[:64]}"
                segment_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()

            candidate_id = segment_id
            duplicate_counter = 1
            while candidate_id in seen_ids:
                duplicate_counter += 1
                candidate_id = f"{segment_id}_{duplicate_counter}"

            segment["segment_id"] = candidate_id
            seen_ids.add(candidate_id)
            segment["char_length"] = len(text_value)

            prepared.append(segment)

        self._compute_segment_positions(prepared, rich_text)
        return prepared

    def _compute_segment_positions(self, segments: List[Dict[str, Any]], rich_text: str) -> None:
        """估算段落在全文中的字符偏移"""
        if not rich_text:
            for segment in segments:
                segment["char_start"] = None
                segment["char_end"] = None
            return

        current_pos = 0
        for segment in segments:
            text_value = (segment.get("text") or "").strip()
            if not text_value:
                segment["char_start"] = None
                segment["char_end"] = None
                continue

            position = rich_text.find(text_value, current_pos)
            if position == -1:
                position = rich_text.find(text_value)
            if position == -1:
                segment["char_start"] = None
                segment["char_end"] = None
                continue

            segment["char_start"] = position
            segment["char_end"] = position + len(text_value)
            current_pos = segment["char_end"]

    @staticmethod
    def _normalize_alignment_text(text: str) -> str:
        """对齐映射时去除空白字符"""
        if not text:
            return ""
        return re.sub(r"\s+", "", text)

    def _build_chunk_annotations(
        self,
        chunk_texts: List[str],
        segments: List[Dict[str, Any]],
        segment_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """根据分块长度映射段落，生成多模态引用信息"""
        if not chunk_texts:
            return []

        normalized_segments: List[Dict[str, Any]] = []
        for segment in segments:
            normalized_text = self._normalize_alignment_text(segment.get("text") or "")
            normalized_segments.append(
                {
                    "segment_id": segment["segment_id"],
                    "length": len(normalized_text),
                    "type": segment.get("type"),
                    "source": segment.get("source"),
                }
            )

        annotations: List[Dict[str, Any]] = []
        segment_index = 0
        segment_consumed = 0

        for chunk_text in chunk_texts:
            normalized_chunk = self._normalize_alignment_text(chunk_text or "")
            chunk_length = len(normalized_chunk)
            if chunk_length <= 0:
                annotations.append(
                    {"segment_ids": [], "segment_types": [], "segment_sources": [], "char_start": None, "char_end": None}
                )
                continue

            remaining = chunk_length
            chunk_segments: List[str] = []
            chunk_types: List[str] = []
            chunk_sources: List[str] = []

            while remaining > 0 and segment_index < len(normalized_segments):
                current_segment = normalized_segments[segment_index]
                segment_length = current_segment["length"] or 1
                available = segment_length - segment_consumed

                if available <= 0:
                    segment_index += 1
                    segment_consumed = 0
                    continue

                if current_segment["segment_id"] not in chunk_segments:
                    chunk_segments.append(current_segment["segment_id"])
                    chunk_types.append(current_segment["type"])
                    chunk_sources.append(current_segment["source"])

                consumed = min(available, remaining)
                remaining -= consumed
                segment_consumed += consumed

                if segment_consumed >= segment_length:
                    segment_index += 1
                    segment_consumed = 0

            annotations.append(
                {
                    "segment_ids": chunk_segments,
                    "segment_types": chunk_types,
                    "segment_sources": chunk_sources,
                }
            )

        # 如果还有剩余的段落，将其附加到最后一个块
        if segment_index < len(normalized_segments) and annotations:
            remaining_segments = normalized_segments[segment_index:]
            last_annotation = annotations[-1]
            for segment in remaining_segments:
                if segment["segment_id"] not in last_annotation["segment_ids"]:
                    last_annotation["segment_ids"].append(segment["segment_id"])
                    last_annotation["segment_types"].append(segment["type"])
                    last_annotation["segment_sources"].append(segment["source"])

        # 基于段落偏移补全块的字符范围
        for annotation in annotations:
            if not annotation["segment_ids"]:
                annotation["char_start"] = None
                annotation["char_end"] = None
                continue

            starts = [
                segment_map[segment_id].get("char_start")
                for segment_id in annotation["segment_ids"]
                if segment_id in segment_map and segment_map[segment_id].get("char_start") is not None
            ]
            ends = [
                segment_map[segment_id].get("char_end")
                for segment_id in annotation["segment_ids"]
                if segment_id in segment_map and segment_map[segment_id].get("char_end") is not None
            ]

            annotation["char_start"] = min(starts) if starts else None
            annotation["char_end"] = max(ends) if ends else None

        return annotations

    def _build_segments_from_chunks(self, chunk_texts: List[str], source: str) -> List[Dict[str, Any]]:
        """在缺乏 MinerU 解析时，以块为单位提供基本段落结构"""
        segments: List[Dict[str, Any]] = []
        for index, chunk_text in enumerate(chunk_texts):
            text_value = chunk_text or ""
            segments.append(
                {
                    "type": "text",
                    "text": text_value,
                    "source": source,
                    "chunk_index": index,
                }
            )
        return segments


if __name__ == "__main__":
    processor = DocumentProcessor(FILES_DIR)
    print(f"目录 {FILES_DIR} 及其子目录中的所有文件:")
    for filepath in processor.file_reader.list_all_files(recursive=True):
        print(f"  {filepath}")

    stats = processor.get_file_stats(recursive=True)
    print("目录文件统计:")
    print(f"总文件数: {stats['total_files']}")
    print(f"子目录数: {stats['directory_count']}")
    if stats["directory_count"] > 0:
        print("子目录列表:")
        for directory in stats["directories"]:
            print(f"  {directory}")

    print("文件类型分布:")
    for ext, count in stats["extension_counts"].items():
        print(f"  {ext} ({processor.get_extension_type(ext)}): {count} 文件")
    print(f"总文本长度: {stats['total_content_length']} 字符")
    print(f"平均文件长度: {stats['average_file_length']:.2f} 字符")

    print("\n开始处理所有文件...")
    results = processor.process_directory(recursive=True)

    for result in results:
        print(f"\n文件: {result.get('filepath')}")
        print(f"类型: {processor.get_extension_type(result.get('extension', ''))}")
        print(f"处理模式: {result.get('processing_mode')}")
        print(f"内容长度: {result.get('content_length', 0)} 字符")
        if result.get("chunks"):
            print(f"分块数量: {result.get('chunk_count', 0)}")
            print(f"平均分块长度: {result.get('average_chunk_length', 0):.2f} 字符")
        else:
            print(f"分块失败: {result.get('chunk_error', '未知错误')}")
