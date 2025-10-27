#!/usr/bin/env python3
"""
MinerU API Server for GraphRAG Agent
为 graphrag-agent 项目提供 MinerU 多模态文档解析服务

功能:
- PDF/图像文档解析
- 公式识别（LaTeX格式）
- 表格提取（HTML结构）
- 图像提取
- 多模态 Markdown 输出
- 批量处理支持
- 异步处理
"""

import os
import sys
import json
import asyncio
import shutil
import uuid
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# 添加 MinerU 路径（默认使用当前仓库同级目录下的 MinerU 项目）
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MINERU_PATH = REPO_ROOT.parent / "MinerU"
MINERU_PATH = Path(os.getenv("MINERU_HOME", DEFAULT_MINERU_PATH))
if not MINERU_PATH.is_absolute():
    MINERU_PATH = (REPO_ROOT / MINERU_PATH).resolve()
if not MINERU_PATH.exists():
    raise RuntimeError(f"MinerU 路径不存在: {MINERU_PATH}. 请设置 MINERU_HOME 环境变量。")
if str(MINERU_PATH) not in sys.path:
    sys.path.insert(0, str(MINERU_PATH))

from mineru.cli.common import do_parse, read_fn, aio_do_parse
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram
from mineru.utils.enum_class import MakeMode


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def secure_filename(filename: str) -> tuple[str, str]:
    """
    生成安全的文件名，清理潜在的路径遍历与非法字符

    Args:
        filename: 原始文件名

    Returns:
        (safe_name, safe_stem)
    """
    name = Path(filename).name  # 去除路径部分
    stem, suffix = os.path.splitext(name)

    safe_stem = SAFE_FILENAME_PATTERN.sub("_", stem).strip("._")
    if not safe_stem:
        safe_stem = f"file_{uuid.uuid4().hex}"
    safe_stem = safe_stem[:120]

    raw_suffix = SAFE_FILENAME_PATTERN.sub("", suffix)
    if raw_suffix and not raw_suffix.startswith("."):
        raw_suffix = f".{raw_suffix}"
    safe_suffix = raw_suffix[:16]

    safe_name = f"{safe_stem}{safe_suffix}"
    return safe_name, safe_stem


# ============== 配置管理 ==============
class MinerUConfig:
    """MinerU 服务配置"""

    # 服务配置
    HOST = os.getenv("MINERU_HOST", "0.0.0.0")
    PORT = int(os.getenv("MINERU_PORT", "8899"))
    WORKERS = int(os.getenv("MINERU_WORKERS", "1"))

    # 输出目录
    _output_raw = os.getenv("MINERU_OUTPUT_DIR", "./mineru_outputs")
    OUTPUT_DIR = Path(_output_raw)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = (REPO_ROOT / OUTPUT_DIR).resolve()

    _temp_raw = os.getenv("MINERU_TEMP_DIR", "./.tmp/mineru_temp")
    TEMP_DIR = Path(_temp_raw)
    if not TEMP_DIR.is_absolute():
        TEMP_DIR = (REPO_ROOT / TEMP_DIR).resolve()

    # 处理配置
    DEFAULT_BACKEND = os.getenv("MINERU_DEFAULT_BACKEND", "pipeline")  # pipeline or vlm-*
    DEFAULT_PARSE_METHOD = os.getenv("MINERU_DEFAULT_PARSE_METHOD", "auto")  # auto, ocr, txt
    DEFAULT_LANG = os.getenv("MINERU_DEFAULT_LANG", "ch")  # ch, en

    # 功能开关
    FORMULA_ENABLE = os.getenv("MINERU_FORMULA_ENABLE", "true").lower() == "true"
    TABLE_ENABLE = os.getenv("MINERU_TABLE_ENABLE", "true").lower() == "true"

    # 输出开关
    DUMP_MD = os.getenv("MINERU_DUMP_MD", "true").lower() == "true"
    DUMP_CONTENT_LIST = os.getenv("MINERU_DUMP_CONTENT_LIST", "true").lower() == "true"
    DUMP_MIDDLE_JSON = os.getenv("MINERU_DUMP_MIDDLE_JSON", "false").lower() == "true"
    DUMP_MODEL_OUTPUT = os.getenv("MINERU_DUMP_MODEL_OUTPUT", "false").lower() == "true"

    # 设备配置
    DEVICE_MODE = os.getenv("MINERU_DEVICE_MODE", "auto")  # auto, cuda, cpu

    # 保留时间（小时）
    OUTPUT_RETENTION_HOURS = int(os.getenv("MINERU_OUTPUT_RETENTION_HOURS", "24"))

    @classmethod
    def setup_environment(cls):
        """设置 MinerU 环境变量"""
        # 设置设备模式
        if os.getenv('MINERU_DEVICE_MODE') is None:
            device = cls.DEVICE_MODE if cls.DEVICE_MODE != 'auto' else get_device()
            os.environ['MINERU_DEVICE_MODE'] = device

        device_mode = os.environ['MINERU_DEVICE_MODE']
        logger.info(f"Using device mode: {device_mode}")

        # 设置 VRAM
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE') is None:
            if device_mode.startswith("cuda") or device_mode.startswith("npu"):
                try:
                    vram = round(get_vram(device_mode))
                    os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram)
                except Exception as e:
                    logger.warning(f"Failed to get VRAM: {e}, using default 8GB")
                    os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '8'
            else:
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '1'

        logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}GB")

        # 创建必要的目录
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============== 数据模型 ==============
class ParseOptions(BaseModel):
    """文档解析选项"""

    backend: str = Field(default="pipeline", description="解析后端: pipeline 或 vlm-*")
    parse_method: str = Field(default="auto", description="解析方法: auto, ocr, txt")
    lang: str = Field(default="ch", description="文档语言: ch, en, 或其他")
    formula_enable: bool = Field(default=True, description="是否识别公式")
    table_enable: bool = Field(default=True, description="是否识别表格")
    start_page: int = Field(default=0, description="起始页码（0-based）")
    end_page: Optional[int] = Field(default=None, description="结束页码（inclusive），None表示到最后")
    output_format: str = Field(default="mm_md", description="输出格式: mm_md, md_only, content_list")


class ParseResult(BaseModel):
    """解析结果"""

    task_id: str = Field(description="任务ID")
    filename: str = Field(description="内部处理使用的安全文件名")
    original_filename: str = Field(description="原始文件名")
    status: str = Field(description="状态: success, failed")
    output_dir: Optional[str] = Field(default=None, description="输出目录路径")
    markdown_path: Optional[str] = Field(default=None, description="Markdown 文件路径")
    content_list_path: Optional[str] = Field(default=None, description="内容列表 JSON 路径")
    images_dir: Optional[str] = Field(default=None, description="图像目录路径")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    processing_time: Optional[float] = Field(default=None, description="处理时间（秒）")


class BatchParseRequest(BaseModel):
    """批量解析请求"""

    options: ParseOptions = Field(default_factory=ParseOptions)


class BatchParseResult(BaseModel):
    """批量解析结果"""

    total: int = Field(description="总文件数")
    success: int = Field(description="成功数量")
    failed: int = Field(description="失败数量")
    results: List[ParseResult] = Field(description="各文件的解析结果")


# ============== 应用生命周期管理 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting MinerU API Server...")
    MinerUConfig.setup_environment()
    logger.info(f"Output directory: {MinerUConfig.OUTPUT_DIR}")
    logger.info(f"Temp directory: {MinerUConfig.TEMP_DIR}")

    # 启动后台清理任务
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    # 关闭时清理
    logger.info("Shutting down MinerU API Server...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# ============== FastAPI 应用 ==============
app = FastAPI(
    title="MinerU API Server",
    description="多模态文档解析服务，支持 PDF、图像的公式、表格、文本提取",
    version="1.0.0",
    lifespan=lifespan
)


# ============== 核心解析逻辑 ==============
async def parse_document_async(
    file_path: Path,
    normalized_name: str,
    options: ParseOptions,
    task_id: str,
    original_filename: Optional[str] = None
) -> ParseResult:
    """
    异步解析单个文档

    Args:
        file_path: 文件路径
        normalized_name: 安全的文件名（不含扩展名）
        options: 解析选项
        task_id: 任务ID
        original_filename: 原始文件名

    Returns:
        ParseResult: 解析结果
    """
    start_time = datetime.now()
    display_name = original_filename or file_path.name

    try:
        # 读取文件字节
        pdf_bytes = read_fn(file_path)

        # 确定输出格式
        if options.output_format == "mm_md":
            make_mode = MakeMode.MM_MD
        elif options.output_format == "md_only":
            make_mode = MakeMode.MD
        elif options.output_format == "content_list":
            make_mode = MakeMode.CONTENT_LIST
        else:
            make_mode = MakeMode.MM_MD

        # 设置输出目录
        output_dir = MinerUConfig.OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 判断是否使用异步
        backend = options.backend
        if backend.startswith("vlm-vllm-async-engine"):
            # 异步 VLM
            await aio_do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[normalized_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[options.lang],
                backend=backend,
                parse_method=options.parse_method,
                formula_enable=options.formula_enable,
                table_enable=options.table_enable,
                f_dump_md=MinerUConfig.DUMP_MD,
                f_dump_content_list=MinerUConfig.DUMP_CONTENT_LIST,
                f_dump_middle_json=MinerUConfig.DUMP_MIDDLE_JSON,
                f_dump_model_output=MinerUConfig.DUMP_MODEL_OUTPUT,
                f_dump_orig_pdf=False,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_make_md_mode=make_mode,
                start_page_id=options.start_page,
                end_page_id=options.end_page
            )
        else:
            # 同步处理（在线程池中执行以避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: do_parse(
                    output_dir=str(output_dir),
                    pdf_file_names=[normalized_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=[options.lang],
                    backend=backend,
                    parse_method=options.parse_method,
                    formula_enable=options.formula_enable,
                    table_enable=options.table_enable,
                    f_dump_md=MinerUConfig.DUMP_MD,
                    f_dump_content_list=MinerUConfig.DUMP_CONTENT_LIST,
                    f_dump_middle_json=MinerUConfig.DUMP_MIDDLE_JSON,
                    f_dump_model_output=MinerUConfig.DUMP_MODEL_OUTPUT,
                    f_dump_orig_pdf=False,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_make_md_mode=make_mode,
                    start_page_id=options.start_page,
                    end_page_id=options.end_page
                )
            )

        # 检查输出文件
        doc_root = output_dir / normalized_name
        method_dir: Optional[Path] = None

        if doc_root.exists():
            # VLM 输出目录固定为 vlm
            if backend.startswith("vlm") and (doc_root / "vlm").exists():
                method_dir = doc_root / "vlm"
            else:
                expected_dir = doc_root / options.parse_method
                if expected_dir.exists():
                    method_dir = expected_dir

            if method_dir is None:
                # 兜底：取第一个子目录
                for candidate in doc_root.iterdir():
                    if candidate.is_dir():
                        method_dir = candidate
                        break

        if method_dir is None:
            raise RuntimeError(f"未找到解析输出目录: {doc_root}")

        markdown_path = method_dir / f"{normalized_name}.md" if MinerUConfig.DUMP_MD else None
        content_list_path = method_dir / f"{normalized_name}_content_list.json" if MinerUConfig.DUMP_CONTENT_LIST else None
        images_dir = method_dir / "images"
        if not images_dir.exists():
            images_dir = None

        # 提取元数据
        metadata = {
            "task_id": task_id,
            "original_filename": display_name,
            "normalized_filename": normalized_name,
            "parse_method": options.parse_method,
            "backend": options.backend,
            "lang": options.lang,
            "formula_enabled": options.formula_enable,
            "table_enabled": options.table_enable,
            "page_range": f"{options.start_page}-{options.end_page if options.end_page else 'end'}",
            "output_dir": str(method_dir)
        }

        # 如果有 content_list，读取并添加统计信息
        if content_list_path and content_list_path.exists():
            try:
                with open(content_list_path, 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
                    metadata["content_stats"] = {
                        "total_items": len(content_list),
                        "item_types": {}
                    }
                    # 统计各类型元素数量
                    for item in content_list:
                        item_type = item.get("type", "unknown")
                        metadata["content_stats"]["item_types"][item_type] = \
                            metadata["content_stats"]["item_types"].get(item_type, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to parse content_list: {e}")

        processing_time = (datetime.now() - start_time).total_seconds()

        return ParseResult(
            task_id=task_id,
            filename=normalized_name,
            original_filename=display_name,
            status="success",
            output_dir=str(method_dir) if method_dir.exists() else None,
            markdown_path=str(markdown_path) if markdown_path and markdown_path.exists() else None,
            content_list_path=str(content_list_path) if content_list_path and content_list_path.exists() else None,
            images_dir=str(images_dir) if images_dir else None,
            metadata=metadata,
            processing_time=processing_time
        )

    except Exception as e:
        logger.exception(f"Failed to parse document {normalized_name}")
        processing_time = (datetime.now() - start_time).total_seconds()

        return ParseResult(
            task_id=task_id,
            filename=normalized_name,
            original_filename=display_name,
            status="failed",
            error=str(e),
            processing_time=processing_time
        )


# ============== API 端点 ==============
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "MinerU API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "parse": "/parse",
            "parse_batch": "/parse/batch",
            "download": "/download/{task_id}/{filename}",
            "health": "/health",
            "config": "/config"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "device_mode": os.environ.get('MINERU_DEVICE_MODE'),
        "vram_size": os.environ.get('MINERU_VIRTUAL_VRAM_SIZE'),
        "output_dir": str(MinerUConfig.OUTPUT_DIR),
        "temp_dir": str(MinerUConfig.TEMP_DIR)
    }


@app.get("/config")
async def get_config():
    """获取配置信息"""
    return {
        "backend": MinerUConfig.DEFAULT_BACKEND,
        "parse_method": MinerUConfig.DEFAULT_PARSE_METHOD,
        "lang": MinerUConfig.DEFAULT_LANG,
        "formula_enable": MinerUConfig.FORMULA_ENABLE,
        "table_enable": MinerUConfig.TABLE_ENABLE,
        "device_mode": os.environ.get('MINERU_DEVICE_MODE'),
        "output_retention_hours": MinerUConfig.OUTPUT_RETENTION_HOURS
    }


@app.post("/parse", response_model=ParseResult)
async def parse_single_file(
    file: UploadFile = File(..., description="要解析的文件（PDF或图像）"),
    backend: str = Form(default=MinerUConfig.DEFAULT_BACKEND),
    parse_method: str = Form(default=MinerUConfig.DEFAULT_PARSE_METHOD),
    lang: str = Form(default=MinerUConfig.DEFAULT_LANG),
    formula_enable: bool = Form(default=MinerUConfig.FORMULA_ENABLE),
    table_enable: bool = Form(default=MinerUConfig.TABLE_ENABLE),
    start_page: int = Form(default=0),
    end_page: Optional[int] = Form(default=None),
    output_format: str = Form(default="mm_md")
):
    """
    解析单个文档

    支持格式: PDF, PNG, JPEG, JP2, WEBP, GIF, BMP, JPG, TIFF
    """
    # 生成任务ID
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    safe_name, safe_stem = secure_filename(file.filename)
    # 保存上传文件到临时目录
    temp_file = MinerUConfig.TEMP_DIR / task_id / safe_name
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 保存文件
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 构建解析选项
        options = ParseOptions(
            backend=backend,
            parse_method=parse_method,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page=start_page,
            end_page=end_page,
            output_format=output_format
        )

        # 解析文档
        result = await parse_document_async(
            temp_file,
            safe_stem,
            options,
            task_id,
            original_filename=file.filename
        )

        return result

    except Exception as e:
        logger.exception(f"Error processing file {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        try:
            temp_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")


@app.post("/parse/batch", response_model=BatchParseResult)
async def parse_batch_files(
    files: List[UploadFile] = File(..., description="要解析的文件列表"),
    backend: str = Form(default=MinerUConfig.DEFAULT_BACKEND),
    parse_method: str = Form(default=MinerUConfig.DEFAULT_PARSE_METHOD),
    lang: str = Form(default=MinerUConfig.DEFAULT_LANG),
    formula_enable: bool = Form(default=MinerUConfig.FORMULA_ENABLE),
    table_enable: bool = Form(default=MinerUConfig.TABLE_ENABLE),
    start_page: int = Form(default=0),
    end_page: Optional[int] = Form(default=None),
    output_format: str = Form(default="mm_md")
):
    """
    批量解析文档

    支持同时上传多个文件进行批量处理
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # 生成批次任务ID
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 构建解析选项
    options = ParseOptions(
        backend=backend,
        parse_method=parse_method,
        lang=lang,
        formula_enable=formula_enable,
        table_enable=table_enable,
        start_page=start_page,
        end_page=end_page,
        output_format=output_format
    )

    results = []
    temp_files = []

    try:
        # 保存所有上传文件
        for idx, file in enumerate(files):
            task_id = f"{batch_id}_{idx}"
            safe_name, safe_stem = secure_filename(file.filename)
            temp_file = MinerUConfig.TEMP_DIR / task_id / safe_name
            temp_file.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            temp_files.append((temp_file, safe_stem, file.filename, task_id))

        # 并行处理所有文件
        tasks = [
            parse_document_async(
                temp_file,
                safe_stem,
                options,
                task_id,
                original_filename=original_name
            )
            for temp_file, safe_stem, original_name, task_id in temp_files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        # 统计结果
        success_count = sum(1 for r in results if r.status == "success")
        failed_count = len(results) - success_count

        return BatchParseResult(
            total=len(results),
            success=success_count,
            failed=failed_count,
            results=results
        )

    except Exception as e:
        logger.exception("Error in batch processing")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        for temp_file, *_ in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """
    下载处理后的文件

    Args:
        task_id: 任务ID
        filename: 文件名（如 document.md, document_content_list.json）
    """
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    safe_name, _ = secure_filename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 查找文件
    output_base = MinerUConfig.OUTPUT_DIR / task_id

    if not output_base.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # 递归查找文件
    for root, dirs, files in os.walk(output_base):
        if filename in files:
            file_path = Path(root) / filename
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type="application/octet-stream"
            )

    raise HTTPException(status_code=404, detail=f"File {filename} not found in task {task_id}")


@app.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """
    清理指定任务的输出文件

    Args:
        task_id: 任务ID
    """
    output_dir = MinerUConfig.OUTPUT_DIR / task_id

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        shutil.rmtree(output_dir)
        return {"status": "success", "message": f"Task {task_id} cleaned up"}
    except Exception as e:
        logger.exception(f"Failed to cleanup task {task_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup/old")
async def cleanup_old_outputs(hours: int = MinerUConfig.OUTPUT_RETENTION_HOURS):
    """
    清理超过指定时间的输出文件

    Args:
        hours: 保留时间（小时）
    """
    cleaned_count = 0
    current_time = datetime.now()

    try:
        for task_dir in MinerUConfig.OUTPUT_DIR.iterdir():
            if task_dir.is_dir():
                # 检查目录修改时间
                mtime = datetime.fromtimestamp(task_dir.stat().st_mtime)
                age_hours = (current_time - mtime).total_seconds() / 3600

                if age_hours > hours:
                    shutil.rmtree(task_dir)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old task: {task_dir.name} (age: {age_hours:.1f}h)")

        return {
            "status": "success",
            "cleaned_count": cleaned_count,
            "retention_hours": hours
        }

    except Exception as e:
        logger.exception("Failed to cleanup old outputs")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 后台任务 ==============
async def periodic_cleanup():
    """定期清理过期文件"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时执行一次

            current_time = datetime.now()
            cleaned_count = 0

            for task_dir in MinerUConfig.OUTPUT_DIR.iterdir():
                if task_dir.is_dir():
                    mtime = datetime.fromtimestamp(task_dir.stat().st_mtime)
                    age_hours = (current_time - mtime).total_seconds() / 3600

                    if age_hours > MinerUConfig.OUTPUT_RETENTION_HOURS:
                        shutil.rmtree(task_dir)
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Periodic cleanup: removed {cleaned_count} old tasks")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Error in periodic cleanup")


# ============== 主函数 ==============
def main():
    """启动服务器"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # 添加文件日志
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "mineru_server_{time}.log",
        rotation="500 MB",
        retention="7 days",
        level="INFO"
    )

    logger.info("=" * 60)
    logger.info("MinerU API Server for GraphRAG Agent")
    logger.info("=" * 60)
    logger.info(f"Host: {MinerUConfig.HOST}")
    logger.info(f"Port: {MinerUConfig.PORT}")
    logger.info(f"Workers: {MinerUConfig.WORKERS}")
    logger.info(f"Output Directory: {MinerUConfig.OUTPUT_DIR}")
    logger.info("=" * 60)

    # 启动服务器
    uvicorn.run(
        "mineru_server:app",
        host=MinerUConfig.HOST,
        port=MinerUConfig.PORT,
        workers=MinerUConfig.WORKERS,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
