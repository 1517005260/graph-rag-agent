"""
图谱构建相关 API 路由

提供：
- WebSocket 进度实时推送
- SSE (Server-Sent Events) 进度推送
- 触发构建任务的 HTTP 端点
- 构建状态查询
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils.progress_broadcaster import get_broadcaster
from utils.progress_manager import get_progress_manager
from graphrag_agent.config.settings import FILES_DIR

_LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/build", tags=["build"])


# ========== WebSocket 端点 ==========


@router.websocket("/ws/progress")
async def websocket_progress_endpoint(websocket: WebSocket):
    """
    WebSocket 端点：实时接收图谱构建进度

    前端连接此端点后，将实时收到以下类型的消息：
    - type: "connected" - 连接成功
    - type: "log" - 日志消息
    - type: "progress" - 进度更新（包含 stage, percent, details）
    - type: "status" - 状态更新（started, running, completed, failed）
    - type: "file_status" - 单个文件处理状态
    - type: "stats" - 统计信息（实体数、关系数等）
    - type: "error" - 错误消息

    Example:
        ```javascript
        const ws = new WebSocket("ws://localhost:8000/build/ws/progress");
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.type, data);
        };
        ```
    """
    broadcaster = get_broadcaster()
    await broadcaster.connect(websocket)

    try:
        # 保持连接，监听客户端消息（如果需要支持暂停/停止）
        while True:
            data = await websocket.receive_text()
            # 可以在这里处理客户端发来的控制命令
            _LOGGER.debug(f"收到客户端消息: {data}")

    except WebSocketDisconnect:
        _LOGGER.info("客户端主动断开 WebSocket 连接")
        await broadcaster.disconnect(websocket)
    except Exception as exc:
        _LOGGER.exception(f"WebSocket 连接异常: {exc}")
        await broadcaster.disconnect(websocket)


# ========== SSE 端点 ==========


async def sse_event_generator():
    """
    SSE 事件生成器

    将 ProgressManager 的状态更新转换为 SSE 格式。
    """
    progress_mgr = get_progress_manager()

    async for event_data in progress_mgr.event_generator():
        # SSE 格式：event: <type>\ndata: <json>\n\n
        event_type = event_data.get("event", "message")
        data = event_data.get("data", "{}")

        yield f"event: {event_type}\ndata: {data}\n\n"


@router.get("/sse/progress")
async def sse_progress_stream():
    """
    SSE (Server-Sent Events) 端点：单向推送图谱构建进度

    **特点**：
    - 基于 HTTP，比 WebSocket 更简单
    - 服务器单向推送到客户端
    - 自动重连机制（浏览器内置）
    - 适合只需接收进度的场景

    **消息格式**：
    - event: connected - 连接成功确认
    - event: status - 状态更新（包含完整进度信息）

    **状态数据结构**：
    ```json
    {
        "percent": 45,
        "stage": "entity_extraction",
        "details": "正在提取实体: 45/100",
        "logs": ["[10:00:00] 开始任务", ...],
        "stats": {
            "l0_files": 10,
            "l1_tasks": 8,
            "entities": 1234,
            "relations": 567
        },
        "last_update": "2025-12-13T10:00:00"
    }
    ```

    **前端使用示例**：
    ```javascript
    const eventSource = new EventSource("http://localhost:8000/build/sse/progress");

    eventSource.addEventListener("connected", (event) => {
        const data = JSON.parse(event.data);
        console.log("连接成功:", data.message);
    });

    eventSource.addEventListener("status", (event) => {
        const status = JSON.parse(event.data);
        console.log("进度:", status.percent + "%");
        console.log("阶段:", status.stage);
        console.log("详情:", status.details);
    });

    eventSource.onerror = (error) => {
        console.error("SSE 错误:", error);
    };
    ```

    Returns:
        StreamingResponse: SSE 数据流
    """
    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )


@router.get("/sse/status")
async def get_sse_current_status() -> Dict[str, Any]:
    """
    获取当前构建状态（快照）

    不需要建立 SSE 连接，直接返回当前状态。
    适合轮询场景或一次性查询。

    Returns:
        当前进度状态
    """
    progress_mgr = get_progress_manager()
    return progress_mgr.get_current_status()


# ========== HTTP 端点 ==========


class BuildRequest(BaseModel):
    """构建请求参数"""

    mode: str = Field(
        default="incremental",
        description="构建模式：incremental（增量）| full（全量）"
    )
    file_paths: Optional[list[str]] = Field(
        default=None,
        description="指定要处理的文件路径列表（None 表示处理所有变更文件）"
    )
    force: bool = Field(
        default=False,
        description="是否强制重新构建（即使文件未变更）"
    )
    skip_l0: bool = Field(
        default=False,
        description="是否跳过 L0 快速索引（仅运行 L1 深度索引）"
    )
    skip_l1: bool = Field(
        default=False,
        description="是否跳过 L1 深度索引（仅运行 L0 快速索引）"
    )


class BuildResponse(BaseModel):
    """构建响应"""

    status: str = Field(description="状态：started, already_running, failed")
    message: str = Field(description="消息")
    task_id: Optional[str] = Field(default=None, description="任务 ID（如果启动成功）")


# 全局标志：防止重复构建
_build_running = False
_build_lock = asyncio.Lock()


async def _run_build_task(request: BuildRequest):
    """
    后台执行构建任务（异步）

    同时更新 WebSocket (broadcaster) 和 SSE (progress_manager) 的进度。

    Args:
        request: 构建请求参数
    """
    global _build_running

    broadcaster = get_broadcaster()
    progress_mgr = get_progress_manager()

    try:
        # 动态导入，避免循环依赖和模块加载时的 Neo4j 连接
        from graphrag_agent.integrations.build.incremental_update_v2 import (
            IncrementalUpdateManagerV2
        )

        # 初始化状态
        await broadcaster.emit_status("started", "图谱构建任务已启动")
        await broadcaster.emit_log("初始化构建管理器...", "INFO")
        await progress_mgr.update(
            percent=0,
            stage="init",
            details="初始化构建管理器...",
            log="图谱构建任务已启动"
        )

        # 创建管理器实例（传入广播器）
        manager = IncrementalUpdateManagerV2(
            files_dir=FILES_DIR,
            broadcaster=broadcaster  # 注入广播器
        )

        # 执行构建
        if request.mode == "incremental":
            # 增量构建
            if not request.skip_l0:
                await broadcaster.emit_log("开始执行 L0 快速索引...", "INFO")
                await progress_mgr.update(
                    percent=10,
                    stage="l0_ingestion",
                    details="开始执行 L0 快速索引...",
                    log="开始 L0 快速索引"
                )

                # 直接 await，因为 manager 方法已经是 async 的了
                l0_result = await manager.run_fast_ingestion(
                    file_paths=request.file_paths
                )

                l0_files = l0_result.get('processed_count', 0)
                await broadcaster.emit_log(f"L0 完成：{l0_files} 个文件", "INFO")
                await progress_mgr.update(
                    percent=50,
                    stage="l0_completed",
                    details=f"L0 完成：{l0_files} 个文件",
                    log=f"L0 快速索引完成，处理 {l0_files} 个文件",
                    stats={"l0_files": l0_files}
                )

            if not request.skip_l1:
                await broadcaster.emit_log("开始执行 L1 深度索引...", "INFO")
                await progress_mgr.update(
                    percent=60,
                    stage="l1_indexing",
                    details="开始执行 L1 深度索引...",
                    log="开始 L1 深度索引"
                )

                # 直接 await，因为 manager 方法已经是 async 的了
                l1_result = await manager.run_deep_indexing(
                    file_paths=request.file_paths
                )

                l1_tasks = l1_result.get('submitted_count', 0)
                await broadcaster.emit_log(f"L1 完成：{l1_tasks} 个任务已提交", "INFO")
                await progress_mgr.update(
                    percent=100,
                    stage="l1_completed",
                    details=f"L1 完成：{l1_tasks} 个任务已提交",
                    log=f"L1 深度索引完成，提交 {l1_tasks} 个任务",
                    stats={"l1_tasks": l1_tasks}
                )

        elif request.mode == "full":
            # 全量构建
            await broadcaster.emit_log("执行全量构建...", "INFO")
            await broadcaster.emit_log("全量构建功能开发中，请使用增量模式", "WARNING")
            await progress_mgr.update(
                stage="error",
                details="全量构建功能开发中",
                log="全量构建功能开发中，请使用增量模式"
            )

        await broadcaster.emit_status("completed", "图谱构建任务完成")
        await progress_mgr.update(
            percent=100,
            stage="completed",
            details="图谱构建任务完成",
            log="✅ 所有任务完成"
        )

    except Exception as exc:
        _LOGGER.exception(f"构建任务执行失败: {exc}")
        await broadcaster.emit_error(f"构建失败: {exc}")
        await broadcaster.emit_status("failed", str(exc))
        await progress_mgr.update(
            stage="error",
            details=f"构建失败: {exc}",
            log=f"❌ 错误: {exc}"
        )

    finally:
        async with _build_lock:
            _build_running = False


@router.post("/run", response_model=BuildResponse)
async def run_build(
    request: BuildRequest,
    background_tasks: BackgroundTasks
):
    """
    触发图谱构建任务

    **注意**：此接口立即返回，构建任务在后台异步执行。

    **监听进度的两种方式**：
    1. WebSocket: `/build/ws/progress` (双向通信，更灵活)
    2. SSE: `/build/sse/progress` (单向推送，更简单)

    Args:
        request: 构建请求参数
        background_tasks: FastAPI 后台任务

    Returns:
        BuildResponse: 包含任务启动状态

    Example:
        ```bash
        # 触发构建
        curl -X POST "http://localhost:8000/build/run" \\
          -H "Content-Type: application/json" \\
          -d '{"mode": "incremental", "force": false}'

        # 查询当前状态（SSE 快照）
        curl http://localhost:8000/build/sse/status
        ```
    """
    global _build_running

    async with _build_lock:
        if _build_running:
            return BuildResponse(
                status="already_running",
                message="已有构建任务正在运行，请等待其完成",
                task_id=None
            )

        _build_running = True

    # 在后台启动构建任务
    background_tasks.add_task(_run_build_task, request)

    return BuildResponse(
        status="started",
        message="构建任务已在后台启动，请通过 WebSocket 监听进度",
        task_id="build_001"  # 可以生成唯一 ID
    )


@router.get("/status")
async def get_build_status() -> Dict[str, Any]:
    """
    查询当前构建状态

    Returns:
        构建状态信息
    """
    global _build_running

    return {
        "is_running": _build_running,
        "active_connections": len(get_broadcaster().active_connections),
    }


@router.post("/stop")
async def stop_build():
    """
    停止当前构建任务（TODO: 需要实现优雅停止机制）

    Returns:
        停止结果
    """
    global _build_running

    # TODO: 实现实际的停止逻辑
    async with _build_lock:
        if not _build_running:
            raise HTTPException(status_code=400, detail="当前没有运行中的构建任务")

        _build_running = False

    await get_broadcaster().emit_status("stopped", "构建任务已被用户停止")

    return {"status": "stopped", "message": "构建任务停止指令已发送"}
