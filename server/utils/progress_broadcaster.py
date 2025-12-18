"""
WebSocket 进度广播器

用于将后端的图谱构建进度实时推送到前端。
支持多个 WebSocket 客户端同时连接和接收进度更新。
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import WebSocket

_LOGGER = logging.getLogger(__name__)


class ProgressBroadcaster:
    """
    WebSocket 进度广播器（单例模式）

    功能：
    - 管理多个 WebSocket 连接
    - 广播进度更新、日志消息
    - 支持不同类型的消息（log, progress, status, error）
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """
        接受一个新的 WebSocket 连接

        Args:
            websocket: FastAPI WebSocket 对象
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        _LOGGER.info(f"新的 WebSocket 连接建立，当前连接数：{len(self.active_connections)}")

        # 发送初始连接成功消息
        await self._send_to_client(websocket, {
            "type": "connected",
            "message": "WebSocket 连接成功",
            "timestamp": datetime.now().isoformat()
        })

    async def disconnect(self, websocket: WebSocket):
        """
        断开一个 WebSocket 连接

        Args:
            websocket: FastAPI WebSocket 对象
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        _LOGGER.info(f"WebSocket 连接断开，当前连接数：{len(self.active_connections)}")

    async def _send_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        发送消息到单个客户端

        Args:
            websocket: WebSocket 连接
            message: 消息内容（字典格式）
        """
        try:
            await websocket.send_json(message)
        except Exception as exc:
            _LOGGER.warning(f"发送消息失败: {exc}")
            # 客户端可能已断开，从列表中移除
            await self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """
        广播消息到所有连接的客户端

        Args:
            message: JSON 格式的消息数据
        """
        if not self.active_connections:
            # 没有连接，仅记录日志
            _LOGGER.debug(f"没有活动的 WebSocket 连接，跳过广播: {message.get('type', 'unknown')}")
            return

        # 添加时间戳
        message["timestamp"] = datetime.now().isoformat()

        # 广播到所有客户端
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as exc:
                _LOGGER.warning(f"广播失败: {exc}")
                disconnected.append(connection)

        # 清理断开的连接
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)

    # ========== 不同类型的消息发送方法 ==========

    async def emit_log(self, text: str, level: str = "INFO"):
        """
        发送日志消息

        Args:
            text: 日志文本
            level: 日志级别（INFO, WARNING, ERROR）
        """
        await self.broadcast({
            "type": "log",
            "level": level,
            "content": text
        })

    async def emit_progress(
        self,
        stage: str,
        percent: int,
        current: Optional[int] = None,
        total: Optional[int] = None,
        details: str = ""
    ):
        """
        发送进度更新

        Args:
            stage: 当前阶段名称（如 "entity_extraction", "graph_indexing"）
            percent: 进度百分比（0-100）
            current: 当前处理数量
            total: 总数量
            details: 详细信息
        """
        message = {
            "type": "progress",
            "stage": stage,
            "percent": min(100, max(0, percent)),  # 限制在 0-100
            "details": details
        }

        if current is not None and total is not None:
            message["current"] = current
            message["total"] = total

        await self.broadcast(message)

    async def emit_status(self, status: str, message: str = ""):
        """
        发送状态更新

        Args:
            status: 状态（started, running, completed, failed, paused）
            message: 状态描述
        """
        await self.broadcast({
            "type": "status",
            "status": status,
            "message": message
        })

    async def emit_error(self, error: str, details: Optional[Dict] = None):
        """
        发送错误消息

        Args:
            error: 错误描述
            details: 错误详情
        """
        message = {
            "type": "error",
            "error": error
        }
        if details:
            message["details"] = details

        await self.broadcast(message)

    async def emit_file_status(
        self,
        file_path: str,
        status: str,
        stage: str = "",
        progress: Optional[int] = None
    ):
        """
        发送单个文件的处理状态

        Args:
            file_path: 文件路径
            status: 状态（processing, completed, failed）
            stage: 处理阶段（L0, L1）
            progress: 文件处理进度（0-100）
        """
        message = {
            "type": "file_status",
            "file_path": file_path,
            "status": status,
        }

        if stage:
            message["stage"] = stage
        if progress is not None:
            message["progress"] = progress

        await self.broadcast(message)

    async def emit_stats(self, stats: Dict[str, Any]):
        """
        发送统计信息

        Args:
            stats: 统计数据（实体数、关系数、文件数等）
        """
        await self.broadcast({
            "type": "stats",
            "data": stats
        })

    def has_active_connections(self) -> bool:
        """
        检查是否有活动的 WebSocket 连接

        Returns:
            是否有活动连接
        """
        return len(self.active_connections) > 0


# 全局单例
broadcaster = ProgressBroadcaster()


def get_broadcaster() -> ProgressBroadcaster:
    """
    获取全局广播器实例

    Returns:
        ProgressBroadcaster 单例
    """
    return broadcaster
