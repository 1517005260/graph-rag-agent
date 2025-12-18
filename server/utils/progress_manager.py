"""
SSE 进度管理器（单例模式）

用于在后台任务和 SSE 接口之间共享构建进度状态。
与 WebSocket 方案互补，提供更轻量的单向推送方案。
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading

_LOGGER = logging.getLogger(__name__)


class ProgressManager:
    """
    SSE 进度管理器（线程安全单例）

    功能：
    - 维护当前构建进度状态
    - 支持多个 SSE 客户端订阅
    - 线程安全的状态更新
    - 自动清理过期日志
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProgressManager, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        """初始化状态"""
        self.current_status: Dict[str, Any] = {
            "percent": 0,
            "stage": "idle",
            "details": "等待开始",
            "logs": [],
            "stats": {
                "l0_files": 0,
                "l1_tasks": 0,
                "entities": 0,
                "relations": 0
            },
            "last_update": datetime.now().isoformat()
        }

        # 用于通知 SSE 有新数据的事件
        self._events: List[asyncio.Event] = []
        self._events_lock = asyncio.Lock()

    async def update(
        self,
        percent: Optional[int] = None,
        stage: Optional[str] = None,
        details: Optional[str] = None,
        log: Optional[str] = None,
        stats: Optional[Dict[str, int]] = None
    ):
        """
        更新进度状态并通知所有订阅者

        Args:
            percent: 进度百分比（0-100）
            stage: 当前阶段（idle, init, detect_changes, entity_extraction, etc.）
            details: 详细描述
            log: 日志消息（会添加到日志列表）
            stats: 统计数据（文件数、实体数等）
        """
        updated = False

        if percent is not None:
            self.current_status["percent"] = min(100, max(0, percent))
            updated = True

        if stage is not None:
            self.current_status["stage"] = stage
            updated = True

        if details is not None:
            self.current_status["details"] = details
            updated = True

        if log is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {log}"
            self.current_status["logs"].append(log_entry)

            # 只保留最近 50 条日志
            if len(self.current_status["logs"]) > 50:
                self.current_status["logs"].pop(0)

            updated = True

        if stats is not None:
            self.current_status["stats"].update(stats)
            updated = True

        if updated:
            self.current_status["last_update"] = datetime.now().isoformat()
            await self._notify_subscribers()

    async def _notify_subscribers(self):
        """通知所有订阅的 SSE 客户端"""
        async with self._events_lock:
            for event in self._events:
                event.set()

    async def subscribe(self) -> asyncio.Event:
        """
        订阅进度更新

        Returns:
            asyncio.Event: 用于等待更新的事件对象
        """
        event = asyncio.Event()
        async with self._events_lock:
            self._events.append(event)
        _LOGGER.info(f"新的 SSE 订阅者，当前订阅数: {len(self._events)}")
        return event

    async def unsubscribe(self, event: asyncio.Event):
        """
        取消订阅

        Args:
            event: 订阅时返回的事件对象
        """
        async with self._events_lock:
            if event in self._events:
                self._events.remove(event)
        _LOGGER.info(f"SSE 订阅者断开，剩余订阅数: {len(self._events)}")

    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前状态快照

        Returns:
            当前进度状态的副本
        """
        return self.current_status.copy()

    async def event_generator(self):
        """
        SSE 事件生成器

        生成 Server-Sent Events 格式的数据流。
        当状态更新时，会自动推送最新状态到客户端。

        Yields:
            Dict: SSE 事件数据（包含 event 和 data 字段）
        """
        # 订阅更新事件
        event = await self.subscribe()

        try:
            # 立即发送当前状态
            yield {
                "event": "connected",
                "data": json.dumps({
                    "message": "SSE 连接成功",
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False)
            }

            yield {
                "event": "status",
                "data": json.dumps(self.current_status, ensure_ascii=False)
            }

            # 持续监听更新
            while True:
                # 等待状态更新
                await event.wait()
                event.clear()

                # 发送更新后的状态
                yield {
                    "event": "status",
                    "data": json.dumps(self.current_status, ensure_ascii=False)
                }

        except asyncio.CancelledError:
            _LOGGER.info("SSE 连接被取消")
        except Exception as exc:
            _LOGGER.exception(f"SSE 事件生成器错误: {exc}")
        finally:
            # 清理订阅
            await self.unsubscribe(event)

    def reset(self):
        """重置状态到初始值"""
        self.current_status = {
            "percent": 0,
            "stage": "idle",
            "details": "等待开始",
            "logs": [],
            "stats": {
                "l0_files": 0,
                "l1_tasks": 0,
                "entities": 0,
                "relations": 0
            },
            "last_update": datetime.now().isoformat()
        }

    def has_subscribers(self) -> bool:
        """
        检查是否有活动的订阅者

        Returns:
            是否有订阅者
        """
        return len(self._events) > 0


# 全局单例
progress_manager = ProgressManager()


def get_progress_manager() -> ProgressManager:
    """
    获取全局进度管理器实例

    Returns:
        ProgressManager 单例
    """
    return progress_manager
