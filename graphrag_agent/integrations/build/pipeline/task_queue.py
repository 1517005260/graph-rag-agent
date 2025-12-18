"""
任务队列系统 - 支持异步图谱构建

实现一个简单的内存任务队列，支持：
1. 文件处理任务的异步提交
2. 后台Worker执行
3. 任务状态跟踪
4. 优先级调度
"""
import time
import threading
import queue
from enum import Enum
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


class TaskPriority(Enum):
    """任务优先级"""
    HIGH = 1    # 高优先级（用户主动上传）
    NORMAL = 2  # 正常优先级（批量处理）
    LOW = 3     # 低优先级（后台重建）


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 正在执行
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败


@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str  # 'entity_extraction', 'graph_build', 'community_detection'
    file_path: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """优先级排序（数字小的优先）"""
        return self.priority.value < other.priority.value


class GraphBuildTaskQueue:
    """
    图谱构建任务队列

    特性：
    1. 支持优先级队列（高优先级任务优先执行）
    2. 后台Worker自动消费任务
    3. 任务状态实时跟踪
    4. 线程安全
    """

    def __init__(self, num_workers: int = 2):
        """
        初始化任务队列

        Args:
            num_workers: Worker线程数量
        """
        self.task_queue = queue.PriorityQueue()
        self.task_registry: Dict[str, Task] = {}  # 任务注册表
        self.task_lock = threading.RLock()

        # Worker管理
        self.num_workers = num_workers
        self.workers: List[threading.Thread] = []
        self.running = False
        self.stop_event = threading.Event()

        # 任务处理器注册
        self.task_handlers: Dict[str, Callable] = {}

        # 统计信息
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "active_tasks": 0
        }

    def register_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器

        Args:
            task_type: 任务类型
            handler: 处理函数 handler(task: Task) -> None
        """
        self.task_handlers[task_type] = handler

    def submit_task(self, task: Task) -> str:
        """
        提交任务到队列

        Args:
            task: 任务对象

        Returns:
            str: 任务ID
        """
        with self.task_lock:
            # 检查是否已存在相同任务
            if task.task_id in self.task_registry:
                existing_task = self.task_registry[task.task_id]
                if existing_task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    return task.task_id  # 已存在且未完成，跳过

            # 注册任务
            self.task_registry[task.task_id] = task

            # 加入队列
            self.task_queue.put(task)

            # 更新统计
            self.stats["total_submitted"] += 1
            self.stats["active_tasks"] += 1

        return task.task_id

    def submit_file_processing(
        self,
        file_path: str,
        task_type: str = "entity_extraction",
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        提交文件处理任务（便捷方法）

        Args:
            file_path: 文件路径
            task_type: 任务类型
            priority: 优先级
            metadata: 元数据

        Returns:
            str: 任务ID
        """
        task_id = f"{task_type}:{file_path}:{int(time.time())}"
        task = Task(
            task_id=task_id,
            task_type=task_type,
            file_path=file_path,
            priority=priority,
            metadata=metadata or {}
        )
        return self.submit_task(task)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Optional[TaskStatus]: 任务状态，不存在则返回None
        """
        with self.task_lock:
            task = self.task_registry.get(task_id)
            return task.status if task else None

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务详情"""
        with self.task_lock:
            return self.task_registry.get(task_id)

    def get_pending_tasks(self) -> List[Task]:
        """获取所有待处理任务"""
        with self.task_lock:
            return [
                task for task in self.task_registry.values()
                if task.status == TaskStatus.PENDING
            ]

    def get_running_tasks(self) -> List[Task]:
        """获取所有运行中任务"""
        with self.task_lock:
            return [
                task for task in self.task_registry.values()
                if task.status == TaskStatus.RUNNING
            ]

    def _worker_loop(self, worker_id: int):
        """
        Worker主循环

        Args:
            worker_id: Worker编号
        """
        print(f"[TaskQueue] Worker-{worker_id} 已启动")

        while not self.stop_event.is_set():
            try:
                # 从队列获取任务（超时1秒，避免死锁）
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 更新任务状态
                with self.task_lock:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()

                print(f"[TaskQueue] Worker-{worker_id} 开始处理: {task.task_id}")

                # 执行任务
                try:
                    handler = self.task_handlers.get(task.task_type)
                    if handler:
                        handler(task)

                        # 标记完成
                        with self.task_lock:
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now()
                            self.stats["total_completed"] += 1
                            self.stats["active_tasks"] -= 1

                        print(f"[TaskQueue] Worker-{worker_id} 完成: {task.task_id}")
                    else:
                        raise ValueError(f"未找到任务处理器: {task.task_type}")

                except Exception as e:
                    # 标记失败
                    with self.task_lock:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = datetime.now()
                        self.stats["total_failed"] += 1
                        self.stats["active_tasks"] -= 1

                    print(f"[TaskQueue] Worker-{worker_id} 失败: {task.task_id}, 错误: {e}")

                finally:
                    self.task_queue.task_done()

            except Exception as e:
                print(f"[TaskQueue] Worker-{worker_id} 异常: {e}")

        print(f"[TaskQueue] Worker-{worker_id} 已停止")

    def start(self):
        """启动Worker"""
        if self.running:
            print("[TaskQueue] 已经在运行中")
            return

        self.running = True
        self.stop_event.clear()

        # 启动Worker线程
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        print(f"[TaskQueue] 已启动 {self.num_workers} 个Worker")

    def stop(self, wait_completion: bool = True):
        """
        停止Worker

        Args:
            wait_completion: 是否等待所有任务完成
        """
        if not self.running:
            print("[TaskQueue] 未在运行中")
            return

        print("[TaskQueue] 正在停止Worker...")

        if wait_completion:
            # 等待队列清空
            print("[TaskQueue] 等待任务队列清空...")
            self.task_queue.join()

        # 发送停止信号
        self.stop_event.set()

        # 等待所有Worker退出
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()
        self.running = False

        print("[TaskQueue] Worker已全部停止")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.task_lock:
            return {
                **self.stats,
                "queue_size": self.task_queue.qsize(),
                "running_tasks": len(self.get_running_tasks()),
                "pending_tasks": len(self.get_pending_tasks())
            }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("任务队列统计")
        print("="*50)
        print(f"总提交任务: {stats['total_submitted']}")
        print(f"已完成: {stats['total_completed']}")
        print(f"失败: {stats['total_failed']}")
        print(f"队列中: {stats['queue_size']}")
        print(f"运行中: {stats['running_tasks']}")
        print(f"待处理: {stats['pending_tasks']}")
        print("="*50 + "\n")


# 全局单例
_global_task_queue: Optional[GraphBuildTaskQueue] = None


def get_global_task_queue() -> GraphBuildTaskQueue:
    """获取全局任务队列单例"""
    global _global_task_queue
    if _global_task_queue is None:
        _global_task_queue = GraphBuildTaskQueue(num_workers=2)
        _global_task_queue.start()
    return _global_task_queue
