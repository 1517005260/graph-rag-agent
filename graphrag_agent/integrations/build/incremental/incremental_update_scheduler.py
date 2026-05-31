import time
import schedule
import threading
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table

from graphrag_agent.config.settings import MAX_WORKERS

class IncrementalUpdateScheduler:
    """
    增量更新调度器，管理不同组件的更新频率，确保高效的增量更新策略。
    
    主要功能：
    1. 根据配置管理不同组件的更新频率
    2. 实现智能调度，避免频繁更新
    3. 支持按需更新和定时更新
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增量更新调度器
        
        Args:
            config: 调度配置，包含不同组件的更新阈值
        """
        self.console = Console()
        
        # 默认配置
        self.default_config = {
            "file_change_threshold": 300,  # 5分钟
            "entity_embedding_threshold": 1800,  # 30分钟
            "chunk_embedding_threshold": 1800,  # 30分钟
            "graph_consistency_threshold": 86400,  # 24小时
            "community_detection_threshold": 172800,  # 48小时
            "manual_edit_check_threshold": 604800  # 7天
        }
        
        # 合并用户配置
        self.config = {**self.default_config, **(config or {})}
        
        # 上次运行时间记录
        self.last_run = {}
        
        # 调度任务
        self.scheduled_jobs = {}
        
        # 调度锁，防止任务重叠
        self.schedule_lock = threading.Lock()
        
        # 设置线程数
        self.max_workers = MAX_WORKERS
    
    def should_update(self, component: str, force: bool = False) -> bool:
        """
        判断组件是否需要更新
        
        Args:
            component: 组件名称
            force: 是否强制更新
            
        Returns:
            bool: 是否需要更新
        """
        if force:
            return True
            
        if component not in self.last_run:
            return True
            
        current_time = time.time()
        elapsed = current_time - self.last_run[component]
        
        # 根据配置决定是否需要更新
        threshold = self.config.get(f"{component}_threshold", 3600)  # 默认1小时
        return elapsed > threshold
    
    def mark_updated(self, component: str):
        """
        标记组件已更新
        
        Args:
            component: 组件名称
        """
        self.last_run[component] = time.time()
    
    def schedule_component(self, component: str, processor: Callable, interval: int = None):
        """
        调度组件更新
        
        Args:
            component: 组件名称
            processor: 处理函数
            interval: 更新间隔（秒），如果为None则使用配置值
        """
        # 获取组件更新间隔
        if interval is None:
            interval = self.config.get(f"{component}_threshold", 3600)
        
        # 计算间隔的小时和分钟表示
        hours = interval // 3600
        minutes = (interval % 3600) // 60
        
        if hours > 0:
            # 如果间隔大于1小时，按小时调度
            job = schedule.every(hours).hours.do(self._run_component, component, processor)
        else:
            # 否则按分钟调度
            job = schedule.every(max(1, minutes)).minutes.do(self._run_component, component, processor)
        
        # 保存调度任务
        self.scheduled_jobs[component] = job
        
        self.console.print(f"[blue]已调度组件 {component} 每 {hours}小时{minutes}分钟 更新一次[/blue]")
    
    def _run_component(self, component: str, processor: Callable):
        """
        运行组件更新
        
        Args:
            component: 组件名称
            processor: 处理函数
        """
        # 使用锁确保任务不重叠
        if not self.schedule_lock.acquire(blocking=False):
            self.console.print(f"[yellow]组件 {component} 更新被跳过，因为另一个任务正在运行[/yellow]")
            return
            
        try:
            # 检查是否应该更新
            if self.should_update(component):
                self.console.print(f"[cyan]开始更新组件: {component}[/cyan]")
                
                # 执行更新
                result = processor()
                
                # 标记已更新
                self.mark_updated(component)
                
                self.console.print(f"[green]组件 {component} 更新完成[/green]")
                return result
            else:
                self.console.print(f"[yellow]组件 {component} 更新被跳过，因为上次更新时间较近[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]组件 {component} 更新出错: {e}[/red]")
        finally:
            self.schedule_lock.release()
    
    def schedule_update(self, processor):
        """
        调度完整增量更新流程
        
        Args:
            processor: 处理对象，必须包含各种处理方法
        """
        # 检查文件变更
        self.schedule_component("file_change", processor.detect_file_changes, 
                               self.config["file_change_threshold"])
        
        # 检查实体Embedding更新
        self.schedule_component("entity_embedding", processor.update_entity_embeddings,
                               self.config["entity_embedding_threshold"])
        
        # 检查Chunk Embedding更新
        self.schedule_component("chunk_embedding", processor.update_chunk_embeddings,
                               self.config["chunk_embedding_threshold"])
        
        # 检查图结构完整性
        self.schedule_component("graph_consistency", processor.verify_graph_consistency,
                               self.config["graph_consistency_threshold"])
        
        # 检查社区检测
        self.schedule_component("community_detection", processor.detect_communities,
                               self.config["community_detection_threshold"])
        
        # 全量重建（低频率）
        self.schedule_component("full_rebuild", processor.rebuild_if_needed,
                               self.config["full_rebuild_threshold"])
    
    def start(self):
        """启动调度器"""
        self.console.print("[cyan]启动增量更新调度器...[/cyan]")
        
        # 启动调度线程
        cease_continuous_run = threading.Event()
        
        class ScheduleThread(threading.Thread):
            @classmethod
            def run(cls):
                while not cease_continuous_run.is_set():
                    schedule.run_pending()
                    time.sleep(60)  # 检查间隔，每分钟
        
        continuous_thread = ScheduleThread()
        continuous_thread.daemon = True
        continuous_thread.start()
        
        return cease_continuous_run
    
    def stop(self, cease_event):
        """
        停止调度器
        
        Args:
            cease_event: 停止事件
        """
        self.console.print("[yellow]正在停止增量更新调度器...[/yellow]")
        cease_event.set()
        
        # 清除所有调度任务
        schedule.clear()
        self.scheduled_jobs.clear()
        
        self.console.print("[green]增量更新调度器已停止[/green]")
    
    def run_once(self, processor):
        """
        立即执行一次完整的增量更新流程
        
        Args:
            processor: 处理对象，必须包含各种处理方法
        """
        with self.schedule_lock:
            self.console.print("[bold cyan]开始执行一次完整增量更新...[/bold cyan]")
            
            try:
                # 检测文件变更
                file_changes = processor.detect_file_changes()
                self.mark_updated("file_change")
                
                # 只有当有文件变更时才执行后续步骤
                if file_changes and (file_changes.get("added") or file_changes.get("modified") or file_changes.get("deleted")):
                    # 更新实体Embedding
                    processor.update_entity_embeddings()
                    self.mark_updated("entity_embedding")
                    
                    # 更新Chunk Embedding
                    processor.update_chunk_embeddings()
                    self.mark_updated("chunk_embedding")
                    
                    # 验证图结构完整性
                    processor.verify_graph_consistency()
                    self.mark_updated("graph_consistency")
                else:
                    self.console.print("[yellow]没有检测到文件变更，跳过后续步骤[/yellow]")
                
                self.console.print("[green]完整增量更新执行完成[/green]")
                
            except Exception as e:
                self.console.print(f"[red]执行增量更新时出错: {e}[/red]")
    
    def force_update(self, component: str, processor: Callable):
        """
        强制更新指定组件
        
        Args:
            component: 组件名称
            processor: 处理函数
        """
        with self.schedule_lock:
            self.console.print(f"[bold cyan]强制更新组件: {component}[/bold cyan]")
            
            try:
                # 执行更新
                result = processor()
                
                # 标记已更新
                self.mark_updated(component)
                
                self.console.print(f"[green]组件 {component} 强制更新完成[/green]")
                return result
                
            except Exception as e:
                self.console.print(f"[red]强制更新组件 {component} 时出错: {e}[/red]")
                
    def print_status(self):
        """打印调度器状态"""
        self.console.print("\n[bold cyan]增量更新调度器状态[/bold cyan]")
        
        status_table = Table(title="组件更新状态")
        status_table.add_column("组件", style="cyan")
        status_table.add_column("上次更新", style="magenta")
        status_table.add_column("下次更新", style="green")
        status_table.add_column("间隔", style="blue")
        
        current_time = time.time()
        
        for component, threshold_key in [
            ("文件变更", "file_change_threshold"),
            ("实体Embedding", "entity_embedding_threshold"),
            ("Chunk Embedding", "chunk_embedding_threshold"),
            ("图结构完整性", "graph_consistency_threshold"),
            ("社区检测", "community_detection_threshold"),
            ("手动编辑检查", "manual_edit_check_threshold")
        ]:
            # 获取上次更新时间
            last_update = self.last_run.get(component.lower().replace(" ", "_"), None)
            last_update_str = "从未" if last_update is None else datetime.fromtimestamp(last_update).strftime("%Y-%m-%d %H:%M:%S")
            
            # 计算下次更新时间
            threshold = self.config.get(threshold_key, 3600)
            if last_update is not None:
                next_update = last_update + threshold
                next_update_str = datetime.fromtimestamp(next_update).strftime("%Y-%m-%d %H:%M:%S")
            else:
                next_update_str = "立即"
            
            # 格式化间隔
            hours = threshold // 3600
            minutes = (threshold % 3600) // 60
            interval_str = f"{hours}小时{minutes}分钟"
            
            status_table.add_row(component, last_update_str, next_update_str, interval_str)
        
        self.console.print(status_table)