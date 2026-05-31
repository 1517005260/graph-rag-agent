import os
import time
import psutil
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from graphrag_agent.graph import ChunkIndexManager
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import MAX_WORKERS, CHUNK_BATCH_SIZE

import shutup
shutup.please()

class ChunkIndexBuilder:
    """
    文本块索引构建器，负责在基础图谱构建后为Chunk节点创建向量索引，以支持naive RAG查询。
    
    主要功能包括：
    1. Chunk节点索引的创建和管理
    2. 向量索引性能统计
    """
    
    def __init__(self):
        """初始化文本块索引构建器"""
        # 初始化终端界面
        self.console = Console()
        
        # 阶段性能统计
        self.performance_stats = {
            "初始化": 0,
            "索引创建": 0
        }
        
        # 添加计时器
        self.start_time = None
        self.end_time = None
        
        # 初始化组件
        self._initialize_components()

    def _create_progress(self):
        """创建进度显示器"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _initialize_components(self):
        """初始化所有必要的组件"""
        init_start = time.time()
        
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=2)
            
            # 初始化图数据库连接
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)
             
            self.index_manager = ChunkIndexManager(
                batch_size=CHUNK_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            
            # 输出使用的参数
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {CHUNK_BATCH_SIZE}[/blue]")
            
            progress.advance(task)
        
        self.performance_stats["初始化"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """显示处理阶段的标题"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格"""
        table = Table(title=title, show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")
        
        for key, value in data.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
        
    def _format_time(self, seconds: float) -> str:
        """格式化时间为小时:分钟:秒.毫秒"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def build_chunk_index(self):
        """
        构建文本块索引
        
        Returns:
            bool: 处理是否成功
        """
        self._display_stage_header("构建文本块索引")
        
        try:
            # 创建Chunk索引
            index_start = time.time()
            self.console.print("[cyan]正在创建文本块索引...[/cyan]")
            
            # 只计算和存储embeddings，不创建新的向量索引
            vector_store = self.index_manager.create_chunk_index()
            
            self.performance_stats["索引创建"] = time.time() - index_start
            
            # 显示嵌入计算性能
            embedding_time = getattr(self.index_manager, 'embedding_time', 0)
            db_time = getattr(self.index_manager, 'db_time', 0)
            index_total = self.performance_stats["索引创建"]
            
            if index_total > 0:
                self.console.print(f"[blue]索引创建完成，总耗时: {index_total:.2f}秒[/blue]")
                self.console.print(f"[blue]其中: 嵌入计算: {embedding_time:.2f}秒 ({embedding_time/index_total*100:.1f}%), "
                                f"数据库操作: {db_time:.2f}秒 ({db_time/index_total*100:.1f}%)[/blue]")
            
            # 查询节点数量
            try:
                node_count = self.graph.query(
                    """
                    MATCH (c:`__Chunk__`)
                    WHERE c.embedding IS NOT NULL
                    RETURN count(c) as count
                    """
                )
                
                self._display_results_table(
                    "索引创建结果",
                    {
                        "已索引节点数量": node_count[0]["count"] if node_count else 0,
                        "总耗时": f"{index_total:.2f}秒",
                        "嵌入计算": f"{embedding_time:.2f}秒 ({embedding_time/index_total*100:.1f}%)" if index_total > 0 else "0.00秒",
                        "数据库操作": f"{db_time:.2f}秒 ({db_time/index_total*100:.1f}%)" if index_total > 0 else "0.00秒"
                    }
                )
            except Exception as e:
                self.console.print(f"[yellow]查询索引状态时出错 (可忽略): {e}[/yellow]")
            
            self.console.print("[green]文本块索引构建完成[/green]")
            
            # 显示性能统计摘要
            performance_table = Table(title="性能统计摘要")
            performance_table.add_column("处理阶段", style="cyan")
            performance_table.add_column("耗时(秒)", justify="right")
            performance_table.add_column("占比(%)", justify="right")
            
            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")
            
            performance_table.add_row("总计", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]文本块索引构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行文本块索引构建流程"""
        try:
            # 记录开始时间
            self.start_time = time.time()
            
            # 显示系统资源信息
            cpu_count = os.cpu_count() or "未知"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # 显示开始面板
            start_text = Text("开始文本块索引构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 构建文本块索引
            self.build_chunk_index()
            
            # 记录结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # 显示完成面板
            success_text = Text("文本块索引构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示总耗时信息
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")
            
            return True
            
        except Exception as e:
            # 记录结束时间（即使出错）
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")
                
            error_text = Text(f"处理过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise