import os
import time
import psutil
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from graphrag_agent.config.settings import community_algorithm
from graphrag_agent.graph import EntityIndexManager
from graphrag_agent.graph import GDSConfig, SimilarEntityDetector
from graphrag_agent.graph import EntityMerger
from graphrag_agent.graph.processing import EntityQualityProcessor
from graphrag_agent.community import CommunityDetectorFactory
from graphrag_agent.community import CommunitySummarizerFactory
from graphdatascience import GraphDataScience

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import MAX_WORKERS, ENTITY_BATCH_SIZE, GDS_MEMORY_LIMIT, NEO4J_CONFIG

import shutup
shutup.please()

class IndexCommunityBuilder:
    """
    索引和社区构建器，负责在基础图谱构建后的进一步处理。
    
    主要功能包括：
    1. 实体索引的创建和管理
    2. 相似实体的检测和合并
    3. 实体消歧和对齐
    4. 社区检测
    5. 社区摘要生成
    """
    
    def __init__(self):
        """初始化索引和社区构建器"""
        # 初始化终端界面
        self.console = Console()
        
        # 阶段性能统计 - 确保在_initialize_components之前定义
        self.performance_stats = {
            "初始化": 0,
            "索引创建": 0,
            "相似实体检测": 0,
            "实体合并": 0,
            "实体质量提升": 0,
            "社区检测": 0,
            "社区摘要": 0
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
            task = progress.add_task("[cyan]初始化组件...", total=4)
            
            # 初始化图数据库连接
            self.gds = GraphDataScience(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
            )
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)
            
            self.index_manager = EntityIndexManager(
                batch_size=ENTITY_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            self.gds_config = GDSConfig(
                memory_limit=GDS_MEMORY_LIMIT,
                batch_size=ENTITY_BATCH_SIZE
            )
            self.entity_detector = SimilarEntityDetector(self.gds_config)
            self.entity_merger = EntityMerger(
                batch_size=ENTITY_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            progress.advance(task)
            
            # 初始化实体质量处理器
            self.quality_processor = EntityQualityProcessor()
            progress.advance(task)
            
            # 输出使用的参数
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {ENTITY_BATCH_SIZE}[/blue]")
            self.console.print(f"[blue]GDS内存限制: {GDS_MEMORY_LIMIT}GB[/blue]")
            
            # 初始化结果存储
            self.processing_results = {
                'index_process': {}
            }
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

    def build_index_and_communities(self):
        """
        构建索引和社区
        
        Returns:
            bool: 处理是否成功
        """
        self._display_stage_header("构建索引和社区")
        
        try:
            # 1. 创建实体索引
            index_start = time.time()
            self.console.print("[cyan]正在创建实体索引...[/cyan]")
            
            vector_store = self.index_manager.create_entity_index()
            if not vector_store:
                self.console.print("[yellow]警告: 实体索引创建可能不完整[/yellow]")
            
            self.performance_stats["索引创建"] = time.time() - index_start
            
            # 显示嵌入计算性能
            embedding_time = getattr(self.index_manager, 'embedding_time', 0)
            db_time = getattr(self.index_manager, 'db_time', 0)
            index_total = self.performance_stats["索引创建"]
            
            self.console.print(f"[blue]索引创建完成，总耗时: {index_total:.2f}秒[/blue]")
            self.console.print(f"[blue]其中: 嵌入计算: {embedding_time:.2f}秒 ({embedding_time/index_total*100:.1f}%), "
                              f"数据库操作: {db_time:.2f}秒 ({db_time/index_total*100:.1f}%)[/blue]")
            
            # 2. 检测和合并相似实体
            similar_start = time.time()
            self.console.print("[cyan]正在检测相似实体...[/cyan]")
            
            duplicates = self.entity_detector.process_entities()
            
            self.performance_stats["相似实体检测"] = time.time() - similar_start
            
            # 显示相似实体检测性能统计
            projection_time = getattr(self.entity_detector, 'projection_time', 0)
            knn_time = getattr(self.entity_detector, 'knn_time', 0)
            wcc_time = getattr(self.entity_detector, 'wcc_time', 0)
            query_time = getattr(self.entity_detector, 'query_time', 0)
            similar_total = self.performance_stats["相似实体检测"]
            
            self.console.print(f"[blue]相似实体检测完成，找到 {len(duplicates)} 组候选实体，总耗时: {similar_total:.2f}秒[/blue]")
            self.console.print(f"[blue]其中: 投影创建: {projection_time:.2f}秒 ({projection_time/similar_total*100:.1f}%), "
                              f"KNN处理: {knn_time:.2f}秒 ({knn_time/similar_total*100:.1f}%), "
                              f"WCC处理: {wcc_time:.2f}秒 ({wcc_time/similar_total*100:.1f}%), "
                              f"查询处理: {query_time:.2f}秒 ({query_time/similar_total*100:.1f}%)[/blue]")
            
            # 3. 执行实体合并
            merge_start = time.time()
            self.console.print("[cyan]正在合并相似实体...[/cyan]")
            
            merged_count = self.entity_merger.process_duplicates(duplicates)
            
            self.performance_stats["实体合并"] = time.time() - merge_start
            
            # 显示实体合并性能统计
            llm_time = getattr(self.entity_merger, 'llm_time', 0)
            parse_time = getattr(self.entity_merger, 'parse_time', 0)
            db_time = getattr(self.entity_merger, 'db_time', 0)
            merge_total = self.performance_stats["实体合并"]
            
            self._display_results_table(
                "实体合并结果",
                {
                    "合并的实体组数": merged_count,
                    "总耗时": f"{merge_total:.2f}秒",
                    "LLM处理": f"{llm_time:.2f}秒 ({llm_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00秒 (0.0%)",
                    "结果解析": f"{parse_time:.2f}秒 ({parse_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00秒 (0.0%)",
                    "数据库操作": f"{db_time:.2f}秒 ({db_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00秒 (0.0%)"
                }
            )
            
            # 4. 实体质量提升（消歧和对齐）
            quality_start = time.time()
            self.console.print("[cyan]正在进行实体消歧和对齐...[/cyan]")
            
            quality_result = self.quality_processor.process()
            
            self.performance_stats["实体质量提升"] = time.time() - quality_start
            
            self._display_results_table(
                "实体质量提升结果",
                {
                    "消歧的实体": quality_result['disambiguated'],
                    "对齐的实体": quality_result['aligned'],
                    "总耗时": f"{self.performance_stats['实体质量提升']:.2f}秒"
                }
            )
            
            # 5. 社区检测
            community_start = time.time()
            self.console.print("[cyan]正在执行社区检测...[/cyan]")

            # 使用工厂类创建检测器
            detector = CommunityDetectorFactory.create(
                algorithm=community_algorithm,
                gds=self.gds,
                graph=self.graph
            )
            community_results = detector.process()

            self.performance_stats["社区检测"] = time.time() - community_start

            self.console.print(f"[blue]社区检测完成，耗时: {self.performance_stats['社区检测']:.2f}秒[/blue]")
            if community_results and community_results.get('status') == 'success':
                community_count = community_results.get('details', {}).get('detection', {}).get('communityCount', 0)
                self.console.print(f"[blue]检测到 {community_count} 个社区[/blue]")

            # 6. 生成社区摘要
            summary_start = time.time()
            self.console.print("[cyan]正在生成社区摘要...[/cyan]")

            # 使用摘要工厂类
            summarizer = CommunitySummarizerFactory.create_summarizer(
                community_algorithm,
                self.graph
            )
            summaries = summarizer.process_communities()

            self.performance_stats["社区摘要"] = time.time() - summary_start

            self._display_results_table(
                "社区摘要结果",
                {
                    "生成的摘要数量": len(summaries) if summaries else 0,
                    "耗时": f"{self.performance_stats['社区摘要']:.2f}秒"
                }
            )
            
            self.console.print("[green]索引和社区构建完成[/green]")
            
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
            self.console.print(f"[red]索引和社区构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行索引和社区构建流程"""
        try:
            # 记录开始时间
            self.start_time = time.time()
            
            # 显示系统资源信息
            cpu_count = os.cpu_count() or "未知"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # 显示开始面板
            start_text = Text("开始索引和社区构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 构建索引和社区
            self.build_index_and_communities()
            
            # 记录结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # 显示完成面板
            success_text = Text("索引和社区构建流程完成", style="bold green")
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

if __name__ == "__main__":
    try:
        builder = IndexCommunityBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")
