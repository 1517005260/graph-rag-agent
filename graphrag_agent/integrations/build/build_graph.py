import time
import os
import psutil
from typing import Dict, Any, List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
from graphrag_agent.config.prompts import (
    system_template_build_graph,
    human_template_build_graph
)
from graphrag_agent.config.settings import (
    entity_types,
    relationship_types,
    theme,
    FILES_DIR,
    CHUNK_SIZE,
    OVERLAP,
    MAX_WORKERS, BATCH_SIZE,
)
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
from graphrag_agent.graph import GraphStructureBuilder
from graphrag_agent.graph import EntityRelationExtractor
from graphrag_agent.graph import GraphWriter

import shutup
shutup.please()


class KnowledgeGraphBuilder:
    """
    知识图谱构建器，负责图谱的基础构建流程。
    
    主要功能包括：
    1. 文件读取和解析
    2. 文本分块
    3. 实体和关系抽取
    4. 构建基础图结构
    5. 写入数据库
    """
    
    def __init__(self):
        """初始化知识图谱构建器"""
        # 初始化终端界面
        self.console = Console()
        self.processed_documents = []
        
        # 添加计时器
        self.start_time = None
        self.end_time = None
        
        # 阶段性能统计
        self.performance_stats = {
            "初始化": 0,
            "文件处理": 0,  # 改为"文件处理"，包含读取和分块
            "图结构构建": 0,
            "实体抽取": 0,
            "写入数据库": 0
        }
        
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
            
            # 初始化模型
            self.llm = get_llm_model()
            self.embeddings = get_embeddings_model()
            progress.advance(task)
            
            # 初始化图数据库连接
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)
            
            # 初始化文档处理器
            self.document_processor = DocumentProcessor(FILES_DIR, CHUNK_SIZE, OVERLAP)
            progress.advance(task)
            
            self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)
            self.entity_extractor = EntityRelationExtractor(
                self.llm,
                system_template_build_graph,
                human_template_build_graph,
                entity_types,
                relationship_types,
                max_workers=MAX_WORKERS,
                batch_size=5  # LLM批处理大小保持小一些以确保质量
            )
            
            # 输出使用的参数
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {BATCH_SIZE}[/blue]")
            
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

    def build_base_graph(self) -> List:
        """
        构建基础知识图谱
        
        Returns:
            List: 处理后的文件内容列表，包含文件名、原文、分块和处理结果
        """
        self._display_stage_header("构建基础知识图谱")
        
        try:
            # 1. 处理文件（读取和分块）
            process_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]处理文件...", total=1)
                
                # 使用DocumentProcessor处理文件
                self.processed_documents = self.document_processor.process_directory()
                progress.update(task, completed=1)
                
                # 显示文件信息
                table = Table(title="文件信息")
                table.add_column("文件名")
                table.add_column("类型", style="cyan")
                table.add_column("内容长度", justify="right")
                table.add_column("分块数量", justify="right")
                
                for doc in self.processed_documents:
                    file_type = self.document_processor.get_extension_type(doc["extension"])
                    chunks_count = doc.get("chunk_count", 0)
                    table.add_row(
                        doc["filename"], 
                        file_type, 
                        str(doc["content_length"]),
                        str(chunks_count)
                    )
                self.console.print(table)
            
            self.performance_stats["文件处理"] = time.time() - process_start
            
            # 显示分块统计
            total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
            total_length = sum(doc["content_length"] for doc in self.processed_documents)
            avg_chunk_size = sum(sum(doc.get("chunk_lengths", [0])) for doc in self.processed_documents) / total_chunks if total_chunks else 0
            
            self.console.print(f"[blue]共处理 {len(self.processed_documents)} 个文件，总计 {total_length} 字符[/blue]")
            self.console.print(f"[blue]共生成 {total_chunks} 个文本块，平均每块 {avg_chunk_size:.1f} 字符[/blue]")
            
            # 3. 构建图结构
            struct_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]构建图结构...", total=3)
                
                # 清空并创建Document节点
                self.struct_builder.clear_database()
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                        self.struct_builder.create_document(
                            type="local",
                            uri=str(FILES_DIR),
                            file_name=doc["filename"],
                            domain=theme
                        )
                progress.advance(task)
                
                # 创建Chunk节点和关系 - 优化：使用并行处理大文件
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                        # 根据chunks数量选择处理方法
                        chunks = doc["chunks"]
                        if doc.get("chunk_count", 0) > 100:
                            # 对于大文件使用并行处理
                            result = self.struct_builder.parallel_process_chunks(
                                doc["filename"],
                                chunks,
                                max_workers=os.cpu_count() or 4
                            )
                        else:
                            # 对于小文件使用标准批处理
                            result = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],
                                chunks
                            )
                        doc["graph_result"] = result
                progress.advance(task)
                progress.advance(task)
            
            self.performance_stats["图结构构建"] = time.time() - struct_start
            
            # 4. 提取实体和关系
            extract_start = time.time()
            with self._create_progress() as progress:
                total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
                task = progress.add_task("[cyan]提取实体和关系...", total=total_chunks)
                
                def progress_callback(chunk_index):
                    progress.advance(task)
                
                # 准备处理的数据格式
                file_contents_format = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        file_contents_format.append([
                            doc["filename"], 
                            doc["content"], 
                            doc["chunks"]
                        ])
                
                # 根据数据集大小选择处理方法
                if total_chunks > 100:
                    # 对于大型数据集使用批处理模式
                    processed_file_contents = self.entity_extractor.process_chunks_batch(
                        file_contents_format,
                        progress_callback
                    )
                else:
                    # 对于小型数据集使用标准并行处理
                    processed_file_contents = self.entity_extractor.process_chunks(
                        file_contents_format,
                        progress_callback
                    )
                
                # 将处理结果合并回文档数据
                file_content_map = {}
                for processed_file in processed_file_contents:
                    if len(processed_file) >= 4:  # 确保有足够的元素
                        filename = processed_file[0]
                        entity_data = processed_file[3]
                        file_content_map[filename] = entity_data
                
                # 使用映射将结果放回到原始文档中
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        filename = doc["filename"]
                        if filename in file_content_map:
                            doc["entity_data"] = file_content_map[filename]
                        else:
                            self.console.print(f"[yellow]警告: 文件 {filename} 的实体抽取结果未找到[/yellow]")
            
            self.performance_stats["实体抽取"] = time.time() - extract_start
            
            # 输出缓存统计
            cache_hits = getattr(self.entity_extractor, 'cache_hits', 0)
            cache_misses = getattr(self.entity_extractor, 'cache_misses', 0)
            total_requests = cache_hits + cache_misses
            cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            self.console.print(f"[blue]LLM调用缓存命中率: {cache_rate:.1f}% ({cache_hits}/{total_requests})[/blue]")
            
            # 5. 写入数据库
            write_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]写入数据库...", total=1)
                
                # 将处理数据转换为GraphWriter所需格式
                graph_writer_data = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"] and "entity_data" in doc:
                        # 获取图构建结果（创建的chunk节点列表）
                        graph_result = doc.get("graph_result", [])
                        entity_data = doc.get("entity_data", [])
                        
                        # 确保graph_result和entity_data存在且长度相等
                        if not graph_result:
                            self.console.print(f"[yellow]警告: 文件 {doc['filename']} 的图结构结果缺失[/yellow]")
                            continue
                            
                        if not entity_data or not isinstance(entity_data, list):
                            self.console.print(f"[yellow]警告: 文件 {doc['filename']} 的实体数据缺失或格式不正确[/yellow]")
                            continue
                            
                        # 调整数据格式以匹配GraphWriter期望的结构
                        graph_writer_data.append([
                            doc["filename"],
                            doc["content"],
                            doc["chunks"],
                            graph_result,  # 这应该是chunks_with_hash数据
                            entity_data,    # 这应该是实体提取结果
                        ])
                
                # 使用优化的GraphWriter
                graph_writer = GraphWriter(
                    self.graph, 
                    batch_size=50,
                    max_workers=os.cpu_count() or 4
                )
                graph_writer.process_and_write_graph_documents(graph_writer_data)
                progress.update(task, completed=1)
            
            self.performance_stats["写入数据库"] = time.time() - write_start
            
            self.console.print("[green]基础知识图谱构建完成[/green]")
            
            # 显示性能统计
            performance_table = Table(title="性能统计")
            performance_table.add_column("处理阶段", style="cyan")
            performance_table.add_column("耗时(秒)", justify="right")
            performance_table.add_column("占比(%)", justify="right")
            
            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")
            
            performance_table.add_row("总计", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)
            
            # 返回处理好的文档列表
            file_contents_compat = []
            for doc in self.processed_documents:
                if "chunks" in doc and doc["chunks"]:
                    content_list = [
                        doc["filename"],
                        doc["content"],
                        doc["chunks"]
                    ]
                    if "entity_data" in doc:
                        content_list.append(doc["entity_data"])
                    file_contents_compat.append(content_list)
            
            return file_contents_compat
            
        except Exception as e:
            self.console.print(f"[red]基础图谱构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行知识图谱构建流程"""
        try:
            # 记录开始时间
            self.start_time = time.time()
            
            # 显示系统资源信息
            cpu_count = os.cpu_count() or "未知"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # 显示开始面板
            start_text = Text("开始知识图谱构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 构建基础图谱
            result = self.build_base_graph()
            
            # 记录结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # 显示完成面板
            success_text = Text("知识图谱构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示总耗时信息
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")
            
            return result
            
        except Exception as e:
            # 记录结束时间（即使出错）
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")
                
            error_text = Text(f"构建过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise

if __name__ == "__main__":
    try:
        builder = KnowledgeGraphBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")