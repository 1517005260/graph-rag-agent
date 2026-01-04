"""
增量更新管理器 V2 - 支持 L0/L1 拆分

核心改进：
1. L0 快速通道：文件上传后 10 秒内可搜索（Naive RAG）
2. L1 慢速通道：后台异步构建知识图谱
3. 任务队列管理：异步处理图谱构建任务
4. 进度追踪：用户可查看图谱构建进度
5. WebSocket 实时推送：前端可监听构建进度
"""
import os
import time
import signal
import argparse
import asyncio
from typing import Dict, List, Optional, TYPE_CHECKING
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from server.utils.progress_broadcaster import ProgressBroadcaster

from .incremental_graph_builder import IncrementalGraphUpdater
from graphrag_agent.graph.graph_consistency_validator import GraphConsistencyValidator
from graphrag_agent.integrations.build.incremental.manual_edit_manager import ManualEditManager
from graphrag_agent.community import CommunityDetectorFactory, CommunitySummarizerFactory
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import (
    FILES_DIR,
    community_algorithm,
    MAX_WORKERS,
    BATCH_SIZE,
    NEO4J_CONFIG
)

# 导入新的管道组件
from graphrag_agent.integrations.build.pipeline import (
    FastIngestionPipeline,
    SlowGraphPipeline,
    GraphBuildTaskQueue,
    Task,
    TaskPriority,
    get_global_task_queue,
    entity_extraction_task_handler
)


class IncrementalUpdateManagerV2:
    """
    增量更新管理器 V2

    新特性：
    1. 支持 L0/L1 拆分：用户上传文件后立即可搜索
    2. 任务队列：异步处理图谱构建
    3. 进度跟踪：实时查看构建状态
    4. 保留原有功能：手动编辑保护、一致性验证、社区检测
    """

    def __init__(
        self,
        files_dir: str = FILES_DIR,
        config=None,
        broadcaster: Optional["ProgressBroadcaster"] = None
    ):
        """
        初始化增量更新管理器 V2

        Args:
            files_dir: 监控的文件目录
            config: 配置参数
            broadcaster: WebSocket 进度广播器（可选）
        """
        self.console = Console()

        # 配置参数
        self.files_dir = files_dir
        self.config = config or {}
        self.broadcaster = broadcaster  # WebSocket 广播器

        # 初始化原有组件
        self.graph = get_db_manager().graph
        self.updater = IncrementalGraphUpdater(files_dir)
        self.validator = GraphConsistencyValidator()
        self.edit_manager = ManualEditManager()

        # 初始化新组件
        self.fast_pipeline = FastIngestionPipeline(files_dir)
        self.slow_pipeline = SlowGraphPipeline()

        # 获取全局任务队列
        self.task_queue = get_global_task_queue()

        # 注册任务处理器
        self.task_queue.register_handler("entity_extraction", entity_extraction_task_handler)

        # 运行状态
        self.running = False

        # 性能统计
        self.stats = {
            "l0_files_processed": 0,
            "l1_files_processed": 0,
            "entities_extracted": 0,
            "communities_detected": 0,
            "errors": 0
        }

    # ===================
    # L0 快速通道
    # ===================

    async def run_fast_ingestion(self, file_paths: Optional[List[str]] = None) -> Dict:
        """
        L0 快速摄取：仅做文本分块和向量化（秒级）

        Args:
            file_paths: 文件路径列表，None 则处理所有新文件

        Returns:
            Dict: 处理结果
        """
        start_time = time.time()

        self.console.print("\n[bold cyan]🚀 L0 快速摄取流程...[/bold cyan]")
        if self.broadcaster:
            await self.broadcaster.emit_log("开始 L0 快速摄取流程", "INFO")
            await self.broadcaster.emit_progress("l0_ingestion", 0, details="检测文件变更...")

        try:
            # 如果未指定文件，检测所有变更
            if file_paths is None:
                changes = self.updater.detect_changes()
                file_paths = changes.get("added", []) + changes.get("modified", [])

            if not file_paths:
                self.console.print("[yellow]没有需要处理的文件[/yellow]")
                if self.broadcaster:
                    await self.broadcaster.emit_log("没有需要处理的文件", "INFO")
                return {"status": "success", "files_processed": 0}

            total_files = len(file_paths)
            if self.broadcaster:
                await self.broadcaster.emit_progress(
                    "l0_ingestion", 10, total=total_files, current=0,
                    details=f"准备处理 {total_files} 个文件..."
                )

            # 批量快速处理（带进度回调）
            results = []
            for idx, file_path in enumerate(file_paths):
                result = self.fast_pipeline.process_single_file(file_path)
                results.append(result)

                # 更新进度
                progress = int(10 + (idx + 1) / total_files * 80)  # 10-90%
                if self.broadcaster:
                    await self.broadcaster.emit_progress(
                        "l0_ingestion", progress,
                        current=idx + 1, total=total_files,
                        details=f"处理中: {Path(file_path).name}"
                    )
                    await self.broadcaster.emit_file_status(
                        file_path, result["status"], stage="L0"
                    )

            # 更新统计
            success_count = sum(1 for r in results if r["status"] == "success")
            self.stats["l0_files_processed"] += success_count

            total_duration = time.time() - start_time

            self.console.print(
                f"[bold green]✅ L0 快速摄取完成: "
                f"{success_count}/{len(file_paths)} 成功, "
                f"耗时 {total_duration:.2f}s[/bold green]"
            )

            if self.broadcaster:
                await self.broadcaster.emit_progress("l0_ingestion", 100, details="L0 快速摄取完成")
                await self.broadcaster.emit_log(
                    f"L0 完成: {success_count}/{total_files} 成功, 耗时 {total_duration:.2f}s", "INFO"
                )

            return {
                "status": "success",
                "files_processed": success_count,
                "processed_count": success_count,  # 添加这个字段用于 API 响应
                "duration": total_duration,
                "results": results
            }

        except Exception as e:
            self.console.print(f"[red]❌ L0 快速摄取失败: {e}[/red]")
            self.stats["errors"] += 1
            if self.broadcaster:
                await self.broadcaster.emit_error(f"L0 快速摄取失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }

    def quick_ingest_single_file(self, file_path: str) -> Dict:
        """
        快速摄取单个文件（用于用户主动上传）

        Args:
            file_path: 文件路径

        Returns:
            Dict: 处理结果
        """
        self.console.print(f"\n[bold cyan]🚀 L0 快速处理: {file_path}[/bold cyan]")

        result = self.fast_pipeline.process_single_file(file_path)

        if result["status"] == "success":
            self.stats["l0_files_processed"] += 1

        return result

    # ===================
    # L1 慢速通道
    # ===================

    async def run_deep_indexing(self, file_paths: Optional[List[str]] = None) -> Dict:
        """
        L1 深度索引：图谱构建与社区发现（慢速，后台执行）

        Args:
            file_paths: 文件路径列表，None 则处理所有新文件

        Returns:
            Dict: 提交结果
        """
        start_time = time.time()

        self.console.print("\n[bold magenta]🏗️  L1 深度索引流程（后台任务）...[/bold magenta]")
        if self.broadcaster:
            await self.broadcaster.emit_log("开始 L1 深度索引流程", "INFO")
            await self.broadcaster.emit_progress("l1_indexing", 0, details="检测需要处理的文件...")

        try:
            # 如果未指定文件，检测所有变更
            if file_paths is None:
                changes = self.updater.detect_changes()
                file_paths = changes.get("added", []) + changes.get("modified", [])

            if not file_paths:
                self.console.print("[yellow]没有需要处理的文件[/yellow]")
                if self.broadcaster:
                    await self.broadcaster.emit_log("没有需要处理的文件", "INFO")
                return {"status": "success", "tasks_submitted": 0, "submitted_count": 0}

            total_files = len(file_paths)
            if self.broadcaster:
                await self.broadcaster.emit_progress(
                    "l1_indexing", 10, total=total_files, current=0,
                    details=f"准备提交 {total_files} 个任务..."
                )

            # 提交到任务队列（异步执行）
            task_ids = []
            for idx, file_path in enumerate(file_paths):
                task_id = self.task_queue.submit_file_processing(
                    file_path=file_path,
                    task_type="entity_extraction",
                    priority=TaskPriority.NORMAL,
                    metadata={"source": "incremental_update"}
                )
                task_ids.append(task_id)

                # 更新进度
                progress = int(10 + (idx + 1) / total_files * 80)  # 10-90%
                if self.broadcaster:
                    await self.broadcaster.emit_progress(
                        "l1_indexing", progress,
                        current=idx + 1, total=total_files,
                        details=f"提交任务: {Path(file_path).name}"
                    )
                    await self.broadcaster.emit_file_status(
                        file_path, "queued", stage="L1"
                    )

            self.console.print(
                f"[green]✓ 已提交 {len(task_ids)} 个任务到后台队列[/green]"
            )

            # 显示任务队列状态
            self.task_queue.print_stats()

            if self.broadcaster:
                await self.broadcaster.emit_progress("l1_indexing", 100, details="所有任务已提交到后台队列")
                await self.broadcaster.emit_log(
                    f"L1 任务提交完成: {len(task_ids)} 个任务已进入后台队列", "INFO"
                )

            return {
                "status": "success",
                "tasks_submitted": len(task_ids),
                "submitted_count": len(task_ids),  # 添加这个字段用于 API 响应
                "task_ids": task_ids,
                "duration": time.time() - start_time
            }

        except Exception as e:
            self.console.print(f"[red]❌ L1 任务提交失败: {e}[/red]")
            self.stats["errors"] += 1
            if self.broadcaster:
                await self.broadcaster.emit_error(f"L1 任务提交失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }

    def submit_entity_extraction_task(
        self,
        file_path: str,
        priority: TaskPriority = TaskPriority.HIGH
    ) -> str:
        """
        提交单个文件的实体提取任务（用于用户主动触发）

        Args:
            file_path: 文件路径
            priority: 任务优先级

        Returns:
            str: 任务ID
        """
        task_id = self.task_queue.submit_file_processing(
            file_path=file_path,
            task_type="entity_extraction",
            priority=priority,
            metadata={"source": "user_upload"}
        )

        self.console.print(
            f"[green]✓ 已提交实体提取任务: {task_id}[/green]"
        )

        return task_id

    # ===================
    # 辅助功能
    # ===================

    def resolve_entities(
        self,
        threshold: float = 0.9,
        entity_type: Optional[str] = None
    ) -> Dict:
        """
        执行实体对齐（合并相似实体）

        Args:
            threshold: 相似度阈值（0-1），默认 0.9
            entity_type: 指定实体类型（None 表示所有类型）

        Returns:
            Dict: 对齐结果
        """
        from graphrag_agent.graph.processing import EntityResolver

        self.console.print(f"[bold cyan]执行实体对齐（阈值: {threshold}）...[/bold cyan]")

        try:
            resolver = EntityResolver(threshold=threshold)
            result = resolver.resolve_entities(entity_type=entity_type)

            if result.get("status") == "success":
                merged_count = result.get("merged_count", 0)
                total_entities = result.get("total_entities", 0)

                self.console.print(
                    f"[green]✅ 实体对齐完成：{merged_count} 组相似实体已合并，"
                    f"共处理 {total_entities} 个实体[/green]"
                )

                # 发送进度更新
                self._emit_sync(self.broadcaster.emit_log(
                    f"实体对齐完成：合并 {merged_count} 组相似实体", "INFO"
                ) if self.broadcaster else None)

                return result
            else:
                self.console.print("[yellow]实体对齐未成功[/yellow]")
                return result

        except Exception as e:
            self.console.print(f"[red]❌ 执行实体对齐时出错: {e}[/red]")
            self.stats["errors"] += 1
            self._emit_sync(self.broadcaster.emit_error(
                f"实体对齐失败: {e}"
            ) if self.broadcaster else None)
            return {"status": "error", "error": str(e)}

    def detect_communities(self) -> Dict:
        """
        执行社区检测和摘要生成

        Returns:
            Dict: 社区检测结果
        """
        self.console.print("[bold cyan]执行社区检测...[/bold cyan]")

        try:
            # 获取数据库连接和GDS对象
            db_manager = get_db_manager()
            graph = db_manager.graph

            # 导入GraphDataScience库
            try:
                from graphdatascience import GraphDataScience
                gds = GraphDataScience(
                    NEO4J_CONFIG["uri"],
                    auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
                )
            except Exception as e:
                self.console.print(f"[yellow]导入GDS库失败，无法执行社区检测: {e}[/yellow]")
                return {"status": "error", "message": str(e)}

            # 创建社区检测器
            self.console.print(f"[blue]使用 {community_algorithm} 算法执行社区检测[/blue]")
            detector = CommunityDetectorFactory.create(
                algorithm=community_algorithm,
                gds=gds,
                graph=graph
            )

            # 执行社区检测
            detection_result = detector.process()

            if detection_result.get('status', '') == 'success':
                community_count = detection_result.get('details', {}).get('detection', {}).get('communityCount', 0)
                self.console.print(f"[green]社区检测完成，共检测到 {community_count} 个社区[/green]")

                # 更新统计信息
                self.stats["communities_detected"] += community_count

                return detection_result
            else:
                self.console.print("[yellow]社区检测未成功[/yellow]")
                return detection_result

        except Exception as e:
            self.console.print(f"[red]执行社区检测时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"status": "error", "message": str(e)}

    def verify_graph_consistency(self, repair=True) -> Dict:
        """
        验证图谱一致性

        Args:
            repair: 是否执行修复

        Returns:
            Dict: 验证结果
        """
        self.console.print("[bold cyan]验证图谱一致性...[/bold cyan]")

        try:
            if repair:
                result = self.validator.repair_graph()
                repaired_count = result["validation_stats"]["repaired_issues"]
                self.console.print(f"[green]图谱一致性验证完成，修复了 {repaired_count} 个问题[/green]")
            else:
                result = self.validator.validate_graph()
                issues_count = result["validation_stats"]["total_issues"]
                self.console.print(f"[green]图谱一致性验证完成，发现 {issues_count} 个问题[/green]")

            return result

        except Exception as e:
            self.console.print(f"[red]验证图谱一致性时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}

    def sync_manual_edits(self, file_paths: List[str]) -> Dict:
        """
        同步手动编辑

        Args:
            file_paths: 文件路径列表

        Returns:
            Dict: 同步结果
        """
        self.console.print("[bold cyan]同步手动编辑...[/bold cyan]")

        try:
            preserved_count = self.edit_manager.preserve_manual_edits(file_paths)
            self.console.print(f"[green]已保护 {preserved_count} 处手动编辑[/green]")

            return {
                "status": "success",
                "preserved_count": preserved_count
            }

        except Exception as e:
            self.console.print(f"[red]同步手动编辑时出错: {e}[/red]")
            return {"status": "error", "error": str(e)}

    # ===================
    # 状态查询
    # ===================

    def get_file_status(self, file_path: str) -> Dict:
        """
        获取文件的处理状态

        Args:
            file_path: 文件路径

        Returns:
            Dict: 状态信息
                {
                    "l0_ready": bool,  # 是否可以 Naive RAG 搜索
                    "l1_ready": bool,  # 是否图谱已构建
                    "entities_count": int,
                    "relationships_count": int
                }
        """
        # L0 状态：检查 Chunk 是否已向量化
        l0_ready = self.fast_pipeline.check_file_ready_for_search(file_path)

        # L1 状态：检查图谱是否已构建
        graph_status = self.slow_pipeline.check_file_graph_ready(file_path)

        return {
            "l0_ready": l0_ready,
            "l1_ready": graph_status["ready"],
            "entities_count": graph_status["entities_count"],
            "relationships_count": graph_status["relationships_count"]
        }

    def get_queue_status(self) -> Dict:
        """获取任务队列状态"""
        return self.task_queue.get_stats()

    def display_status(self):
        """显示完整状态信息"""
        self.console.print("\n[bold cyan]增量更新管理器状态[/bold cyan]")

        # 创建状态表格
        table = Table(title="处理统计")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")

        table.add_row("L0 文件处理数", str(self.stats["l0_files_processed"]))
        table.add_row("L1 文件处理数", str(self.stats["l1_files_processed"]))
        table.add_row("实体提取数", str(self.stats["entities_extracted"]))
        table.add_row("社区检测数", str(self.stats["communities_detected"]))
        table.add_row("错误数", str(self.stats["errors"]))

        self.console.print(table)

        # 显示队列状态
        self.console.print("\n")
        self.task_queue.print_stats()

    # ===================
    # 完整流程
    # ===================

    async def run_full_pipeline(self, file_paths: Optional[List[str]] = None) -> Dict:
        """
        执行完整的增量更新流程（L0 + L1）

        Args:
            file_paths: 文件路径列表

        Returns:
            Dict: 处理结果
        """
        start_time = time.time()

        self.console.print("\n[bold cyan]开始完整增量更新流程...[/bold cyan]")

        results = {}

        try:
            # 步骤 1: L0 快速摄取
            l0_result = await self.run_fast_ingestion(file_paths)
            results["l0"] = l0_result

            # 步骤 2: L1 任务提交
            l1_result = await self.run_deep_indexing(file_paths)
            results["l1"] = l1_result

            # 步骤 3: 验证图谱一致性（仅在有变更时）
            if l0_result.get("files_processed", 0) > 0:
                consistency_result = self.verify_graph_consistency()
                results["consistency"] = consistency_result

            # 步骤 4: 社区检测（仅在有变更时）
            if l0_result.get("files_processed", 0) > 0:
                community_result = self.detect_communities()
                results["community"] = community_result

            total_duration = time.time() - start_time

            self.console.print(
                f"[bold green]✅ 完整流程完成，总耗时: {total_duration:.2f}s[/bold green]"
            )

            results["total_duration"] = total_duration

            return results

        except Exception as e:
            self.console.print(f"[red]❌ 执行完整流程时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }


# ===================
# 便捷函数
# ===================

def quick_upload_file(file_path: str) -> Dict:
    """
    用户上传单个文件的快速处理（便捷函数）

    流程：
    1. L0 快速摄取（10秒内完成）
    2. 返回成功，用户可立即搜索
    3. 后台提交 L1 任务

    Args:
        file_path: 文件路径

    Returns:
        Dict: 处理结果
    """
    manager = IncrementalUpdateManagerV2()

    # L0 快速处理
    l0_result = manager.quick_ingest_single_file(file_path)

    if l0_result["status"] == "success":
        # L1 后台任务
        task_id = manager.submit_entity_extraction_task(
            file_path,
            priority=TaskPriority.HIGH
        )

        return {
            "status": "success",
            "l0_result": l0_result,
            "l1_task_id": task_id,
            "message": "文件已可搜索，图谱构建中..."
        }
    else:
        return l0_result


# ===================
# CLI 入口
# ===================

def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="增量更新管理器 V2")
    parser.add_argument("--mode", choices=["l0", "l1", "full"], default="full",
                        help="运行模式: l0(快速), l1(慢速), full(完整)")
    parser.add_argument("--file", type=str, help="处理单个文件")
    parser.add_argument("--status", action="store_true", help="显示状态")

    args = parser.parse_args()

    manager = IncrementalUpdateManagerV2()

    if args.status:
        manager.display_status()
    elif args.file:
        if args.mode == "l0":
            manager.quick_ingest_single_file(args.file)
        elif args.mode == "l1":
            manager.submit_entity_extraction_task(args.file)
        else:
            quick_upload_file(args.file)
    else:
        if args.mode == "l0":
            manager.run_fast_ingestion()
        elif args.mode == "l1":
            manager.run_deep_indexing()
        else:
            manager.run_full_pipeline()


if __name__ == "__main__":
    main()
