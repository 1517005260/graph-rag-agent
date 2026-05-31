import os
import time
import signal
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from incremental_graph_builder import IncrementalGraphUpdater
from graphrag_agent.graph.graph_consistency_validator import GraphConsistencyValidator
from graphrag_agent.integrations.build.incremental.manual_edit_manager import ManualEditManager
from graphrag_agent.graph.indexing.embedding_manager import EmbeddingManager
from graphrag_agent.community import CommunityDetectorFactory, CommunitySummarizerFactory
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import FILES_DIR, community_algorithm, MAX_WORKERS, BATCH_SIZE, NEO4J_CONFIG
from graphrag_agent.integrations.build.incremental.incremental_update_scheduler import IncrementalUpdateScheduler

class IncrementalUpdateManager:
    """
    增量更新管理器，整合所有增量更新功能，支持后台运行和定期检测。
    
    主要功能：
    1. 检测文件变更并更新图谱
    2. 更新实体和Chunk的Embedding
    3. 验证图谱一致性
    4. 处理社区检测和摘要生成
    5. 支持手动编辑同步
    6. 后台运行和定时调度
    """
    
    def __init__(self, files_dir: str = FILES_DIR, config=None):
        """
        初始化增量更新管理器
        
        Args:
            files_dir: 监控的文件目录
            config: 配置参数
        """
        self.console = Console()
        
        # 配置参数
        self.files_dir = files_dir
        self.config = config or {}
        
        # 初始化子组件
        self.graph = get_db_manager().graph
        self.updater = IncrementalGraphUpdater(files_dir)
        self.validator = GraphConsistencyValidator()
        self.edit_manager = ManualEditManager()
        self.embedding_manager = EmbeddingManager(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
        
        # 初始化调度器
        self.scheduler = IncrementalUpdateScheduler(self.config)
        
        # 运行状态
        self.running = False
        self.stop_event = None
        
        # 性能统计
        self.stats = {
            "updates_performed": 0,
            "files_processed": 0,
            "entities_updated": 0,
            "communities_detected": 0,
            "errors": 0
        }
    
    def detect_file_changes(self):
        """
        检测文件变更并更新图谱
        
        Returns:
            Dict: 变更信息
        """
        self.console.print("[bold cyan]检测文件变更...[/bold cyan]")
        
        try:
            # 检测变更
            changes = self.updater.detect_changes()
            
            # 更新统计信息
            added_count = len(changes.get("added", []))
            modified_count = len(changes.get("modified", []))
            deleted_count = len(changes.get("deleted", []))
            total_changed = added_count + modified_count + deleted_count
            
            if total_changed > 0:
                self.console.print(f"[green]检测到 {total_changed} 个文件变更：[/green]")
                self.console.print(f"[green]新增: {added_count}, 修改: {modified_count}, 删除: {deleted_count}[/green]")
                
                # 如果有变更，执行增量更新
                self.updater.process_incremental_update()
                
                # 如果有文件删除，执行图一致性检查
                if deleted_count > 0:
                    self.verify_graph_consistency()
                
                # 更新统计信息
                self.stats["updates_performed"] += 1
                self.stats["files_processed"] += total_changed
            else:
                self.console.print("[yellow]未检测到文件变更[/yellow]")
            
            return changes
            
        except Exception as e:
            self.console.print(f"[red]检测文件变更时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def update_entity_embeddings(self):
        """
        更新实体Embedding
        
        Returns:
            int: 更新的实体数量
        """
        self.console.print("[bold cyan]更新实体Embedding...[/bold cyan]")
        
        try:
            # 获取需要更新的实体
            entities = self.embedding_manager.get_entities_needing_update()
            
            if entities:
                self.console.print(f"[green]发现 {len(entities)} 个需要更新Embedding的实体[/green]")
                
                # 执行更新
                updated_count = self.embedding_manager.update_entity_embeddings()
                
                # 更新统计信息
                self.stats["entities_updated"] += updated_count
                
                return updated_count
            else:
                self.console.print("[yellow]没有需要更新Embedding的实体[/yellow]")
                return 0
                
        except Exception as e:
            self.console.print(f"[red]更新实体Embedding时出错: {e}[/red]")
            self.stats["errors"] += 1
            return 0
    
    def update_chunk_embeddings(self):
        """
        更新Chunk Embedding
        
        Returns:
            int: 更新的Chunk数量
        """
        self.console.print("[bold cyan]更新Chunk Embedding...[/bold cyan]")
        
        try:
            # 获取需要更新的Chunk
            chunks = self.embedding_manager.get_chunks_needing_update()
            
            if chunks:
                self.console.print(f"[green]发现 {len(chunks)} 个需要更新Embedding的Chunk[/green]")
                
                # 执行更新
                updated_count = self.embedding_manager.update_chunk_embeddings()
                
                return updated_count
            else:
                self.console.print("[yellow]没有需要更新Embedding的Chunk[/yellow]")
                return 0
                
        except Exception as e:
            self.console.print(f"[red]更新Chunk Embedding时出错: {e}[/red]")
            self.stats["errors"] += 1
            return 0
    
    def verify_graph_consistency(self, repair=True):
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
                # 执行验证和修复
                result = self.validator.repair_graph()
                
                # 显示修复结果
                repaired_count = result["validation_stats"]["repaired_issues"]
                self.console.print(f"[green]图谱一致性验证完成，修复了 {repaired_count} 个问题[/green]")
            else:
                # 仅执行验证
                result = self.validator.validate_graph()
                
                # 显示验证结果
                issues_count = result["validation_stats"]["total_issues"]
                self.console.print(f"[green]图谱一致性验证完成，发现 {issues_count} 个问题[/green]")
            
            return result
            
        except Exception as e:
            self.console.print(f"[red]验证图谱一致性时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def detect_communities(self):
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
                
                # 执行社区摘要生成
                self.console.print("[blue]开始生成社区摘要...[/blue]")
                summarizer = CommunitySummarizerFactory.create_summarizer(
                    community_algorithm,
                    graph
                )
                summaries = summarizer.process_communities()
                
                self.console.print(f"[green]社区摘要生成完成，共生成 {len(summaries) if summaries else 0} 个摘要[/green]")
                
                return {
                    "status": "success", 
                    "communities": community_count,
                    "summaries": len(summaries) if summaries else 0
                }
            else:
                self.console.print(f"[yellow]社区检测失败: {detection_result.get('message', '未知错误')}[/yellow]")
                return detection_result
                
        except Exception as e:
            self.console.print(f"[red]执行社区检测时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"status": "error", "message": str(e)}
    
    def sync_manual_edits(self, changed_files=None):
        """
        同步手动编辑
        
        Args:
            changed_files: 变更的文件列表
            
        Returns:
            Dict: 同步结果
        """
        self.console.print("[bold cyan]同步手动编辑...[/bold cyan]")
        
        try:
            # 如果没有提供变更文件列表，获取所有变更
            if changed_files is None:
                changes = self.updater.detect_changes()
                changed_files = changes.get("added", []) + changes.get("modified", [])
            
            if changed_files:
                # 执行手动编辑同步
                result = self.edit_manager.process(changed_files)
                
                return result
            else:
                self.console.print("[yellow]没有变更的文件，跳过手动编辑同步[/yellow]")
                return {"status": "skipped"}
                
        except Exception as e:
            self.console.print(f"[red]同步手动编辑时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def check_manual_edits(self):
        """
        检查Neo4j中的手动编辑，并确保这些编辑在增量更新中被保留
        
        Returns:
            Dict: 手动编辑的统计信息
        """
        self.console.print("[bold cyan]检查手动编辑...[/bold cyan]")
        
        try:
            # 使用ManualEditManager检测手动编辑
            edit_stats = self.edit_manager.detect_manual_edits()
            
            # 输出手动编辑统计
            manual_entities = edit_stats.get("manual_entities", 0)
            manual_relations = edit_stats.get("manual_relations", 0)
            
            if manual_entities > 0 or manual_relations > 0:
                self.console.print(f"[green]检测到 {manual_entities} 个手动编辑的实体和 {manual_relations} 个手动编辑的关系[/green]")
                
                # 确保增量更新时保留这些手动编辑
                changes = self.updater.detect_changes()
                changed_files = []
                if changes:
                    changed_files = changes.get("added", []) + changes.get("modified", [])
                
                # 应用手动编辑保护
                if changed_files:
                    preserved_count = self.edit_manager.preserve_manual_edits(changed_files)
                    self.console.print(f"[green]已保护 {preserved_count} 个手动编辑，确保增量更新不会覆盖它们[/green]")
                
                return {
                    "manual_entities": manual_entities,
                    "manual_relations": manual_relations,
                    "preserved_edits": preserved_count if changed_files else 0
                }
            else:
                self.console.print("[blue]没有检测到手动编辑[/blue]")
                return {
                    "manual_entities": 0,
                    "manual_relations": 0
                }
                
        except Exception as e:
            self.console.print(f"[red]检查手动编辑时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def run_once(self):
        """
        执行一次完整的增量更新流程
        
        Returns:
            Dict: 更新结果
        """
        start_time = time.time()
        
        self.console.print("\n[bold cyan]开始执行增量更新流程...[/bold cyan]")
        
        results = {}
        
        try:
            # 1. 检测文件变更
            changes = self.detect_file_changes()
            results["file_changes"] = changes
            
            # 2. 更新实体Embedding
            entity_updates = self.update_entity_embeddings()
            results["entity_updates"] = entity_updates
            
            # 3. 更新Chunk Embedding
            chunk_updates = self.update_chunk_embeddings()
            results["chunk_updates"] = chunk_updates
            
            # 4. 验证图谱一致性
            consistency_check = self.verify_graph_consistency()
            results["consistency_check"] = consistency_check
            
            # 5. 同步手动编辑
            if changes and (changes.get("added") or changes.get("modified")):
                edit_sync = self.sync_manual_edits(
                    changes.get("added", []) + changes.get("modified", [])
                )
                results["edit_sync"] = edit_sync
            
            # 6. 执行社区检测
            # 只有当文件有变更时才执行社区检测，避免不必要的计算
            if changes and (changes.get("added") or changes.get("modified") or changes.get("deleted")):
                community_detection = self.detect_communities()
                results["community_detection"] = community_detection
            
            # 计算总耗时
            end_time = time.time()
            total_time = end_time - start_time
            
            self.console.print(f"[bold green]增量更新流程完成，总耗时: {total_time:.2f}秒[/bold green]")
            
            return results
            
        except Exception as e:
            self.console.print(f"[red]执行增量更新流程时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def start_scheduler(self):
        """
        启动调度器，开始后台运行增量更新流程
        """
        self.console.print("[bold cyan]启动增量更新调度器...[/bold cyan]")
        
        # 注册处理方法
        self.scheduler.schedule_component("file_change", self.detect_file_changes)
        self.scheduler.schedule_component("entity_embedding", self.update_entity_embeddings)
        self.scheduler.schedule_component("chunk_embedding", self.update_chunk_embeddings)
        self.scheduler.schedule_component("graph_consistency", self.verify_graph_consistency)
        self.scheduler.schedule_component("community_detection", self.detect_communities)
        self.scheduler.schedule_component("manual_edit_check", self.check_manual_edits)
        
        # 启动调度器
        self.stop_event = self.scheduler.start()
        self.running = True
        
        self.console.print("[green]增量更新调度器已启动，正在后台运行...[/green]")
        
        # 显示调度器状态
        self.scheduler.print_status()
    
    def stop_scheduler(self):
        """停止调度器"""
        if self.running and self.stop_event:
            self.scheduler.stop(self.stop_event)
            self.running = False
            self.stop_event = None
            
            self.console.print("[yellow]增量更新调度器已停止[/yellow]")
        else:
            self.console.print("[yellow]调度器未运行[/yellow]")
    
    def display_stats(self):
        """显示统计信息"""
        self.console.print("\n[bold cyan]增量更新统计信息[/bold cyan]")
        self.console.print(f"[blue]执行的更新次数: {self.stats['updates_performed']}[/blue]")
        self.console.print(f"[blue]处理的文件数: {self.stats['files_processed']}[/blue]")
        self.console.print(f"[blue]更新的实体数: {self.stats['entities_updated']}[/blue]")
        self.console.print(f"[blue]检测的社区数: {self.stats['communities_detected']}[/blue]")
        self.console.print(f"[blue]错误数: {self.stats['errors']}[/blue]")
        
        if self.running:
            self.scheduler.print_status()
    
    def signal_handler(self, sig, frame):
        """
        信号处理函数，用于处理终止信号
        
        Args:
            sig: 信号
            frame: 帧
        """
        self.console.print("\n[yellow]正在退出...[/yellow]")
        
        if self.running:
            self.stop_scheduler()
            
        self.display_stats()
        
        self.console.print("[green]增量更新管理器已安全退出[/green]")
        exit(0)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="增量更新管理器")
    parser.add_argument("--dir", type=str, default=FILES_DIR, help="监控的文件目录")
    parser.add_argument("--once", action="store_true", help="执行一次更新后退出")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式运行")
    parser.add_argument("--interval", type=int, default=300, help="检查间隔（秒）")
    parser.add_argument("--community-interval", type=int, default=1800, help="社区检测间隔（秒）")
    parser.add_argument("--manual-check-interval", type=int, default=900, help="手动编辑检查间隔（秒）")
    args = parser.parse_args()
    
    # 创建控制台
    console = Console()
    
    # 显示启动信息
    start_text = Text("启动增量更新管理器", style="bold cyan")
    console.print(Panel(start_text, border_style="cyan"))
    
    # 配置参数
    config = {
        "file_change_threshold": args.interval,
        "entity_embedding_threshold": args.interval * 2,
        "chunk_embedding_threshold": args.interval * 2,
        "graph_consistency_threshold": args.interval * 6,
        "community_detection_threshold": args.community_interval,
        "manual_edit_check_threshold": args.manual_check_interval
    }
    
    # 初始化管理器
    manager = IncrementalUpdateManager(args.dir, config)
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    try:
        if args.once:
            # 执行一次更新后退出
            console.print("[cyan]执行一次更新后退出...[/cyan]")
            manager.run_once()
        else:
            # 启动调度器
            manager.start_scheduler()
            
            if args.daemon:
                # 以守护进程模式运行
                console.print("[cyan]以守护进程模式运行，按Ctrl+C终止...[/cyan]")
                
                while True:
                    time.sleep(60)
            else:
                # 交互模式
                console.print("[cyan]增量更新管理器已启动，输入 'exit' 退出[/cyan]")
                
                while True:
                    cmd = input(">>> ").strip().lower()
                    
                    if cmd == "exit":
                        manager.stop_scheduler()
                        break
                    elif cmd == "stats":
                        manager.display_stats()
                    elif cmd == "run":
                        manager.run_once()
                    elif cmd == "help":
                        console.print("[blue]命令列表:[/blue]")
                        console.print("[blue]  exit: 退出程序[/blue]")
                        console.print("[blue]  stats: 显示统计信息[/blue]")
                        console.print("[blue]  run: 执行一次更新[/blue]")
                        console.print("[blue]  help: 显示帮助[/blue]")
                    else:
                        console.print("[yellow]未知命令，输入 'help' 获取帮助[/yellow]")
    finally:
        # 确保调度器被停止
        if manager.running:
            manager.stop_scheduler()
            
        # 显示统计信息
        manager.display_stats()
        
        # 显示结束信息
        end_text = Text("增量更新管理器已退出", style="bold green")
        console.print(Panel(end_text, border_style="green"))

if __name__ == "__main__":
    main()
