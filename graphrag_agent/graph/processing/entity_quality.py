import time
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

from graphrag_agent.graph.processing.entity_disambiguation import EntityDisambiguator
from graphrag_agent.graph.processing.entity_alignment import EntityAligner

class EntityQualityProcessor:
    """
    实体质量处理器：整合消歧和对齐流程
    在相似实体检测和合并之后运行，进一步提升实体质量
    """
    
    def __init__(self):
        self.console = Console()
        self.disambiguator = EntityDisambiguator()
        self.aligner = EntityAligner()
        
        # 性能统计
        self.stats = {
            'total_time': 0,
            'disambig_time': 0,
            'align_time': 0
        }
    
    def process(self) -> Dict[str, Any]:
        """
        执行完整的实体质量提升流程
        
        Returns:
            Dict: 处理结果统计
        """
        start_time = time.time()
        
        self.console.print("[bold cyan]开始实体质量提升流程[/bold cyan]")
        
        # 1. 实体消歧
        self.console.print("\n[cyan]阶段1: 实体消歧[/cyan]")
        disambig_start = time.time()
        
        disambiguated = self.disambiguator.apply_to_graph()
        
        self.stats['disambig_time'] = time.time() - disambig_start
        
        self.console.print(f"[green]消歧完成: 处理了 {disambiguated} 个实体[/green]")
        
        # 显示消歧统计
        self._display_stats_table("消歧统计", {
            '设置canonical_id的实体数': disambiguated,
            '耗时': f"{self.stats['disambig_time']:.2f}秒"
        })
        
        # 2. 实体对齐
        self.console.print("\n[cyan]阶段2: 实体对齐[/cyan]")
        align_start = time.time()
        
        align_result = self.aligner.align_all()
        
        self.stats['align_time'] = time.time() - align_start
        
        self.console.print(f"[green]对齐完成: 合并了 {align_result['entities_aligned']} 个实体[/green]")
        
        # 显示对齐统计
        self._display_stats_table("对齐统计", {
            '处理的分组数': align_result['groups_processed'],
            '检测到的冲突': align_result['conflicts_detected'],
            '合并的实体数': align_result['entities_aligned'],
            '耗时': f"{self.stats['align_time']:.2f}秒"
        })
        
        # 总计
        self.stats['total_time'] = time.time() - start_time
        
        self.console.print(f"\n[bold green]实体质量提升完成，总耗时: {self.stats['total_time']:.2f}秒[/bold green]")
        
        return {
            'disambiguated': disambiguated,
            'aligned': align_result['entities_aligned'],
            'stats': self.stats
        }
    
    def _display_stats_table(self, title: str, data: Dict[str, Any]):
        """显示统计表格"""
        table = Table(title=title)
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right")
        
        for key, value in data.items():
            table.add_row(str(key), str(value))
        
        self.console.print(table)