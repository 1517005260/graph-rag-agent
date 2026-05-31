import time
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.table import Table
from graphrag_agent.config.neo4jdb import get_db_manager

class GraphConsistencyValidator:
    """
    图谱一致性验证器，负责验证图谱结构和内容的完整性。
    
    主要功能：
    1. 检查孤立节点
    2. 验证关系链完整性
    3. 修复常见的一致性问题
    """
    
    def __init__(self):
        """初始化图谱一致性验证器"""
        self.console = Console()
        self.graph = get_db_manager().graph
        
        # 性能计时器
        self.validation_time = 0
        self.repair_time = 0
        
        # 验证统计
        self.validation_stats = {
            "orphan_entities": 0,
            "dangling_chunks": 0,
            "empty_chunks": 0,
            "broken_doc_links": 0,
            "broken_chunk_chains": 0,
            "total_issues": 0,
            "repaired_issues": 0
        }
    
    def check_orphan_entities(self) -> Tuple[List[str], int]:
        """
        检查孤立的实体节点（没有被任何Chunk引用）
        
        Returns:
            Tuple: 孤立实体ID列表和计数
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE NOT (e)<-[:MENTIONS]-()
          AND NOT e.manual_edit = true
          AND NOT e.protected = true
        RETURN e.id AS entity_id, count(e) AS count
        """
        
        result = self.graph.query(query)
        
        orphan_ids = []
        orphan_count = 0
        
        if result:
            orphan_count = result[0]["count"]
            # 获取最多1000个孤立实体ID
            id_query = """
            MATCH (e:`__Entity__`)
            WHERE NOT (e)<-[:MENTIONS]-()
              AND NOT e.manual_edit = true
              AND NOT e.protected = true
            RETURN e.id AS entity_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            orphan_ids = [r["entity_id"] for r in id_result]
        
        self.validation_stats["orphan_entities"] = orphan_count
        
        return orphan_ids, orphan_count
    
    def check_dangling_chunks(self) -> Tuple[List[str], int]:
        """
        检查悬空的Chunk节点（没有关联到Document）
        
        Returns:
            Tuple: 悬空Chunk ID列表和计数
        """
        query = """
        MATCH (c:`__Chunk__`)
        WHERE NOT (c)-[:PART_OF]->()
        RETURN c.id AS chunk_id, count(c) AS count
        """
        
        result = self.graph.query(query)
        
        dangling_ids = []
        dangling_count = 0
        
        if result:
            dangling_count = result[0]["count"]
            # 获取最多1000个悬空Chunk ID
            id_query = """
            MATCH (c:`__Chunk__`)
            WHERE NOT (c)-[:PART_OF]->()
            RETURN c.id AS chunk_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            dangling_ids = [r["chunk_id"] for r in id_result]
        
        self.validation_stats["dangling_chunks"] = dangling_count
        
        return dangling_ids, dangling_count
    
    def check_empty_chunks(self) -> Tuple[List[str], int]:
        """
        检查空的Chunk节点（没有文本内容）
        
        Returns:
            Tuple: 空Chunk ID列表和计数
        """
        query = """
        MATCH (c:`__Chunk__`)
        WHERE c.text IS NULL OR c.text = ''
        RETURN c.id AS chunk_id, count(c) AS count
        """
        
        result = self.graph.query(query)
        
        empty_ids = []
        empty_count = 0
        
        if result:
            empty_count = result[0]["count"]
            # 获取最多1000个空Chunk ID
            id_query = """
            MATCH (c:`__Chunk__`)
            WHERE c.text IS NULL OR c.text = ''
            RETURN c.id AS chunk_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            empty_ids = [r["chunk_id"] for r in id_result]
        
        self.validation_stats["empty_chunks"] = empty_count
        
        return empty_ids, empty_count
    
    def check_broken_doc_links(self) -> int:
        """
        检查文档链接是否完整（Document应该有FIRST_CHUNK关系）
        
        Returns:
            int: 有问题的文档数量
        """
        query = """
        MATCH (d:`__Document__`)
        WHERE NOT (d)-[:FIRST_CHUNK]->()
        RETURN count(d) AS count
        """
        
        result = self.graph.query(query)
        
        count = result[0]["count"] if result else 0
        self.validation_stats["broken_doc_links"] = count
        
        return count
    
    def check_broken_chunk_chains(self) -> int:
        """
        检查文本块链是否完整（前后关系）
        
        Returns:
            int: 有问题的链数量
        """
        query = """
        MATCH (c:`__Chunk__`)-[:PART_OF]->(d:`__Document__`)
        WHERE c.position > 1 AND NOT (c)<-[:NEXT_CHUNK]-()
        RETURN count(c) AS count
        """
        
        result = self.graph.query(query)
        
        count = result[0]["count"] if result else 0
        self.validation_stats["broken_chunk_chains"] = count
        
        return count
    
    def validate_graph(self) -> Dict[str, Any]:
        """
        执行全面的图谱验证
        
        Returns:
            Dict: 验证结果统计
        """
        start_time = time.time()
        
        # 1. 检查孤立实体
        orphan_ids, orphan_count = self.check_orphan_entities()
        if orphan_count > 0:
            self.console.print(f"[yellow]发现 {orphan_count} 个孤立实体节点[/yellow]")
        
        # 2. 检查悬空Chunk
        dangling_ids, dangling_count = self.check_dangling_chunks()
        if dangling_count > 0:
            self.console.print(f"[yellow]发现 {dangling_count} 个悬空Chunk节点[/yellow]")
        
        # 3. 检查空Chunk
        empty_ids, empty_count = self.check_empty_chunks()
        if empty_count > 0:
            self.console.print(f"[yellow]发现 {empty_count} 个空Chunk节点[/yellow]")
        
        # 4. 检查文档链接
        broken_doc_count = self.check_broken_doc_links()
        if broken_doc_count > 0:
            self.console.print(f"[yellow]发现 {broken_doc_count} 个没有FIRST_CHUNK关系的文档[/yellow]")
        
        # 5. 检查块链完整性
        broken_chain_count = self.check_broken_chunk_chains()
        if broken_chain_count > 0:
            self.console.print(f"[yellow]发现 {broken_chain_count} 个断开的Chunk链[/yellow]")
        
        # 计算总问题数
        total_issues = (orphan_count + dangling_count + empty_count + 
                       broken_doc_count + broken_chain_count)
        self.validation_stats["total_issues"] = total_issues
        
        # 计算验证时间
        self.validation_time = time.time() - start_time
        
        self.console.print(f"[blue]图谱验证完成，耗时: {self.validation_time:.2f}秒[/blue]")
        self.console.print(f"[blue]共发现 {total_issues} 个一致性问题[/blue]")
        
        return {
            "validation_time": self.validation_time,
            "validation_stats": self.validation_stats,
            "orphan_ids": orphan_ids,
            "dangling_ids": dangling_ids,
            "empty_ids": empty_ids
        }
    
    def repair_orphan_entities(self, orphan_ids: List[str] = None) -> int:
        """
        修复孤立实体节点（删除或标记）
        
        Args:
            orphan_ids: 要修复的孤立实体ID列表，如果为None则自动检测
            
        Returns:
            int: 修复的节点数量
        """
        if orphan_ids is None:
            orphan_ids, _ = self.check_orphan_entities()
        
        if not orphan_ids:
            return 0
        
        # 删除孤立实体
        delete_query = """
        UNWIND $orphan_ids AS entity_id
        MATCH (e:`__Entity__` {id: entity_id})
        WHERE NOT (e)<-[:MENTIONS]-()
          AND NOT e.manual_edit = true
          AND NOT e.protected = true
        DELETE e
        RETURN count(*) AS deleted
        """
        
        result = self.graph.query(delete_query, params={"orphan_ids": orphan_ids})
        
        deleted = result[0]["deleted"] if result else 0
        
        self.console.print(f"[green]已删除 {deleted} 个孤立实体节点[/green]")
        
        return deleted
    
    def repair_dangling_chunks(self, dangling_ids: List[str] = None) -> int:
        """
        修复悬空Chunk节点（删除）
        
        Args:
            dangling_ids: 要修复的悬空Chunk ID列表，如果为None则自动检测
            
        Returns:
            int: 修复的节点数量
        """
        if dangling_ids is None:
            dangling_ids, _ = self.check_dangling_chunks()
        
        if not dangling_ids:
            return 0
        
        # 删除悬空Chunk
        delete_query = """
        UNWIND $dangling_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        WHERE NOT (c)-[:PART_OF]->()
        DETACH DELETE c
        RETURN count(*) AS deleted
        """
        
        result = self.graph.query(delete_query, params={"dangling_ids": dangling_ids})
        
        deleted = result[0]["deleted"] if result else 0
        
        self.console.print(f"[green]已删除 {deleted} 个悬空Chunk节点[/green]")
        
        return deleted
    
    def repair_empty_chunks(self, empty_ids: List[str] = None) -> int:
        """
        修复空Chunk节点（添加占位符文本或删除）
        
        Args:
            empty_ids: 要修复的空Chunk ID列表，如果为None则自动检测
            
        Returns:
            int: 修复的节点数量
        """
        if empty_ids is None:
            empty_ids, _ = self.check_empty_chunks()
        
        if not empty_ids:
            return 0
        
        # 为空Chunk添加占位符文本
        repair_query = """
        UNWIND $empty_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        WHERE c.text IS NULL OR c.text = ''
        SET c.text = '[Empty Chunk]', c.repaired = true
        RETURN count(*) AS repaired
        """
        
        result = self.graph.query(repair_query, params={"empty_ids": empty_ids})
        
        repaired = result[0]["repaired"] if result else 0
        
        self.console.print(f"[green]已修复 {repaired} 个空Chunk节点[/green]")
        
        return repaired
    
    def repair_broken_doc_links(self) -> int:
        """
        修复断开的文档链接（创建缺失的FIRST_CHUNK关系）
        
        Returns:
            int: 修复的关系数量
        """
        repair_query = """
        MATCH (d:`__Document__`)
        WHERE NOT (d)-[:FIRST_CHUNK]->()
        
        MATCH (c:`__Chunk__`)-[:PART_OF]->(d)
        WHERE c.position = 1 OR c.position IS NULL
        
        WITH d, c ORDER BY c.position ASC LIMIT 1
        MERGE (d)-[r:FIRST_CHUNK]->(c)
        
        RETURN count(r) AS repaired
        """
        
        result = self.graph.query(repair_query)
        
        repaired = result[0]["repaired"] if result else 0
        
        self.console.print(f"[green]已修复 {repaired} 个断开的文档链接[/green]")
        
        return repaired
    
    def repair_broken_chunk_chains(self) -> int:
        """
        修复断开的Chunk链（重建NEXT_CHUNK关系）
        
        Returns:
            int: 修复的关系数量
        """
        repair_query = """
        MATCH (d:`__Document__`)
        WITH d
        MATCH (c1:`__Chunk__`)-[:PART_OF]->(d)
        WHERE c1.position IS NOT NULL
        WITH d, c1 ORDER BY c1.position ASC
        WITH d, collect(c1) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH d, chunks[i] AS current, chunks[i+1] AS next
        WHERE NOT (current)-[:NEXT_CHUNK]->(next)
        MERGE (current)-[r:NEXT_CHUNK]->(next)
        RETURN count(r) AS repaired
        """
        
        result = self.graph.query(repair_query)
        
        repaired = result[0]["repaired"] if result else 0
        
        self.console.print(f"[green]已修复 {repaired} 个断开的Chunk链[/green]")
        
        return repaired
    
    def repair_graph(self) -> Dict[str, Any]:
        """
        执行图谱修复操作
        
        Returns:
            Dict: 修复结果统计
        """
        start_time = time.time()
        
        # 先进行全面验证
        validation_result = self.validate_graph()
        
        # 根据验证结果进行修复
        repairs = {
            "orphan_entities": self.repair_orphan_entities(validation_result.get("orphan_ids", [])),
            "dangling_chunks": self.repair_dangling_chunks(validation_result.get("dangling_ids", [])),
            "empty_chunks": self.repair_empty_chunks(validation_result.get("empty_ids", [])),
            "broken_doc_links": self.repair_broken_doc_links(),
            "broken_chunk_chains": self.repair_broken_chunk_chains()
        }
        
        # 计算总修复数量
        total_repaired = sum(repairs.values())
        self.validation_stats["repaired_issues"] = total_repaired
        
        # 计算修复时间
        self.repair_time = time.time() - start_time
        
        self.console.print(f"[blue]图谱修复完成，耗时: {self.repair_time:.2f}秒[/blue]")
        self.console.print(f"[blue]共修复 {total_repaired} 个一致性问题[/blue]")
        
        return {
            "validation_time": self.validation_time,
            "repair_time": self.repair_time,
            "validation_stats": self.validation_stats,
            "repairs": repairs
        }
    
    def display_graph_stats(self):
        """显示图谱统计信息"""
        # 获取图谱节点和关系统计
        stats_query = """
        MATCH (n)
        RETURN 
            count(n) AS total_nodes,
            sum(CASE WHEN n:`__Document__` THEN 1 ELSE 0 END) AS doc_count,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunk_count,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entity_count
        """
        
        stats_result = self.graph.query(stats_query)
        
        if not stats_result:
            self.console.print("[yellow]无法获取图谱统计信息[/yellow]")
            return
        
        node_stats = stats_result[0]
        
        # 获取关系统计
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """
        
        rel_result = self.graph.query(rel_query)
        
        # 显示节点统计表
        node_table = Table(title="图谱节点统计")
        node_table.add_column("节点类型", style="cyan")
        node_table.add_column("数量", justify="right")
        
        node_table.add_row("__Document__", str(node_stats["doc_count"]))
        node_table.add_row("__Chunk__", str(node_stats["chunk_count"]))
        node_table.add_row("__Entity__", str(node_stats["entity_count"]))
        node_table.add_row("总计", str(node_stats["total_nodes"]), style="bold")
        
        self.console.print(node_table)
        
        # 显示关系统计表
        if rel_result:
            rel_table = Table(title="图谱关系统计")
            rel_table.add_column("关系类型", style="cyan")
            rel_table.add_column("数量", justify="right")
            
            total_rels = 0
            for rel in rel_result:
                rel_table.add_row(rel["rel_type"], str(rel["count"]))
                total_rels += rel["count"]
                
            rel_table.add_row("总计", str(total_rels), style="bold")
            
            self.console.print(rel_table)
    
    def process(self, repair: bool = True) -> Dict[str, Any]:
        """
        执行完整的图谱一致性验证和修复流程
        
        Args:
            repair: 是否执行修复操作
            
        Returns:
            Dict: 处理结果统计
        """
        try:
            # 显示图谱基本统计
            self.display_graph_stats()
            
            # 验证图谱一致性
            validation_result = self.validate_graph()
            
            # 如果需要修复并且存在问题，执行修复
            if repair and validation_result["validation_stats"]["total_issues"] > 0:
                repair_result = self.repair_graph()
                return {
                    "validation_result": validation_result,
                    "repair_result": repair_result,
                    "total_time": self.validation_time + self.repair_time
                }
            
            return {
                "validation_result": validation_result,
                "total_time": self.validation_time
            }
            
        except Exception as e:
            self.console.print(f"[red]图谱一致性验证过程中出现错误: {e}[/red]")
            raise