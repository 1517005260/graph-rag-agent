"""
L1 慢速图谱构建管道 - 实体提取与知识图谱构建

在后台异步执行：
1. 实体与关系提取
2. 实体消歧与对齐
3. 图谱写入 Neo4j
4. 实体向量化

特性：利用流式处理，边提取边写入
"""
import time
from typing import List, Dict, Optional
from rich.console import Console

from graphrag_agent.graph.extraction.entity_extractor import EntityRelationExtractor
from graphrag_agent.graph.indexing.embedding_manager import EmbeddingManager
from graphrag_agent.integrations.build.pipeline.task_queue import Task, TaskStatus
from graphrag_agent.config.settings import (
    BATCH_SIZE,
    MAX_WORKERS,
    entity_types,
    relationship_types
)


class SlowGraphPipeline:
    """
    L1 慢速图谱构建管道

    职责：
    1. 从文件 Chunk 中提取实体和关系
    2. 实体消歧与对齐
    3. 构建知识图谱
    4. 生成实体 Embedding

    优化：
    - 利用 stream_process_large_files 流式写入
    - 支持任务队列异步执行
    """

    def __init__(self, user_examples: Optional[List[str]] = None):
        """
        初始化慢速图谱管道

        Args:
            user_examples: 用户修正的示例（用于 Few-Shot Prompt）
        """
        self.console = Console()

        # 初始化实体提取器
        self.entity_extractor = EntityRelationExtractor(
            entity_types=entity_types,
            relationship_types=relationship_types,
            # user_examples=user_examples  # TODO: 需要在 EntityRelationExtractor 中实现
        )

        # 初始化 Embedding 管理器
        self.embedding_manager = EmbeddingManager(
            batch_size=BATCH_SIZE,
            max_workers=MAX_WORKERS
        )

    def process_file_entity_extraction(self, file_path: str, task: Optional[Task] = None) -> Dict:
        """
        处理单个文件的实体提取

        Args:
            file_path: 文件路径
            task: 任务对象（用于状态更新）

        Returns:
            Dict: 处理结果
                {
                    "status": "success" | "error",
                    "file_path": str,
                    "entities_extracted": int,
                    "relationships_extracted": int,
                    "duration": float,
                    "error": str (可选)
                }
        """
        start_time = time.time()

        self.console.print(f"[bold magenta][L1] 实体提取: {file_path}[/bold magenta]")

        try:
            # 更新任务状态
            if task:
                task.metadata["stage"] = "entity_extraction"

            # 步骤 1: 获取文件对应的 Chunks
            self.console.print("[magenta]  → 步骤 1/3: 加载文本块...[/magenta]")
            chunks = self._load_chunks_for_file(file_path)
            self.console.print(f"[green]  ✓ 加载 {len(chunks)} 个文本块[/green]")

            # 步骤 2: 实体提取（流式）
            self.console.print("[magenta]  → 步骤 2/3: 实体提取（流式）...[/magenta]")

            extraction_start = time.time()

            # 使用流式处理（如果文件大）
            if len(chunks) > 50:  # 大文件使用流式
                entities, relationships = self._stream_extract_entities(chunks, file_path)
            else:  # 小文件使用批量
                entities, relationships = self._batch_extract_entities(chunks, file_path)

            extraction_duration = time.time() - extraction_start

            self.console.print(
                f"[green]  ✓ 提取 {len(entities)} 个实体, "
                f"{len(relationships)} 个关系 "
                f"({extraction_duration:.2f}s)[/green]"
            )

            # 步骤 3: 生成实体 Embedding
            self.console.print("[magenta]  → 步骤 3/3: 生成实体向量...[/magenta]")

            embed_start = time.time()
            embedded_count = self.embedding_manager.update_entity_embeddings()
            embed_duration = time.time() - embed_start

            self.console.print(
                f"[green]  ✓ 向量化 {embedded_count} 个实体 "
                f"({embed_duration:.2f}s)[/green]"
            )

            total_duration = time.time() - start_time

            self.console.print(
                f"[bold green]✅ L1 处理完成，总耗时: {total_duration:.2f}s[/bold green]"
            )

            return {
                "status": "success",
                "file_path": file_path,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "entities_vectorized": embedded_count,
                "duration": total_duration
            }

        except Exception as e:
            error_msg = str(e)
            self.console.print(f"[red]❌ L1 处理失败: {error_msg}[/red]")

            # 更新任务状态
            if task:
                task.status = TaskStatus.FAILED
                task.error = error_msg

            return {
                "status": "error",
                "file_path": file_path,
                "error": error_msg,
                "duration": time.time() - start_time
            }

    def _load_chunks_for_file(self, file_path: str) -> List[Dict]:
        """
        加载文件对应的所有 Chunks

        Args:
            file_path: 文件路径

        Returns:
            List[Dict]: Chunk列表
        """
        # 从 Neo4j 查询文件对应的所有 Chunk
        # 这需要一个查询接口

        # 临时实现：调用 entity_extractor 的方法
        # 假设 entity_extractor 有一个方法可以获取 chunks

        # TODO: 实现从 Neo4j 查询 chunks
        # 目前先返回空列表或模拟数据
        import os
        from graphrag_agent.config.neo4jdb import get_db_manager

        graph = get_db_manager().graph

        # Cypher 查询：获取文件对应的所有 Chunk
        file_name = os.path.basename(file_path)

        query = """
        MATCH (c:Chunk)
        WHERE c.file_name = $file_name
        RETURN c.chunk_id AS chunk_id,
               c.text AS text,
               c.file_name AS file_name,
               c.chunk_index AS chunk_index
        ORDER BY c.chunk_index
        """

        result = graph.query(query, params={"file_name": file_name})

        chunks = [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "file_name": row["file_name"],
                "chunk_index": row["chunk_index"]
            }
            for row in result
        ]

        return chunks

    def _stream_extract_entities(self, chunks: List[Dict], file_path: str) -> tuple:
        """
        流式提取实体（边提取边写入）

        利用 EntityRelationExtractor.stream_process_large_files

        Args:
            chunks: Chunk列表
            file_path: 文件路径

        Returns:
            tuple: (entities, relationships)
        """
        self.console.print("[blue]    使用流式提取模式（大文件优化）[/blue]")

        # 调用流式处理方法
        # 注意：stream_process_large_files 需要文件路径或 chunks
        # 需要确认其接口

        # 假设接口是: stream_process_large_files(file_path) -> (entities, relationships)
        if hasattr(self.entity_extractor, 'stream_process_large_files'):
            entities, relationships = self.entity_extractor.stream_process_large_files(file_path)
        else:
            # 如果没有流式接口，回退到批量处理
            self.console.print("[yellow]    流式接口不可用，回退到批量处理[/yellow]")
            entities, relationships = self._batch_extract_entities(chunks, file_path)

        return entities, relationships

    def _batch_extract_entities(self, chunks: List[Dict], file_path: str) -> tuple:
        """
        批量提取实体

        Args:
            chunks: Chunk列表
            file_path: 文件路径

        Returns:
            tuple: (entities, relationships)
        """
        self.console.print("[blue]    使用批量提取模式[/blue]")

        # 调用批量处理方法
        # 注意：process_chunks_batch 的接口

        if hasattr(self.entity_extractor, 'process_chunks_batch'):
            entities, relationships = self.entity_extractor.process_chunks_batch(
                chunks=chunks,
                batch_size=BATCH_SIZE
            )
        else:
            # 兜底：逐个处理
            entities = []
            relationships = []

            for i, chunk in enumerate(chunks):
                self.console.print(f"    处理 Chunk {i+1}/{len(chunks)}...", end="\r")

                # 假设有单个 chunk 处理方法
                chunk_entities, chunk_relationships = self._process_single_chunk(chunk)

                entities.extend(chunk_entities)
                relationships.extend(chunk_relationships)

            print()  # 换行

        return entities, relationships

    def _process_single_chunk(self, chunk: Dict) -> tuple:
        """
        处理单个 Chunk（兜底方法）

        Args:
            chunk: Chunk数据

        Returns:
            tuple: (entities, relationships)
        """
        # 这是最基础的处理逻辑
        # 实际应该调用 EntityRelationExtractor 的方法

        # TODO: 实现单 chunk 处理
        # 临时返回空列表
        return [], []

    def check_file_graph_ready(self, file_path: str) -> Dict:
        """
        检查文件的图谱是否已构建完成

        Args:
            file_path: 文件路径

        Returns:
            Dict: 状态信息
                {
                    "ready": bool,
                    "entities_count": int,
                    "relationships_count": int
                }
        """
        import os
        from graphrag_agent.config.neo4jdb import get_db_manager

        graph = get_db_manager().graph
        file_name = os.path.basename(file_path)

        # 查询该文件对应的实体和关系数量
        query = """
        MATCH (c:Chunk {file_name: $file_name})-[:HAS_ENTITY]->(e:Entity)
        WITH count(DISTINCT e) as entity_count
        MATCH (c:Chunk {file_name: $file_name})-[:HAS_ENTITY]->(e1:Entity)-[r]-(e2:Entity)
        RETURN entity_count, count(DISTINCT r) as relationship_count
        """

        result = graph.query(query, params={"file_name": file_name})

        if result:
            row = result[0]
            return {
                "ready": row["entity_count"] > 0,
                "entities_count": row["entity_count"],
                "relationships_count": row["relationship_count"]
            }
        else:
            return {
                "ready": False,
                "entities_count": 0,
                "relationships_count": 0
            }


# 任务处理器（用于任务队列）
def entity_extraction_task_handler(task: Task):
    """
    实体提取任务处理器

    Args:
        task: 任务对象
    """
    pipeline = SlowGraphPipeline()
    result = pipeline.process_file_entity_extraction(task.file_path, task)

    # 更新任务元数据
    task.metadata.update(result)


# 便捷函数
def extract_entities_from_file(file_path: str) -> Dict:
    """
    从文件提取实体（便捷函数）

    Args:
        file_path: 文件路径

    Returns:
        Dict: 处理结果
    """
    pipeline = SlowGraphPipeline()
    return pipeline.process_file_entity_extraction(file_path)
