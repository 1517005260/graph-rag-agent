"""
L0 快速摄取管道 - 文本分块与向量化

用户上传文件后，10秒内完成：
1. 文件读取与文本提取
2. 文本分块 (Chunking)
3. 向量化 (Embedding)
4. 写入向量数据库

目标：让用户立即使用 Naive RAG 搜索
"""
import time
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console

from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
from graphrag_agent.graph.indexing.embedding_manager import EmbeddingManager
from graphrag_agent.config.settings import FILES_DIR, BATCH_SIZE, MAX_WORKERS


class FastIngestionPipeline:
    """
    L0 快速摄取管道

    职责：
    1. 快速处理文件 → Chunk
    2. 快速生成 Embedding
    3. 快速写入向量数据库

    不做：
    - 实体提取（慢）
    - 图谱构建（慢）
    - 社区检测（很慢）
    """

    def __init__(self, files_dir: str = FILES_DIR):
        """
        初始化快速摄取管道

        Args:
            files_dir: 文件目录
        """
        self.console = Console()
        self.files_dir = files_dir

        # 初始化组件
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(
            batch_size=BATCH_SIZE,
            max_workers=MAX_WORKERS
        )

    def process_single_file(self, file_path: str, force: bool = False) -> Dict:
        """
        快速处理单个文件

        Args:
            file_path: 文件路径
            force: 是否强制重新处理

        Returns:
            Dict: 处理结果
                {
                    "status": "success" | "error",
                    "file_path": str,
                    "chunks_created": int,
                    "chunks_vectorized": int,
                    "duration": float,
                    "error": str (可选)
                }
        """
        start_time = time.time()

        self.console.print(f"[bold cyan][L0] 快速处理文件: {file_path}[/bold cyan]")

        try:
            # 步骤 1: 文本提取与分块
            self.console.print("[cyan]  → 步骤 1/2: 文本提取与分块...[/cyan]")
            chunk_start = time.time()

            # 调用文档处理器
            chunks = self._extract_and_chunk(file_path)

            chunk_duration = time.time() - chunk_start
            self.console.print(
                f"[green]  ✓ 生成 {len(chunks)} 个文本块 "
                f"({chunk_duration:.2f}s)[/green]"
            )

            # 步骤 2: 向量化
            self.console.print("[cyan]  → 步骤 2/2: 向量化与写入...[/cyan]")
            embed_start = time.time()

            # 批量生成并写入向量
            vectorized_count = self._vectorize_and_store(chunks, file_path)

            embed_duration = time.time() - embed_start
            self.console.print(
                f"[green]  ✓ 向量化 {vectorized_count} 个块 "
                f"({embed_duration:.2f}s)[/green]"
            )

            total_duration = time.time() - start_time
            self.console.print(
                f"[bold green]✅ L0 处理完成，总耗时: {total_duration:.2f}s[/bold green]"
            )

            return {
                "status": "success",
                "file_path": file_path,
                "chunks_created": len(chunks),
                "chunks_vectorized": vectorized_count,
                "duration": total_duration
            }

        except Exception as e:
            error_msg = str(e)
            self.console.print(f"[red]❌ L0 处理失败: {error_msg}[/red]")

            return {
                "status": "error",
                "file_path": file_path,
                "error": error_msg,
                "duration": time.time() - start_time
            }

    def process_batch_files(self, file_paths: List[str]) -> List[Dict]:
        """
        批量快速处理多个文件

        Args:
            file_paths: 文件路径列表

        Returns:
            List[Dict]: 处理结果列表
        """
        self.console.print(
            f"[bold cyan][L0] 批量处理 {len(file_paths)} 个文件[/bold cyan]"
        )

        results = []
        for file_path in file_paths:
            result = self.process_single_file(file_path)
            results.append(result)

        # 统计
        success_count = sum(1 for r in results if r["status"] == "success")
        total_chunks = sum(r.get("chunks_created", 0) for r in results)
        total_time = sum(r["duration"] for r in results)

        self.console.print(
            f"[bold green]✅ 批量处理完成: "
            f"{success_count}/{len(file_paths)} 成功, "
            f"共 {total_chunks} 个块, "
            f"总耗时 {total_time:.2f}s[/bold green]"
        )

        return results

    def _extract_and_chunk(self, file_path: str) -> List[Dict]:
        """
        提取文本并分块

        Args:
            file_path: 文件路径

        Returns:
            List[Dict]: Chunk列表，每个Chunk包含:
                {
                    "text": str,
                    "chunk_id": str,
                    "file_name": str,
                    "metadata": dict
                }
        """
        # 使用文档处理器提取文本
        # 注意：DocumentProcessor 需要支持单文件处理
        # 如果当前实现是批量处理，需要调整

        # 假设 DocumentProcessor 有 process_single_file 方法
        # 如果没有，需要适配
        try:
            # 方案 A: 如果 DocumentProcessor 支持单文件
            if hasattr(self.doc_processor, 'process_single_file'):
                chunks = self.doc_processor.process_single_file(file_path)
            # 方案 B: 使用 process_directory 但只处理单个文件
            else:
                # 创建临时目录或直接处理
                import os
                file_dir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)

                # 调用处理器（可能需要适配）
                all_chunks = self.doc_processor.process_directory(file_dir)

                # 过滤出当前文件的chunks
                chunks = [
                    chunk for chunk in all_chunks
                    if chunk.get("file_name") == file_name
                ]

            return chunks

        except Exception as e:
            self.console.print(f"[red]文本提取失败: {e}[/red]")
            raise

    def _vectorize_and_store(self, chunks: List[Dict], file_path: str) -> int:
        """
        向量化并存储到向量数据库

        Args:
            chunks: Chunk列表
            file_path: 原始文件路径

        Returns:
            int: 成功向量化的数量
        """
        if not chunks:
            return 0

        try:
            # 调用 EmbeddingManager 进行批量向量化
            # 注意：需要确保 chunks 已经写入 Neo4j
            # 或者 EmbeddingManager 支持直接传入 chunk 数据

            # 方案：先确保 chunks 写入 Neo4j，再调用 update_chunk_embeddings
            # 这需要与 Neo4j 写入逻辑集成

            # 临时方案：假设 chunks 已经通过 DocumentProcessor 写入了 Neo4j
            # 直接调用 update_chunk_embeddings

            updated_count = self.embedding_manager.update_chunk_embeddings()

            return updated_count

        except Exception as e:
            self.console.print(f"[red]向量化失败: {e}[/red]")
            raise

    def check_file_ready_for_search(self, file_path: str) -> bool:
        """
        检查文件是否已经可以搜索（Chunk已向量化）

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否可搜索
        """
        # 查询 Neo4j，检查文件对应的 Chunk 是否都有 embedding
        # 这需要一个查询方法

        # 临时实现：通过 embedding_manager 检查
        try:
            chunks = self.embedding_manager.get_chunks_needing_update()

            # 过滤出当前文件的chunks
            file_chunks_needing_update = [
                c for c in chunks
                if c.get("file_path") == file_path or c.get("file_name") == Path(file_path).name
            ]

            # 如果没有需要更新的chunk，说明都已向量化
            return len(file_chunks_needing_update) == 0

        except Exception as e:
            self.console.print(f"[yellow]检查文件状态失败: {e}[/yellow]")
            return False

    def get_processing_stats(self) -> Dict:
        """
        获取处理统计

        Returns:
            Dict: 统计信息
        """
        # 查询数据库获取统计信息
        # 例如：总chunk数、已向量化数、待处理数
        try:
            all_chunks = self.embedding_manager.get_chunks_needing_update()

            return {
                "total_chunks_needing_embedding": len(all_chunks),
                "status": "ready" if len(all_chunks) == 0 else "processing"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# 便捷函数
def quick_ingest_file(file_path: str) -> Dict:
    """
    快速摄取单个文件（便捷函数）

    Args:
        file_path: 文件路径

    Returns:
        Dict: 处理结果
    """
    pipeline = FastIngestionPipeline()
    return pipeline.process_single_file(file_path)


def quick_ingest_directory(directory: str) -> List[Dict]:
    """
    快速摄取整个目录（便捷函数）

    Args:
        directory: 目录路径

    Returns:
        List[Dict]: 处理结果列表
    """
    import os

    # 获取目录下所有文件
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 过滤支持的文件类型
            if file.endswith(('.txt', '.pdf', '.md', '.docx', '.doc', '.csv', '.json', '.yaml')):
                file_paths.append(os.path.join(root, file))

    # 批量处理
    pipeline = FastIngestionPipeline()
    return pipeline.process_batch_files(file_paths)
