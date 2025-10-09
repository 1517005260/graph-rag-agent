import time
import concurrent.futures
from typing import List, Dict, Any, Optional

from rich.console import Console

from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import EMBEDDING_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EmbeddingManager:
    """
    Embedding管理器，支持增量更新嵌入向量。
    
    主要功能：
    1. 仅处理需要更新的实体和Chunk的Embedding
    2. 高效的批处理和并行计算
    3. 维护Embedding更新状态
    """
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        初始化Embedding管理器
        
        Args:
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        self.console = Console()
        self.graph = get_db_manager().graph
        self.embeddings_model = get_embeddings_model()
        
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 性能监控
        self.embedding_time = 0
        self.db_time = 0
        self.total_time = 0
        
        # 处理统计
        self.stats = {
            "entity_updates": 0,
            "chunk_updates": 0,
            "total_updates": 0,
            "errors": 0
        }
    
    def setup_embedding_tracking(self):
        """设置Embedding更新追踪"""
        try:
            # 添加实体修改时间追踪
            self.graph.query("""
                MATCH (e:`__Entity__`)
                WHERE e.created_at IS NULL
                SET e.created_at = datetime()
            """)

            # 添加Chunk修改时间追踪
            self.graph.query("""
                MATCH (c:`__Chunk__`)
                WHERE c.created_at IS NULL
                SET c.created_at = datetime()
            """)

            self.console.print("[green]Embedding更新追踪设置完成[/green]")
            
        except Exception as e:
            self.console.print(f"[yellow]设置Embedding追踪时出错: {e}[/yellow]")
    
    def get_entities_needing_update(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        获取需要更新Embedding的实体
        
        Args:
            limit: 返回的最大实体数量
            
        Returns:
            List[Dict]: 需要更新的实体列表
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.embedding IS NULL 
        OR (e.needs_reembedding IS NOT NULL AND e.needs_reembedding = true)
        RETURN elementId(e) AS neo4j_id,
            e.id AS entity_id, 
            CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text
        LIMIT $limit
        """
        
        result = self.graph.query(query, params={"limit": limit})
        return result if result else []
    
    def get_chunks_needing_update(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        获取需要更新Embedding的Chunk
        
        Args:
            limit: 返回的最大Chunk数量
            
        Returns:
            List[Dict]: 需要更新的Chunk列表
        """
        query = """
        MATCH (c:`__Chunk__`)
        WHERE c.embedding IS NULL 
            OR c.needs_reembedding = true
            OR (c.last_updated IS NOT NULL AND 
                (c.last_embedded IS NULL OR c.last_updated > c.last_embedded))
        RETURN elementId(c) AS neo4j_id,
               c.id AS chunk_id, 
               c.text AS text
        LIMIT $limit
        """
        
        result = self.graph.query(query, params={"limit": limit})
        return result if result else []
    
    def update_entity_embeddings(self, entity_ids: Optional[List[str]] = None) -> int:
        """
        更新实体Embedding
        
        Args:
            entity_ids: 要更新的实体ID列表，如果为None则自动检测
            
        Returns:
            int: 更新的实体数量
        """
        start_time = time.time()
        
        # 获取需要更新的实体
        if entity_ids:
            # 如果提供了特定的实体ID列表
            id_list = ", ".join([f"'{eid}'" for eid in entity_ids])
            query = f"""
            MATCH (e:`__Entity__`)
            WHERE e.id IN [{id_list}]
            RETURN elementId(e) AS neo4j_id,
                   e.id AS entity_id, 
                   CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text
            """
            entities = self.graph.query(query)
        else:
            # 自动检测需要更新的实体
            entities = self.get_entities_needing_update(limit=self.batch_size * 5)
        
        if not entities:
            self.console.print("[yellow]没有需要更新Embedding的实体[/yellow]")
            return 0
        
        self.console.print(f"[cyan]开始更新 {len(entities)} 个实体的Embedding...[/cyan]")
        
        # 批量处理实体
        updated_count = 0
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i+self.batch_size]
            
            # 提取文本和ID
            texts = [entity["text"] for entity in batch]
            entity_ids = [entity["entity_id"] for entity in batch]
            neo4j_ids = [entity["neo4j_id"] for entity in batch]
            
            # 计算Embedding
            embedding_start = time.time()
            try:
                embeddings = self._compute_embeddings_batch(texts)
                self.embedding_time += time.time() - embedding_start
                
                # 准备更新数据
                updates = []
                for j, entity_id in enumerate(entity_ids):
                    if j < len(embeddings) and embeddings[j] is not None:
                        updates.append({
                            "neo4j_id": neo4j_ids[j],
                            "embedding": embeddings[j]
                        })
                
                # 更新数据库
                db_start = time.time()
                if updates:
                    query = """
                    UNWIND $updates AS update
                    MATCH (e) WHERE elementId(e) = update.neo4j_id
                    SET e.embedding = update.embedding,
                        e.last_embedded = datetime(),
                        e.needs_reembedding = false
                    RETURN count(e) AS updated
                    """
                    
                    result = self.graph.query(query, params={"updates": updates})
                    batch_updated = result[0]["updated"] if result else 0
                    updated_count += batch_updated
                    
                self.db_time += time.time() - db_start
                
                self.console.print(f"[green]批次 {i//self.batch_size + 1} 更新完成，"
                                  f"处理了 {len(batch)} 个实体，"
                                  f"成功更新 {batch_updated} 个[/green]")
                
            except Exception as e:
                self.console.print(f"[red]更新实体Embedding时出错: {e}[/red]")
                self.stats["errors"] += 1
        
        # 更新统计
        self.stats["entity_updates"] += updated_count
        self.stats["total_updates"] += updated_count
        
        # 计算总时间
        self.total_time += time.time() - start_time
        
        self.console.print(f"[blue]实体Embedding更新完成，共更新 {updated_count} 个实体，"
                          f"耗时: {time.time() - start_time:.2f}秒[/blue]")
        
        return updated_count
    
    def update_chunk_embeddings(self, chunk_ids: Optional[List[str]] = None) -> int:
        """
        更新Chunk Embedding
        
        Args:
            chunk_ids: 要更新的Chunk ID列表，如果为None则自动检测
            
        Returns:
            int: 更新的Chunk数量
        """
        start_time = time.time()
        
        # 获取需要更新的Chunk
        if chunk_ids:
            # 如果提供了特定的Chunk ID列表
            id_list = ", ".join([f"'{cid}'" for cid in chunk_ids])
            query = f"""
            MATCH (c:`__Chunk__`)
            WHERE c.id IN [{id_list}]
            RETURN elementId(c) AS neo4j_id,
                   c.id AS chunk_id, 
                   c.text AS text
            """
            chunks = self.graph.query(query)
        else:
            # 自动检测需要更新的Chunk
            chunks = self.get_chunks_needing_update(limit=self.batch_size * 5)
        
        if not chunks:
            self.console.print("[yellow]没有需要更新Embedding的Chunk[/yellow]")
            return 0
        
        self.console.print(f"[cyan]开始更新 {len(chunks)} 个Chunk的Embedding...[/cyan]")
        
        # 批量处理Chunk
        updated_count = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            
            # 提取文本和ID
            texts = [chunk["text"] for chunk in batch]
            chunk_ids = [chunk["chunk_id"] for chunk in batch]
            neo4j_ids = [chunk["neo4j_id"] for chunk in batch]
            
            # 计算Embedding
            embedding_start = time.time()
            try:
                embeddings = self._compute_embeddings_batch(texts)
                self.embedding_time += time.time() - embedding_start
                
                # 准备更新数据
                updates = []
                for j, chunk_id in enumerate(chunk_ids):
                    if j < len(embeddings) and embeddings[j] is not None:
                        updates.append({
                            "neo4j_id": neo4j_ids[j],
                            "embedding": embeddings[j]
                        })
                
                # 更新数据库
                db_start = time.time()
                if updates:
                    query = """
                    UNWIND $updates AS update
                    MATCH (c) WHERE elementId(c) = update.neo4j_id
                    SET c.embedding = update.embedding,
                        c.last_embedded = datetime(),
                        c.needs_reembedding = false
                    RETURN count(c) AS updated
                    """
                    
                    result = self.graph.query(query, params={"updates": updates})
                    batch_updated = result[0]["updated"] if result else 0
                    updated_count += batch_updated
                    
                self.db_time += time.time() - db_start
                
                self.console.print(f"[green]批次 {i//self.batch_size + 1} 更新完成，"
                                  f"处理了 {len(batch)} 个Chunk，"
                                  f"成功更新 {batch_updated} 个[/green]")
                
            except Exception as e:
                self.console.print(f"[red]更新Chunk Embedding时出错: {e}[/red]")
                self.stats["errors"] += 1
        
        # 更新统计
        self.stats["chunk_updates"] += updated_count
        self.stats["total_updates"] += updated_count
        
        # 计算总时间
        self.total_time += time.time() - start_time
        
        self.console.print(f"[blue]Chunk Embedding更新完成，共更新 {updated_count} 个Chunk，"
                          f"耗时: {time.time() - start_time:.2f}秒[/blue]")
        
        return updated_count
    
    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        计算一批文本的embedding向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: embedding向量列表
        """
        embeddings = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 预创建嵌入任务
            embedding_tasks = []
            for text in texts:
                # 添加强健性处理，确保文本不为空
                safe_text = text if text and text.strip() else "empty content"
                embedding_tasks.append(safe_text)
            
            # 分析批处理的最佳大小
            embed_batch_size = min(32, len(embedding_tasks))
            
            # 批量执行嵌入任务
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # 尝试使用批量嵌入方法
                    if hasattr(self.embeddings_model, 'embed_documents'):
                        sub_batch_embeddings = self.embeddings_model.embed_documents(sub_batch)
                        embeddings.extend(sub_batch_embeddings)
                    else:
                        # 回退到单个嵌入
                        futures = [executor.submit(self.embeddings_model.embed_query, text) for text in sub_batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                embeddings.append(future.result())
                            except Exception as e:
                                self.console.print(f"[yellow]嵌入计算失败: {e}[/yellow]")
                                # 添加零向量作为备用
                                if hasattr(self.embeddings_model, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings_model.embedding_size)
                                else:
                                    # 假设使用通用嵌入大小
                                    embeddings.append([0.0] * 1536)
                except Exception as e:
                    self.console.print(f"[yellow]批量嵌入处理失败: {e}[/yellow]")
                    # 尝试单个嵌入作为回退
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings_model.embed_query(text))
                        except Exception as e2:
                            self.console.print(f"[yellow]单个嵌入计算失败: {e2}[/yellow]")
                            # 添加零向量作为备用
                            if hasattr(self.embeddings_model, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings_model.embedding_size)
                            else:
                                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def mark_entities_for_update(self, entity_ids: List[str]) -> int:
        """
        标记实体需要更新Embedding
        
        Args:
            entity_ids: 实体ID列表
            
        Returns:
            int: 标记的实体数量
        """
        if not entity_ids:
            return 0
            
        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:`__Entity__` {id: entity_id})
        SET e.needs_reembedding = true,
            e.last_updated = datetime()
        RETURN count(e) AS marked
        """
        
        result = self.graph.query(query, params={"entity_ids": entity_ids})
        marked = result[0]["marked"] if result else 0
        
        self.console.print(f"[blue]已标记 {marked} 个实体需要更新Embedding[/blue]")
        
        return marked
    
    def mark_chunks_for_update(self, chunk_ids: List[str]) -> int:
        """
        标记Chunk需要更新Embedding
        
        Args:
            chunk_ids: Chunk ID列表
            
        Returns:
            int: 标记的Chunk数量
        """
        if not chunk_ids:
            return 0
            
        query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        SET c.needs_reembedding = true,
            c.last_updated = datetime()
        RETURN count(c) AS marked
        """
        
        result = self.graph.query(query, params={"chunk_ids": chunk_ids})
        marked = result[0]["marked"] if result else 0
        
        self.console.print(f"[blue]已标记 {marked} 个Chunk需要更新Embedding[/blue]")
        
        return marked
    
    def mark_document_chunks_for_update(self, filename: str) -> int:
        """
        标记文档的所有Chunk需要更新Embedding
        
        Args:
            filename: 文件名
            
        Returns:
            int: 标记的Chunk数量
        """
        query = """
        MATCH (d:`__Document__` {fileName: $filename})<-[:PART_OF]-(c:`__Chunk__`)
        SET c.needs_reembedding = true,
            c.last_updated = datetime()
        RETURN count(c) AS marked
        """
        
        result = self.graph.query(query, params={"filename": filename})
        marked = result[0]["marked"] if result else 0
        
        self.console.print(f"[blue]已标记文件 {filename} 的 {marked} 个Chunk需要更新Embedding[/blue]")
        
        return marked
    
    def mark_changed_files_chunks(self, changed_files: List[str]) -> int:
        """
        标记变更文件的所有Chunk需要更新Embedding
        
        Args:
            changed_files: 变更的文件列表
            
        Returns:
            int: 标记的Chunk数量
        """
        if not changed_files:
            return 0
            
        total_marked = 0
        for filename in changed_files:
            # 获取文件名（不包含路径）
            file_name = filename.split("/")[-1]
            marked = self.mark_document_chunks_for_update(file_name)
            total_marked += marked
        
        return total_marked
    
    def display_stats(self):
        """显示统计信息"""
        self.console.print("\n[bold cyan]Embedding更新统计[/bold cyan]")
        self.console.print(f"[blue]实体更新: {self.stats['entity_updates']} 个[/blue]")
        self.console.print(f"[blue]Chunk更新: {self.stats['chunk_updates']} 个[/blue]")
        self.console.print(f"[blue]总更新: {self.stats['total_updates']} 个[/blue]")
        self.console.print(f"[blue]错误: {self.stats['errors']} 个[/blue]")
        
        self.console.print(f"[blue]总耗时: {self.total_time:.2f}秒，"
                          f"其中: 嵌入计算: {self.embedding_time:.2f}秒 ({self.embedding_time/self.total_time*100:.1f}%)，"
                          f"数据库操作: {self.db_time:.2f}秒 ({self.db_time/self.total_time*100:.1f}%)[/blue]")
    
    def process(self, entity_limit: int = 500, chunk_limit: int = 500) -> Dict[str, Any]:
        """
        执行完整的Embedding更新流程
        
        Args:
            entity_limit: 处理的最大实体数量
            chunk_limit: 处理的最大Chunk数量
            
        Returns:
            Dict: 处理结果统计
        """
        start_time = time.time()
        
        try:
            # 设置Embedding追踪
            self.setup_embedding_tracking()
            
            # 更新实体Embedding
            entity_count = self.update_entity_embeddings(limit=entity_limit)
            
            # 更新Chunk Embedding
            chunk_count = self.update_chunk_embeddings(limit=chunk_limit)
            
            # 显示统计信息
            self.display_stats()
            
            # 计算总时间
            self.total_time = time.time() - start_time
            
            return {
                "entity_updates": entity_count,
                "chunk_updates": chunk_count,
                "total_updates": entity_count + chunk_count,
                "total_time": self.total_time,
                "embedding_time": self.embedding_time,
                "db_time": self.db_time
            }
            
        except Exception as e:
            self.console.print(f"[red]Embedding更新过程中出现错误: {e}[/red]")
            raise