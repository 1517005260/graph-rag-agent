import time
from typing import Dict, List, Any
from pathlib import Path
import os
import tempfile
import shutil

from rich.console import Console
from rich.table import Table

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.config.prompt import system_template_build_graph, human_template_build_graph
from graphrag_agent.config.settings import (
    entity_types, relationship_types, CHUNK_SIZE, OVERLAP, MAX_WORKERS, BATCH_SIZE,
    FILE_REGISTRY_PATH
)
from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
from graphrag_agent.graph import EntityRelationExtractor, GraphWriter, GraphStructureBuilder
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.integrations.build.incremental.file_change_manager import FileChangeManager
from graphrag_agent.graph.indexing.embedding_manager import EmbeddingManager

class IncrementalGraphUpdater:
    """
    增量图谱更新器，基于LightRAG理念实现高效的增量更新。
    
    主要功能：
    1. 无缝集成新数据到现有图结构
    2. 仅更新变更部分，避免重建整个索引
    3. 高效合并新旧图结构
    4. 保护现有图谱的完整性
    """
    
    def __init__(self, files_dir: str, registry_path: str = None):
        """
        初始化增量图谱更新器

        Args:
            files_dir: 文件目录
            registry_path: 文件注册表路径，默认使用配置中的路径
        """
        if registry_path is None:
            registry_path = str(FILE_REGISTRY_PATH)

        self.console = Console()
        self.graph = get_db_manager().graph
        
        # 初始化文件变更管理器
        self.file_manager = FileChangeManager(files_dir, registry_path)
        
        # 初始化优化的Embedding管理器
        self.embedding_manager = EmbeddingManager(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
        
        # 保存文件目录路径
        self.files_dir = files_dir
        
        # 初始化LLM模型
        self.llm = get_llm_model()
        
        # 初始化文档处理器
        self.document_processor = DocumentProcessor(files_dir, CHUNK_SIZE, OVERLAP)
        
        # 初始化图结构构建器
        self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)
        
        # 初始化实体关系抽取器
        self.entity_extractor = EntityRelationExtractor(
            self.llm,
            system_template_build_graph,
            human_template_build_graph,
            entity_types,
            relationship_types,
            max_workers=MAX_WORKERS
        )
        
        # 初始化图写入器
        self.graph_writer = GraphWriter(self.graph, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
        
        # 处理统计
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_time": 0,
            "files_processed": 0,
            "entities_integrated": 0,
            "relations_integrated": 0,
            "entities_updated": 0,
            "chunks_updated": 0
        }
    
    def detect_changes(self) -> Dict[str, List[str]]:
        """
        检测文件变更
        
        Returns:
            Dict: 文件变更信息
        """
        return self.file_manager.detect_changes()
    
    def process_new_files(self, added_files: List[str]) -> Dict[str, Any]:
        """
        对新文件执行完整的处理流程（分块、实体抽取、关系创建）
        
        Args:
            added_files: 新增文件路径列表
            
        Returns:
            Dict: 处理结果统计
        """
        if not added_files:
            return {"files_processed": 0, "entities_extracted": 0, "relations_created": 0}
        
        results = {
            "files_processed": 0,
            "entities_extracted": 0,
            "relations_created": 0
        }
        
        self.console.print(f"[bold cyan]正在处理 {len(added_files)} 个新文件...[/bold cyan]")
        
        # 打印文件路径以便调试
        for file_path in added_files:
            self.console.print(f"[blue]处理文件路径: {file_path}[/blue]")
            if not os.path.exists(file_path):
                self.console.print(f"[red]警告: 文件不存在: {file_path}[/red]")
        
        # 使用临时目录
        # 1. 创建临时目录并复制新文件
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 复制文件到临时目录
                copy_success = False
                for file_path in added_files:
                    try:
                        if os.path.exists(file_path):
                            file_name = os.path.basename(file_path)
                            dest_path = os.path.join(temp_dir, file_name)
                            shutil.copy2(file_path, dest_path)
                            self.console.print(f"[green]已复制 {file_path} 到临时目录[/green]")
                            copy_success = True
                        elif os.path.exists(os.path.join(self.files_dir, file_path)):
                            # 尝试将文件路径视为相对于files_dir的路径
                            full_path = os.path.join(self.files_dir, file_path)
                            file_name = os.path.basename(file_path)
                            dest_path = os.path.join(temp_dir, file_name)
                            shutil.copy2(full_path, dest_path)
                            self.console.print(f"[green]已复制 {full_path} 到临时目录[/green]")
                            copy_success = True
                        else:
                            # 最后尝试直接使用文件名
                            file_name = os.path.basename(file_path)
                            full_path = os.path.join(self.files_dir, file_name)
                            if os.path.exists(full_path):
                                dest_path = os.path.join(temp_dir, file_name)
                                shutil.copy2(full_path, dest_path)
                                self.console.print(f"[green]已复制 {full_path} 到临时目录[/green]")
                                copy_success = True
                            else:
                                self.console.print(f"[red]复制文件失败，文件不存在: {file_path}[/red]")
                    except Exception as e:
                        self.console.print(f"[red]复制文件 {file_path} 到临时目录时出错: {e}[/red]")
                
                if not copy_success:
                    self.console.print("[red]没有成功复制任何文件到临时目录，无法继续处理[/red]")
                    return results
                
                # 2. 保存原始目录并临时修改
                original_dir = self.document_processor.directory_path
                self.document_processor.directory_path = temp_dir
                self.document_processor.file_reader.directory_path = temp_dir
                
                # 3. 处理临时目录中的文件
                processed_documents = self.document_processor.process_directory()
                
                # 4. 恢复原始目录
                self.document_processor.directory_path = original_dir
                self.document_processor.file_reader.directory_path = original_dir
                
                # 记录处理的文件数
                if processed_documents:
                    results["files_processed"] = len(processed_documents)
                    self.console.print(f"[green]成功处理 {len(processed_documents)} 个文件[/green]")
                    
                    # 5. 构建图谱结构
                    for doc in processed_documents:
                        if "chunks" in doc and doc["chunks"]:
                            # 创建文档节点
                            self.console.print(f"[blue]为文件 {doc['filename']} 创建文档节点[/blue]")
                            self.struct_builder.create_document(
                                type="local",
                                uri=str(self.files_dir),
                                file_name=doc["filename"],
                                domain="document"
                            )
                            
                            # 创建chunk节点和关系
                            chunks_count = len(doc['chunks']) if doc['chunks'] else 0
                            self.console.print(f"[blue]为文件 {doc['filename']} 创建 {chunks_count} 个文本块节点[/blue]")
                            doc["graph_result"] = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],
                                doc["chunks"]
                            )
                    
                    # 6. 准备实体抽取的数据
                    file_contents_format = []
                    for doc in processed_documents:
                        if "chunks" in doc and doc["chunks"]:
                            file_contents_format.append([
                                doc["filename"], 
                                doc["content"], 
                                doc["chunks"]
                            ])
                    
                    # 7. 抽取实体和关系
                    if file_contents_format:
                        self.console.print(f"[cyan]开始抽取实体和关系，文件数: {len(file_contents_format)}[/cyan]")
                        
                        total_chunks = sum(len(content[2]) for content in file_contents_format)
                        self.console.print(f"[blue]总计 {total_chunks} 个文本块需要处理[/blue]")
                        
                        processed_chunk_count = 0
                        def progress_callback(i):
                            nonlocal processed_chunk_count
                            processed_chunk_count += 1
                            if processed_chunk_count % 5 == 0 or processed_chunk_count == total_chunks:
                                self.console.print(f"[blue]已处理 {processed_chunk_count}/{total_chunks} 个文本块[/blue]")
                        
                        # 确保禁用缓存以处理新文件
                        original_cache_setting = getattr(self.entity_extractor, 'enable_cache', True)
                        self.entity_extractor.enable_cache = False
                        
                        try:
                            processed_contents = self.entity_extractor.process_chunks(
                                file_contents_format, 
                                progress_callback=progress_callback
                            )
                            
                            # 恢复缓存设置
                            self.entity_extractor.enable_cache = original_cache_setting
                            
                            # 输出处理结果
                            if processed_contents:
                                self.console.print(f"[green]实体抽取完成，已处理 {len(processed_contents)} 个文件[/green]")
                            
                                # 打印调试信息
                                for i, content in enumerate(processed_contents):
                                    if len(content) > 3:
                                        entity_data = content[3]
                                        entity_count = sum(1 for data in entity_data if '("entity"' in str(data))
                                        relation_count = sum(1 for data in entity_data if '("relationship"' in str(data))
                                        self.console.print(f"[blue]文件 {i+1}: {content[0]}, 抽取了 {entity_count} 个实体和 {relation_count} 个关系[/blue]")
                                    else:
                                        self.console.print(f"[yellow]文件 {i+1}: {content[0]}, 没有返回实体数据[/yellow]")
                            
                                # 8. 处理结果并写入图数据库
                                graph_writer_data = []
                                for doc in processed_documents:
                                    if "chunks" in doc and doc["chunks"] and "graph_result" in doc:
                                        # 查找对应的处理结果
                                        entity_data = None
                                        for content in processed_contents:
                                            if content[0] == doc["filename"] and len(content) > 3:
                                                entity_data = content[3]
                                                break
                                        
                                        if entity_data:
                                            # 估算实体和关系数量
                                            entity_count = sum(1 for data in entity_data if '("entity"' in str(data))
                                            relation_count = sum(1 for data in entity_data if '("relationship"' in str(data))
                                            self.console.print(f"[green]文件 {doc['filename']} 中识别出 {entity_count} 个实体和 {relation_count} 个关系[/green]")
                                            
                                            # 添加到写入数据
                                            graph_writer_data.append([
                                                doc["filename"],
                                                doc["content"],
                                                doc["chunks"],
                                                doc["graph_result"],
                                                entity_data
                                            ])
                                            
                                            # 更新统计
                                            results["entities_extracted"] += entity_count
                                            results["relations_created"] += relation_count
                                
                                # 9. 写入图数据库
                                if graph_writer_data:
                                    self.console.print(f"[cyan]开始写入 {len(graph_writer_data)} 个文件的图数据...[/cyan]")
                                    self.graph_writer.process_and_write_graph_documents(graph_writer_data)
                                    self.console.print(f"[green]图数据写入完成[/green]")
                                else:
                                    self.console.print("[yellow]没有有效的图数据需要写入[/yellow]")
                            else:
                                self.console.print("[yellow]实体抽取过程没有返回有效结果[/yellow]")
                        
                        except Exception as e:
                            self.console.print(f"[red]实体抽取过程中出错: {e}[/red]")
                            import traceback
                            self.console.print(f"[red]{traceback.format_exc()}[/red]")
                    else:
                        self.console.print("[yellow]没有找到可用于抽取实体的文本块[/yellow]")
                else:
                    self.console.print("[yellow]没有处理到任何文件[/yellow]")
            
            except Exception as e:
                self.console.print(f"[red]处理新文件时发生错误: {e}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
        
        self.console.print(f"[green]已完成处理 {results['files_processed']} 个新文件[/green]")
        if results["entities_extracted"] > 0 or results["relations_created"] > 0:
            self.console.print(f"[green]抽取了 {results['entities_extracted']} 个实体和 {results['relations_created']} 个关系[/green]")
        
        return results
    
    def integrate_new_entities(self, new_entities: List[Dict[str, Any]]) -> int:
        """
        无缝集成新实体到现有图结构
        
        Args:
            new_entities: 新实体列表
            
        Returns:
            int: 集成的实体数量
        """
        if not new_entities:
            return 0
            
        # 合并实体
        query = """
        UNWIND $entities AS entity
        MERGE (e:`__Entity__` {id: entity.id})
        ON CREATE 
            SET e += entity.properties,
                e.created_at = datetime(),
                e.needs_reembedding = true
        ON MATCH 
            SET e += entity.properties,
                e.last_updated = datetime(),
                e.needs_reembedding = true
        RETURN count(e) AS entity_count
        """
        
        # 准备实体数据
        entities_data = []
        for entity in new_entities:
            entity_data = {
                "id": entity.get("id", ""),
                "properties": {
                    k: v for k, v in entity.items() if k != "id"
                }
            }
            entities_data.append(entity_data)
        
        # 执行查询
        result = self.graph.query(query, params={"entities": entities_data})
        entity_count = result[0]["entity_count"] if result else 0
        
        # 更新统计
        self.stats["entities_integrated"] += entity_count
        
        self.console.print(f"[green]已集成 {entity_count} 个实体[/green]")
        
        return entity_count
    
    def integrate_new_relationships(self, new_relationships: List[Dict[str, Any]]) -> int:
        """
        无缝集成新关系到现有图结构
        
        Args:
            new_relationships: 新关系列表
            
        Returns:
            int: 集成的关系数量
        """
        if not new_relationships:
            return 0
            
        # 合并关系
        query = """
        UNWIND $relationships AS rel
        MATCH (s:`__Entity__` {id: rel.source_id})
        MATCH (t:`__Entity__` {id: rel.target_id})
        CALL apoc.merge.relationship(s, rel.type, 
            {}, 
            rel.properties, 
            t, 
            {
                onMatch: {
                    properties: rel.properties,
                    last_updated: datetime()
                },
                onCreateProperties: {
                    created_at: datetime()
                }
            }
        )
        YIELD rel as created
        RETURN count(created) AS rel_count
        """
        
        try:
            # 执行查询
            result = self.graph.query(query, params={"relationships": new_relationships})
            rel_count = result[0]["rel_count"] if result else 0
            
            # 更新统计
            self.stats["relations_integrated"] += rel_count
            
            self.console.print(f"[green]已集成 {rel_count} 个关系[/green]")
            
            return rel_count
        except Exception as e:
            self.console.print(f"[yellow]集成关系时出错 (可能APOC未安装): {e}[/yellow]")
            
            # 使用备用方法
            integrated = 0
            for rel in new_relationships:
                source_id = rel.get("source_id", "")
                target_id = rel.get("target_id", "")
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                
                if source_id and target_id:
                    try:
                        backup_query = """
                        MATCH (s:`__Entity__` {id: $source_id})
                        MATCH (t:`__Entity__` {id: $target_id})
                        MERGE (s)-[r:`%s`]->(t)
                        ON CREATE SET r += $properties, r.created_at = datetime()
                        ON MATCH SET r += $properties, r.last_updated = datetime()
                        RETURN count(r) AS created
                        """ % rel_type
                        
                        backup_result = self.graph.query(
                            backup_query, 
                            params={
                                "source_id": source_id, 
                                "target_id": target_id,
                                "properties": properties
                            }
                        )
                        
                        if backup_result and backup_result[0]["created"] > 0:
                            integrated += 1
                    except Exception as e2:
                        self.console.print(f"[red]使用备用方法集成关系时出错: {e2}[/red]")
            
            # 更新统计
            self.stats["relations_integrated"] += integrated
            
            self.console.print(f"[green]使用备用方法集成了 {integrated} 个关系[/green]")
            
            return integrated
    
    def merge_graph_structures(self, old_graph: Dict[str, Any], new_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并原有图结构与新图结构
        
        Args:
            old_graph: 原有图结构
            new_graph: 新图结构
        
        Returns:
            Dict: 合并后的图结构
        """
        # 节点集合并
        merged_nodes = {**old_graph.get("nodes", {})}
        for node_id, node in new_graph.get("nodes", {}).items():
            if node_id in merged_nodes:
                # 如果节点已存在，根据时间戳决定是否更新
                old_timestamp = merged_nodes[node_id].get("last_updated", 0)
                new_timestamp = node.get("last_updated", time.time())
                
                if new_timestamp > old_timestamp:
                    # 新数据更新，更新节点
                    merged_nodes[node_id] = {**merged_nodes[node_id], **node}
                    merged_nodes[node_id]["last_updated"] = new_timestamp
                    merged_nodes[node_id]["needs_reembedding"] = True
            else:
                # 新节点直接添加
                merged_nodes[node_id] = node
                # 确保新节点有标记需要嵌入
                if "needs_reembedding" not in merged_nodes[node_id]:
                    merged_nodes[node_id]["needs_reembedding"] = True
        
        # 边集合并，避免重复关系
        merged_edges = {}
        
        # 先添加旧图的边
        for edge in old_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")
            
            # 创建唯一键
            key = f"{source}_{rel_type}_{target}"
            merged_edges[key] = edge
        
        # 再添加或更新新图的边
        for edge in new_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")
            
            # 创建唯一键
            key = f"{source}_{rel_type}_{target}"
            
            if key in merged_edges:
                # 如果关系已存在，根据时间戳决定是否更新
                old_timestamp = merged_edges[key].get("last_updated", 0)
                new_timestamp = edge.get("last_updated", time.time())
                
                if new_timestamp > old_timestamp:
                    # 新数据更新，更新关系
                    merged_edges[key] = {**merged_edges[key], **edge}
                    merged_edges[key]["last_updated"] = new_timestamp
            else:
                # 新关系直接添加
                merged_edges[key] = edge
        
        return {
            "nodes": merged_nodes,
            "edges": list(merged_edges.values())
        }
    
    def update_changed_file_embeddings(self, changed_files: List[str]) -> Dict[str, int]:
        """
        更新变更文件相关的Embedding
        
        Args:
            changed_files: 变更的文件列表
            
        Returns:
            Dict: 更新统计
        """
        if not changed_files:
            return {"entities": 0, "chunks": 0}
            
        # 标记变更文件的Chunk需要更新Embedding
        marked_chunks = self.embedding_manager.mark_changed_files_chunks(changed_files)
        
        # 查找这些Chunk关联的实体，标记需要更新Embedding
        query = """
        MATCH (c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
        WHERE c.fileName IN $filenames OR c.needs_reembedding = true
        SET e.needs_reembedding = true,
            e.last_updated = datetime()
        RETURN count(DISTINCT e) AS entity_count
        """
        
        # 获取文件名（不包含路径）
        filenames = [Path(file).name for file in changed_files]
        
        result = self.graph.query(query, params={"filenames": filenames})
        marked_entities = result[0]["entity_count"] if result else 0
        
        self.console.print(f"[blue]已标记 {marked_chunks} 个Chunk和 {marked_entities} 个实体需要更新Embedding[/blue]")
        
        # 执行更新
        updated_entities = self.embedding_manager.update_entity_embeddings()
        updated_chunks = self.embedding_manager.update_chunk_embeddings()
        
        # 更新统计
        self.stats["entities_updated"] += updated_entities
        self.stats["chunks_updated"] += updated_chunks
        
        return {
            "entities": updated_entities,
            "chunks": updated_chunks
        }
    
    def process_deleted_files(self, deleted_files: List[str]) -> int:
        """
        处理已删除的文件
        
        Args:
            deleted_files: 删除的文件列表
            
        Returns:
            int: 删除的节点数量
        """
        if not deleted_files:
            return 0
            
        self.console.print(f"[cyan]处理 {len(deleted_files)} 个已删除的文件...[/cyan]")
        
        deleted_count = 0
        for file_path in deleted_files:
            file_name = Path(file_path).name
            
            # 查找与文件关联的所有Chunk节点
            chunk_query = """
            MATCH (d:`__Document__` {fileName: $fileName})<-[:PART_OF]-(c:`__Chunk__`)
            RETURN collect(c.id) AS chunk_ids, count(c) AS chunk_count
            """
            
            chunk_result = self.graph.query(chunk_query, params={"fileName": file_name})
            
            if not chunk_result or not chunk_result[0]["chunk_ids"]:
                self.console.print(f"[yellow]文件 {file_name} 没有相关的数据需要删除[/yellow]")
                continue
                
            chunk_ids = chunk_result[0]["chunk_ids"]
            chunk_count = chunk_result[0]["chunk_count"]
            
            # 查找这些Chunk关联的实体，但排除被其他Chunk引用的实体
            entity_query = """
            MATCH (c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
            WHERE c.id IN $chunk_ids
            WITH e, count(c) AS references
            MATCH (chunk:`__Chunk__`)-[:MENTIONS]->(e)
            WITH e, references, count(chunk) AS total_references
            WHERE references = total_references 
              AND NOT e.manual_edit = true  // 排除手动编辑的实体
              AND NOT e.protected = true    // 排除受保护的实体
            RETURN collect(e.id) AS entity_ids, count(e) AS entity_count
            """
            
            entity_result = self.graph.query(entity_query, params={"chunk_ids": chunk_ids})
            
            entity_ids = []
            entity_count = 0
            if entity_result and entity_result[0]["entity_ids"]:
                entity_ids = entity_result[0]["entity_ids"]
                entity_count = entity_result[0]["entity_count"]
            
            # 删除与文件关联的所有数据
            delete_query = """
            // 删除文档节点和关系
            MATCH (d:`__Document__` {fileName: $fileName})
            OPTIONAL MATCH (d)-[r]-()
            DELETE r
            
            // 删除Chunk节点和关系
            WITH d
            MATCH (c:`__Chunk__`)-[r1:PART_OF]->(d)
            OPTIONAL MATCH (c)-[r2]-()
            WHERE NOT type(r2) = 'PART_OF'
            DELETE r2
            
            // 删除孤立的实体节点
            WITH d, collect(c.id) as chunk_ids
            UNWIND $entity_ids AS entity_id
            MATCH (e:`__Entity__` {id: entity_id})
            WHERE NOT e.manual_edit = true AND NOT e.protected = true // 再次检查保护
            DELETE e
            
            // 删除Chunk节点
            WITH d, chunk_ids
            MATCH (c:`__Chunk__`)
            WHERE c.id IN chunk_ids
            DELETE c
            
            // 最后删除文档节点
            DELETE d
            
            RETURN count(d) AS deleted_docs
            """
            
            delete_result = self.graph.query(delete_query, params={
                "fileName": file_name, 
                "entity_ids": entity_ids
            })
            
            deleted_docs = delete_result[0]["deleted_docs"] if delete_result else 0
            file_deleted_count = chunk_count + entity_count + deleted_docs
            deleted_count += file_deleted_count
            
            self.console.print(f"[blue]已删除文件 {file_name} 相关数据: "
                              f"{chunk_count} 个Chunk节点, "
                              f"{entity_count} 个实体节点, "
                              f"{deleted_docs} 个文档节点[/blue]")
        
        self.console.print(f"[green]已完成删除文件处理，共删除 {deleted_count} 个节点[/green]")
        return deleted_count
    
    def export_graph_structure(self) -> Dict[str, Any]:
        """
        导出当前图谱结构
        
        Returns:
            Dict: 图谱结构数据
        """
        # 查询所有实体节点
        node_query = """
        MATCH (e:`__Entity__`)
        RETURN e.id AS id, 
               labels(e) AS labels,
               properties(e) AS properties,
               e.last_updated AS last_updated
        """
        
        node_result = self.graph.query(node_query)
        
        # 查询所有关系
        edge_query = """
        MATCH (s:`__Entity__`)-[r]->(t:`__Entity__`)
        RETURN s.id AS source,
               t.id AS target,
               type(r) AS type,
               properties(r) AS properties,
               CASE WHEN r.last_updated IS NOT NULL THEN r.last_updated ELSE null END AS last_updated
        """
        
        edge_result = self.graph.query(edge_query)
        
        # 构建图谱结构
        nodes = {}
        for node in node_result:
            node_id = node["id"]
            nodes[node_id] = {
                "id": node_id,
                "labels": node["labels"],
                "properties": node["properties"],
                "last_updated": node["last_updated"] if node["last_updated"] else time.time()
            }
        
        edges = []
        for edge in edge_result:
            edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                "properties": edge["properties"],
                "last_updated": edge["last_updated"] if edge["last_updated"] else time.time()
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def import_graph_structure(self, graph_structure: Dict[str, Any]) -> Dict[str, int]:
        """
        导入图谱结构
        
        Args:
            graph_structure: 图谱结构数据
            
        Returns:
            Dict: 导入统计
        """
        if not graph_structure:
            return {"nodes": 0, "edges": 0}
            
        # 导入节点
        nodes = list(graph_structure.get("nodes", {}).values())
        node_count = 0
        
        if nodes:
            # 准备节点数据
            node_data = []
            for node in nodes:
                node_data.append({
                    "id": node["id"],
                    "labels": node.get("labels", ["__Entity__"]),
                    "properties": node.get("properties", {})
                })
            
            # 执行导入
            node_query = """
            UNWIND $nodes AS node
            CALL apoc.merge.node(
                node.labels,
                {id: node.id},
                node.properties,
                node.properties
            )
            YIELD node as n
            RETURN count(n) AS node_count
            """
            
            try:
                node_result = self.graph.query(node_query, params={"nodes": node_data})
                node_count = node_result[0]["node_count"] if node_result else 0
            except Exception as e:
                self.console.print(f"[yellow]导入节点时出错 (可能APOC未安装): {e}[/yellow]")
                
                # 使用备用方法
                simple_node_query = """
                UNWIND $nodes AS node
                MERGE (n:`__Entity__` {id: node.id})
                SET n += node.properties
                RETURN count(n) AS node_count
                """
                
                node_result = self.graph.query(simple_node_query, params={"nodes": node_data})
                node_count = node_result[0]["node_count"] if node_result else 0
        
        # 导入边
        edges = graph_structure.get("edges", [])
        edge_count = 0
        
        if edges:
            # 执行导入
            self.integrate_new_relationships(edges)
            edge_count = len(edges)
        
        return {
            "nodes": node_count,
            "edges": edge_count
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 节点统计
        node_query = """
        MATCH (n)
        RETURN 
            count(n) AS total_nodes,
            sum(CASE WHEN n:`__Document__` THEN 1 ELSE 0 END) AS doc_count,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunk_count,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entity_count
        """
        
        node_result = self.graph.query(node_query)
        
        # 关系统计
        rel_query = """
        MATCH ()-[r]->()
        RETURN count(r) AS total_relations,
               count(DISTINCT type(r)) AS relation_types
        """
        
        rel_result = self.graph.query(rel_query)
        
        # 嵌入统计
        embedding_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN 
            count(n) AS nodes_with_embedding,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entities_with_embedding,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunks_with_embedding
        """
        
        embedding_result = self.graph.query(embedding_query)
        
        # 合并统计信息
        stats = {
            "total_nodes": node_result[0]["total_nodes"] if node_result else 0,
            "document_count": node_result[0]["doc_count"] if node_result else 0,
            "chunk_count": node_result[0]["chunk_count"] if node_result else 0,
            "entity_count": node_result[0]["entity_count"] if node_result else 0,
            "total_relations": rel_result[0]["total_relations"] if rel_result else 0,
            "relation_types": rel_result[0]["relation_types"] if rel_result else 0,
            "nodes_with_embedding": embedding_result[0]["nodes_with_embedding"] if embedding_result else 0,
            "entities_with_embedding": embedding_result[0]["entities_with_embedding"] if embedding_result else 0,
            "chunks_with_embedding": embedding_result[0]["chunks_with_embedding"] if embedding_result else 0
        }
        
        return stats
    
    def display_graph_statistics(self):
        """显示图谱统计信息"""
        stats = self.get_graph_statistics()
        
        # 创建统计表格
        stats_table = Table(title="图谱统计信息")
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("数量", justify="right")
        
        # 添加节点统计
        stats_table.add_row("总节点数", str(stats["total_nodes"]))
        stats_table.add_row("文档节点数", str(stats["document_count"]))
        stats_table.add_row("文本块节点数", str(stats["chunk_count"]))
        stats_table.add_row("实体节点数", str(stats["entity_count"]))
        
        # 添加关系统计
        stats_table.add_row("总关系数", str(stats["total_relations"]))
        stats_table.add_row("关系类型数", str(stats["relation_types"]))
        
        # 添加嵌入统计
        stats_table.add_row("具有嵌入的节点数", str(stats["nodes_with_embedding"]))
        stats_table.add_row("具有嵌入的实体数", str(stats["entities_with_embedding"]))
        stats_table.add_row("具有嵌入的文本块数", str(stats["chunks_with_embedding"]))
        
        # 显示统计表格
        self.console.print(stats_table)
    
    def process_incremental_update(self) -> Dict[str, Any]:
        """
        执行增量更新流程
        
        Returns:
            Dict: 更新结果统计
        """
        start_time = time.time()
        self.stats["start_time"] = start_time
        
        try:
            # 1. 检测文件变更
            self.console.print("[bold cyan]检测文件变更...[/bold cyan]")
            changes = self.detect_changes()
            
            # 分别处理新增、修改和删除的文件
            added_files = changes.get("added", [])
            modified_files = changes.get("modified", [])
            deleted_files = changes.get("deleted", [])
            
            changed_files = modified_files  # 只有修改的文件需要更新embedding
            self.stats["files_processed"] = len(added_files) + len(modified_files) + len(deleted_files)
            
            if not added_files and not changed_files and not deleted_files:
                self.console.print("[yellow]未检测到文件变更[/yellow]")
                return self.stats
            
            # 2. 处理已删除的文件
            if deleted_files:
                self.console.print("[bold cyan]处理已删除的文件...[/bold cyan]")
                self.process_deleted_files(deleted_files)
            
            # 3. 处理新文件 - 执行完整的处理流程
            if added_files:
                self.console.print("[bold cyan]处理新增文件...[/bold cyan]")
                new_file_results = self.process_new_files(added_files)
                # 更新统计信息
                self.stats["entities_integrated"] += new_file_results.get("entities_extracted", 0)
                self.stats["relations_integrated"] += new_file_results.get("relations_created", 0)
            
            # 4. 更新变更文件的Embedding
            if changed_files:
                self.console.print("[bold cyan]更新变更文件的Embedding...[/bold cyan]")
                embedding_stats = self.update_changed_file_embeddings(changed_files)
                
                # 显示Embedding更新结果
                self.console.print(f"[green]更新的实体Embedding: {embedding_stats['entities']}[/green]")
                self.console.print(f"[green]更新的Chunk Embedding: {embedding_stats['chunks']}[/green]")
            
            # 5. 更新文件注册表
            self.file_manager.update_registry()
            
            # 6. 显示图谱统计信息
            self.console.print("[bold cyan]图谱统计信息[/bold cyan]")
            self.display_graph_statistics()
            
            # 计算结束时间和总时间
            end_time = time.time()
            self.stats["end_time"] = end_time
            self.stats["total_time"] = end_time - start_time
            
            # 显示处理结果
            self.console.print("\n[bold green]增量更新完成![/bold green]")
            self.console.print(f"[green]总耗时: {self.stats['total_time']:.2f}秒[/green]")
            self.console.print(f"[green]处理的文件数: {self.stats['files_processed']}[/green]")
            if added_files:
                self.console.print(f"[green]新增实体数: {self.stats['entities_integrated']}[/green]")
                self.console.print(f"[green]新增关系数: {self.stats['relations_integrated']}[/green]")
            
            return self.stats
            
        except Exception as e:
            self.console.print(f"[red]增量更新过程中出现错误: {e}[/red]")
            
            # 记录结束时间和总时间
            end_time = time.time()
            self.stats["end_time"] = end_time
            self.stats["total_time"] = end_time - start_time
            
            raise