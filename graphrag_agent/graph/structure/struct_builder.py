import time
import concurrent.futures
from typing import List, Dict
from langchain_core.documents import Document

from graphrag_agent.graph.core import connection_manager, generate_hash
from graphrag_agent.config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE
from graphrag_agent.config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS

class GraphStructureBuilder:
    """
    图结构构建器，负责创建和管理Neo4j中的文档和块节点结构。
    处理文档节点、Chunk节点的创建，以及它们之间关系的建立。
    """
    
    def __init__(self, batch_size=100):
        """
        初始化图结构构建器
        
        Args:
            batch_size: 批处理大小
        """
        self.graph = connection_manager.get_connection()
        self.graph.refresh_schema()
        
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
            
    def clear_database(self):
        """清空数据库"""
        clear_query = """
            MATCH (n)
            DETACH DELETE n
            """
        self.graph.query(clear_query)
        
    def create_document(self, type: str, uri: str, file_name: str, domain: str) -> Dict:
        """
        创建Document节点
        
        Args:
            type: 文档类型
            uri: 文档URI
            file_name: 文件名
            domain: 文档域
            
        Returns:
            Dict: 创建的文档节点信息
        """
        query = """
        MERGE(d:`__Document__` {fileName: $file_name}) 
        SET d.type=$type, d.uri=$uri, d.domain=$domain
        RETURN d;
        """
        doc = self.graph.query(
            query,
            {"file_name": file_name, "type": type, "uri": uri, "domain": domain}
        )
        return doc
        
    def create_relation_between_chunks(self, file_name: str, chunks: List) -> List[Dict]:
        """
        创建Chunk节点并建立关系 - 批处理优化版本
        
        Args:
            file_name: 文件名
            chunks: 文本块列表
            
        Returns:
            List[Dict]: 带有ID和文档的块列表
        """
        t0 = time.time()
        
        current_chunk_id = ""
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset = 0
        
        # 处理每个chunk
        for i, chunk in enumerate(chunks):
            page_content = ''.join(chunk)
            current_chunk_id = generate_hash(page_content)
            position = i + 1
            previous_chunk_id = current_chunk_id if i == 0 else lst_chunks_including_hash[-1]['chunk_id']
            
            if i > 0:
                last_page_content = ''.join(chunks[i-1])
                offset += len(last_page_content)
                
            firstChunk = (i == 0)
            
            # 创建metadata和Document对象
            metadata = {
                "position": position,
                "length": len(page_content),
                "content_offset": offset,
                "tokens": len(chunk)
            }
            chunk_document = Document(page_content=page_content, metadata=metadata)
            
            # 准备batch数据
            chunk_data = {
                "id": current_chunk_id,
                "pg_content": chunk_document.page_content,
                "position": position,
                "length": chunk_document.metadata["length"],
                "f_name": file_name,
                "previous_id": previous_chunk_id,
                "content_offset": offset,
                "tokens": len(chunk)
            }
            batch_data.append(chunk_data)
            
            lst_chunks_including_hash.append({
                'chunk_id': current_chunk_id,
                'chunk_doc': chunk_document
            })
            
            # 创建关系数据
            if firstChunk:
                relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
            else:
                relationships.append({
                    "type": "NEXT_CHUNK",
                    "previous_chunk_id": previous_chunk_id,
                    "current_chunk_id": current_chunk_id
                })
            
            # 当累积了一定量的数据时，进行批处理
            if len(batch_data) >= self.batch_size:
                self._process_batch(file_name, batch_data, relationships)
                batch_data = []
                relationships = []
        
        # 处理剩余的数据
        if batch_data:
            self._process_batch(file_name, batch_data, relationships)
        
        t1 = time.time()
        print(f"创建关系耗时: {t1-t0:.2f}秒")
        
        return lst_chunks_including_hash
    
    def _process_batch(self, file_name: str, batch_data: List[Dict], relationships: List[Dict]):
        """
        批量处理一组chunks和关系
        
        Args:
            file_name: 文件名
            batch_data: 批处理数据
            relationships: 关系数据
        """
        if not batch_data:
            return
            
        # 分离FIRST_CHUNK和NEXT_CHUNK关系
        first_relationships = [r for r in relationships if r.get("type") == "FIRST_CHUNK"]
        next_relationships = [r for r in relationships if r.get("type") == "NEXT_CHUNK"]
        
        # 使用优化的数据库操作
        self._create_chunks_and_relationships_optimized(file_name, batch_data, first_relationships, next_relationships)
    
    def _create_chunks_and_relationships_optimized(self, file_name: str, batch_data: List[Dict], 
                                                  first_relationships: List[Dict], next_relationships: List[Dict]):
        """
        优化的创建chunks和关系的查询 - 减少数据库往返
        
        Args:
            file_name: 文件名
            batch_data: 批处理数据
            first_relationships: FIRST_CHUNK关系列表
            next_relationships: NEXT_CHUNK关系列表
        """
        # 合并查询：创建Chunk节点和PART_OF关系
        query_chunks_and_part_of = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content, 
            c.position = data.position, 
            c.length = data.length, 
            c.fileName = data.f_name,
            c.content_offset = data.content_offset, 
            c.tokens = data.tokens
        WITH c, data
        MATCH (d:`__Document__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunks_and_part_of, params={"batch_data": batch_data})
        
        # 处理FIRST_CHUNK关系
        if first_relationships:
            query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """
            self.graph.query(query_first_chunk, params={
                "f_name": file_name,
                "relationships": first_relationships
            })
        
        # 处理NEXT_CHUNK关系
        if next_relationships:
            query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            MERGE (pc)-[:NEXT_CHUNK]->(c)
            """
            self.graph.query(query_next_chunk, params={"relationships": next_relationships})
    
    def parallel_process_chunks(self, file_name: str, chunks: List, max_workers=None) -> List[Dict]:
        """
        并行处理chunks，提高大量数据的处理速度
        
        Args:
            file_name: 文件名
            chunks: 文本块列表
            max_workers: 并行工作线程数
            
        Returns:
            List[Dict]: 带有ID和文档的块列表
        """
        max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        if len(chunks) < 100:  # 对于小数据集，使用标准方法
            return self.create_relation_between_chunks(file_name, chunks)
        
        # 将chunks分为多个批次
        chunk_batches = []
        batch_size = max(10, len(chunks) // max_workers)
        
        for i in range(0, len(chunks), batch_size):
            chunk_batches.append(chunks[i:i+batch_size])
        
        print(f"并行处理 {len(chunks)} 个块，每批次 {batch_size} 个，共 {len(chunk_batches)} 批次")
        
        # 为每个批次准备处理函数
        def process_chunk_batch(batch, start_index):
            results = []
            current_chunk_id = ""
            batch_data = []
            relationships = []
            offset = 0
            
            if start_index > 0 and start_index < len(chunks):
                # 获取前一个chunk的ID作为起始点
                prev_chunk = chunks[start_index - 1]
                prev_content = ''.join(prev_chunk)
                current_chunk_id = generate_hash(prev_content)
                # 计算前面所有chunk的offset
                for j in range(start_index):
                    offset += len(''.join(chunks[j]))
            
            # 处理批次内的每个chunk
            for i, chunk in enumerate(batch):
                abs_index = start_index + i
                page_content = ''.join(chunk)
                previous_chunk_id = current_chunk_id
                current_chunk_id = generate_hash(page_content)
                position = abs_index + 1
                
                if i > 0:
                    last_page_content = ''.join(batch[i-1])
                    offset += len(last_page_content)
                    
                firstChunk = (abs_index == 0)
                
                # 创建metadata和Document对象
                metadata = {
                    "position": position,
                    "length": len(page_content),
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                chunk_document = Document(page_content=page_content, metadata=metadata)
                
                # 准备batch数据
                chunk_data = {
                    "id": current_chunk_id,
                    "pg_content": chunk_document.page_content,
                    "position": position,
                    "length": chunk_document.metadata["length"],
                    "f_name": file_name,
                    "previous_id": previous_chunk_id,
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                batch_data.append(chunk_data)
                
                results.append({
                    'chunk_id': current_chunk_id,
                    'chunk_doc': chunk_document
                })
                
                # 创建关系数据
                if firstChunk:
                    relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
                else:
                    relationships.append({
                        "type": "NEXT_CHUNK",
                        "previous_chunk_id": previous_chunk_id,
                        "current_chunk_id": current_chunk_id
                    })
            
            return {
                "batch_data": batch_data,
                "relationships": relationships,
                "results": results
            }
        
        # 并行处理所有批次
        start_time = time.time()
        all_batch_data = []
        all_relationships = []
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_chunk_batch, batch, i * batch_size): i
                for i, batch in enumerate(chunk_batches)
            }
            
            # 收集所有处理结果
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    all_batch_data.extend(result["batch_data"])
                    all_relationships.extend(result["relationships"])
                    all_results.extend(result["results"])
                except Exception as e:
                    print(f"处理批次时出错: {e}")
        
        # 写入数据库
        print(f"并行处理完成，共 {len(all_batch_data)} 个块，开始写入数据库")
        
        # 按批次写入数据库
        db_batch_size = 500
        for i in range(0, len(all_batch_data), db_batch_size):
            batch = all_batch_data[i:i+db_batch_size]
            rel_batch = [r for r in all_relationships 
                         if r.get("type") == "FIRST_CHUNK" and any(b["id"] == r["chunk_id"] for b in batch)
                         or r.get("type") == "NEXT_CHUNK" and any(b["id"] == r["current_chunk_id"] for b in batch)]
            
            self._create_chunks_and_relationships(file_name, batch, rel_batch)
            print(f"已写入批次 {i//db_batch_size + 1}/{(len(all_batch_data) + db_batch_size - 1) // db_batch_size}")
        
        end_time = time.time()
        print(f"写入数据库完成，耗时: {end_time - start_time:.2f}秒")
        
        return all_results
    
    def _create_chunks_and_relationships(self, file_name: str, batch_data: List[Dict], relationships: List[Dict]):
        """
        执行创建chunks和关系的查询
        
        Args:
            file_name: 文件名
            batch_data: 批处理数据
            relationships: 关系数据
        """
        # 创建Chunk节点和PART_OF关系
        query_chunk_part_of = """
            UNWIND $batch_data AS data
            MERGE (c:`__Chunk__` {id: data.id})
            SET c.text = data.pg_content, 
                c.position = data.position, 
                c.length = data.length, 
                c.fileName = data.f_name,
                c.content_offset = data.content_offset, 
                c.tokens = data.tokens
            WITH data, c
            MATCH (d:`__Document__` {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunk_part_of, params={"batch_data": batch_data})
        
        # 创建FIRST_CHUNK关系
        query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c))
        """
        self.graph.query(query_first_chunk, params={
            "f_name": file_name,
            "relationships": relationships
        })
        
        # 创建NEXT_CHUNK关系
        query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                    MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
        self.graph.query(query_next_chunk, params={"relationships": relationships})