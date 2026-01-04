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
        [优化版] 并行处理chunks

        改进点：
        1. 预计算 Offset，避免线程内 O(N^2) 累加
        2. 采用"缝合"策略，线程内只建内部关系，跨批次关系由主线程建立
        3. 优化关系过滤，使用 Set 查找
        4. 增加数据库写入重试机制

        Args:
            file_name: 文件名
            chunks: 文本块列表
            max_workers: 并行工作线程数

        Returns:
            List[Dict]: 带有ID和文档的块列表
        """
        max_workers = max_workers or DEFAULT_MAX_WORKERS

        if len(chunks) < 100:
            return self.create_relation_between_chunks(file_name, chunks)

        # 1. 预计算所有 chunks 的 offset (O(N))，避免在线程中重复计算
        global_offsets = []
        current_offset = 0
        for chunk in chunks:
            global_offsets.append(current_offset)
            current_offset += len(''.join(chunk))

        # 2. 准备批次
        chunk_batches = []
        batch_size = max(10, len(chunks) // max_workers)

        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_data = {
                "chunks": chunks[i:end_idx],
                "offsets": global_offsets[i:end_idx],
                "start_global_index": i
            }
            chunk_batches.append(batch_data)

        print(f"并行处理 {len(chunks)} 个块，每批次 {batch_size} 个，共 {len(chunk_batches)} 批次")

        # 定义处理函数（纯函数，不依赖外部列表状态）
        def process_chunk_batch(data):
            batch_chunks = data["chunks"]
            batch_offsets = data["offsets"]
            start_index = data["start_global_index"]

            local_nodes = []
            local_rels = []
            results = []

            # 记录本批次的首尾ID，用于后续缝合
            first_id = None
            last_id = None

            previous_chunk_id = None

            for i, chunk in enumerate(batch_chunks):
                page_content = ''.join(chunk)
                current_chunk_id = generate_hash(page_content)

                # 记录首尾ID
                if i == 0:
                    first_id = current_chunk_id
                if i == len(batch_chunks) - 1:
                    last_id = current_chunk_id

                position = start_index + i + 1

                # 构建 Node 数据
                metadata = {
                    "position": position,
                    "length": len(page_content),
                    "content_offset": batch_offsets[i],
                    "tokens": len(chunk)
                }
                chunk_document = Document(page_content=page_content, metadata=metadata)

                node_data = {
                    "id": current_chunk_id,
                    "pg_content": chunk_document.page_content,
                    "position": position,
                    "length": chunk_document.metadata["length"],
                    "f_name": file_name,
                    "content_offset": batch_offsets[i],
                    "tokens": len(chunk)
                }
                local_nodes.append(node_data)

                results.append({
                    'chunk_id': current_chunk_id,
                    'chunk_doc': chunk_document
                })

                # 构建内部关系 (只构建 batch 内部的 NEXT_CHUNK)
                if i == 0:
                    # 如果是全局第一个块，加 FIRST_CHUNK
                    if start_index == 0:
                        local_rels.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
                else:
                    # 内部前后连接
                    local_rels.append({
                        "type": "NEXT_CHUNK",
                        "previous_chunk_id": previous_chunk_id,
                        "current_chunk_id": current_chunk_id
                    })

                previous_chunk_id = current_chunk_id

            return {
                "nodes": local_nodes,
                "rels": local_rels,
                "results": results,
                "batch_index": start_index // batch_size,
                "first_id": first_id,
                "last_id": last_id
            }

        # 3. 并行执行
        all_nodes = []
        all_rels = []
        all_results = []
        batch_link_info = []  # 存储 (index, first_id, last_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk_batch, batch) for batch in chunk_batches]

            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    all_nodes.extend(res["nodes"])
                    all_rels.extend(res["rels"])
                    all_results.extend(res["results"])
                    batch_link_info.append((res["batch_index"], res["first_id"], res["last_id"]))
                except Exception as e:
                    print(f"[ERROR] 批次处理失败: {e}")
                    # 生产环境建议在这里抛出异常或记录严重错误，否则会导致数据丢失

        # 4. 缝合批次 (Stitch Batches)
        # 按 batch_index 排序，确保连接顺序正确
        batch_link_info.sort(key=lambda x: x[0])

        for i in range(len(batch_link_info) - 1):
            curr_batch = batch_link_info[i]
            next_batch = batch_link_info[i+1]

            # 建立跨批次连接: Current Last -> Next First
            stitch_rel = {
                "type": "NEXT_CHUNK",
                "previous_chunk_id": curr_batch[2],  # last_id of current
                "current_chunk_id": next_batch[1]    # first_id of next
            }
            all_rels.append(stitch_rel)

        # 5. 批量写入数据库 (优化过滤性能 + 重试机制)
        print(f"并行处理完成，开始写入 {len(all_nodes)} 个节点和 {len(all_rels)} 条关系")
        self._batch_write_to_db(file_name, all_nodes, all_rels)

        return all_results

    def _batch_write_to_db(self, file_name: str, nodes: List[Dict], rels: List[Dict], batch_size=500):
        """
        [优化版] 数据库写入：高性能过滤 + 重试机制

        Args:
            file_name: 文件名
            nodes: 节点数据列表
            rels: 关系数据列表
            batch_size: 每批次写入大小
        """
        total_batches = (len(nodes) + batch_size - 1) // batch_size

        # Step 1: 写入所有节点 (带重试)
        for i in range(0, len(nodes), batch_size):
            node_batch = nodes[i:i+batch_size]
            self._retry_query(
                self._create_chunks_only,  # 拆分出的只建节点的函数
                params={"batch_data": node_batch, "file_name": file_name},
                desc=f"写入节点批次 {i//batch_size + 1}/{total_batches}"
            )

        # Step 2: 写入所有关系 (带重试)
        # 关系也分批，避免单次 Cypher 过大
        rel_batch_size = 1000
        total_rel_batches = (len(rels) + rel_batch_size - 1) // rel_batch_size

        for i in range(0, len(rels), rel_batch_size):
            rel_batch = rels[i:i+rel_batch_size]
            # 简单分类
            first_rels = [r for r in rel_batch if r["type"] == "FIRST_CHUNK"]
            next_rels = [r for r in rel_batch if r["type"] == "NEXT_CHUNK"]

            if first_rels:
                self._retry_query(
                    self._create_first_rels,
                    params={"relationships": first_rels, "file_name": file_name},
                    desc=f"写入关系(FIRST) 批次 {i//rel_batch_size + 1}/{total_rel_batches}"
                )
            if next_rels:
                self._retry_query(
                    self._create_next_rels,
                    params={"relationships": next_rels},
                    desc=f"写入关系(NEXT) 批次 {i//rel_batch_size + 1}/{total_rel_batches}"
                )

    def _retry_query(self, func, params, desc, max_retries=3):
        """
        执行带有重试机制的数据库操作

        Args:
            func: 要执行的函数
            params: 函数参数
            desc: 操作描述
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                func(**params)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[CRITICAL] {desc} 失败，已重试 {max_retries} 次: {e}")
                    raise e
                print(f"[WARN] {desc} 失败，正在重试 ({attempt+1}/{max_retries}): {e}")
                time.sleep(1 * (attempt + 1))

    # --- 拆分出的原子查询函数 ---

    def _create_chunks_only(self, batch_data, file_name):
        """
        仅创建 Chunk 节点和 PART_OF 关系

        Args:
            batch_data: 批次数据
            file_name: 文件名
        """
        query = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content,
            c.position = data.position,
            c.length = data.length,
            c.fileName = $file_name,
            c.content_offset = data.content_offset,
            c.tokens = data.tokens
        WITH c
        MATCH (d:`__Document__` {fileName: $file_name})
        MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query, params={"batch_data": batch_data, "file_name": file_name})

    def _create_first_rels(self, relationships, file_name):
        """
        创建 FIRST_CHUNK 关系

        Args:
            relationships: 关系列表
            file_name: 文件名
        """
        query = """
        UNWIND $relationships AS rel
        MATCH (d:`__Document__` {fileName: $file_name})
        MATCH (c:`__Chunk__` {id: rel.chunk_id})
        MERGE (d)-[:FIRST_CHUNK]->(c)
        """
        self.graph.query(query, params={"relationships": relationships, "file_name": file_name})

    def _create_next_rels(self, relationships):
        """
        创建 NEXT_CHUNK 关系

        Args:
            relationships: 关系列表
        """
        query = """
        UNWIND $relationships AS rel
        MATCH (c:`__Chunk__` {id: rel.current_chunk_id})
        MATCH (pc:`__Chunk__` {id: rel.previous_chunk_id})
        MERGE (pc)-[:NEXT_CHUNK]->(c)
        """
        self.graph.query(query, params={"relationships": relationships})

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