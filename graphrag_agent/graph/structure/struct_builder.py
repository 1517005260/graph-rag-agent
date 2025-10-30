import time
import concurrent.futures
from typing import List, Dict, Optional, Any
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
        
    def create_document(
        self,
        type: str,
        uri: str,
        file_name: str,
        domain: str,
        extra_properties: Optional[Dict[str, Any]] = None,
    ) -> Dict:
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
        set_fragments = [
            "d.type = $type",
            "d.uri = $uri",
            "d.domain = $domain",
        ]
        params: Dict[str, Any] = {
            "file_name": file_name,
            "type": type,
            "uri": uri,
            "domain": domain,
        }

        if extra_properties:
            for key, value in extra_properties.items():
                if value is None:
                    continue
                param_key = f"extra_{key}"
                params[param_key] = value
                set_fragments.append(f"d.{key} = ${param_key}")

        set_clause = ", ".join(set_fragments)
        query = f"""
        MERGE (d:`__Document__` {{fileName: $file_name}})
        SET {set_clause}
        RETURN d
        """
        doc = self.graph.query(query, params)
        return doc
        
    def create_relation_between_chunks(
        self,
        file_name: str,
        chunks: List,
        chunk_annotations: Optional[List[Dict]] = None,
        segments: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        创建Chunk节点并建立关系 - 批处理优化版本
        
        Args:
            file_name: 文件名
            chunks: 文本块列表
            chunk_annotations: 分块的多模态段落映射信息
            segments: 文档的多模态段落列表
            
        Returns:
            List[Dict]: 带有ID和文档的块列表
        """
        t0 = time.time()
        
        current_chunk_id = ""
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset = 0
        
        annotations = chunk_annotations or []
        segments = segments or []

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
            annotation = annotations[i] if i < len(annotations) else {}
            modal_segment_ids = annotation.get("segment_ids") or []
            modal_segment_types = annotation.get("segment_types") or []
            modal_segment_sources = annotation.get("segment_sources") or []
            modal_char_start = annotation.get("char_start")
            modal_char_end = annotation.get("char_end")

            metadata = {
                "position": position,
                "length": len(page_content),
                "content_offset": offset,
                "tokens": len(chunk)
            }
            if modal_segment_ids:
                metadata["modal_segment_ids"] = modal_segment_ids
                metadata["modal_segment_types"] = modal_segment_types
                metadata["modal_segment_sources"] = modal_segment_sources
            if modal_char_start is not None:
                metadata["modal_char_start"] = modal_char_start
            if modal_char_end is not None:
                metadata["modal_char_end"] = modal_char_end

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
                "tokens": len(chunk),
                "modal_segment_ids": modal_segment_ids,
                "modal_segment_types": modal_segment_types,
                "modal_segment_sources": modal_segment_sources,
                "modal_char_start": modal_char_start,
                "modal_char_end": modal_char_end,
            }
            batch_data.append(chunk_data)
            
            lst_chunks_including_hash.append({
                'chunk_id': current_chunk_id,
                'chunk_doc': chunk_document,
                'modal_segment_ids': modal_segment_ids,
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
        
        chunk_ids = [item['chunk_id'] for item in lst_chunks_including_hash]
        if segments:
            self._create_modal_segments(file_name, segments)
        if segments and annotations:
            self._link_chunks_to_segments(chunk_ids, annotations)

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
    
    def _create_modal_segments(self, file_name: str, segments: List[Dict[str, Any]]) -> None:
        """创建多模态段落节点并建立与文档的关系"""
        if not segments:
            return

        payload: List[Dict[str, Any]] = []
        for segment in segments:
            segment_id = segment.get("segment_id")
            if not segment_id:
                continue
            payload.append(
                {
                    "id": segment_id,
                    "file_name": file_name,
                    "text": segment.get("text") or "",
                    "type": segment.get("type") or "text",
                    "order": segment.get("segment_index"),
                    "source": segment.get("source"),
                    "page_idx": segment.get("page_idx"),
                    "bbox": segment.get("bbox"),
                    "char_start": segment.get("char_start"),
                    "char_end": segment.get("char_end"),
                    "image_rel_path": segment.get("image_relative_path"),
                    "image_path": segment.get("image_path"),
                    "table_html": segment.get("table_html"),
                    "table_caption": segment.get("table_caption"),
                    "table_footnote": segment.get("table_footnote") or [],
                    "latex": segment.get("latex"),
                    "image_caption": segment.get("image_caption") or [],
                    "image_footnote": segment.get("image_footnote") or [],
                    "vision_summary": segment.get("vision_summary"),
                }
            )

        if not payload:
            return

        query = """
        UNWIND $segments AS data
        MERGE (s:`__ModalSegment__` {id: data.id})
        SET s.text = data.text,
            s.modalType = data.type,
            s.fileName = data.file_name,
            s.order = data.order,
            s.source = data.source,
            s.pageIndex = data.page_idx,
            s.bbox = data.bbox,
            s.charStart = data.char_start,
            s.charEnd = data.char_end,
            s.imageRelativePath = data.image_rel_path,
            s.imagePath = data.image_path,
            s.tableHtml = data.table_html,
            s.tableCaption = data.table_caption,
            s.tableFootnote = data.table_footnote,
            s.latex = data.latex,
            s.imageCaption = data.image_caption,
            s.imageFootnote = data.image_footnote,
            s.visionSummary = data.vision_summary
        WITH s, data
        MATCH (d:`__Document__` {fileName: data.file_name})
        MERGE (d)-[:HAS_MODAL_SEGMENT]->(s)
        """
        self.graph.query(query, params={"segments": payload})

    def _link_chunks_to_segments(self, chunk_ids: List[str], annotations: List[Dict[str, Any]]) -> None:
        """建立Chunk与多模态段落之间的联系"""
        if not chunk_ids or not annotations:
            return

        relations: List[Dict[str, Any]] = []
        for index, chunk_id in enumerate(chunk_ids):
            if index >= len(annotations):
                break
            segment_ids = annotations[index].get("segment_ids") or []
            for order, segment_id in enumerate(segment_ids):
                if not segment_id:
                    continue
                relations.append(
                    {
                        "chunk_id": chunk_id,
                        "segment_id": segment_id,
                        "order": order,
                    }
                )

        if not relations:
            return

        query = """
        UNWIND $relations AS rel
        MATCH (c:`__Chunk__` {id: rel.chunk_id})
        MATCH (s:`__ModalSegment__` {id: rel.segment_id})
        MERGE (c)-[r:HAS_MODAL]->(s)
        SET r.sequence = rel.order
        """
        self.graph.query(query, params={"relations": relations})
    
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
            c.tokens = data.tokens,
            c.modal_segment_ids = CASE WHEN data.modal_segment_ids IS NULL THEN [] ELSE data.modal_segment_ids END,
            c.modal_segment_types = CASE WHEN data.modal_segment_types IS NULL THEN [] ELSE data.modal_segment_types END,
            c.modal_segment_sources = CASE WHEN data.modal_segment_sources IS NULL THEN [] ELSE data.modal_segment_sources END,
            c.modal_char_start = data.modal_char_start,
            c.modal_char_end = data.modal_char_end
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
    
    def parallel_process_chunks(
        self,
        file_name: str,
        chunks: List,
        chunk_annotations: Optional[List[Dict]] = None,
        segments: Optional[List[Dict]] = None,
        max_workers=None,
    ) -> List[Dict]:
        """
        并行处理chunks，提高大量数据的处理速度
        
        Args:
            file_name: 文件名
            chunks: 文本块列表
            chunk_annotations: 分块的多模态段落映射信息
            segments: 文档的多模态段落列表
            max_workers: 并行工作线程数
            
        Returns:
            List[Dict]: 带有ID和文档的块列表
        """
        annotations = chunk_annotations or []
        segments = segments or []
        max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        if len(chunks) < 100:  # 对于小数据集，使用标准方法
            return self.create_relation_between_chunks(
                file_name,
                chunks,
                chunk_annotations=annotations,
                segments=segments,
            )
        
        ordered_chunk_ids = [generate_hash(''.join(chunk)) for chunk in chunks]

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
                
                annotation = annotations[abs_index] if abs_index < len(annotations) else {}
                modal_segment_ids = annotation.get("segment_ids") or []
                modal_segment_types = annotation.get("segment_types") or []
                modal_segment_sources = annotation.get("segment_sources") or []
                modal_char_start = annotation.get("char_start")
                modal_char_end = annotation.get("char_end")

                # 创建metadata和Document对象
                metadata = {
                    "position": position,
                    "length": len(page_content),
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                if modal_segment_ids:
                    metadata["modal_segment_ids"] = modal_segment_ids
                    metadata["modal_segment_types"] = modal_segment_types
                    metadata["modal_segment_sources"] = modal_segment_sources
                if modal_char_start is not None:
                    metadata["modal_char_start"] = modal_char_start
                if modal_char_end is not None:
                    metadata["modal_char_end"] = modal_char_end

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
                    "tokens": len(chunk),
                    "modal_segment_ids": modal_segment_ids,
                    "modal_segment_types": modal_segment_types,
                    "modal_segment_sources": modal_segment_sources,
                    "modal_char_start": modal_char_start,
                    "modal_char_end": modal_char_end,
                }
                batch_data.append(chunk_data)
                
                results.append({
                    'chunk_id': current_chunk_id,
                    'chunk_doc': chunk_document,
                    'modal_segment_ids': modal_segment_ids,
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

        if segments:
            self._create_modal_segments(file_name, segments)
        if segments and annotations:
            self._link_chunks_to_segments(ordered_chunk_ids, annotations)
        
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
                c.tokens = data.tokens,
                c.modal_segment_ids = CASE WHEN data.modal_segment_ids IS NULL THEN [] ELSE data.modal_segment_ids END,
                c.modal_segment_types = CASE WHEN data.modal_segment_types IS NULL THEN [] ELSE data.modal_segment_types END,
                c.modal_segment_sources = CASE WHEN data.modal_segment_sources IS NULL THEN [] ELSE data.modal_segment_sources END,
                c.modal_char_start = data.modal_char_start,
                c.modal_char_end = data.modal_char_end
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
