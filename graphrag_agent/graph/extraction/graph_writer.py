import re
import concurrent.futures
from typing import List, Set
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class GraphWriter:
    """
    图写入器，负责将提取的实体和关系写入Neo4j图数据库。
    处理实体和关系的解析、转换为GraphDocument，以及批量写入图数据库。
    """
    
    def __init__(self, graph: Neo4jGraph = None, batch_size: int = 50, max_workers: int = 4):
        """
        初始化图写入器
        
        Args:
            graph: Neo4j图数据库对象，如果为None则使用连接管理器获取
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        self.graph = graph or connection_manager.get_connection()
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 节点缓存，用于减少重复节点的创建
        self.node_cache = {}
        
        # 用于跟踪已经处理的节点，减少重复操作
        self.processed_nodes: Set[str] = set()
        
    def convert_to_graph_document(self, chunk_id: str, input_text: str, result: str) -> GraphDocument:
        """
        将提取的实体关系文本转换为GraphDocument对象
        
        Args:
            chunk_id: 文本块ID
            input_text: 输入文本
            result: 提取结果
            
        Returns:
            GraphDocument: 转换后的图文档对象
        """
        node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
        relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

        nodes = {}
        relationships = []

        # 使用高效的正则匹配处理
        try:
            # 解析节点 - 使用缓存提高效率
            for match in node_pattern.findall(result):
                node_id, node_type, description = match
                # 检查节点缓存
                if node_id in self.node_cache:
                    nodes[node_id] = self.node_cache[node_id]
                elif node_id not in nodes:
                    new_node = Node(
                        id=node_id,
                        type=node_type,
                        properties={'description': description}
                    )
                    nodes[node_id] = new_node
                    self.node_cache[node_id] = new_node

            # 解析关系
            for match in relationship_pattern.findall(result):
                source_id, target_id, rel_type, description, weight = match
                # 确保源节点存在，先检查缓存
                if source_id not in nodes:
                    if source_id in self.node_cache:
                        nodes[source_id] = self.node_cache[source_id]
                    else:
                        new_node = Node(
                            id=source_id,
                            type="未知",
                            properties={'description': 'No additional data'}
                        )
                        nodes[source_id] = new_node
                        self.node_cache[source_id] = new_node
                        
                # 确保目标节点存在，先检查缓存
                if target_id not in nodes:
                    if target_id in self.node_cache:
                        nodes[target_id] = self.node_cache[target_id]
                    else:
                        new_node = Node(
                            id=target_id,
                            type="未知",
                            properties={'description': 'No additional data'}
                        )
                        nodes[target_id] = new_node
                        self.node_cache[target_id] = new_node
                    
                relationships.append(
                    Relationship(
                        source=nodes[source_id],
                        target=nodes[target_id],
                        type=rel_type,
                        properties={
                            "description": description,
                            "weight": float(weight)
                        }
                    )
                )
        except Exception as e:
            print(f"解析文本时出错: {e}")
            # 返回空的GraphDocument而不是引发异常
            return GraphDocument(
                nodes=[],
                relationships=[],
                source=Document(
                    page_content=input_text,
                    metadata={"chunk_id": chunk_id, "error": str(e)}
                )
            )

        # 创建并返回GraphDocument对象
        return GraphDocument(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=Document(
                page_content=input_text,
                metadata={"chunk_id": chunk_id}
            )
        )
        
    def process_and_write_graph_documents(self, file_contents: List) -> None:
        """
        处理并写入所有文件的GraphDocument对象 - 使用并行处理和批处理优化
        
        Args:
            file_contents: 文件内容列表
        """
        all_graph_documents = []
        all_chunk_ids = []
        
        # 预分配列表大小
        total_chunks = sum(len(file_content[3]) for file_content in file_contents)
        all_graph_documents = [None] * total_chunks
        all_chunk_ids = [None] * total_chunks
        
        chunk_index = 0
        error_count = 0
        
        print(f"开始处理 {total_chunks} 个chunks的GraphDocument")
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}
            
            # 提交所有任务
            for file_content in file_contents:
                chunks = file_content[3]  # chunks_with_hash在索引3的位置
                results = file_content[4]  # 提取结果在索引4的位置
                
                for i, (chunk, result) in enumerate(zip(chunks, results)):
                    future = executor.submit(
                        self.convert_to_graph_document,
                        chunk["chunk_id"],
                        chunk["chunk_doc"].page_content,
                        result
                    )
                    future_to_index[future] = chunk_index
                    chunk_index += 1
            
            # 收集处理结果
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    graph_document = future.result()
                    
                    # 只保留有效的图文档
                    if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                        all_graph_documents[idx] = graph_document
                        all_chunk_ids[idx] = graph_document.source.metadata.get("chunk_id")
                    else:
                        all_graph_documents[idx] = None
                        all_chunk_ids[idx] = None
                        
                except Exception as e:
                    error_count += 1
                    print(f"处理chunk时出错 (已有{error_count}个错误): {e}")
                    all_graph_documents[idx] = None
                    all_chunk_ids[idx] = None
        
        # 过滤掉None值
        all_graph_documents = [doc for doc in all_graph_documents if doc is not None]
        all_chunk_ids = [id for id in all_chunk_ids if id is not None]
        
        print(f"共处理 {total_chunks} 个chunks, 有效文档 {len(all_graph_documents)}, 错误 {error_count}")
        
        # 批量写入图文档
        self._batch_write_graph_documents(all_graph_documents)
        
        # 批量合并chunk关系
        if all_chunk_ids:
            self.merge_chunk_relationships(all_chunk_ids)
    
    def _batch_write_graph_documents(self, documents: List[GraphDocument]) -> None:
        """
        批量写入图文档
        
        Args:
            documents: 图文档列表
        """
        if not documents:
            return
            
        # 增加批处理大小的动态调整
        optimal_batch_size = min(self.batch_size, max(10, len(documents) // 10))
        total_batches = (len(documents) + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"开始批量写入 {len(documents)} 个文档，批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 批量写入图文档
        for i in range(0, len(documents), optimal_batch_size):
            batch = documents[i:i+optimal_batch_size]
            if batch:
                try:
                    self.graph.add_graph_documents(
                        batch,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    print(f"已写入批次 {i//optimal_batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"写入图文档批次时出错: {e}")
                    # 如果批次写入失败，尝试逐个写入以避免整批失败
                    for doc in batch:
                        try:
                            self.graph.add_graph_documents(
                                [doc],
                                baseEntityLabel=True,
                                include_source=True
                            )
                        except Exception as e2:
                            print(f"单个文档写入失败: {e2}")
    
    def merge_chunk_relationships(self, chunk_ids: List[str]) -> None:
        """
        合并Chunk节点与Document节点的关系
        
        Args:
            chunk_ids: 块ID列表
        """
        if not chunk_ids:
            return
        
        # 去除重复的chunk_id以减少操作数量
        unique_chunk_ids = list(set(chunk_ids))
        print(f"开始合并 {len(unique_chunk_ids)} 个唯一chunk关系")
            
        # 动态批处理大小
        optimal_batch_size = min(self.batch_size, max(20, len(unique_chunk_ids) // 5))
        total_batches = (len(unique_chunk_ids) + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"合并关系批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 分批处理，避免一次性处理过多数据
        for i in range(0, len(unique_chunk_ids), optimal_batch_size):
            batch_chunk_ids = unique_chunk_ids[i:i+optimal_batch_size]
            batch_data = [{"chunk_id": chunk_id} for chunk_id in batch_chunk_ids]
            
            try:
                # 使用原始的查询，确保兼容性
                merge_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
                    WITH c, d
                    MATCH (d)-[r:MENTIONS]->(e)
                    MERGE (c)-[newR:MENTIONS]->(e)
                    ON CREATE SET newR += properties(r)
                    DETACH DELETE d
                """
                
                self.graph.query(merge_query, params={"batch_data": batch_data})
                print(f"已处理合并关系批次 {i//optimal_batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"合并关系批次时出错: {e}")
                # 如果批处理失败，尝试逐个处理
                for chunk_id in batch_chunk_ids:
                    try:
                        single_query = """
                            MATCH (c:`__Chunk__` {id: $chunk_id}), (d:Document{chunk_id:$chunk_id})
                            WITH c, d
                            MATCH (d)-[r:MENTIONS]->(e)
                            MERGE (c)-[newR:MENTIONS]->(e)
                            ON CREATE SET newR += properties(r)
                            DETACH DELETE d
                        """
                        self.graph.query(single_query, params={"chunk_id": chunk_id})
                    except Exception as e2:
                        print(f"处理单个chunk关系时出错: {e2}")