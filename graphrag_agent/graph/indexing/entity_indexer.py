import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector

from graphrag_agent.models.get_models import get_embeddings_model, get_llm_model
from graphrag_agent.graph.core import BaseIndexer, connection_manager
from graphrag_agent.config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EntityIndexManager(BaseIndexer):
    """
    实体索引管理器，负责在Neo4j数据库中创建和管理实体的向量索引。
    处理实体节点的embedding向量计算和索引创建，支持后续基于向量相似度的实体查询。
    """
    
    def __init__(self, refresh_schema: bool = True, batch_size: int = 100, max_workers: int = 4):
        """
        初始化实体索引管理器
        
        Args:
            refresh_schema: 是否刷新Neo4j图数据库的schema
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """        
        batch_size = batch_size or ENTITY_BATCH_SIZE
        max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        super().__init__(batch_size, max_workers)
        
        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()
        
        # 初始化模型
        self.embeddings = get_embeddings_model()
        self.llm = get_llm_model()
        
        # 创建必要的索引
        self._create_indexes()
    
    def _create_indexes(self) -> None:
        """创建必要的索引以优化查询性能"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"
        ]
        
        connection_manager.create_multiple_indexes(index_queries)
        
    def clear_existing_index(self) -> None:
        """清除已存在的实体embedding索引，为了防止有的时候embedding模型的切换问题，这里顺便清下vector索引"""
        connection_manager.drop_index("entity_embedding")
        connection_manager.drop_index("vector")

    def create_entity_index(self, 
                          node_label: str = '__Entity__',
                          text_properties: List[str] = ['id', 'description'],
                          embedding_property: str = 'embedding') -> Optional[Neo4jVector]:
        """
        创建实体的向量索引，带批处理和并行优化
        
        Args:
            node_label: 实体节点的标签
            text_properties: 用于计算embedding的文本属性列表
            embedding_property: 存储embedding的属性名
            
        Returns:
            Neo4jVector: 创建的向量存储对象
        """
        start_time = time.time()
        
        # 先清除已有索引
        self.clear_existing_index()
        
        # 获取所有实体节点以准备批处理
        entities = self.graph.query(
            f"""
            MATCH (e:`{node_label}`)
            WHERE e.{embedding_property} IS NULL
            RETURN id(e) AS neo4j_id, e.id AS entity_id
            """
        )
        
        if not entities:
            print("没有找到需要处理的实体节点")
            return None
            
        print(f"开始为 {len(entities)} 个实体生成embeddings")
        
        # 批量处理所有实体
        self._process_embeddings_in_batches(entities, node_label, text_properties, embedding_property)
        
        # 创建新的向量索引
        try:
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=text_properties,
                embedding_node_property=embedding_property
            )
            
            end_time = time.time()
            print(f"索引创建成功，总耗时: {end_time - start_time:.2f}秒")
            print(f"其中: embedding计算: {self.embedding_time:.2f}秒, 数据库操作: {self.db_time:.2f}秒")
            
            return vector_store
        except Exception as e:
            print(f"创建向量索引时出错: {e}")
            return None
    
    def _process_embeddings_in_batches(self, entities: List[Dict[str, Any]], 
                                      node_label: str, text_properties: List[str], 
                                      embedding_property: str) -> None:
        """
        批量处理实体embedding的生成
        
        Args:
            entities: 实体列表
            node_label: 实体标签
            text_properties: 文本属性
            embedding_property: embedding属性名
        """
        # 获取最优批处理大小
        entity_count = len(entities)
        optimal_batch_size = self.get_optimal_batch_size(entity_count)
        
        def process_batch(batch, batch_index):
            # 获取批次内所有实体的文本
            entity_texts = self._get_entity_texts_batch(batch, text_properties)
            
            # 计算embeddings
            embedding_start = time.time()
            embeddings = self._compute_embeddings_batch(entity_texts)
            embedding_end = time.time()
            self.embedding_time += (embedding_end - embedding_start)
            
            # 更新数据库
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            self.db_time += (db_end - db_start)
        
        # 使用通用批处理方法
        self.batch_process_with_progress(
            entities, 
            process_batch, 
            optimal_batch_size, 
            "处理实体embedding"
        )
    
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
                safe_text = text if text and text.strip() else "unknown entity"
                embedding_tasks.append(safe_text)
            
            # 分析批处理的最佳大小
            embed_batch_size = min(32, len(embedding_tasks))
            
            # 批量执行嵌入任务
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # 尝试使用批量嵌入方法（如果可用）
                    if hasattr(self.embeddings, 'embed_documents'):
                        sub_batch_embeddings = self.embeddings.embed_documents(sub_batch)
                        embeddings.extend(sub_batch_embeddings)
                    else:
                        # 回退到单个嵌入
                        futures = [executor.submit(self.embeddings.embed_query, text) for text in sub_batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                embeddings.append(future.result())
                            except Exception as e:
                                print(f"嵌入计算失败: {e}")
                                # 添加零向量作为备用
                                if hasattr(self.embeddings, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings.embedding_size)
                                else:
                                    embeddings.append([0.0] * 1536)
                except Exception as e:
                    print(f"批量嵌入处理失败: {e}")
                    # 尝试单个嵌入作为回退
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings.embed_query(text))
                        except Exception as e2:
                            print(f"单个嵌入计算失败: {e2}")
                            # 添加零向量作为备用
                            if hasattr(self.embeddings, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings.embedding_size)
                            else:
                                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def _get_entity_texts_batch(self, entities: List[Dict[str, Any]], text_properties: List[str]) -> List[str]:
        """
        获取批量实体的文本内容
        
        Args:
            entities: 实体列表
            text_properties: 文本属性列表
            
        Returns:
            List[str]: 实体文本列表
        """
        # 构建查询参数
        entity_ids = [entity['neo4j_id'] for entity in entities]
        
        # 使用高效的文本提取查询
        property_selections = ", ".join([
            f"CASE WHEN e.{prop} IS NOT NULL THEN e.{prop} ELSE '' END AS {prop}_text"
            for prop in text_properties
        ])
        
        query = f"""
        UNWIND $entity_ids AS id
        MATCH (e) WHERE id(e) = id
        RETURN id, {property_selections}
        """
        
        results = self.graph.query(query, params={"entity_ids": entity_ids})
        
        # 组合文本属性
        entity_texts = []
        for row in results:
            text_parts = []
            for prop in text_properties:
                prop_text = row.get(f"{prop}_text", "")
                if prop_text:
                    text_parts.append(prop_text)
            
            # 组合所有文本属性，确保至少有一些内容
            combined_text = " ".join(text_parts).strip()
            if not combined_text:
                combined_text = f"entity_{row['id']}"
                
            entity_texts.append(combined_text)
        
        return entity_texts
    
    def _update_embeddings_batch(self, entities: List[Dict[str, Any]], 
                                embeddings: List[List[float]], 
                                embedding_property: str) -> None:
        """
        批量更新实体embeddings
        
        Args:
            entities: 实体列表
            embeddings: 对应的embedding列表
            embedding_property: embedding属性名
        """
        # 构建更新数据
        update_data = []
        for i, entity in enumerate(entities):
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": entity['neo4j_id'],
                    "embedding": embeddings[i]
                })
        
        # 批量更新
        if update_data:
            try:
                query = f"""
                UNWIND $updates AS update
                MATCH (e) WHERE id(e) = update.id
                SET e.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"批量更新embeddings失败: {e}")
                # 回退到单个更新模式
                for update in update_data:
                    try:
                        single_query = f"""
                        MATCH (e) WHERE id(e) = $id
                        SET e.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        print(f"单个embedding更新失败 (ID: {update['id']}): {e2}")