from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time

from langchain_core.tools import BaseTool

from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
from graphrag_agent.cache_manager.manager import CacheManager, ContextAndKeywordAwareCacheKeyStrategy, MemoryCacheBackend
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.search.utils import VectorUtils
from graphrag_agent.config.settings import BASE_SEARCH_CONFIG


class BaseSearchTool(ABC):
    """搜索工具基础类，为各种搜索实现提供通用功能"""
    
    def __init__(self, cache_dir: str = "./cache/search"):
        """
        初始化搜索工具
        
        参数:
            cache_dir: 缓存目录，用于存储搜索结果
        """
        # 初始化大语言模型和嵌入模型
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        self.default_vector_limit = BASE_SEARCH_CONFIG["vector_limit"]
        self.default_text_limit = BASE_SEARCH_CONFIG["text_limit"]
        self.default_semantic_top_k = BASE_SEARCH_CONFIG["semantic_top_k"]
        self.default_relevance_top_k = BASE_SEARCH_CONFIG["relevance_top_k"]
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(
            key_strategy=ContextAndKeywordAwareCacheKeyStrategy(),
            storage_backend=MemoryCacheBackend(
                max_size=BASE_SEARCH_CONFIG["cache_max_size"]
            ),
            cache_dir=cache_dir
        )
        
        # 性能监控指标
        self.performance_metrics = {
            "query_time": 0,  # 数据库查询时间
            "llm_time": 0,    # 大语言模型处理时间
            "total_time": 0   # 总处理时间
        }
        
        # 初始化Neo4j连接
        self._setup_neo4j()
    
    def _setup_neo4j(self):
        """设置Neo4j连接"""
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        
        # 获取图数据库实例
        self.graph = db_manager.get_graph()
        
        # 获取驱动（用于直接执行查询）
        self.driver = db_manager.get_driver()
    
    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        """
        执行Cypher查询
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数
            
        返回:
            查询结果
        """
        # 使用连接管理器执行查询
        return get_db_manager().execute_query(cypher, params)
        
    @abstractmethod
    def _setup_chains(self):
        """
        设置处理链，子类必须实现
        用于配置各种LLM处理链和提示模板
        """
        pass
    
    @abstractmethod
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词
        """
        pass
    
    @abstractmethod
    def search(self, query: Any) -> str:
        """
        执行搜索
        
        参数:
            query: 查询内容，可以是字符串或包含更多信息的字典
            
        返回:
            str: 搜索结果
        """
        pass

    def vector_search(self, query: str, limit: int = None) -> List[str]:
        """
        基于向量相似度的搜索方法
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            limit = limit or self.default_vector_limit
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 构建Neo4j向量搜索查询
            cypher = """
            CALL db.index.vector.queryNodes('vector', $limit, $embedding)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score DESC
            """
            
            # 执行搜索
            results = self.db_query(cypher, {
                "embedding": query_embedding,
                "limit": limit
            })
            
            # 提取实体ID
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"向量搜索失败: {e}")
            # 如果向量搜索失败，尝试使用文本搜索作为备用
            return self.text_search(query, limit)
    
    def text_search(self, query: str, limit: int = None) -> List[str]:
        """
        基于文本匹配的搜索方法（作为向量搜索的备选）
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            limit = limit or self.default_text_limit
            # 构建全文搜索查询
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """
            
            results = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })
            
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"文本搜索失败: {e}")
            return []
            
    def semantic_search(self, query: str, entities: List[Dict],
                        embedding_field: str = "embedding",
                        top_k: int = None) -> List[Dict]:
        """
        对一组实体进行语义相似度搜索
        
        参数:
            query: 搜索查询
            entities: 实体列表
            embedding_field: 嵌入向量的字段名
            top_k: 返回的最大结果数
            
        返回:
            按相似度排序的实体列表，每项增加"score"字段表示相似度
        """
        try:
            top_k = top_k or self.default_semantic_top_k
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 使用工具类进行排序
            return VectorUtils.rank_by_similarity(
                query_embedding, 
                entities, 
                embedding_field, 
                top_k
            )
        except Exception as e:
            print(f"语义搜索失败: {e}")
            return entities[:top_k] if top_k else entities
    
    def filter_by_relevance(self, query: str, docs: List, top_k: int = None) -> List:
        """
        根据相关性过滤文档
        
        参数:
            query: 查询字符串
            docs: 文档列表
            top_k: 返回的最大结果数
            
        返回:
            按相关性排序的文档列表
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            limit = top_k or self.default_relevance_top_k
            return VectorUtils.filter_documents_by_relevance(
                query_embedding,
                docs,
                top_k=limit
            )
        except Exception as e:
            print(f"文档过滤失败: {e}")
            limit = top_k or self.default_relevance_top_k
            return docs[:limit] if limit else docs
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具实例
        
        返回:
            BaseTool: 搜索工具
        """
        # 创建动态工具类
        class DynamicSearchTool(BaseTool):
            name : str= f"{self.__class__.__name__.lower()}"
            description : str = "高级搜索工具，用于在知识库中查找信息"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DynamicSearchTool()
    
    def _log_performance(self, operation: str, start_time: float):
        """
        记录性能指标
        
        参数:
            operation: 操作名称
            start_time: 开始时间
        """
        duration = time.time() - start_time
        self.performance_metrics[operation] = duration
        print(f"性能指标 - {operation}: {duration:.4f}s")
    
    def close(self):
        """关闭资源连接"""
        # 关闭Neo4j连接
        if hasattr(self, 'graph'):
            # 如果Neo4jGraph有close方法，调用它
            if hasattr(self.graph, 'close'):
                self.graph.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源被正确释放"""
        self.close()
