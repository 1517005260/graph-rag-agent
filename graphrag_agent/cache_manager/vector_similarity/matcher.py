import faiss
import pickle
import os
import threading
from typing import List, Tuple, Dict, Any
from .embeddings import EmbeddingProvider, get_cache_embedding_provider

from graphrag_agent.config.settings import similarity_threshold as st


class VectorSimilarityMatcher:
    """向量相似性匹配器，支持基于向量相似度的缓存匹配"""

    def __init__(self,
                 embedding_provider: EmbeddingProvider = None,
                 similarity_threshold: float = st,
                 max_vectors: int = 10000,
                 index_file: str = None):
        """
        初始化向量相似性匹配器

        参数:
            embedding_provider: 嵌入向量提供者，如果为None则根据配置自动选择
            similarity_threshold: 相似度阈值
            max_vectors: 最大向量数量
            index_file: 索引文件路径
        """
        self.embedding_provider = embedding_provider or get_cache_embedding_provider()
        self.similarity_threshold = similarity_threshold
        self.max_vectors = max_vectors
        self.index_file = index_file
        
        # 初始化FAISS索引
        self.dimension = self.embedding_provider.get_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 存储键到向量的映射
        self.key_to_index = {}
        self.index_to_key = {}
        self.key_to_context = {}
        self.key_to_query = {}  # 存储原始查询
        
        self._lock = threading.RLock()
        self._next_index = 0
        
        # 如果指定了索引文件，尝试加载
        if self.index_file and os.path.exists(f"{self.index_file}.pkl"):
            self._load_index()
    
    def add_vector(self, cache_key: str, query: str, context_info: Dict[str, Any] = None):
        """添加向量到索引"""
        with self._lock:
            # 如果已存在，先删除
            if cache_key in self.key_to_index:
                self.remove_vector(cache_key)

            # 生成嵌入向量
            embedding = self.embedding_provider.encode(query)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            # 添加到FAISS索引
            faiss_index = self._next_index
            self.index.add(embedding)

            # 更新映射
            self.key_to_index[cache_key] = faiss_index
            self.index_to_key[faiss_index] = cache_key
            self.key_to_context[cache_key] = context_info or {}
            self.key_to_query[cache_key] = query

            self._next_index += 1

            # 检查是否超出最大容量
            if self._next_index > self.max_vectors:
                self._cleanup_old_vectors()
    
    def find_similar(self, query: str, context_info: Dict[str, Any] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """查找相似的缓存键"""
        with self._lock:
            if self.index.ntotal == 0:
                return []

            # 生成查询向量
            query_embedding = self.embedding_provider.encode(query)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # 搜索相似向量
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.index_to_key):
                    continue

                if idx in self.index_to_key:
                    cache_key = self.index_to_key[idx]

                    # 检查上下文匹配
                    if self._context_matches(context_info, self.key_to_context.get(cache_key, {})):
                        if score >= self.similarity_threshold:
                            results.append((cache_key, float(score)))

            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def remove_vector(self, cache_key: str):
        """从索引中移除向量"""
        with self._lock:
            if cache_key not in self.key_to_index:
                return
            
            faiss_index = self.key_to_index[cache_key]
            
            # 从映射中删除
            del self.key_to_index[cache_key]
            if faiss_index in self.index_to_key:
                del self.index_to_key[faiss_index]
            if cache_key in self.key_to_context:
                del self.key_to_context[cache_key]
            if cache_key in self.key_to_query:
                del self.key_to_query[cache_key]
    
    def clear(self):
        """清空所有向量"""
        with self._lock:
            self.index.reset()
            self.key_to_index.clear()
            self.index_to_key.clear()
            self.key_to_context.clear()
            self.key_to_query.clear()
            self._next_index = 0
    
    def _context_matches(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> bool:
        """检查两个上下文是否匹配"""
        if not context1 and not context2:
            return True
        
        if not context1 or not context2:
            return False
        
        # 检查线程ID是否匹配
        thread_id1 = context1.get('thread_id', 'default')
        thread_id2 = context2.get('thread_id', 'default')
        
        return thread_id1 == thread_id2
    
    def _cleanup_old_vectors(self):
        """清理旧向量以保持在最大容量内"""
        # 重建索引，保留最近的向量
        pass
    
    def save_index(self, file_path: str = None):
        """保存索引到文件"""
        if file_path is None:
            file_path = self.index_file
        
        if file_path is None:
            return
        
        with self._lock:
            try:
                data = {
                    'key_to_index': self.key_to_index,
                    'index_to_key': self.index_to_key,
                    'key_to_context': self.key_to_context,
                    'key_to_query': self.key_to_query,
                    'next_index': self._next_index
                }
                
                # 保存FAISS索引
                if self.index.ntotal > 0:
                    faiss.write_index(self.index, f"{file_path}.faiss")
                
                # 保存映射关系
                with open(f"{file_path}.pkl", 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"保存向量索引失败: {e}")
    
    def _load_index(self):
        """从文件加载索引"""
        try:
            # 加载映射关系
            with open(f"{self.index_file}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.key_to_index = data.get('key_to_index', {})
                self.index_to_key = data.get('index_to_key', {})
                self.key_to_context = data.get('key_to_context', {})
                self.key_to_query = data.get('key_to_query', {})
                self._next_index = data.get('next_index', 0)
            
            # 加载FAISS索引
            faiss_file = f"{self.index_file}.faiss"
            if os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
            else:
                # 如果FAISS文件不存在，重建索引
                self._rebuild_index()
                
        except Exception as e:
            print(f"加载向量索引失败: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.key_to_index.clear()
            self.index_to_key.clear()
            self.key_to_context.clear()
            self.key_to_query.clear()
            self._next_index = 0
    
    def _rebuild_index(self):
        """重建FAISS索引"""
        if not self.key_to_query:
            return
        
        # 重新创建索引
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 重新添加所有向量
        for cache_key, query in self.key_to_query.items():
            embedding = self.embedding_provider.encode(query)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            self.index.add(embedding)