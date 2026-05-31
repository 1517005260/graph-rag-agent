import numpy as np
from typing import List, Dict, Any, Union

class VectorUtils:
    """向量搜索和相似度计算的统一工具类"""
    
    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                         vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的余弦相似度
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 相似度值 (0-1)
        """
        # 确保向量是numpy数组
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # 避免被零除
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def rank_by_similarity(query_embedding: List[float], 
                          candidates: List[Dict[str, Any]], 
                          embedding_field: str = "embedding",
                          top_k: int = None) -> List[Dict[str, Any]]:
        """
        对候选项按与查询向量的相似度排序
        
        参数:
            query_embedding: 查询向量
            candidates: 候选项列表，每项都包含embedding_field指定的字段
            embedding_field: 包含嵌入向量的字段名
            top_k: 返回的最大结果数，None表示返回所有结果
            
        返回:
            按相似度排序的候选项列表，每项增加"score"字段表示相似度
        """
        scored_items = []
        
        for item in candidates:
            if embedding_field in item and item[embedding_field]:
                # 计算相似度
                similarity = VectorUtils.cosine_similarity(query_embedding, item[embedding_field])
                # 复制item并添加分数
                scored_item = item.copy()
                scored_item["score"] = similarity
                scored_items.append(scored_item)
        
        # 按相似度排序（降序）
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # 如果指定了top_k，则返回前top_k个结果
        if top_k is not None:
            return scored_items[:top_k]
            
        return scored_items
    
    @staticmethod
    def filter_documents_by_relevance(query_embedding: List[float],
                                     docs: List, 
                                     embedding_attr: str = "embedding",
                                     threshold: float = 0.0,
                                     top_k: int = None) -> List:
        """
        基于相似度过滤文档
        
        参数:
            query_embedding: 查询向量
            docs: 文档列表，可以是具有embedding属性的对象
            embedding_attr: 嵌入向量的属性名称
            threshold: 最小相似度阈值
            top_k: 返回的最大结果数
            
        返回:
            按相似度排序的文档列表
        """
        scored_docs = []
        
        for doc in docs:
            # 获取文档的向量表示
            doc_embedding = getattr(doc, embedding_attr, None) if hasattr(doc, embedding_attr) else None
            
            if doc_embedding:
                similarity = VectorUtils.cosine_similarity(query_embedding, doc_embedding)
                # 只添加超过阈值的文档
                if similarity >= threshold:
                    scored_docs.append({
                        'document': doc,
                        'score': similarity
                    })
            else:
                # 如果没有向量，给一个基础分数
                scored_docs.append({
                    'document': doc,
                    'score': 0.0
                })
        
        # 按分数排序（降序）
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 提取排序后的文档
        if top_k is not None:
            top_docs = [item['document'] for item in scored_docs[:top_k]]
        else:
            top_docs = [item['document'] for item in scored_docs]
            
        return top_docs
    
    @staticmethod
    def batch_cosine_similarity(query_embedding: np.ndarray, 
                            embeddings: List[np.ndarray]) -> np.ndarray:
        """
        批量计算余弦相似度，提高效率
        
        参数:
            query_embedding: 查询向量
            embeddings: 多个向量的列表
            
        返回:
            包含每个向量相似度的numpy数组
        """
        # 将列表转换为二维数组
        matrix = np.vstack(embeddings)
        
        # 规范化查询向量
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings))
        query_normalized = query_embedding / query_norm
        
        # 规范化所有向量（按行）
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        # 避免除以零
        matrix_norm[matrix_norm == 0] = 1.0
        matrix_normalized = matrix / matrix_norm
        
        # 一次性计算所有相似度（矩阵乘法提高效率）
        similarities = np.dot(matrix_normalized, query_normalized)
        
        return similarities