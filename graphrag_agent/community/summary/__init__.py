from langchain_community.graphs import Neo4jGraph
from typing import Union
from .base import BaseSummarizer
from .leiden import LeidenSummarizer
from .sllpa import SLLPASummarizer

class CommunitySummarizerFactory:
    """社区摘要生成器工厂类"""
    
    SUMMARIZERS = {
        'leiden': LeidenSummarizer,
        'sllpa': SLLPASummarizer
    }
    
    @classmethod
    def create_summarizer(cls, 
                         algorithm: str,
                         graph: Neo4jGraph) -> BaseSummarizer:
        """
        创建社区摘要生成器实例
        
        Args:
            algorithm: 算法类型 ('leiden' 或 'sllpa')
            graph: Neo4j图实例
            
        Returns:
            BaseSummarizer: 摘要生成器实例
        """
        algorithm = algorithm.lower()
        if algorithm not in cls.SUMMARIZERS:
            raise ValueError(f"不支持的算法类型: {algorithm}")
            
        summarizer_class = cls.SUMMARIZERS[algorithm]
        return summarizer_class(graph)

__all__ = ['CommunitySummarizerFactory', 'BaseSummarizer', 
           'LeidenSummarizer', 'SLLPASummarizer']