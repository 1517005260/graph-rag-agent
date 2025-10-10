import os
from typing import Dict, Any
import pandas as pd
from neo4j import GraphDatabase, Result
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv


class DBConnectionManager:
    """数据库连接管理器，实现单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 避免重复初始化
        if self._initialized:
            return
            
        # 加载环境变量
        load_dotenv()
        
        # 从环境变量获取连接信息
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        # 初始化Neo4j驱动
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # 初始化LangChain Neo4j图实例
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            refresh_schema=False,
        )
        
        # 连接池配置
        self.session_pool = []
        self.max_pool_size = 10
        
        # 标记为已初始化
        self._initialized = True
    
    def get_driver(self):
        """获取Neo4j驱动实例"""
        return self.driver
    
    def get_graph(self):
        """获取LangChain Neo4j图实例"""
        return self.graph
    
    def execute_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        执行Cypher查询并返回结果
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数
            
        返回:
            pd.DataFrame: 查询结果DataFrame
        """
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )
    
    def get_session(self):
        """
        从连接池获取会话
        
        返回:
            neo4j.Session: Neo4j会话
        """
        if self.session_pool:
            # 从池中获取会话
            return self.session_pool.pop()
        else:
            # 创建新会话
            return self.driver.session()
    
    def release_session(self, session):
        """
        释放会话回连接池
        
        参数:
            session: Neo4j会话
        """
        if len(self.session_pool) < self.max_pool_size:
            self.session_pool.append(session)
        else:
            # 池已满，关闭会话
            session.close()
    
    def close(self):
        """关闭所有资源"""
        # 关闭所有池中的会话
        for session in self.session_pool:
            try:
                session.close()
            except:
                pass
        
        # 清空池
        self.session_pool = []
        
        # 关闭驱动
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 提供便捷的全局访问点
db_manager = DBConnectionManager()


def get_db_manager() -> DBConnectionManager:
    """获取数据库连接管理器实例"""
    return db_manager