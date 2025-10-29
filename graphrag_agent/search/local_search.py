from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
from neo4j import Result
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from graphrag_agent.config.prompts import LC_SYSTEM_PROMPT, LOCAL_SEARCH_CONTEXT_PROMPT
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import LOCAL_SEARCH_SETTINGS
from graphrag_agent.search.modal_enricher import ModalEnricher, ModalSummary


class _ModalAwareRetriever(BaseRetriever):
    """包装向量检索器，在返回结果前补充多模态信息。"""

    def __init__(self, backend: BaseRetriever, enrich_fn):
        super().__init__()
        self._backend = backend
        self._enrich_fn = enrich_fn

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        docs = self._backend.get_relevant_documents(query)
        return self._enrich_fn(docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        if hasattr(self._backend, "aget_relevant_documents"):
            docs = await self._backend.aget_relevant_documents(query)
            return self._enrich_fn(docs)
        return self._get_relevant_documents(query, run_manager=run_manager)

class LocalSearch:
    """
    本地搜索类：使用Neo4j和LangChain实现基于向量检索的本地搜索功能
    
    该类通过向量相似度搜索在知识图谱中查找相关内容，并生成回答
    主要功能包括：
    1. 基于向量相似度的文本检索
    2. 社区内容和关系的检索
    3. 使用LLM生成最终答案
    """
    
    def __init__(self, llm, embeddings, response_type: str = "多个段落"):
        """
        初始化本地搜索类
        
        参数:
            llm: 大语言模型实例
            embeddings: 向量嵌入模型
            response_type: 响应类型，默认为"多个段落"
        """
        # 保存模型实例和配置
        self.llm = llm
        self.embeddings = embeddings
        self.response_type = response_type
        
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        self.driver = db_manager.get_driver()
        
        # 设置检索参数
        self.top_chunks = LOCAL_SEARCH_SETTINGS["top_chunks"]
        self.top_communities = LOCAL_SEARCH_SETTINGS["top_communities"]
        self.top_outside_rels = LOCAL_SEARCH_SETTINGS[
            "top_outside_relationships"
        ]
        self.top_inside_rels = LOCAL_SEARCH_SETTINGS[
            "top_inside_relationships"
        ]
        self.top_entities = LOCAL_SEARCH_SETTINGS["top_entities"]
        self.index_name = LOCAL_SEARCH_SETTINGS["index_name"]
        self.modal_enricher = ModalEnricher(self.driver)
        
        # 初始化社区节点权重
        self._init_community_weights()
        
        # 配置Neo4j URI和认证信息
        self.neo4j_uri = db_manager.neo4j_uri
        self.neo4j_username = db_manager.neo4j_username
        self.neo4j_password = db_manager.neo4j_password
        
    def _init_community_weights(self):
        """初始化Neo4j中社区节点的权重"""
        self.db_query("""
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount
        """)
        
    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        执行Cypher查询并返回结果
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数
            
        返回:
            pandas.DataFrame: 查询结果
        """
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )
    
    def enrich_documents(self, documents: Sequence[Document]) -> List[Document]:
        """为检索到的文档补充多模态段落及辅助信息。"""
        return self.modal_enricher.enrich_documents(documents)

    def aggregate_modal_summary(
        self,
        documents: Optional[Sequence[Document]] = None,
        modal_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ModalSummary:
        """聚合多模态段落，返回统一结构。"""
        return self.modal_enricher.aggregate_modal_summary(
            documents=documents,
            modal_map=modal_map,
        )
        
    @property
    def retrieval_query(self) -> str:
        """
        获取Neo4j检索查询语句
        
        返回:
            str: Cypher查询语句，用于检索相关内容
        """
        return """
        WITH collect(node) as nodes
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
            WITH distinct c, count(distinct n) as freq
            RETURN {id:c.id, text: c.text} AS chunkText
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH distinct c, c.community_rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
            LIMIT $topCommunities
        } AS report_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topOutsideRels
        } as outsideRels,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topInsideRels
        } as insideRels,
        collect {
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        } as entities
        RETURN {
            Chunks: text_mapping, 
            Reports: report_mapping, 
            Relationships: outsideRels + insideRels, 
            Entities: entities
        } AS text, 1.0 AS score, {} AS metadata
        """
    
    def as_retriever(self, **kwargs):
        """
        返回检索器实例，用于链式调用
        
        返回:
            检索器实例
        """
        # 生成包含所有检索参数的查询
        final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks))\
            .replace("$topCommunities", str(self.top_communities))\
            .replace("$topOutsideRels", str(self.top_outside_rels))\
            .replace("$topInsideRels", str(self.top_inside_rels))

        db_manager = get_db_manager()
        
        # 初始化向量存储
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=db_manager.neo4j_uri,
            username=db_manager.neo4j_username,
            password=db_manager.neo4j_password,
            index_name=self.index_name,
            retrieval_query=final_query
        )
        
        # 返回带多模态增强的检索器
        backend_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.top_entities}
        )
        return _ModalAwareRetriever(backend_retriever, self.enrich_documents)
        
    def search(self, query: str) -> str:
        """
        执行本地搜索
        
        参数:
            query: 搜索查询字符串
            
        返回:
            str: 生成的最终答案
        """
        # 初始化对话提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", LOCAL_SEARCH_CONTEXT_PROMPT),
        ])
        
        # 创建搜索链
        chain = prompt | self.llm | StrOutputParser()
        
        # 初始化向量存储
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            retrieval_query=self.retrieval_query
        )
        
        # 执行相似度搜索
        docs = vector_store.similarity_search(
            query,
            k=self.top_entities,
            params={
                "topChunks": self.top_chunks,
                "topCommunities": self.top_communities,
                "topOutsideRels": self.top_outside_rels,
                "topInsideRels": self.top_inside_rels,
            }
        )
        docs = self.enrich_documents(docs)
        
        # 使用LLM生成响应
        combined_context = "\n\n".join(
            [getattr(doc, "page_content", "") for doc in docs]
        ) if docs else ""

        response = chain.invoke({
            "context": combined_context,
            "input": query,
            "response_type": self.response_type
        })
        
        return response
        
    def close(self):
        """关闭Neo4j驱动连接"""
        pass
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
