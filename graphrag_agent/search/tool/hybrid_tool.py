import time
import json
from typing import List, Dict, Any
import pandas as pd
from neo4j import Result

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompt import LC_SYSTEM_PROMPT
from graphrag_agent.config.settings import gl_description, response_type, HYBRID_SEARCH_SETTINGS
from graphrag_agent.search.tool.base import BaseSearchTool


class HybridSearchTool(BaseSearchTool):
    """
    混合搜索工具，实现类似LightRAG的双级检索策略
    结合了局部细节检索和全局主题检索
    """
    
    def __init__(self):
        """初始化混合搜索工具"""
        # 检索参数
        self.entity_limit = HYBRID_SEARCH_SETTINGS["entity_limit"]
        self.max_hop_distance = HYBRID_SEARCH_SETTINGS["max_hop_distance"]
        self.top_communities = HYBRID_SEARCH_SETTINGS["top_communities"]
        self.batch_size = HYBRID_SEARCH_SETTINGS["batch_size"]
        self.community_level = HYBRID_SEARCH_SETTINGS["community_level"]
        
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/hybrid_search")

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 创建主查询处理链 - 用于生成最终答案
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，以下内容组合了低级详细信息和高级主题概念。

                ## 低级内容（实体详细信息）:
                {low_level}
                
                ## 高级内容（主题和概念）:
                {high_level}

                用户的问题是：
                {query}
                
                请综合利用上述信息回答问题，确保回答全面且有深度。
                回答格式应包含：
                1. 主要内容（使用清晰的段落展示）
                2. 在末尾标明引用的数据来源
                """
            )
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专门从用户查询中提取搜索关键词的助手。你需要将关键词分为两类：
                1. 低级关键词：具体实体名称、人物、地点、具体事件等
                2. 高级关键词：主题、概念、关系类型等
                
                返回格式必须是JSON格式：
                {{
                    "low_level": ["关键词1", "关键词2", ...], 
                    "high_level": ["关键词1", "关键词2", ...]
                }}
                
                注意：
                - 每类提取3-5个关键词即可
                - 不要添加任何解释或其他文本，只返回JSON
                - 如果某类无关键词，则返回空列表
                """),
            ("human", "{query}")
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取双级关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
        """
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            llm_start = time.time()
            
            # 调用LLM提取关键词
            result = self.keyword_chain.invoke({"query": query})
            
            print(f"DEBUG - LLM关键词结果: {result[:100]}...") if len(str(result)) > 100 else print(f"DEBUG - LLM关键词结果: {result}")
            
            # 解析JSON结果
            try:
                # 尝试直接解析
                if isinstance(result, dict):
                    # 结果已经是字典，无需解析
                    keywords = result
                elif isinstance(result, str):
                    # 清理字符串，移除可能导致解析失败的字符
                    result = result.strip()
                    # 检查字符串是否以JSON格式开始
                    if result.startswith('{') and result.endswith('}'):
                        keywords = json.loads(result)
                    else:
                        # 尝试提取JSON部分 - 寻找第一个{和最后一个}
                        start_idx = result.find('{')
                        end_idx = result.rfind('}')
                        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                            json_str = result[start_idx:end_idx+1]
                            keywords = json.loads(json_str)
                        else:
                            # 没有有效的JSON结构，使用简单的关键词提取
                            raise ValueError("No valid JSON structure found")
                else:
                    # 不是字符串也不是字典
                    raise TypeError(f"Unexpected result type: {type(result)}")
                    
            except (json.JSONDecodeError, ValueError, TypeError) as json_err:
                print(f"JSON解析失败: {json_err}，尝试备用方法提取关键词")
                
                # 备用方法：手动提取关键词
                if isinstance(result, str):
                    # 简单分词提取关键词
                    import re
                    # 移除标点符号，按空格分词
                    words = re.findall(r'\b\w+\b', query.lower())
                    # 过滤停用词（简化版，实际需要更完整的停用词表）
                    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                                "in", "on", "at", "to", "for", "with", "by", "about", "of", "and", "or"}
                    keywords = {
                        "high_level": [word for word in words if len(word) > 5 and word not in stopwords][:3],
                        "low_level": [word for word in words if 3 <= len(word) <= 5 and word not in stopwords][:5]
                    }
                else:
                    # 如果不是字符串，返回基于原始查询的简单关键词
                    keywords = {
                        "high_level": [query],
                        "low_level": []
                    }
            
            # 记录LLM处理时间
            self.performance_metrics["llm_time"] += time.time() - llm_start
            
            # 确保包含必要的键
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []
                
            # 确保列表类型
            if not isinstance(keywords["low_level"], list):
                keywords["low_level"] = [str(keywords["low_level"])]
            if not isinstance(keywords["high_level"], list):
                keywords["high_level"] = [str(keywords["high_level"])]
                
            # 缓存结果
            self.cache_manager.set(f"keywords:{query}", keywords)
            
            return keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 返回基于原始查询的默认值
            return {"low_level": [query], "high_level": [query.split()[0] if query.split() else query]}
    
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
    
    def _vector_search(self, query: str, limit: int = 5) -> List[str]:
        """
        使用基类的向量搜索方法
        
        参数:
            query: 查询字符串
            limit: 最大结果数
            
        返回:
            List[str]: 实体ID列表
        """
        return self.vector_search(query, limit)

    def _fallback_text_search(self, query: str, limit: int = 5) -> List[str]:
        """
        基于文本匹配的备用搜索方法
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
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
            print(f"文本搜索也失败: {e}")
            return []
    
    def _retrieve_low_level_content(self, query: str, keywords: List[str]) -> str:
        """
        检索低级内容（具体实体和关系）
        
        参数:
            query: 查询字符串
            keywords: 低级关键词列表
            
        返回:
            str: 格式化的低级内容
        """
        query_start = time.time()
        
        # 首先使用关键词查询获取相关实体
        entity_ids = []
        
        if keywords:
            keyword_params = {}
            keyword_conditions = []
            
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                keyword_params[param_name] = keyword
                keyword_conditions.append(f"e.id CONTAINS ${param_name} OR e.description CONTAINS ${param_name}")
            
            # 构建查询
            if keyword_conditions:
                keyword_query = """
                MATCH (e:__Entity__)
                WHERE """ + " OR ".join(keyword_conditions) + """
                RETURN e.id AS id
                LIMIT $limit
                """
                
                try:
                    keyword_results = self.db_query(keyword_query, 
                                                {**keyword_params, "limit": self.entity_limit})
                    if not keyword_results.empty:
                        entity_ids = keyword_results['id'].tolist()
                except Exception as e:
                    print(f"关键词查询失败: {e}")
        
        # 如果关键词搜索没有结果或没有提供关键词，尝试使用向量搜索
        if not entity_ids:
            try:
                # 使用我们的自定义向量搜索方法
                vector_entity_ids = self._vector_search(query, limit=self.entity_limit)
                if vector_entity_ids:
                    entity_ids = vector_entity_ids
            except Exception as e:
                print(f"向量搜索失败: {e}")
        
        # 如果仍然没有实体，使用基本文本匹配
        if not entity_ids:
            try:
                entity_ids = self._fallback_text_search(query, limit=self.entity_limit)
            except Exception as e:
                print(f"文本搜索失败: {e}")
        
        # 如果仍然没有实体，返回空内容
        if not entity_ids:
            self.performance_metrics["query_time"] += time.time() - query_start
            return "没有找到相关的低级内容。"
        
        # 获取实体信息 - 不使用多跳关系以避免复杂查询
        entity_query = """
        // 从种子实体开始
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids
        
        RETURN collect({
            id: e.id, 
            type: CASE WHEN size(labels(e)) > 1 
                     THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0] 
                     ELSE 'Unknown' 
                  END, 
            description: e.description
        }) AS entities
        """
        
        # 获取关系信息 - 分别查询，避免复杂路径
        relation_query = """
        // 查找实体间的关系
        MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
        WHERE e1.id IN $entity_ids 
          AND e2.id IN $entity_ids
          AND e1.id < e2.id  // 避免重复关系
        
        RETURN collect({
            start: e1.id, 
            type: type(r), 
            end: e2.id,
            description: CASE WHEN r.description IS NULL THEN '' ELSE r.description END
        }) AS relationships
        """
        
        # 获取文本块信息
        chunk_query = """
        // 查找包含这些实体的文本块
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE e.id IN $entity_ids
        
        RETURN collect(DISTINCT {
            id: c.id, 
            text: c.text
        })[0..5] AS chunks
        """
        
        try:
            # 获取实体信息
            entity_results = self.db_query(entity_query, {"entity_ids": entity_ids})
            
            # 获取关系信息
            relation_results = self.db_query(relation_query, {"entity_ids": entity_ids})
            
            # 获取文本块信息
            chunk_results = self.db_query(chunk_query, {"entity_ids": entity_ids})
            
            self.performance_metrics["query_time"] += time.time() - query_start
            
            # 构建结果
            low_level = []
            
            # 添加实体信息
            if not entity_results.empty and 'entities' in entity_results.columns:
                entities = entity_results.iloc[0]['entities']
                if entities:
                    low_level.append("### 相关实体")
                    for entity in entities:
                        entity_desc = f"- **{entity['id']}** ({entity['type']}): {entity['description']}"
                        low_level.append(entity_desc)
            
            # 添加关系信息
            if not relation_results.empty and 'relationships' in relation_results.columns:
                relationships = relation_results.iloc[0]['relationships']
                if relationships:
                    low_level.append("\n### 实体关系")
                    for rel in relationships:
                        rel_desc = f"- **{rel['start']}** -{rel['type']}-> **{rel['end']}**: {rel['description']}"
                        low_level.append(rel_desc)
            
            # 添加文本块信息
            if not chunk_results.empty and 'chunks' in chunk_results.columns:
                chunks = chunk_results.iloc[0]['chunks']
                if chunks:
                    low_level.append("\n### 相关文本")
                    for chunk in chunks:
                        chunk_text = f"- ID: {chunk['id']}\n  内容: {chunk['text']}"
                        low_level.append(chunk_text)
            
            if not low_level:
                return "没有找到相关的低级内容。"
                
            return "\n".join(low_level)
        except Exception as e:
            self.performance_metrics["query_time"] += time.time() - query_start
            print(f"实体查询失败: {e}")
            return "查询实体信息时出错。"
    
    def _retrieve_high_level_content(self, query: str, keywords: List[str]) -> str:
        """
        检索高级内容（社区和主题概念）
        
        参数:
            query: 查询字符串
            keywords: 高级关键词列表
            
        返回:
            str: 格式化的高级内容
        """
        query_start = time.time()
        
        # 构建关键词条件
        keyword_conditions = []
        params = {"level": self.community_level, "limit": self.top_communities}
        
        if keywords:
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                params[param_name] = keyword
                keyword_conditions.append(f"c.summary CONTAINS ${param_name} OR c.full_content CONTAINS ${param_name}")
        
        # 构建查询
        community_query = """
        // 使用关键词过滤社区
        MATCH (c:__Community__ {level: $level})
        """
        
        if keyword_conditions:
            community_query += "WHERE " + " OR ".join(keyword_conditions)
        else:
            # 如果没有关键词，则使用查询文本
            params["query"] = query
            community_query += "WHERE c.summary CONTAINS $query OR c.full_content CONTAINS $query"
        
        # 添加排序和限制
        community_query += """
        WITH c
        ORDER BY CASE WHEN c.community_rank IS NULL THEN 0 ELSE c.community_rank END DESC
        LIMIT $limit
        RETURN c.id AS id, c.summary AS summary
        """
        
        try:
            community_results = self.db_query(community_query, params)
            
            self.performance_metrics["query_time"] += time.time() - query_start
            
            # 处理结果
            if community_results.empty:
                return "没有找到相关的高级内容。"
                
            # 构建格式化的高级内容
            high_level = ["### 相关主题概念"]
            
            for _, row in community_results.iterrows():
                community_desc = f"- **社区 {row['id']}**:\n  {row['summary']}"
                high_level.append(community_desc)
            
            return "\n".join(high_level)
        except Exception as e:
            self.performance_metrics["query_time"] += time.time() - query_start
            print(f"社区查询失败: {e}")
            return "查询社区信息时出错。"
    
    def search(self, query_input: Any) -> str:
        """
        执行混合搜索，结合低级和高级内容
        
        参数:
            query_input: 字符串查询或包含查询和关键词的字典
            
        返回:
            str: 生成的最终答案
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            # 支持直接传入分类的关键词
            low_keywords = query_input.get("low_level_keywords", [])
            high_keywords = query_input.get("high_level_keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            keywords = self.extract_keywords(query)
            low_keywords = keywords.get("low_level", [])
            high_keywords = keywords.get("high_level", [])
        
        # 检查缓存
        cache_key = query
        if low_keywords or high_keywords:
            cache_key = self.cache_manager.key_strategy.generate_key(
                query, 
                low_level_keywords=low_keywords, 
                high_level_keywords=high_keywords
            )
            
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 1. 检索低级内容（实体和关系）
            low_level_content = self._retrieve_low_level_content(query, low_keywords)
            
            # 2. 检索高级内容（社区和主题）
            high_level_content = self._retrieve_high_level_content(query, high_keywords)
            
            # 3. 生成最终答案
            llm_start = time.time()
            
            # 调用LLM生成最终答案
            result = self.query_chain.invoke({
                "query": query,
                "low_level": low_level_content,
                "high_level": high_level_content,
                "response_type": response_type
            })
            
            self.performance_metrics["llm_time"] += time.time() - llm_start
            
            # 缓存结果
            self.cache_manager.set(
                query, 
                result, 
                low_level_keywords=low_keywords,
                high_level_keywords=high_keywords
            )
            
            self.performance_metrics["total_time"] = time.time() - overall_start

            if not result:
                return "未找到相关信息"
            return result
            
        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_global_tool(self) -> BaseTool:
        """
        获取全局搜索工具
        
        返回:
            BaseTool: 全局搜索工具实例
        """
        class GlobalSearchTool(BaseTool):
            name : str = "global_retriever"
            description : str= gl_description
            
            def _run(self_tool, query: Any) -> str:
                # 设置为仅使用高级内容
                if isinstance(query, dict) and "query" in query:
                    original_query = query["query"]
                    keywords = query.get("keywords", [])
                    # 转换为高级关键词
                    high_keywords = keywords
                    query = {
                        "query": original_query,
                        "high_level_keywords": high_keywords,
                        "low_level_keywords": []  # 不使用低级关键词
                    }
                else:
                    # 提取关键词
                    keywords = self.extract_keywords(str(query))
                    query = {
                        "query": str(query),
                        "high_level_keywords": keywords.get("high_level", []),
                        "low_level_keywords": []
                    }
                
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
                
        return GlobalSearchTool()
    
    def close(self):
        """关闭资源"""
        super().close()
