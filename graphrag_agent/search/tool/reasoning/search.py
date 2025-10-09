from typing import Dict, List
import re

class DualPathSearcher:
    """
    双路径搜索器：支持同时使用多种方式搜索知识库
    """
    
    def __init__(self, kb_retriever, kg_retriever=None, kb_name=""):
        """
        初始化双路径搜索器
        
        参数:
            kb_retriever: 知识库搜索函数
            kg_retriever: 知识图谱搜索函数
            kb_name: 知识库名称，用于构建查询
        """
        self.kb_retriever = kb_retriever
        self.kg_retriever = kg_retriever
        self.kb_name = kb_name
    
    def search(self, query: str) -> Dict:
        """
        执行双路径搜索
        
        参数:
            query: 搜索查询
            
        返回:
            Dict: 搜索结果
        """
        # 精确查询
        precise_query = query.replace(self.kb_name, "").strip()
        # 带名称的查询
        kb_query = f"{self.kb_name} {query}" if self.kb_name.lower() not in query.lower() else query
        
        # 执行两种查询
        precise_results = self.kb_retriever(precise_query)
        kb_results = self.kb_retriever(kb_query)
        
        # 提取文本内容以便LLM评估
        precise_text = self._extract_text_for_evaluation(precise_results)
        kb_text = self._extract_text_for_evaluation(kb_results)
        
        # 检查是否有内容可供评估
        precise_has_content = len(precise_text.strip()) > 50
        kb_has_content = len(kb_text.strip()) > 50
        
        # 如果只有一个结果有内容，直接返回那个
        if precise_has_content and not kb_has_content:
            print("[双路径搜索] 只有精确查询返回有效结果")
            return precise_results
        elif kb_has_content and not precise_has_content:
            print("[双路径搜索] 只有带知识库名查询返回有效结果")
            return kb_results
        elif not precise_has_content and not kb_has_content:
            print("[双路径搜索] 两种查询均未返回有效结果")
            # 合并可能的部分结果
            return self._merge_results(precise_results, kb_results)
        
        # 两种查询都有内容，使用LLM评估
        evaluation = self._evaluate_results_with_llm(query, precise_text, kb_text)
        
        if evaluation == "precise":
            print("[双路径搜索] LLM评估: 精确查询结果更具体更有价值")
            return precise_results
        elif evaluation == "kb":
            print("[双路径搜索] LLM评估: 带知识库名查询结果更具体更有价值")
            return kb_results
        else:
            # 评估结果不明确，合并结果
            print("[双路径搜索] LLM评估: 两种结果均有价值，合并结果")
            return self._merge_results(precise_results, kb_results)

    def _extract_text_for_evaluation(self, results: Dict) -> str:
        """从结果中提取文本用于评估"""
        texts = []
        
        # 从chunks中提取文本
        for chunk in results.get("chunks", []):
            if "text" in chunk:
                texts.append(chunk["text"])
        
        return "\n\n".join(texts)

    def _evaluate_results_with_llm(self, query: str, text1: str, text2: str) -> str:
        """
        使用LLM评估哪个结果更具体、更有价值
        
        参数:
            query: 原始查询
            text1: 精确查询结果
            text2: 带知识库名查询结果
            
        返回:
            str: "precise"表示精确查询更好，"kb"表示带知识库名查询更好，
                "both"表示两者都有价值
        """
        try:
            # 构建评估提示
            prompt = f"""请评估以下两个搜索结果，判断哪个更具体、包含更多有价值的信息，特别是具体数字、规定或明确标准。
                
            原始查询: {query}

            结果1:
            {text1}

            结果2:
            {text2}

            请评估哪个结果包含更具体的信息（如明确的规定、数字、标准等）。
            只返回以下三个选项之一：
            - "precise"：如果结果1更具体、更有价值
            - "kb"：如果结果2更具体、更有价值
            - "both"：如果两者具有相当的价值或包含不同的有价值信息

            只返回选项，不要包含其他解释。
            """
            
            # 调用LLM进行评估
            if hasattr(self, "llm"):
                response = self.llm.invoke(prompt)
                result = response.content if hasattr(response, "content") else str(response)
            else:
                # 如果没有llm属性，尝试从外部获取
                from model.get_models import get_llm_model
                llm = get_llm_model()
                response = llm.invoke(prompt)
                result = response.content if hasattr(response, "content") else str(response)
            
            # 提取评估结果
            result = result.strip().lower()
            if "precise" in result:
                return "precise"
            elif "kb" in result:
                return "kb"
            else:
                return "both"
                
        except Exception as e:
            print(f"[LLM评估失败] {str(e)}")
            # 评估失败时默认合并结果
            return "both"
    
    def _merge_results(self, result1: Dict, result2: Dict) -> Dict:
        """
        合并两个搜索结果
        
        参数:
            result1: 第一个搜索结果
            result2: 第二个搜索结果
            
        返回:
            Dict: 合并后的结果
        """
        # 初始化结果字典
        result = {
            "chunks": result1.get("chunks", []).copy(),
            "doc_aggs": result1.get("doc_aggs", []).copy()
        }
        
        # 如果第一个结果没有chunks，直接使用第二个结果
        if not result["chunks"]:
            return result2
        
        # 已存在的chunk_id和doc_id集合
        existing_chunk_ids = set(c.get("chunk_id") for c in result["chunks"] if "chunk_id" in c)
        existing_doc_ids = set(d.get("doc_id") for d in result["doc_aggs"] if "doc_id" in d)
        
        # 合并chunks，避免重复
        for chunk in result2.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            # 只添加不存在的chunks
            if chunk_id and chunk_id not in existing_chunk_ids:
                result["chunks"].append(chunk)
                existing_chunk_ids.add(chunk_id)
            elif not chunk_id:
                # 如果没有chunk_id，使用内容作为唯一性判断
                content = chunk.get("text", "")
                if not any(c.get("text") == content for c in result["chunks"]):
                    result["chunks"].append(chunk)
        
        # 合并doc_aggs，避免重复
        for doc in result2.get("doc_aggs", []):
            doc_id = doc.get("doc_id")
            if doc_id and doc_id not in existing_doc_ids:
                result["doc_aggs"].append(doc)
                existing_doc_ids.add(doc_id)
        
        # 复制其他字段
        for key in result2:
            if key not in ["chunks", "doc_aggs"]:
                if key not in result:
                    result[key] = result2[key]
                elif isinstance(result[key], list) and isinstance(result2[key], list):
                    # 合并列表类型的字段
                    result[key].extend([item for item in result2[key] if item not in result[key]])
        
        return result
        

class QueryGenerator:
    """查询生成器：生成子查询和跟进查询"""
    
    def __init__(self, llm, sub_query_prompt, followup_query_prompt):
        """
        初始化查询生成器
        
        参数:
            llm: 大语言模型实例
            sub_query_prompt: 子查询提示模板
            followup_query_prompt: 跟进查询提示模板
        """
        self.llm = llm
        self.sub_query_prompt = sub_query_prompt
        self.followup_query_prompt = followup_query_prompt
    
    def generate_sub_queries(self, original_query: str) -> List[str]:
        """
        将原始查询分解为多个子查询
        
        参数:
            original_query: 原始用户查询
            
        返回:
            List[str]: 子查询列表
        """
        try:
            # 调用LLM生成子查询
            response = self.llm.invoke(self.sub_query_prompt.format(original_query=original_query))
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取列表文本
            list_text = re.search(r'\[.*\]', content, re.DOTALL)
            if list_text:
                try:
                    # 解析列表
                    sub_queries = eval(list_text.group(0))
                    return sub_queries
                except Exception as e:
                    print(f"[子查询生成] 解析列表失败: {str(e)}")
            
            # 如果无法解析，返回原始查询
            return [original_query]
        except Exception as e:
            print(f"[子查询生成错误] {str(e)}")
            return [original_query]
    
    def generate_multiple_hypotheses(query: str, llm) -> List[str]:
        """
        为查询生成多个假设
        
        Args:
            query: 查询字符串
            llm: 语言模型
            
        Returns:
            List[str]: 假设列表
        """
        prompt = f"""
        为以下问题生成2-3个可能的假设，这些假设应该代表不同角度或思路：
        
        问题: "{query}"
        
        每个假设应该:
        1. 不同于其他假设
        2. 提供一种可能的思考方向
        3. 有助于深入分析问题
        
        以列表形式返回假设，每个假设简短明了。
        """
        
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 使用正则表达式提取假设
            import re
            
            # 尝试匹配编号列表 (1. xxx 2. xxx)
            numbered_pattern = re.compile(r'\d+\.\s*(.*?)(?=\d+\.|$)', re.DOTALL)
            numbered_matches = numbered_pattern.findall(content)
            
            if numbered_matches:
                return [match.strip() for match in numbered_matches if match.strip()]
            
            # 尝试匹配破折号列表 (- xxx)
            dash_pattern = re.compile(r'-\s*(.*?)(?=-|$)', re.DOTALL)
            dash_matches = dash_pattern.findall(content)
            
            if dash_matches:
                return [match.strip() for match in dash_matches if match.strip()]
            
            # 如果上述方法失败，按行分割并过滤
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            potential_hypotheses = [line for line in lines if len(line) > 10 and not line.startswith("假设") and not line.startswith("以下是")]
            
            return potential_hypotheses[:3]  # 最多返回3个假设
            
        except Exception as e:
            print(f"生成假设失败: {e}")
            return []
        
    def generate_followup_queries(self, original_query: str, retrieved_info: List[str]) -> List[str]:
        """
        基于已检索的信息生成跟进查询
        
        参数:
            original_query: 原始查询
            retrieved_info: 已检索的信息列表
            
        返回:
            List[str]: 跟进查询列表，如果不需要则为空列表
        """
        # 如果没有检索到任何信息，或信息不足，返回空列表
        if not retrieved_info or len(retrieved_info) < 2:
            return []
        
        try:
            # 合并已检索信息（但限制长度）
            info_text = "\n\n".join(retrieved_info[-3:])  # 只使用最近的3条信息
            
            # 调用LLM生成跟进查询
            response = self.llm.invoke(self.followup_query_prompt.format(
                original_query=original_query,
                retrieved_info=info_text
            ))
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取列表文本
            list_text = re.search(r'\[.*\]', content, re.DOTALL)
            if list_text:
                try:
                    # 解析列表
                    followup_queries = eval(list_text.group(0))
                    
                    # 确保没有重复查询
                    unique_queries = []
                    for q in followup_queries:
                        if q not in unique_queries:
                            unique_queries.append(q)
                    
                    return unique_queries
                except Exception as e:
                    print(f"[跟进查询生成] 解析列表失败: {str(e)}")
            
            # 如果无法解析，返回空列表
            return []
        except Exception as e:
            print(f"[跟进查询生成错误] {str(e)}")
            return []