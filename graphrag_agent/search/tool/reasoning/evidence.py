from typing import Dict, List
import time
import hashlib

from graphrag_agent.models.get_models import get_llm_model

class EvidenceChainTracker:
    """
    证据链收集和推理跟踪器
    
    收集和管理深度研究过程中的证据链，
    追踪推理步骤使用的证据来源和推理逻辑
    """
    
    def __init__(self):
        """初始化证据链跟踪器"""
        self.llm = get_llm_model()
        self.reasoning_steps = []  # 推理步骤
        self.evidence_items = {}   # 证据项
        self.query_contexts = {}   # 查询上下文
        self.step_counter = 0      # 步骤计数器
        self.confidence_scores = {}  # 证据的可信度评分
        self.contradictions = {}     # 记录相互矛盾的证据
        self.citation_index = {}     # 引用索引
        
    def start_new_query(self, query: str, keywords: Dict[str, List[str]]) -> str:
        """
        开始新的查询跟踪
        
        参数:
            query: 用户查询
            keywords: 查询关键词
            
        返回:
            str: 查询ID
        """
        # 生成查询ID
        query_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()[:10]
        
        # 存储查询上下文
        self.query_contexts[query_id] = {
            "query": query,
            "keywords": keywords,
            "start_time": time.time(),
            "step_ids": []
        }
        
        return query_id
    
    def add_reasoning_step(self, 
                         query_id: str, 
                         search_query: str, 
                         reasoning: str) -> str:
        """
        添加推理步骤
        
        参数:
            query_id: 查询ID
            search_query: 搜索查询
            reasoning: 推理过程
            
        返回:
            str: 步骤ID
        """
        # 生成步骤ID
        step_id = f"step_{self.step_counter}"
        self.step_counter += 1
        
        # 创建步骤记录
        step = {
            "step_id": step_id,
            "query_id": query_id,
            "search_query": search_query,
            "reasoning": reasoning,
            "evidence_ids": [],
            "timestamp": time.time()
        }
        
        # 添加步骤到列表并关联到查询
        self.reasoning_steps.append(step)
        if query_id in self.query_contexts:
            self.query_contexts[query_id]["step_ids"].append(step_id)
        
        return step_id
    
    def add_evidence(self, 
                   step_id: str, 
                   source_id: str, 
                   content: str, 
                   source_type: str) -> str:
        """
        添加证据项
        
        参数:
            step_id: 步骤ID
            source_id: 来源ID（如块ID）
            content: 证据内容
            source_type: 来源类型
            
        返回:
            str: 证据ID
        """
        # 生成证据ID
        evidence_id = hashlib.md5(f"{source_id}:{content[:50]}".encode()).hexdigest()[:10]
        
        # 创建证据记录
        evidence = {
            "evidence_id": evidence_id,
            "source_id": source_id,
            "content": content,
            "source_type": source_type,
            "timestamp": time.time()
        }
        
        # 存储证据并关联到步骤
        self.evidence_items[evidence_id] = evidence
        
        # 查找步骤并添加证据ID
        for step in self.reasoning_steps:
            if step["step_id"] == step_id:
                if evidence_id not in step["evidence_ids"]:
                    step["evidence_ids"].append(evidence_id)
                break
        
        return evidence_id

    def add_evidence_with_confidence(
        self, 
        step_id: str, 
        source_id: str, 
        content: str, 
        source_type: str, 
        confidence=0.5,
        metadata=None
    ):
        """
        添加带可信度得分的证据
        
        参数:
            step_id: 步骤ID
            source_id: 来源ID
            content: 证据内容
            source_type: 来源类型
            confidence: 可信度评分(0-1)
            metadata: 元数据字典
            
        返回:
            str: 证据ID
        """
        # 添加基础证据
        evidence_id = self.add_evidence(step_id, source_id, content, source_type)
        
        # 保存可信度评分
        self.confidence_scores[evidence_id] = confidence
        
        # 添加元数据（如果有）
        if metadata:
            if evidence_id in self.evidence_items:
                self.evidence_items[evidence_id]["metadata"] = metadata
        
        # 更新引用索引
        self._update_citation_index(evidence_id, content)
        
        return evidence_id
    
    def _update_citation_index(self, evidence_id, content):
        """
        更新引用索引，提取关键短语作为索引
        
        参数:
            evidence_id: 证据ID
            content: 证据内容
        """
        # 分析内容，提取关键短语
        key_phrases = self._extract_key_phrases(content)
        
        # 将关键短语添加到引用索引
        for phrase in key_phrases:
            if phrase not in self.citation_index:
                self.citation_index[phrase] = []
            
            if evidence_id not in self.citation_index[phrase]:
                self.citation_index[phrase].append(evidence_id)
    
    def _extract_key_phrases(self, content):
        """
        从文本中提取关键短语
        
        参数:
            content: 文本内容
            
        返回:
            list: 关键短语列表
        """
        # 使用简单的启发式方法提取关键短语
        # 1. 划分为句子
        import re
        sentences = re.split(r'[.!?。！？]', content)
        
        # 2. 从每个句子中提取名词短语和数值
        key_phrases = []
        
        # 数值模式
        number_pattern = r'\d+(?:[.,]\d+)?(?:\s*%|\s*元|\s*美元|\s*人民币)?'
        
        # 名词短语模式（简化版）
        noun_phrase_pattern = r'[A-Z][a-z]+\s+(?:[a-z]+\s+){0,2}[a-z]+'
        
        for sentence in sentences:
            # 提取数值
            numbers = re.findall(number_pattern, sentence)
            key_phrases.extend(numbers)
            
            # 提取英文名词短语
            noun_phrases = re.findall(noun_phrase_pattern, sentence)
            key_phrases.extend(noun_phrases)
            
            # 提取中文名词短语（简化）
            if len(sentence) > 3:
                # 使用滑动窗口提取短语
                for i in range(len(sentence) - 3):
                    phrase = sentence[i:i+4]
                    if len(phrase.strip()) >= 2:
                        key_phrases.append(phrase.strip())
        
        # 去重并保留最有意义的短语
        return list(set([p for p in key_phrases if len(p) > 1]))
    
    def detect_contradictions(self, evidence_ids):
        """
        检测证据之间的矛盾
        
        参数:
            evidence_ids: 证据ID列表
            
        返回:
            list: 矛盾信息列表
        """
        if len(evidence_ids) < 2:
            return []
            
        contradictions = []
        evidences = [self.evidence_items[eid] for eid in evidence_ids if eid in self.evidence_items]
        
        # 1. 检测数值类矛盾（通过正则表达式）
        import re
        for i in range(len(evidences)):
            for j in range(i+1, len(evidences)):
                # 提取第一个证据中的数值
                content1 = evidences[i]["content"]
                numbers1 = self._extract_numbers_with_context(content1)
                
                # 提取第二个证据中的数值
                content2 = evidences[j]["content"]
                numbers2 = self._extract_numbers_with_context(content2)
                
                # 比较数值
                for num1_info in numbers1:
                    for num2_info in numbers2:
                        # 检查上下文是否相似
                        if self._context_similarity(num1_info["context"], num2_info["context"]) > 0.7:
                            # 检查数值是否不同
                            if abs(num1_info["value"] - num2_info["value"]) > 0.001 * max(num1_info["value"], num2_info["value"]):
                                contradictions.append({
                                    "type": "numerical",
                                    "evidence1": evidence_ids[i],
                                    "evidence2": evidence_ids[j],
                                    "context": num1_info["context"],
                                    "value1": num1_info["value"],
                                    "value2": num2_info["value"]
                                })
        
        # 2. 使用LLM检测语义矛盾
        if hasattr(self, 'llm') and self.llm:
            for i in range(len(evidences)):
                for j in range(i+1, len(evidences)):
                    # 检查是否已经发现数值矛盾
                    if any(c["evidence1"] == evidence_ids[i] and c["evidence2"] == evidence_ids[j] for c in contradictions):
                        continue
                    
                    # 提取内容
                    content1 = evidences[i]["content"]
                    content2 = evidences[j]["content"]
                    
                    # 使用LLM检测矛盾
                    contradiction = self._detect_semantic_contradiction(content1, content2, evidence_ids[i], evidence_ids[j])
                    if contradiction:
                        contradictions.append(contradiction)
        
        # 保存矛盾信息
        for contradiction in contradictions:
            contradiction_id = f"contradiction_{len(self.contradictions)}"
            self.contradictions[contradiction_id] = contradiction
            
        return contradictions
    
    def _extract_numbers_with_context(self, text):
        """
        提取文本中的数值及其上下文
        
        参数:
            text: 文本内容
            
        返回:
            list: 包含数值和上下文的对象列表
        """
        import re
        
        # 数值模式
        number_pattern = r'(\d+(?:[.,]\d+)?(?:\s*%|\s*元|\s*美元|\s*人民币)?)'
        
        # 查找所有匹配
        matches = list(re.finditer(number_pattern, text))
        results = []
        
        for match in matches:
            # 获取数值
            value_str = match.group(1)
            
            # 清理并转换为浮点数
            clean_value = re.sub(r'[^\d.,]', '', value_str).replace(',', '.')
            try:
                value = float(clean_value)
            except:
                continue
            
            # 获取上下文（数值前后20个字符）
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            
            results.append({
                "value": value,
                "original": value_str,
                "context": context
            })
            
        return results
    
    def _context_similarity(self, context1, context2):
        """
        计算两个上下文的相似度
        
        参数:
            context1: 第一个上下文
            context2: 第二个上下文
            
        返回:
            float: 相似度得分(0-1)
        """
        # 实现简单的基于单词重叠的相似度计算
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        
        if not words1 or not words2:
            return 0
            
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def _detect_semantic_contradiction(self, content1, content2, evidence_id1, evidence_id2):
        """
        使用LLM检测语义矛盾
        
        参数:
            content1: 第一个内容
            content2: 第二个内容
            evidence_id1: 第一个证据ID
            evidence_id2: 第二个证据ID
            
        返回:
            Dict或None: 矛盾信息或None（如果没有矛盾）
        """
        prompt = f"""
        分析以下两段内容，判断它们之间是否存在矛盾或不一致：
        
        内容1:
        {content1}
        
        内容2:
        {content2}
        
        如果存在矛盾，请具体说明矛盾点。如果不存在矛盾，请回答"没有矛盾"。
        """
        
        # 调用LLM
        response = self.llm.invoke(prompt)
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        # 判断是否发现矛盾
        if "没有矛盾" in analysis:
            return None
        
        # 提取矛盾点
        contradiction_point = analysis.replace("矛盾点：", "").strip()
        if len(contradiction_point) > 300:
            contradiction_point = contradiction_point[:300] + "..."
            
        return {
            "type": "semantic",
            "evidence1": evidence_id1,
            "evidence2": evidence_id2,
            "analysis": contradiction_point
        }
    
    def generate_citations(self, answer):
        """
        在答案中生成引用标记
        
        参数:
            answer: 答案文本
            
        返回:
            Dict: 包含带引用的答案和引用信息
        """
        citations = []
        
        # 从答案中提取关键语句
        key_statements = self._extract_key_statements(answer)
        
        # 为每个语句查找最匹配的证据
        for statement in key_statements:
            matching_evidence = self._find_matching_evidence(statement)
            if matching_evidence:
                citation = {
                    "statement": statement,
                    "evidence_id": matching_evidence["evidence_id"],
                    "source_id": matching_evidence["source_id"],
                    "confidence": self.confidence_scores.get(matching_evidence["evidence_id"], 0.5)
                }
                citations.append(citation)
        
        # 生成带引用的答案
        cited_answer = self._add_citations_to_answer(answer, citations)
        
        return {
            "cited_answer": cited_answer,
            "citations": citations
        }
    
    def _extract_key_statements(self, text):
        """
        从文本中提取关键语句
        
        参数:
            text: 文本内容
            
        返回:
            list: 关键语句列表
        """
        import re
        
        # 按句子划分
        sentences = re.split(r'([.!?。！？]\s*)', text)
        
        # 合并分隔符和句子
        merged_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                merged_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                merged_sentences.append(sentences[i])
                i += 1
        
        # 筛选出有意义的句子（长度大于10个字符）
        key_statements = [s.strip() for s in merged_sentences if len(s.strip()) > 10]
        
        return key_statements
    
    def _find_matching_evidence(self, statement):
        """
        为语句查找最匹配的证据
        
        参数:
            statement: 语句
            
        返回:
            Dict或None: 匹配的证据或None
        """
        # 提取语句中的关键短语
        key_phrases = self._extract_key_phrases(statement)
        
        # 收集可能匹配的证据
        candidate_evidence_ids = []
        for phrase in key_phrases:
            if phrase in self.citation_index:
                candidate_evidence_ids.extend(self.citation_index[phrase])
        
        # 如果没有候选证据，返回None
        if not candidate_evidence_ids:
            return None
            
        # 计算每个证据的匹配得分
        evidence_scores = {}
        for evidence_id in set(candidate_evidence_ids):
            if evidence_id in self.evidence_items:
                evidence = self.evidence_items[evidence_id]
                # 基础得分 - 出现频率
                base_score = candidate_evidence_ids.count(evidence_id)
                # 可信度加权
                confidence = self.confidence_scores.get(evidence_id, 0.5)
                # 最终得分
                evidence_scores[evidence_id] = base_score * confidence
        
        # 找出得分最高的证据
        if evidence_scores:
            best_evidence_id = max(evidence_scores, key=evidence_scores.get)
            return self.evidence_items[best_evidence_id]
        
        return None
    
    def _add_citations_to_answer(self, answer, citations):
        """
        在答案中添加引用标记
        
        参数:
            answer: 原始答案
            citations: 引用信息列表
            
        返回:
            str: 带引用的答案
        """
        # 为每个引用添加标记
        cited_answer = answer
        
        # 按语句长度从长到短排序，避免替换冲突
        sorted_citations = sorted(citations, key=lambda x: len(x["statement"]), reverse=True)
        
        for i, citation in enumerate(sorted_citations):
            statement = citation["statement"]
            source_id = citation["source_id"]
            citation_mark = f"[{i+1}]"
            
            # 替换语句添加引用标记
            if statement in cited_answer:
                cited_answer = cited_answer.replace(statement, f"{statement}{citation_mark}")
        
        # 添加引用列表
        if citations:
            cited_answer += "\n\n#### 引用\n"
            for i, citation in enumerate(citations):
                cited_answer += f"[{i+1}] {citation['source_id']}\n"
            
        return cited_answer
    
    def get_reasoning_chain(self, query_id: str) -> Dict:
        """
        获取完整的推理链
        
        参数:
            query_id: 查询ID
            
        返回:
            Dict: 推理链，包含步骤和证据
        """
        if query_id not in self.query_contexts:
            return {}
        
        # 获取查询相关的步骤ID
        step_ids = self.query_contexts[query_id]["step_ids"]
        
        # 按时间顺序收集步骤
        steps = []
        for step_id in step_ids:
            for step in self.reasoning_steps:
                if step["step_id"] == step_id:
                    # 复制步骤并添加完整证据
                    step_copy = step.copy()
                    step_copy["evidence"] = []
                    
                    # 添加证据详情
                    for evidence_id in step["evidence_ids"]:
                        if evidence_id in self.evidence_items:
                            evidence_copy = self.evidence_items[evidence_id].copy()
                            # 添加可信度评分
                            evidence_copy["confidence"] = self.confidence_scores.get(evidence_id, 0.5)
                            step_copy["evidence"].append(evidence_copy)
                    
                    steps.append(step_copy)
                    break
        
        # 按时间戳排序
        steps.sort(key=lambda x: x["timestamp"])
        
        # 构建完整推理链
        reasoning_chain = {
            "query": self.query_contexts[query_id]["query"],
            "keywords": self.query_contexts[query_id]["keywords"],
            "start_time": self.query_contexts[query_id]["start_time"],
            "end_time": time.time(),
            "steps": steps,
            "contradiction_count": len([c for c in self.contradictions.values() 
                                     if any(c.get("evidence1", "") == e_id or c.get("evidence2", "") == e_id 
                                         for e_id in 
                                         [e for s in steps for e in s["evidence_ids"]])
                                    ])
        }
        
        return reasoning_chain
    
    def get_step_evidence(self, step_id: str) -> List[Dict]:
        """
        获取特定步骤的证据
        
        Args:
            step_id: 步骤ID
            
        Returns:
            List[Dict]: 证据列表
        """
        # 查找步骤
        for step in self.reasoning_steps:
            if step["step_id"] == step_id:
                # 收集证据
                evidence_list = []
                for evidence_id in step["evidence_ids"]:
                    if evidence_id in self.evidence_items:
                        evidence_list.append(
                            self.evidence_items[evidence_id]
                        )
                return evidence_list
        
        return []
    
    def summarize_reasoning(self, query_id: str) -> Dict:
        """
        总结推理过程
        
        参数:
            query_id: 查询ID
            
        返回:
            Dict: 推理摘要
        """
        chain = self.get_reasoning_chain(query_id)
        if not chain:
            return {"summary": "没有找到相关推理链"}
        
        # 计算统计信息
        steps_count = len(chain.get("steps", []))
        evidence_count = sum(len(step.get("evidence", [])) 
                           for step in chain.get("steps", []))
        
        # 识别关键步骤（有最多证据的步骤）
        key_steps = []
        if steps_count > 0:
            # 按证据数量排序
            sorted_steps = sorted(
                chain.get("steps", []),
                key=lambda x: len(x.get("evidence", [])),
                reverse=True
            )
            
            # 取前3个关键步骤
            key_steps = sorted_steps[:min(3, len(sorted_steps))]
        
        # 计算处理时间
        duration = chain.get("end_time", time.time()) - chain.get("start_time", time.time())
        
        # 生成摘要
        summary = {
            "query": chain.get("query", ""),
            "steps_count": steps_count,
            "evidence_count": evidence_count,
            "duration_seconds": duration,
            "contradiction_count": chain.get("contradiction_count", 0),
            "key_steps": [
                {
                    "step_id": step.get("step_id"),
                    "search_query": step.get("search_query"),
                    "evidence_count": len(step.get("evidence", []))
                }
                for step in key_steps
            ]
        }
        
        return summary
    
    def get_evidence_source_stats(self, query_id: str) -> Dict:
        """
        获取证据来源统计
        
        参数:
            query_id: 查询ID
            
        返回:
            Dict: 证据来源统计
        """
        chain = self.get_reasoning_chain(query_id)
        if not chain:
            return {"sources": {}}
        
        # 收集所有证据
        all_evidence = []
        for step in chain.get("steps", []):
            all_evidence.extend(step.get("evidence", []))
        
        # 按来源类型分组
        sources = {}
        for evidence in all_evidence:
            source_type = evidence.get("source_type", "unknown")
            if source_type not in sources:
                sources[source_type] = 0
            sources[source_type] += 1
        
        return {"sources": sources}