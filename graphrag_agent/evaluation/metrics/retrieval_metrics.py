import re
from typing import Dict, List, Tuple
from graphrag_agent.evaluation.core.base_metric import BaseMetric
from graphrag_agent.evaluation.core.evaluation_data import RetrievalEvaluationData
from graphrag_agent.evaluation.preprocessing.reference_extractor import extract_references_from_answer
from graphrag_agent.evaluation.preprocessing.text_cleaner import clean_references, clean_thinking_process

class RetrievalPrecision(BaseMetric):
    """检索精确率评估指标"""
    
    metric_name = "retrieval_precision"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索精确率
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== RetrievalPrecision 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        self.log(f"LLM可用: {'是' if self.llm else '否'}")
        
        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"检索实体列表长度: {len(retrieved_entities)}")
        self.log(f"引用实体列表长度: {len(referenced_entities)}")
        
        precision_scores = []
        for idx, (retr_entities, ref_entities) in enumerate(zip(retrieved_entities, referenced_entities)):
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  检索到的实体数量: {len(retr_entities) if retr_entities else 0}")
            self.log(f"  引用的实体数量: {len(ref_entities) if ref_entities else 0}")
                
            # 详细打印实体信息
            if retr_entities:
                self.log(f"  检索实体: {retr_entities[:5]}{'...' if len(retr_entities) > 5 else ''}")
            if ref_entities:
                self.log(f"  引用实体: {ref_entities[:5]}{'...' if len(ref_entities) > 5 else ''}")
            
            # 如果没有检索到实体或引用实体，给予基础分
            if not retr_entities or not ref_entities:
                base_score = 0.3
                self.log(f"  没有检索到实体或引用实体，使用基础分: {base_score}")
                
                # 如果有LLM，尝试回退评估
                if self.llm:
                    # 获取样本
                    sample = data.samples[idx]
                    llm_score = self._get_llm_precision_score(sample, retr_entities, ref_entities)
                    
                    if llm_score > base_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        precision_scores.append(llm_score)
                        continue
                
                precision_scores.append(base_score)
                continue
            
            # 规则匹配评分
            matched, rule_score = self._calculate_rule_precision(retr_entities, ref_entities)
            
            self.log(f"  匹配的实体数量: {matched}")
            self.log(f"  规则精确率分数: {rule_score:.4f}")
            
            # 如果规则评分不佳，使用LLM回退
            if rule_score <= 0.5 and self.llm:
                self.log(f"  规则精确率不理想，尝试使用LLM评估")
                
                # 获取样本
                sample = data.samples[idx]
                llm_score = self._get_llm_precision_score(sample, retr_entities, ref_entities)
                
                # 采用较高的分数
                if llm_score > rule_score:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    precision_scores.append(llm_score)
                    continue
            
            precision_scores.append(rule_score)
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.3
        
        self.log(f"总体评分分布: 最低={min(precision_scores):.4f}, 最高={max(precision_scores):.4f}, 平均={avg_precision:.4f}")
        self.log("完成检索精确率评估")
        
        return {"retrieval_precision": avg_precision}, precision_scores

    def _calculate_rule_precision(self, retr_entities, ref_entities):
        """计算规则匹配精确率"""
        # 实体字符串预处理
        retr_entities_str = [str(e).lower() for e in retr_entities]
        ref_entities_str = [str(e).lower() for e in ref_entities]

        # 1. 直接ID匹配 - 检查引用实体ID是否出现在检索实体文本中
        direct_matches = 0
        for ref_id in ref_entities_str:
            for retr_entity in retr_entities_str:
                if ref_id in retr_entity:
                    direct_matches += 1
                    break
        
        # 2. 数字ID匹配
        num_matches = 0
        for ref_id in ref_entities_str:
            ref_num = re.search(r'\d+', ref_id)
            if ref_num and any(ref_num.group() in retr for retr in retr_entities_str):
                num_matches += 1
        
        # 使用最高的匹配数
        matched = max(direct_matches, num_matches)
        
        # 3. 计算分数
        if matched > 0:
            # 有匹配 - 根据匹配比例给分
            return matched, max(0.3, 0.3 + 0.7 * (matched / len(ref_entities_str)))
        else:
            # 无匹配 - 返回基础分
            return 0, 0.3
    
    def _get_llm_precision_score(self, sample, retr_entities, ref_entities) -> float:
        """
        使用LLM评估检索精确率
        
        Args:
            sample: 评估样本
            retr_entities: 检索到的实体
            ref_entities: 引用的实体
            
        Returns:
            float: LLM评估的精确率分数
        """
        question = sample.question
        agent_type = sample.agent_type
        answer = sample.system_answer
        
        # 准备LLM提示
        retr_str = ", ".join([str(e) for e in retr_entities[:10]]) if retr_entities else "无检索实体"
        ref_str = ", ".join([str(e) for e in ref_entities[:10]]) if ref_entities else "无引用实体"
        
        prompt = f"""
        请评估以下检索到的实体与用户引用实体的匹配程度，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        检索到的实体: [{retr_str}]
        用户引用的实体: [{ref_str}]
        
        回答(部分): {answer[:150]}...
        
        评分标准:
        - 高分(0.8-1.0): 引用实体全部或大部分存在于检索实体中，且高度相关
        - 中分(0.4-0.7): 引用实体部分存在于检索实体中，或存在一定的相关性
        - 低分(0.0-0.3): 引用实体几乎不在检索实体中，或相关性很低
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类的LLM回退评分方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)

class RetrievalUtilization(BaseMetric):
    """检索利用率评估指标"""
    
    metric_name = "retrieval_utilization"

    def __init__(self, config):
        """初始化评估指标"""
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索利用率
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== RetrievalUtilization 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        self.log(f"LLM可用: {'是' if self.llm else '否'}")
        
        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"检索实体列表长度: {len(retrieved_entities)}")
        self.log(f"引用实体列表长度: {len(referenced_entities)}")
        
        utilization_scores = []
        for idx, (retr_entities, ref_entities) in enumerate(zip(retrieved_entities, referenced_entities)):
            self.log(f"\n样本 {idx+1}:")
                
            # 检查数据格式
            if not isinstance(retr_entities, list):
                self.log(f"  检索实体不是列表类型，而是 {type(retr_entities)}")
                retr_entities = []
            if not isinstance(ref_entities, list):
                self.log(f"  引用实体不是列表类型，而是 {type(ref_entities)}")
                ref_entities = []
                    
            # 确保所有元素都是字符串
            retr_entities = [str(e) for e in retr_entities]
            ref_entities = [str(e) for e in ref_entities]
                
            self.log(f"  检索到的实体数量: {len(retr_entities)}")
            self.log(f"  引用的实体数量: {len(ref_entities)}")
                
            # 详细打印实体ID
            if retr_entities:
                self.log(f"  检索实体: {retr_entities[:5]}{'...' if len(retr_entities) > 5 else ''}")
            if ref_entities:
                self.log(f"  引用实体: {ref_entities[:5]}{'...' if len(ref_entities) > 5 else ''}")
            
            # 如果没有引用实体或检索实体，给予基础分
            if not ref_entities or not retr_entities:
                base_score = 0.3
                self.log(f"  没有引用实体或检索实体，使用基础分: {base_score}")
                
                # 如果有LLM，尝试回退评估
                if self.llm:
                    # 获取样本
                    sample = data.samples[idx]
                    llm_score = self._get_llm_utilization_score(sample, retr_entities, ref_entities)
                    
                    if llm_score > base_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        utilization_scores.append(llm_score)
                        continue
                
                utilization_scores.append(base_score)
                continue
            
            # 规则匹配评分
            matches_found, rule_score = self._calculate_rule_utilization(retr_entities, ref_entities)
            
            self.log(f"  在检索结果中找到的引用实体数量: {matches_found}")
            self.log(f"  规则利用率分数: {rule_score:.4f}")
            
            # 如果规则评分不佳，使用LLM回退
            if rule_score <= 0.5 and self.llm:
                self.log(f"  规则利用率不理想，尝试使用LLM评估")
                
                # 获取样本
                sample = data.samples[idx]
                llm_score = self._get_llm_utilization_score(sample, retr_entities, ref_entities)
                
                # 采用较高的分数
                if llm_score > rule_score:
                    self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                    utilization_scores.append(llm_score)
                    continue
            
            utilization_scores.append(rule_score)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.3
        
        self.log(f"总体评分分布: 最低={min(utilization_scores):.4f}, 最高={max(utilization_scores):.4f}, 平均={avg_utilization:.4f}")
        self.log("完成检索利用率评估")
        
        return {"retrieval_utilization": avg_utilization}, utilization_scores

    def _calculate_rule_utilization(self, retr_entities, ref_entities):
        """计算规则匹配利用率"""
        # 标准化处理
        retr_norm = [str(e).lower() for e in retr_entities]
        ref_norm = [str(e).lower() for e in ref_entities]
        
        # 1. 直接ID匹配
        direct_matches = 0
        for ref_id in ref_norm:
            if any(ref_id in retr for retr in retr_norm):
                direct_matches += 1
        
        # 2. 数字ID匹配
        num_matches = 0
        for ref_id in ref_norm:
            ref_num = re.search(r'\d+', ref_id)
            if ref_num and any(ref_num.group() in retr for retr in retr_norm):
                num_matches += 1
        
        # 使用最高的匹配数
        matched = max(direct_matches, num_matches)
        
        # 计算利用率
        if matched > 0:
            # 有匹配，计算基于匹配比例的分数
            return matched, max(0.3, 0.3 + 0.7 * (matched / len(ref_norm)))
        else:
            # 无匹配，但检查字符串相似性
            combined_retr = " ".join(retr_norm)
            for ref in ref_norm:
                # 检查部分匹配
                if any(token in combined_retr for token in ref.split() if len(token) > 3):
                    return 1, 0.4  # 有部分匹配，给予略高于基础的分数
            
            # 无任何匹配
            return 0, 0.3
    
    def _get_llm_utilization_score(self, sample, retr_entities, ref_entities) -> float:
        """
        使用LLM评估检索利用率
        
        Args:
            sample: 评估样本
            retr_entities: 检索到的实体
            ref_entities: 引用的实体
            
        Returns:
            float: LLM评估的利用率分数
        """
        question = sample.question
        agent_type = sample.agent_type
        answer = sample.system_answer
        
        # 准备LLM提示
        retr_str = ", ".join([str(e) for e in retr_entities[:10]]) if retr_entities else "无检索实体"
        ref_str = ", ".join([str(e) for e in ref_entities[:10]]) if ref_entities else "无引用实体"
        
        prompt = f"""
        请评估系统在回答用户问题时对检索实体的利用程度，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        检索到的实体: [{retr_str}]
        用户引用的实体: [{ref_str}]
        
        系统回答(部分): {answer[:200]}...
        
        评分标准:
        - 高分(0.8-1.0): 系统充分利用了检索到的实体中的关键信息，将它们有效地整合到回答中
        - 中分(0.4-0.7): 系统部分利用了检索到的实体信息，但可能没有完全发挥其价值
        - 低分(0.0-0.3): 系统几乎没有利用检索到的实体信息，或利用不当
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类的LLM回退评分方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)

class RetrievalLatency(BaseMetric):
    """检索延迟评估指标"""
    
    metric_name = "retrieval_latency"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索延迟
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== RetrievalLatency 计算日志 ========")
        
        latency_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            # 获取检索时间
            retrieval_time = sample.retrieval_time
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  检索时间: {retrieval_time:.4f}秒")
            
            # 添加到结果列表
            latency_scores.append(retrieval_time)
        
        # 计算平均延迟
        avg_latency = sum(latency_scores) / len(latency_scores) if latency_scores else 0.0
        
        self.log(f"\n检索平均延迟: {avg_latency:.4f}秒")
        self.log("======== RetrievalLatency 计算结束 ========\n")
        
        return {"retrieval_latency": avg_latency}, latency_scores


class ChunkUtilization(BaseMetric):
    """文本块利用率评估指标"""
    
    metric_name = "chunk_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算文本块利用率 - 从引用数据中提取的chunk被利用的程度
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== ChunkUtilization 计算日志 ========")
        
        chunk_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        self.log(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            self.log(f"\n样本 {idx+1}:")
            question = sample.question
            answer = sample.system_answer
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            # 从原始回答中提取引用的chunks
            refs = extract_references_from_answer(sample.system_answer)
            chunk_ids = refs.get("chunks", [])
            
            self.log(f"  问题: {question[:50]}...")
            self.log(f"  Agent类型: {agent_type}")
            self.log(f"  提取的文本块ID数量: {len(chunk_ids)}")
            if chunk_ids:
                self.log(f"  文本块ID样例: {chunk_ids[:3]}{'...' if len(chunk_ids) > 3 else ''}")
            
            # 如果没有找到文本块ID，给予基础分并尝试LLM回退
            if not chunk_ids:
                base_score = 0.3
                self.log("  没有找到文本块ID，使用基础分: 0.3")
                
                # 尝试使用LLM评估
                if self.llm:
                    llm_score = self._llm_fallback_for_chunk(sample, [])
                    if llm_score > base_score:
                        self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                        chunk_scores.append(llm_score)
                        continue
                
                chunk_scores.append(base_score)
                continue
            
            # 在回答中查找chunk内容的使用情况
            answer_text = clean_references(sample.system_answer)
            answer_text = clean_thinking_process(answer_text)
            self.log(f"  清理后的答案长度: {len(answer_text)}")
            
            # 如果没有Neo4j客户端，尝试LLM回退
            if not self.neo4j_client:
                # 使用LLM评估，但不提供文本块内容
                if self.llm:
                    self.log("  Neo4j客户端不可用，使用LLM评估")
                    score = self._llm_fallback_for_chunk(sample, chunk_ids)
                    chunk_scores.append(score)
                else:
                    # 没有LLM，使用默认分数
                    self.log("  Neo4j客户端不可用，且无法使用LLM，使用默认分数: 0.4")
                    chunk_scores.append(0.4)
                continue
            
            # 从Neo4j获取chunk内容
            try:
                chunk_texts = []
                total_matches = 0
                chunk_contents = {}  # 用于存储文本块ID到文本内容的映射
                
                for chunk_id in chunk_ids:
                    # 查询文本块内容
                    query = """
                    MATCH (n:__Chunk__) 
                    WHERE n.id = $id 
                    RETURN n.text AS text
                    """
                    
                    result = self.neo4j_client.execute_query(query, {"id": chunk_id})
                    
                    if result.records and len(result.records) > 0:
                        chunk_text = result.records[0].get("text", "")
                        if chunk_text:
                            chunk_texts.append(chunk_text)
                            chunk_contents[chunk_id] = chunk_text
                            self.log(f"  获取到文本块[{chunk_id}]，长度: {len(chunk_text)}")
                            
                            # 计算文本块内容在回答中的利用率
                            # 将文本块分成关键短语
                            key_phrases = re.findall(r'\b[\w\u4e00-\u9fa5]{4,}\b', chunk_text)
                            key_phrases = list(set([p for p in key_phrases if len(p) > 3]))
                            
                            if key_phrases:
                                # 计算关键短语在回答中出现的比例
                                matched_phrases = sum(1 for phrase in key_phrases 
                                                    if phrase.lower() in answer_text.lower())
                                match_ratio = matched_phrases / len(key_phrases)
                                total_matches += match_ratio
                                
                                self.log(f"  文本块关键短语数: {len(key_phrases)}, 匹配数: {matched_phrases}")
                                self.log(f"  文本块匹配率: {match_ratio:.4f}")
                
                # 计算平均利用率
                if chunk_texts:
                    chunk_utilization = total_matches / len(chunk_texts)
                    self.log(f"  总体文本块利用率: {chunk_utilization:.4f}")
                    
                    # 如果规则评分较低，尝试LLM回退
                    if chunk_utilization <= 0.4 and self.llm:
                        llm_score = self._llm_fallback_for_chunk(sample, chunk_ids, chunk_contents)
                        if llm_score > chunk_utilization:
                            self.log(f"  LLM回退分数更高({llm_score:.4f})，采用LLM评分")
                            chunk_scores.append(llm_score)
                            continue
                    
                    chunk_scores.append(chunk_utilization)
                else:
                    # 未获取到任何文本块内容，尝试LLM回退
                    self.log("  未能获取任何文本块内容")
                    if self.llm:
                        score = self._llm_fallback_for_chunk(sample, chunk_ids)
                        chunk_scores.append(score)
                    else:
                        # 没有LLM，使用基础分数
                        self.log("  使用基础分数: 0.3")
                        chunk_scores.append(0.3)
                    
            except Exception as e:
                self.log(f"  计算文本块利用率时出错: {e}")
                # 出错时尝试LLM回退
                if self.llm:
                    score = self._llm_fallback_for_chunk(sample, chunk_ids)
                    chunk_scores.append(score)
                else:
                    # 使用默认值
                    self.log("  使用默认分数: 0.4")
                    chunk_scores.append(0.4)
        
        avg_chunk_utilization = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        
        self.log(f"\n文本块平均利用率: {avg_chunk_utilization:.4f}")
        self.log("======== ChunkUtilization 计算结束 ========\n")
        
        return {"chunk_utilization": avg_chunk_utilization}, chunk_scores
    
    def _llm_fallback_for_chunk(self, sample, chunk_ids, chunk_contents=None) -> float:
        """
        使用LLM评估文本块利用率
        
        Args:
            sample: 评估样本
            chunk_ids: 文本块ID列表
            chunk_contents: 文本块ID到内容的映射（可选）
            
        Returns:
            float: LLM评估的文本块利用率分数
        """
        question = sample.question
        answer = sample.system_answer
        agent_type = sample.agent_type
        
        # 清理答案
        cleaned_answer = clean_references(answer)
        cleaned_answer = clean_thinking_process(cleaned_answer)
        
        # 构建提示
        prompt = f"""
        请评估以下AI回答对检索文本块的利用程度，给出0到1的分数。
        
        问题: {question}
        Agent类型: {agent_type}
        
        引用的文本块数量: {len(chunk_ids)}
        """
        
        # 如果有文本块内容，添加到提示中
        if chunk_contents and len(chunk_contents) > 0:
            prompt += "\n\n文本块内容样例:\n"
            # 最多添加3个文本块样例
            for i, (chunk_id, content) in enumerate(chunk_contents.items()):
                if i >= 3:
                    break
                # 截取内容前150个字符
                short_content = content[:150] + ("..." if len(content) > 150 else "")
                prompt += f"文本块[{chunk_id}]: {short_content}\n"
        else:
            prompt += f"\n文本块ID: {', '.join(chunk_ids[:5])}\n"
        
        prompt += f"""
        AI回答(部分):
        {cleaned_answer[:]}...
        
        评分标准:
        - 高分(0.8-1.0): 回答充分利用了文本块内容，有效整合了其中的关键信息
        - 中分(0.4-0.7): 回答部分利用了文本块内容，但可能有漏用或欠缺的信息
        - 低分(0.0-0.3): 回答几乎没有利用文本块内容，或利用度很低
        
        只返回一个0到1之间的数字表示分数，不要有任何其他文字。
        """
        
        # 使用基类的LLM回退评分方法
        return self.get_llm_fallback_score(prompt, default_score=0.4)