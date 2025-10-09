from typing import Dict, List, Tuple
import re
from graphrag_agent.evaluation.core.base_metric import BaseMetric

class ReasoningCoherence(BaseMetric):
    """评估推理过程的连贯性"""
    
    metric_name = "reasoning_coherence"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算推理连贯性得分"""
        self.log("\n======== ReasoningCoherence 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        coherence_scores = []
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            
            # 提取思考过程 - 适配DeepResearchTool的标签格式
            thinking_process = ""
            if hasattr(sample, "thinking_process") and sample.thinking_process:
                thinking_process = sample.thinking_process
                self.log(f"  从thinking_process字段获取思考过程，长度: {len(thinking_process)}")
            else:
                # 尝试从答案中提取思考过程 - 作为后备方案
                thinking_pattern = r'<think>(.*?)</think>'
                thinking_match = re.search(thinking_pattern, answer, re.DOTALL)
                if thinking_match:
                    thinking_process = thinking_match.group(1).strip()
                    self.log(f"  从答案中提取思考过程，长度: {len(thinking_process)}")

            if thinking_process:
                thinking_process = thinking_match.group(1).strip()
                self.log(f"  成功提取思考过程，长度: {len(thinking_process)}")
                
                # 评估思考过程的结构
                # 1. 检查搜索查询标记的存在
                search_queries = re.findall(r'>\s*\d+\.\s*(.*?)\n', thinking_process)
                
                # 2. 检查段落和思考步骤
                paragraphs = re.split(r'\n\n+', thinking_process)
                
                # 评估结构
                has_queries = len(search_queries) > 0
                has_structure = len(paragraphs) > 1
                
                self.log(f"  检测到 {len(search_queries)} 个搜索查询")
                self.log(f"  检测到 {len(paragraphs)} 个思考段落")
                
                # 基于结构特性计算规则得分
                structure_score = 0.6  # 默认给予较高的基础分数
                if has_queries:
                    structure_score += 0.1 * min(3, len(search_queries))  # 每个查询最多加0.1，最多0.3
                
                if has_structure and len(paragraphs) > 3:
                    structure_score += 0.1  # 如果有明确的段落结构，再加0.1
                
                # 确保分数不超过1.0
                structure_score = min(1.0, structure_score)
                
                self.log(f"  结构特性得分: {structure_score:.4f}")
                
                # 尝试使用LLM评估连贯性
                if self.llm:
                    prompt = f"""
                    评估以下思考过程的连贯性和逻辑性，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 思考过程逻辑清晰，步骤连贯，每一步都有合理的推导；搜索查询针对性强，相互补充；结论与推理过程一致
                    - 中分(0.4-0.7): 思考过程基本合理，但可能存在一些逻辑跳跃或冗余；查询之间关联度一般；结论与推理过程大致吻合
                    - 低分(0.0-0.3): 思考过程混乱，缺乏逻辑性和连贯性；查询无关联或重复；结论与推理过程脱节
                    
                    问题: {question}
                    思考过程:
                    {thinking_process}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            llm_score = float(score_match.group(1))
                            # 确保在0-1范围内
                            llm_score = max(0.0, min(1.0, llm_score))
                            self.log(f"  LLM评估的连贯性得分: {llm_score:.4f}")
                            
                            # 取规则得分和LLM得分中的较高值
                            coherence = max(structure_score, llm_score)
                            self.log(f"  最终连贯性得分(取较高值): {coherence:.4f}")
                        else:
                            # 如果LLM未返回有效分数，使用规则得分
                            coherence = structure_score
                            self.log(f"  LLM未返回有效分数，使用规则得分: {coherence:.4f}")
                    except Exception as e:
                        # LLM调用出错，使用规则得分
                        self.log(f"  LLM评估时出错: {e}")
                        coherence = structure_score
                        self.log(f"  使用规则得分: {coherence:.4f}")
                else:
                    # 无LLM可用，使用规则得分
                    coherence = structure_score
                    self.log(f"  无LLM可用，使用规则得分: {coherence:.4f}")
            else:
                # 没有找到思考过程，尝试LLM回退
                if self.llm:
                    self.log("  未找到思考过程，尝试LLM回退评估整体回答的连贯性")
                    
                    prompt = f"""
                    未找到显式的思考过程，请评估整体回答的连贯性和逻辑性，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 回答逻辑清晰，步骤连贯，有明确的推导过程
                    - 中分(0.4-0.7): 回答基本合理，但可能存在一些逻辑跳跃
                    - 低分(0.0-0.3): 回答混乱，缺乏逻辑性和连贯性
                    
                    问题: {question}
                    回答:
                    {answer}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            coherence = float(score_match.group(1))
                            # 确保在0-1范围内
                            coherence = max(0.0, min(1.0, coherence))
                            self.log(f"  LLM回退评估的连贯性得分: {coherence:.4f}")
                        else:
                            # 使用默认分数
                            coherence = 0.6
                            self.log(f"  LLM未返回有效分数，使用默认分数: {coherence:.4f}")
                    except Exception as e:
                        # LLM调用出错，使用默认分数
                        self.log(f"  LLM回退评估时出错: {e}")
                        coherence = 0.6
                        self.log(f"  使用默认分数: {coherence:.4f}")
                else:
                    # 无LLM可用，使用默认分数
                    coherence = 0.6
                    self.log(f"  未找到思考过程且无LLM可用，使用默认分数: {coherence:.4f}")
            
            coherence_scores.append(coherence)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        self.log(f"\n推理连贯性平均得分: {avg_coherence:.4f}")
        self.log("======== ReasoningCoherence 计算结束 ========\n")
        
        return {"reasoning_coherence": avg_coherence}, coherence_scores


class ReasoningDepth(BaseMetric):
    """评估推理过程的深度"""
    
    metric_name = "reasoning_depth"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算推理深度得分"""
        self.log("\n======== ReasoningDepth 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        depth_scores = []
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            
            # 提取思考过程
            thinking_process = ""
            if hasattr(sample, "thinking_process") and sample.thinking_process:
                thinking_process = sample.thinking_process
                self.log(f"  从thinking_process字段获取思考过程，长度: {len(thinking_process)}")
            else:
                # 尝试从答案中提取思考过程 - 作为后备方案
                thinking_pattern = r'<think>(.*?)</think>'
                thinking_match = re.search(thinking_pattern, answer, re.DOTALL)
                if thinking_match:
                    thinking_process = thinking_match.group(1).strip()
                    self.log(f"  从答案中提取思考过程，长度: {len(thinking_process)}")

            if thinking_process:
                thinking_process = thinking_match.group(1).strip()
                self.log(f"  成功提取思考过程，长度: {len(thinking_process)}")
                
                # 分析思考复杂度:
                
                # 1. 检测搜索查询数量
                search_queries = re.findall(r'>\s*\d+\.\s*(.*?)\n', thinking_process)
                query_count = len(search_queries)
                
                # 2. 检测思考深度（段落层级）
                paragraphs = re.split(r'\n\n+', thinking_process)
                depth_level = len(paragraphs)
                
                # 3. 检测"Final Information"部分（有效信息提取）
                info_sections = re.findall(r'\*\*Final Information\*\*(.*?)(?=\n\n|\Z)', thinking_process, re.DOTALL)
                info_count = len(info_sections)
                
                self.log(f"  检测到 {query_count} 个搜索查询")
                self.log(f"  检测到 {depth_level} 个思考段落")
                self.log(f"  检测到 {info_count} 个信息提取部分")
                
                # 基于统计特征计算深度分数
                base_score = 0.6  # 给予较高的基础分
                query_score = min(0.2, 0.05 * query_count)  # 查询数量得分，最高0.2
                para_score = min(0.1, 0.02 * depth_level)   # 段落深度得分，最高0.1
                info_score = min(0.1, 0.05 * info_count)    # 信息提取得分，最高0.1
                feature_score = base_score + query_score + para_score + info_score
                
                self.log(f"  基于特征计算的深度得分: {feature_score:.4f}")
                
                # 尝试使用LLM评估推理深度
                if self.llm:
                    prompt = f"""
                    评估以下思考过程的深度和复杂性，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 思考过程深入，包含多轮迭代查询，检索了多种类型的信息，对检索结果进行了深入分析和整合
                    - 中分(0.4-0.7): 思考过程有一定深度，包含几轮查询，对检索结果进行了基本分析
                    - 低分(0.0-0.3): 思考过程浅显，查询次数少，缺乏深入分析
                    
                    问题: {question}
                    思考过程:
                    {thinking_process}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            llm_score = float(score_match.group(1))
                            # 确保在0-1范围内
                            llm_score = max(0.0, min(1.0, llm_score))
                            self.log(f"  LLM评估的深度得分: {llm_score:.4f}")
                            
                            # 取规则得分和LLM得分中的较高值
                            depth = max(feature_score, llm_score)
                            self.log(f"  最终深度得分(取较高值): {depth:.4f}")
                        else:
                            # 如果LLM未返回有效分数，使用规则得分
                            depth = feature_score
                            self.log(f"  LLM未返回有效分数，使用规则得分: {depth:.4f}")
                    except Exception as e:
                        # LLM调用出错，使用规则得分
                        self.log(f"  LLM评估时出错: {e}")
                        depth = feature_score
                        self.log(f"  使用规则得分: {depth:.4f}")
                else:
                    # 无LLM可用，使用规则得分
                    depth = feature_score
                    self.log(f"  无LLM可用，使用规则得分: {depth:.4f}")
            else:
                # 没有找到思考过程，尝试LLM回退
                if self.llm:
                    self.log("  未找到思考过程，尝试LLM回退评估整体回答的深度")
                    
                    prompt = f"""
                    未找到显式的思考过程，请评估整体回答的分析深度和复杂性，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 回答深入分析问题，包含多个层次的推理和丰富的信息
                    - 中分(0.4-0.7): 回答有一定深度，包含基本的推理和必要信息
                    - 低分(0.0-0.3): 回答浅显，缺乏深入分析
                    
                    问题: {question}
                    回答:
                    {answer}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            depth = float(score_match.group(1))
                            # 确保在0-1范围内
                            depth = max(0.0, min(1.0, depth))
                            self.log(f"  LLM回退评估的深度得分: {depth:.4f}")
                        else:
                            # 使用默认分数
                            depth = 0.6
                            self.log(f"  LLM未返回有效分数，使用默认分数: {depth:.4f}")
                    except Exception as e:
                        # LLM调用出错，使用默认分数
                        self.log(f"  LLM回退评估时出错: {e}")
                        depth = 0.6
                        self.log(f"  使用默认分数: {depth:.4f}")
                else:
                    # 无LLM可用，使用默认分数
                    depth = 0.6
                    self.log(f"  未找到思考过程且无LLM可用，使用默认分数: {depth:.4f}")
            
            depth_scores.append(depth)
        
        avg_depth = sum(depth_scores) / len(depth_scores) if depth_scores else 0.0
        
        self.log(f"\n推理深度平均得分: {avg_depth:.4f}")
        self.log("======== ReasoningDepth 计算结束 ========\n")
        
        return {"reasoning_depth": avg_depth}, depth_scores


class IterativeImprovementMetric(BaseMetric):
    """评估迭代改进效果"""
    
    metric_name = "iterative_improvement"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算迭代改进得分"""
        self.log("\n======== IterativeImprovement 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        improvement_scores = []
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            
            # 提取思考过程
            thinking_process = ""
            if hasattr(sample, "thinking_process") and sample.thinking_process:
                thinking_process = sample.thinking_process
                self.log(f"  从thinking_process字段获取思考过程，长度: {len(thinking_process)}")
            else:
                # 尝试从答案中提取思考过程 - 作为后备方案
                thinking_pattern = r'<think>(.*?)</think>'
                thinking_match = re.search(thinking_pattern, answer, re.DOTALL)
                if thinking_match:
                    thinking_process = thinking_match.group(1).strip()
                    self.log(f"  从答案中提取思考过程，长度: {len(thinking_process)}")

            if thinking_process:
                thinking_process = thinking_match.group(1).strip()
                self.log(f"  成功提取思考过程，长度: {len(thinking_process)}")
                
                # 识别迭代结构 - DeepResearchTool使用的是带编号的查询和信息提取段落
                # 查找类似 "> 1. 查询内容" 的模式
                iterations = re.findall(r'>\s*(\d+)\.\s*(.*?)\n', thinking_process)
                iteration_count = len(iterations)
                
                # 提取每轮迭代后的信息提取部分
                info_sections = re.findall(r'\*\*Final Information\*\*(.*?)(?=\n\n|\Z)', thinking_process, re.DOTALL)
                
                self.log(f"  检测到 {iteration_count} 个迭代查询")
                self.log(f"  检测到 {len(info_sections)} 个信息提取部分")
                
                # 基于迭代数量计算基础得分
                base_score = 0.5  # 给予中等基础分
                
                if iteration_count >= 1:
                    # 迭代次数越多，得分越高
                    iteration_bonus = min(0.3, 0.1 * iteration_count)  # 每次迭代加0.1，最多0.3
                    base_score += iteration_bonus
                
                self.log(f"  基于迭代次数的得分: {base_score:.4f}")
                
                # 尝试LLM评估
                if self.llm and iteration_count >= 2:
                    # 对比第一次和最后一次查询
                    first_query = iterations[0][1] if iterations else ""
                    last_query = iterations[-1][1] if iterations else ""
                    
                    # 对比第一次和最后一次信息提取
                    first_info = info_sections[0] if info_sections else ""
                    last_info = info_sections[-1] if info_sections else ""
                    
                    self.log(f"  首次查询: {first_query}")
                    self.log(f"  最终查询: {last_query}")
                    
                    # 使用LLM评估迭代改进程度
                    prompt = f"""
                    评估以下推理过程中的迭代改进程度，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 多轮迭代中查询策略明显改进，后期查询更精准；后期提取的信息质量明显高于初期
                    - 中分(0.4-0.7): 多轮迭代中查询有一定演进，信息质量有所提升
                    - 低分(0.0-0.3): 多轮迭代但查询几乎无变化，信息质量无明显提升
                    
                    问题: {question}
                    迭代次数: {iteration_count}
                    
                    首次查询: {first_query}
                    最终查询: {last_query}
                    
                    首次提取信息: {first_info}
                    最终提取信息: {last_info}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            llm_score = float(score_match.group(1))
                            # 确保在0-1范围内
                            llm_score = max(0.0, min(1.0, llm_score))
                            self.log(f"  LLM评估的改进得分: {llm_score:.4f}")
                            
                            # 取规则得分和LLM得分中的较高值
                            improvement = max(base_score, llm_score)
                            self.log(f"  最终改进得分(取较高值): {improvement:.4f}")
                        else:
                            # 如果LLM未返回有效分数，使用规则得分
                            improvement = base_score
                            self.log(f"  LLM未返回有效分数，使用规则得分: {improvement:.4f}")
                    except Exception as e:
                        # LLM调用出错，使用规则得分
                        self.log(f"  LLM评估时出错: {e}")
                        improvement = base_score
                        self.log(f"  使用规则得分: {improvement:.4f}")
                else:
                    # 无LLM可用或迭代次数不足，使用规则得分
                    improvement = base_score
                    if iteration_count < 2:
                        self.log(f"  迭代次数不足2次，使用规则得分: {improvement:.4f}")
                    else:
                        self.log(f"  无LLM可用，使用规则得分: {improvement:.4f}")
            else:
                # 没有找到思考过程，尝试LLM回退
                if self.llm:
                    self.log("  未找到思考过程，尝试LLM回退评估整体回答的迭代性")
                    
                    prompt = f"""
                    未找到显式的思考过程，请评估整体回答是否展现了从简单到复杂、从浅层到深层的推理过程，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 回答明显展示了递进式的分析过程，从基础概念逐步深入
                    - 中分(0.4-0.7): 回答包含一定的递进分析，但层次不够分明
                    - 低分(0.0-0.3): 回答几乎没有递进分析过程
                    
                    问题: {question}
                    回答:
                    {answer}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            improvement = float(score_match.group(1))
                            # 确保在0-1范围内
                            improvement = max(0.0, min(1.0, improvement))
                            self.log(f"  LLM回退评估的迭代性得分: {improvement:.4f}")
                        else:
                            # 使用默认分数
                            improvement = 0.5
                            self.log(f"  LLM未返回有效分数，使用默认分数: {improvement:.4f}")
                    except Exception as e:
                        # LLM调用出错，使用默认分数
                        self.log(f"  LLM回退评估时出错: {e}")
                        improvement = 0.5
                        self.log(f"  使用默认分数: {improvement:.4f}")
                else:
                    # 无LLM可用，使用默认分数
                    improvement = 0.5
                    self.log(f"  未找到思考过程且无LLM可用，使用默认分数: {improvement:.4f}")
            
            improvement_scores.append(improvement)
        
        avg_improvement = sum(improvement_scores) / len(improvement_scores) if improvement_scores else 0.0
        
        self.log(f"\n迭代改进平均得分: {avg_improvement:.4f}")
        self.log("======== IterativeImprovement 计算结束 ========\n")
        
        return {"iterative_improvement": avg_improvement}, improvement_scores


class KnowledgeGraphUtilizationMetric(BaseMetric):
    """评估知识图谱利用程度（专为DeeperResearchTool设计）"""
    
    metric_name = "knowledge_graph_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算知识图谱利用率得分"""
        self.log("\n======== KnowledgeGraphUtilization 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        utilization_scores = []
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            
            # 尝试从答案中提取知识图谱信息
            entity_pattern = r'核心相关实体[：:]\s*(.*?)(?=\n|$)'
            entity_match = re.search(entity_pattern, answer, re.IGNORECASE)
            
            # 从答案和思考过程中提取实体和关系信息
            entity_refs_in_answer = len(re.findall(r'实体|entity', answer, re.IGNORECASE))
            relation_refs_in_answer = len(re.findall(r'关系|relation', answer, re.IGNORECASE))
            
            # 默认分数较高
            has_graph_info = False
            has_community_info = False
            
            if entity_match:
                entities = entity_match.group(1).strip()
                self.log(f"  找到实体信息: {entities}")
                has_graph_info = True
            
            # 检查是否提到社区信息
            community_pattern = r'相关知识社区[：:]\s*(.*?)(?=\n|$)'
            community_match = re.search(community_pattern, answer, re.IGNORECASE)
            
            if community_match:
                community = community_match.group(1).strip()
                self.log(f"  找到社区信息: {community}")
                has_community_info = True
            
            # 获取思考过程 - 从thinking_process字段
            thinking_process = ""
            if hasattr(sample, "thinking_process") and sample.thinking_process:
                thinking_process = sample.thinking_process
                self.log(f"  从thinking_process字段获取思考过程，长度: {len(thinking_process)}")
            else:
                # 尝试从答案中提取思考过程 - 作为后备方案
                thinking_pattern = r'<think>(.*?)</think>'
                thinking_match = re.search(thinking_pattern, answer, re.DOTALL)
                if thinking_match:
                    thinking_process = thinking_match.group(1).strip()
                    self.log(f"  从答案中提取思考过程，长度: {len(thinking_process)}")
            
            if thinking_process:
                # 检查思考过程中是否提到了图谱分析
                graph_mentions = re.findall(r'知识图谱|实体|关系|社区', thinking_process, re.IGNORECASE)
                graph_mention_count = len(graph_mentions)
                self.log(f"  思考过程中提到图谱相关概念 {graph_mention_count} 次")
                
                # 基础分数计算
                base_score = 0.2  # 基础分
                
                # 思考过程中提到图谱分数
                mention_score = min(0.3, graph_mention_count * 0.03)
                
                # 实体信息和社区信息分数
                entity_score = 0.25 if has_graph_info else 0.0
                community_score = 0.25 if has_community_info else 0.0
                
                # 总分
                rule_score = base_score + mention_score + entity_score + community_score
                self.log(f"  规则计算的图谱利用率: {rule_score:.4f}")
                
                # 尝试使用LLM进行更全面的评估
                if self.llm:
                    prompt = f"""
                    评估以下回答和思考过程对知识图谱信息的利用程度，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 回答充分利用了知识图谱中的实体、关系和社区信息，进行了深入的图谱分析
                    - 中分(0.4-0.7): 回答部分利用了知识图谱信息，对图谱进行了基本分析
                    - 低分(0.0-0.3): 回答几乎不利用知识图谱信息
                    
                    问题: {question}
                    思考过程:
                    {thinking_process[:1000]}...
                    
                    回答:
                    {answer[:500]}...
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            llm_score = float(score_match.group(1))
                            # 确保在0-1范围内
                            llm_score = max(0.0, min(1.0, llm_score))
                            self.log(f"  LLM评估的图谱利用率: {llm_score:.4f}")
                            
                            # 取规则分数和LLM分数中的较高值
                            utilization = max(rule_score, llm_score)
                            self.log(f"  最终图谱利用率得分(取较高值): {utilization:.4f}")
                        else:
                            utilization = rule_score
                            self.log(f"  LLM未返回有效分数，使用规则得分: {utilization:.4f}")
                    except Exception as e:
                        self.log(f"  LLM评估时出错: {e}")
                        utilization = rule_score
                        self.log(f"  出错，使用规则得分: {utilization:.4f}")
                else:
                    utilization = rule_score
                    self.log(f"  无LLM可用，使用规则得分: {utilization:.4f}")
            else:
                # 没有找到思考过程，基于答案内容计算简化得分
                base_score = 0.2
                entity_score = 0.25 if has_graph_info else 0.0
                community_score = 0.25 if has_community_info else 0.0
                mentions_score = min(0.3, (entity_refs_in_answer + relation_refs_in_answer) * 0.05)
                
                rule_score = base_score + entity_score + community_score + mentions_score
                self.log(f"  未找到思考过程，简化计算的图谱利用率: {rule_score:.4f}")
                
                # 尝试LLM回退
                if self.llm:
                    prompt = f"""
                    未找到显式的思考过程，请评估整体回答中对知识图谱(实体、关系、社区等)信息的利用程度，给出0到1的分数。
                    评分标准:
                    - 高分(0.8-1.0): 回答中明确引用了知识图谱中的实体和关系，并进行了相关分析
                    - 中分(0.4-0.7): 回答中部分引用了知识图谱信息，或隐含地使用了这些信息
                    - 低分(0.0-0.3): 回答几乎未利用知识图谱信息
                    
                    问题: {question}
                    回答:
                    {answer}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            llm_score = float(score_match.group(1))
                            # 确保在0-1范围内
                            llm_score = max(0.0, min(1.0, llm_score))
                            self.log(f"  LLM回退评估的图谱利用率: {llm_score:.4f}")
                            
                            # 取规则分数和LLM分数中的较高值
                            utilization = max(rule_score, llm_score)
                            self.log(f"  最终图谱利用率得分(取较高值): {utilization:.4f}")
                        else:
                            utilization = rule_score
                            self.log(f"  LLM未返回有效分数，使用规则得分: {utilization:.4f}")
                    except Exception as e:
                        self.log(f"  LLM回退评估时出错: {e}")
                        utilization = rule_score
                        self.log(f"  出错，使用规则得分: {utilization:.4f}")
                else:
                    utilization = rule_score
                    self.log(f"  无LLM可用，使用规则得分: {utilization:.4f}")
            
            utilization_scores.append(utilization)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.0
        
        self.log(f"\n知识图谱利用率平均得分: {avg_utilization:.4f}")
        self.log("======== KnowledgeGraphUtilization 计算结束 ========\n")
        
        return {"knowledge_graph_utilization": avg_utilization}, utilization_scores