import re
import json
import time
from typing import List, Dict, Any
import logging
import traceback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graphrag_agent.search.tool.reasoning.nlp import extract_between
from graphrag_agent.config.prompts import (
    BEGIN_SEARCH_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
    REASON_PROMPT,
    END_SEARCH_QUERY,
    INITIAL_THINKING_PROMPT,
    HYPOTHESIS_GENERATION_PROMPT,
    HYPOTHESIS_VERIFICATION_PROMPT,
    VERIFICATION_STATUS_PROMPT,
    UPDATE_THINKING_PROMPT,
    COUNTERFACTUAL_ANALYSIS_PROMPT,
    COUNTERFACTUAL_COMPARISON_PROMPT,
)


class ThinkingEngine:
    """
    思考引擎类：负责管理多轮迭代的思考过程
    提供思考历史管理和转换功能
    """
    
    def __init__(self, llm):
        """
        初始化思考引擎
        
        参数:
            llm: 大语言模型实例，用于生成思考内容
        """
        self.llm = llm
        self.all_reasoning_steps = []
        self.msg_history = []
        self.executed_search_queries = []
        self.hypotheses = []       # 存储假设
        self.verification_chain = [] # 验证步骤
        self.reasoning_tree = {}   # 推理树结构
        self.current_branch = "main" # 当前推理分支
    
    def initialize_with_query(self, query: str):
        """使用初始查询初始化思考历史"""
        self.all_reasoning_steps = []
        self.msg_history = [{"role": "user", "content": f'问题:"{query}"\n'}]
        self.executed_search_queries = []
        self.hypotheses = []
        self.verification_chain = []
        self.reasoning_tree = {"main": []} # 初始化主分支
        self.current_branch = "main"
    
    def generate_initial_thinking(self):
        """生成初步思考过程"""
        prompt = INITIAL_THINKING_PROMPT
        response = self.llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": self.msg_history[0]["content"]}
        ])
        
        content = response.content if hasattr(response, 'content') else str(response)
        self.add_reasoning_step(content)
        
        return content
    
    def generate_hypotheses(self, initial_thinking):
        """
        生成多个可能的假设
        
        参数:
            initial_thinking: 初步思考内容
            
        返回:
            List[Dict]: 假设列表
        """
        prompt = HYPOTHESIS_GENERATION_PROMPT.format(initial_thinking=initial_thinking)
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析假设
        try:            
            # 寻找JSON部分
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                hypotheses = json.loads(json_match.group(0))
                self.hypotheses = hypotheses
                
                # 添加假设到推理步骤
                hypothesis_step = "生成的假设：\n"
                for i, hyp in enumerate(hypotheses):
                    hypothesis_step += f"假设 {i+1}: {hyp['hypothesis']}\n"
                    hypothesis_step += f"理由: {hyp['reasoning']}\n\n"
                
                self.add_reasoning_step(hypothesis_step)
                return hypotheses
            else:
                # 使用正则表达式提取假设
                return self._extract_hypotheses_fallback(content)
        except Exception as e:
            print(f"解析假设失败: {e}")
            return self._extract_hypotheses_fallback(content)
    
    def _extract_hypotheses_fallback(self, content):
        """
        当JSON解析失败时，使用正则表达式提取假设
        
        参数:
            content: 包含假设的文本
            
        返回:
            List[Dict]: 假设列表
        """        
        hypotheses = []
        
        # 查找假设模式
        hypothesis_pattern = re.compile(r'假设\s*\d+[:：]?\s*(.*?)(?=假设\s*\d+[:：]?|$)', re.DOTALL)
        matches = hypothesis_pattern.findall(content)
        
        for i, match in enumerate(matches):
            # 尝试分离假设和理由
            parts = re.split(r'理由[:：]', match, 1)
            
            if len(parts) == 2:
                hypothesis, reasoning = parts
            else:
                hypothesis = parts[0]
                reasoning = ""
                
            hypotheses.append({
                "hypothesis": hypothesis.strip(),
                "reasoning": reasoning.strip()
            })
        
        # 如果没有找到假设，创建一个默认假设
        if not hypotheses:
            hypotheses = [{
                "hypothesis": "问题可能需要更多背景信息",
                "reasoning": "初步思考中没有明确的答案方向"
            }]
            
        # 添加假设到思考引擎状态
        self.hypotheses = hypotheses
        
        # 添加假设到推理步骤
        hypothesis_step = "生成的假设：\n"
        for i, hyp in enumerate(hypotheses):
            hypothesis_step += f"假设 {i+1}: {hyp['hypothesis']}\n"
            hypothesis_step += f"理由: {hyp['reasoning']}\n\n"
        
        self.add_reasoning_step(hypothesis_step)
        
        return hypotheses
    
    def verify_hypothesis(self, hypothesis):
        """
        验证假设
        
        参数:
            hypothesis: 要验证的假设
            
        返回:
            Dict: 验证结果
        """
        prompt = HYPOTHESIS_VERIFICATION_PROMPT.format(
            hypothesis_text=hypothesis["hypothesis"],
            reasoning_text=hypothesis["reasoning"],
        )
        
        response = self.llm.invoke(prompt)
        verification = response.content if hasattr(response, 'content') else str(response)
        
        # 创建验证结果
        verification_result = {
            "hypothesis": hypothesis['hypothesis'],
            "verification": verification,
            "status": self._assess_verification_status(verification)
        }
        
        # 添加到验证链
        self.verification_chain.append(verification_result)
        
        # 添加到推理步骤
        self.add_reasoning_step(f"验证假设: {hypothesis['hypothesis']}\n\n{verification}")
        
        return verification_result
    
    def _assess_verification_status(self, verification):
        """
        评估验证状态
        
        参数:
            verification: 验证文本
            
        返回:
            str: 验证状态 (supported/rejected/uncertain)
        """
        # 分析验证文本，确定假设状态
        prompt = VERIFICATION_STATUS_PROMPT.format(verification=verification)
        
        try:
            response = self.llm.invoke(prompt)
            status = response.content if hasattr(response, 'content') else str(response)
            
            # 清理并标准化状态
            status = status.strip().lower()
            
            if "support" in status:
                return "supported"
            elif "reject" in status:
                return "rejected"
            else:
                return "uncertain"
        except:
            # 默认不确定
            return "uncertain"
    
    def think_deeper(self, query, context=None):
        """
        启动深度思考模式
        
        参数:
            query: 用户问题
            context: 上下文信息
            
        返回:
            str: 深度思考结果
        """
        # 初始化思考历史
        self.initialize_with_query(query)
        
        # 添加上下文信息（如果有）
        if context:
            self.add_reasoning_step(f"考虑以下背景信息：\n{context}")
        
        # 生成初步思考
        initial_thinking = self.generate_initial_thinking()
        
        # 提出假设
        hypotheses = self.generate_hypotheses(initial_thinking)
        
        # 对每个假设进行验证
        verifications = []
        for hypothesis in hypotheses:
            verification = self.verify_hypothesis(hypothesis)
            verifications.append(verification)
            
        # 基于验证结果更新思考
        updated_thinking = self.update_thinking_based_on_verification(verifications)
        
        # 整合所有思考过程
        final_thinking = self.integrate_thinking_process(
            initial_thinking,
            hypotheses,
            verifications,
            updated_thinking
        )
        
        return final_thinking

    def update_thinking_based_on_verification(self, verifications):
        """
        基于验证结果更新思考
        
        参数:
            verifications: 验证结果列表
            
        返回:
            str: 更新后的思考
        """
        # 汇总验证结果
        verification_summary = "验证结果汇总:\n"
        
        supported = []
        rejected = []
        uncertain = []
        
        for v in verifications:
            if v["status"] == "supported":
                supported.append(v["hypothesis"])
            elif v["status"] == "rejected":
                rejected.append(v["hypothesis"])
            else:
                uncertain.append(v["hypothesis"])
                
        verification_summary += f"- 被支持的假设: {len(supported)}\n"
        if supported:
            verification_summary += "  " + "\n  ".join(supported) + "\n"
            
        verification_summary += f"- 被拒绝的假设: {len(rejected)}\n"
        if rejected:
            verification_summary += "  " + "\n  ".join(rejected) + "\n"
            
        verification_summary += f"- 不确定的假设: {len(uncertain)}\n"
        if uncertain:
            verification_summary += "  " + "\n  ".join(uncertain) + "\n"
        
        # 添加汇总到推理步骤
        self.add_reasoning_step(verification_summary)
        
        # 基于验证结果更新思考
        prompt = UPDATE_THINKING_PROMPT.format(
            verification_summary=verification_summary
        )
        
        response = self.llm.invoke(prompt)
        updated_thinking = response.content if hasattr(response, 'content') else str(response)
        
        # 添加到推理步骤
        self.add_reasoning_step(f"更新后的思考:\n\n{updated_thinking}")
        
        return updated_thinking
    
    def integrate_thinking_process(self, initial_thinking, hypotheses, verifications, updated_thinking):
        """
        整合所有思考过程
        
        参数:
            initial_thinking: 初步思考
            hypotheses: 假设列表
            verifications: 验证结果
            updated_thinking: 更新后的思考
            
        返回:
            str: 整合后的思考过程
        """
        # 构建完整的思考过程
        integrated_thinking = "# 思考过程\n\n"
        integrated_thinking += "## 初步分析\n\n"
        integrated_thinking += initial_thinking + "\n\n"
        
        integrated_thinking += "## 假设生成\n\n"
        for i, hyp in enumerate(hypotheses):
            integrated_thinking += f"### 假设 {i+1}: {hyp['hypothesis']}\n"
            integrated_thinking += f"{hyp['reasoning']}\n\n"
        
        integrated_thinking += "## 假设验证\n\n"
        for i, ver in enumerate(verifications):
            status_map = {
                "supported": "✅ 支持",
                "rejected": "❌ 拒绝", 
                "uncertain": "❓ 不确定"
            }
            status = status_map.get(ver["status"], "未知")
            
            integrated_thinking += f"### 验证 {i+1}: {ver['hypothesis']} [{status}]\n"
            integrated_thinking += f"{ver['verification']}\n\n"
        
        integrated_thinking += "## 最终思考\n\n"
        integrated_thinking += updated_thinking
        
        return integrated_thinking
    
    def add_reasoning_step(self, content: str):
        """
        添加推理步骤
        
        参数:
            content: 步骤内容
        """
        self.all_reasoning_steps.append(content)
        
        # 更新推理树
        if self.current_branch not in self.reasoning_tree:
            self.reasoning_tree[self.current_branch] = []
            
        self.reasoning_tree[self.current_branch].append({
            "content": content,
            "timestamp": time.time()
        })
    
    def branch_reasoning(self, branch_name: str, base_branch: str = "main"):
        """
        创建推理分支
        
        参数:
            branch_name: 分支名称
            base_branch: 基础分支
        """
        # 确保基础分支存在
        if base_branch not in self.reasoning_tree:
            base_branch = "main"
            
        # 创建新分支
        self.reasoning_tree[branch_name] = []
        
        # 复制基础分支内容
        for step in self.reasoning_tree[base_branch]:
            self.reasoning_tree[branch_name].append(step.copy())
            
        # 切换到新分支
        self.current_branch = branch_name
        
        # 添加分支创建记录
        self.add_reasoning_step(f"创建推理分支: {branch_name}，基于: {base_branch}")
    
    def switch_branch(self, branch_name: str):
        """
        切换推理分支
        
        参数:
            branch_name: 分支名称
        """
        # 确保分支存在
        if branch_name not in self.reasoning_tree:
            return False
            
        # 切换分支
        self.current_branch = branch_name
        return True
    
    def merge_branches(self, source_branch: str, target_branch: str = "main"):
        """
        合并推理分支
        
        参数:
            source_branch: 源分支
            target_branch: 目标分支
            
        返回:
            bool: 是否成功合并
        """
        # 确保分支存在
        if source_branch not in self.reasoning_tree or target_branch not in self.reasoning_tree:
            return False
            
        # 获取源分支独有的步骤
        source_steps = self.reasoning_tree[source_branch]
        target_steps = self.reasoning_tree[target_branch]
        
        # 找出源分支中独有的步骤
        source_unique_steps = []
        target_step_contents = [step["content"] for step in target_steps]
        
        for step in source_steps:
            if step["content"] not in target_step_contents:
                source_unique_steps.append(step)
        
        # 将源分支独有步骤添加到目标分支
        for step in source_unique_steps:
            self.reasoning_tree[target_branch].append(step.copy())
            
        # 添加合并记录
        merged_step = {
            "content": f"合并分支: {source_branch} → {target_branch}",
            "timestamp": time.time()
        }
        self.reasoning_tree[target_branch].append(merged_step)
        
        # 切换到目标分支
        self.current_branch = target_branch
        
        return True
    
    def counter_factual_analysis(self, hypothesis: str):
        """
        执行反事实分析
        
        参数:
            hypothesis: 假设内容
            
        返回:
            str: 反事实分析结果
        """
        # 创建反事实分支
        branch_name = f"counter_factual_{int(time.time())}"
        self.branch_reasoning(branch_name)
        
        # 添加反事实假设
        self.add_reasoning_step(f"反事实假设: {hypothesis}")
        
        # 基于反事实假设进行推理
        prompt = COUNTERFACTUAL_ANALYSIS_PROMPT.format(hypothesis=hypothesis)
        
        response = self.llm.invoke(prompt)
        counter_analysis = response.content if hasattr(response, 'content') else str(response)
        
        # 添加分析结果
        self.add_reasoning_step(f"反事实分析结果:\n\n{counter_analysis}")
        
        # 对比原始推理和反事实推理
        prompt = COUNTERFACTUAL_COMPARISON_PROMPT.format(hypothesis=hypothesis)
        
        response = self.llm.invoke(prompt)
        comparison = response.content if hasattr(response, 'content') else str(response)
        
        # 添加比较结果
        self.add_reasoning_step(f"原始推理与反事实推理对比:\n\n{comparison}")
        
        # 回到主分支
        self.switch_branch("main")
        
        # 添加反事实分析的总结
        self.add_reasoning_step(f"反事实分析总结: 如果 {hypothesis}，那么 {self._extract_conclusion(counter_analysis)}")
        
        return comparison
    
    def _extract_conclusion(self, analysis):
        """
        从分析中提取结论
        
        参数:
            analysis: 分析文本
            
        返回:
            str: 提取的结论
        """
        # 查找结论标记
        conclusion_markers = ["结论", "总结", "因此", "所以", "综上所述"]
        
        for marker in conclusion_markers:
            marker_index = analysis.find(marker)
            if marker_index != -1:
                # 提取标记后的内容作为结论
                conclusion = analysis[marker_index:]
                # 限制长度
                conclusion = conclusion.split("\n")[0]
                if len(conclusion) > 100:
                    conclusion = conclusion[:100] + "..."
                return conclusion
                
        # 如果没有找到标记，返回最后一段
        paragraphs = analysis.split("\n\n")
        if paragraphs:
            last_paragraph = paragraphs[-1]
            if len(last_paragraph) > 100:
                last_paragraph = last_paragraph[:100] + "..."
            return last_paragraph
            
        # 如果分析内容为空，返回默认文本
        return "无法提取明确结论"
    
    def remove_query_tags(self, text: str) -> str:
        """
        移除文本中的查询标签
        
        参数:
            text: 包含标签的文本
            
        返回:
            str: 移除标签后的文本
        """
        pattern = re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def remove_result_tags(self, text: str) -> str:
        """
        移除文本中的结果标签
        
        参数:
            text: 包含标签的文本
            
        返回:
            str: 移除标签后的文本
        """
        pattern = re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def extract_queries(self, text: str) -> List[str]:
        """
        从文本中提取搜索查询
        
        参数:
            text: 包含查询的文本
            
        返回:
            List[str]: 提取的查询列表
        """
        return extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
    
    def generate_next_query(self) -> Dict[str, Any]:
        """
        生成下一步搜索查询
        
        返回:
            Dict: 包含查询和状态信息的字典
        """
        # 使用LLM进行推理分析，获取下一个搜索查询
        formatted_messages = [SystemMessage(content=REASON_PROMPT)] + self.msg_history
        
        try:
            # 调用LLM生成查询
            msg = self.llm.invoke(formatted_messages)
            query_think = msg.content if hasattr(msg, 'content') else str(msg)
            
            # 清理响应
            query_think = re.sub(r"<think>.*</think>", "", query_think, flags=re.DOTALL)
            if not query_think:
                return {"status": "empty", "content": None, "queries": []}
                
            # 更新思考过程
            clean_think = self.remove_query_tags(query_think)
            self.add_reasoning_step(query_think)
            
            # 从AI响应中提取搜索查询
            queries = self.extract_queries(query_think)
            
            # 如果没有生成搜索查询，检查是否应该结束
            if not queries:
                # 检查是否包含最终答案标记
                if "**回答**" in query_think or "足够的信息" in query_think:
                    return {
                        "status": "answer_ready", 
                        "content": query_think,
                        "queries": []
                    }
                
                # 没有明确结束标志，就继续
                return {
                    "status": "no_query", 
                    "content": query_think,
                    "queries": []
                }
            
            # 有查询，继续搜索
            return {
                "status": "has_query", 
                "content": query_think,
                "queries": queries
            }
            
        except Exception as e:
            error_msg = f"生成查询时出错: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return {"status": "error", "error": error_msg, "queries": []}
    
    def add_ai_message(self, content: str):
        """
        添加AI消息到历史记录
        
        参数:
            content: 消息内容
        """
        self.msg_history.append(AIMessage(content=content))
    
    def add_human_message(self, content: str):
        """
        添加用户消息到历史记录
        
        参数:
            content: 消息内容
        """
        self.msg_history.append(HumanMessage(content=content))
    
    def update_continue_message(self):
        """更新最后的消息，请求继续推理"""
        if len(self.msg_history) > 0:
            # 检查最后一条消息的类型
            last_message = self.msg_history[-1]
            
            if isinstance(last_message, dict) and "role" in last_message:
                # 处理字典类型的消息
                if last_message["role"] == "assistant":
                    self.add_human_message("继续基于新信息进行推理分析。\n")
                else:
                    # 更新最后的用户消息
                    last_content = last_message.get("content", "")
                    self.msg_history[-1] = {"role": "user", "content": last_content + "\n\n继续基于新信息进行推理分析。\n"}
            else:
                # 处理对象类型的消息 (如AIMessage, HumanMessage等)
                if hasattr(last_message, "role") and last_message.role == "assistant":
                    self.add_human_message("继续基于新信息进行推理分析。\n")
                elif hasattr(last_message, "content"):
                    # 更新最后的用户消息
                    last_content = last_message.content
                    self.msg_history[-1] = {"role": "user", "content": last_content + "\n\n继续基于新信息进行推理分析。\n"}
        
    def prepare_truncated_reasoning(self) -> str:
        """
        准备截断的推理历史，保留关键部分以减少token使用
        
        返回:
            str: 截断的推理历史
        """
        all_reasoning_steps = self.all_reasoning_steps
        
        if not all_reasoning_steps:
            return ""
            
        # 如果步骤少于5个，保留全部
        if len(all_reasoning_steps) <= 5:
            steps_text = ""
            for i, step in enumerate(all_reasoning_steps):
                steps_text += f"Step {i + 1}: {step}\n\n"
            return steps_text.strip()
        
        # 否则，保留第一步、最后4步和包含查询/结果的步骤
        important_steps = []
        
        # 总是包含第一步
        important_steps.append((0, all_reasoning_steps[0]))
        
        # 包含最后4步
        for i in range(max(1, len(all_reasoning_steps) - 4), len(all_reasoning_steps)):
            important_steps.append((i, all_reasoning_steps[i]))
        
        # 包含中间包含搜索查询或结果的步骤
        for i in range(1, len(all_reasoning_steps) - 4):
            step = all_reasoning_steps[i]
            if BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                important_steps.append((i, step))
        
        # 按原始顺序排序
        important_steps.sort(key=lambda x: x[0])
        
        # 格式化结果
        truncated = ""
        prev_idx = -1
        
        for idx, step in important_steps:
            # 如果有间隔，添加省略号
            if idx > prev_idx + 1:
                truncated += "...\n\n"
            
            truncated += f"Step {idx + 1}: {step}\n\n"
            prev_idx = idx
        
        return truncated.strip()
    
    def get_full_thinking(self) -> str:
        """
        获取完整的思考过程
        
        返回:
            str: 格式化的思考过程
        """
        thinking = "<think>\n"
        
        for step in self.all_reasoning_steps:
            clean_step = self.remove_query_tags(step)
            clean_step = self.remove_result_tags(clean_step)
            thinking += clean_step + "\n\n"
            
        thinking += "</think>"
        return thinking
    
    def has_executed_query(self, query: str) -> bool:
        """
        检查是否已经执行过相同的查询
        
        参数:
            query: 查询字符串
            
        返回:
            bool: 是否已执行过
        """
        return query in self.executed_search_queries
    
    def add_executed_query(self, query: str):
        """
        添加已执行的查询
        
        参数:
            query: 已执行的查询字符串
        """
        self.executed_search_queries.append(query)
