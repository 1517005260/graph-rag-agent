import re
import ast
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.config.prompt import system_template_build_index, user_template_build_index
from graphrag_agent.config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS
from graphrag_agent.graph.core import connection_manager, timer, get_performance_stats, print_performance_stats

class EntityMerger:
    """
    实体合并管理器，负责基于LLM决策合并相似实体。
    主要功能包括使用LLM分析实体相似性、解析合并建议，以及执行实体合并操作。
    """
    
    def __init__(self, batch_size: int = 20, max_workers: int = 4):
        """
        初始化实体合并管理器
        
        Args:
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()
        
        # 获取语言模型
        self.llm = get_llm_model()
        
        # 批处理和并行参数
        self.batch_size = batch_size or ENTITY_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 设置LLM处理链
        self._setup_llm_chain()
        
        # 创建索引
        self._create_indexes()
        
        # 性能监控
        self.llm_time = 0
        self.db_time = 0
        self.parse_time = 0
    
    def _create_indexes(self) -> None:
        """创建必要的索引以优化查询性能"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"
        ]
        
        connection_manager.create_multiple_indexes(index_queries)

    def _setup_llm_chain(self) -> None:
        """
        设置LLM处理链，用于实体合并决策
        包括创建提示模板和构建处理链
        """
        # 检查模型能力
        if not hasattr(self.llm, 'with_structured_output'):
            print("当前LLM模型不支持结构化输出")

        # 创建提示模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template_build_index
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            user_template_build_index
        )
        
        # 构建对话链
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])
        
        # 创建最终的处理链
        self.chain = self.chat_prompt | self.llm

    def _convert_to_list(self, result: str) -> List[List[str]]:
        """
        将LLM返回的实体列表文本转换为Python列表
        
        Args:
            result: LLM返回的文本结果，包含实体列表
            
        Returns:
            List[List[str]]: 二维列表，每个子列表包含一组可合并的实体
        """
        start_time = time.time()
        
        # 使用正则表达式匹配所有方括号包含的内容
        list_pattern = re.compile(r'\[.*?\]')
        entity_lists = []
        
        # 先尝试直接用ast.literal_eval解析整个结果
        try:
            # 查找可能的列表开始位置
            list_start = result.find('[')
            if list_start >= 0:
                # 尝试找出完整列表部分
                nested_level = 0
                for i in range(list_start, len(result)):
                    if result[i] == '[':
                        nested_level += 1
                    elif result[i] == ']':
                        nested_level -= 1
                        if nested_level == 0:
                            # 提取出可能是列表的部分
                            list_portion = result[list_start:i+1]
                            try:
                                parsed_list = ast.literal_eval(list_portion)
                                if isinstance(parsed_list, list):
                                    # 检查是否是二维列表
                                    if all(isinstance(item, list) for item in parsed_list):
                                        entity_lists = parsed_list
                                    else:
                                        entity_lists = [parsed_list]
                                    break
                            except:
                                pass  # 如果解析失败，继续使用正则方法
        except:
            pass  # 如果上述方法失败，回退到正则表达式方法
        
        # 如果直接解析失败，使用正则表达式方法
        if not entity_lists:
            # 解析每个匹配的列表字符串
            for match in list_pattern.findall(result):
                try:
                    # 将字符串转换为Python列表
                    entity_list = ast.literal_eval(match)
                    # 只添加非空列表
                    if entity_list and isinstance(entity_list, list):
                        if all(isinstance(item, list) for item in entity_list):
                            # 如果是嵌套列表，扩展它
                            entity_lists.extend(entity_list)
                        else:
                            # 如果是单个列表，添加它
                            entity_lists.append(entity_list)
                except Exception as e:
                    print(f"解析实体列表时出错: {str(e)}, 原文本: {match}")
        
        # 过滤和规范化结果
        valid_lists = []
        for entity_list in entity_lists:
            # 确保列表中的所有项目都是字符串
            if all(isinstance(item, str) for item in entity_list):
                # 去除重复项
                unique_list = list(dict.fromkeys(entity_list))
                if len(unique_list) > 1:  # 只保留至少有2个实体的组
                    valid_lists.append(unique_list)
        
        self.parse_time += time.time() - start_time
        return valid_lists

    def get_merge_suggestions(self, duplicate_candidates: List[Any]) -> List[List[str]]:
        """
        使用LLM分析并提供实体合并建议 - 并行优化版本
        
        Args:
            duplicate_candidates: 潜在的重复实体候选列表
            
        Returns:
            List[List[str]]: 建议合并的实体分组列表
        """
        # 检查是否有候选实体
        if not duplicate_candidates:
            return []
        
        llm_start_time = time.time()
            
        # 收集LLM的合并建议
        merged_entities = []
        
        # 动态调整批处理大小
        candidate_count = len(duplicate_candidates)
        optimal_batch_size = min(self.max_workers * 2, max(1, candidate_count // 5))
        
        print(f"处理 {candidate_count} 个候选实体组，批次大小: {optimal_batch_size}")
        
        # 分批处理，避免创建过多线程
        for batch_start in range(0, candidate_count, optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, candidate_count)
            batch = duplicate_candidates[batch_start:batch_end]
            
            # 使用线程池并行处理LLM请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_candidates = {
                    executor.submit(self._process_candidate_group, candidates): i
                    for i, candidates in enumerate(batch)
                }
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_candidates):
                    try:
                        result = future.result()
                        if result:
                            merged_entities.append(result)
                    except Exception as e:
                        print(f"处理候选实体组时出错: {e}")
            
            # 报告进度
            print(f"已处理 {batch_end}/{candidate_count} 个候选实体组")
        
        self.llm_time += time.time() - llm_start_time
        
        parse_start_time = time.time()
        # 解析并整理最终的合并建议
        results = []
        for candidates in merged_entities:
            # 将每个建议转换为列表格式
            temp = self._convert_to_list(candidates)
            results.extend(temp)
        
        self.parse_time += time.time() - parse_start_time
        
        # 合并具有相同实体的组
        merged_results = self._merge_overlapping_groups(results)
        
        print(f"LLM分析完成，找到 {len(merged_results)} 组可合并实体")
        return merged_results
    
    def _merge_overlapping_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """
        合并有重叠的实体组
        
        Args:
            groups: 实体组列表
            
        Returns:
            List[List[str]]: 合并后的实体组列表
        """
        if not groups:
            return []
            
        # 创建实体到组的映射
        entity_to_groups = {}
        for i, group in enumerate(groups):
            for entity in group:
                if entity not in entity_to_groups:
                    entity_to_groups[entity] = []
                entity_to_groups[entity].append(i)
        
        # 使用并查集合并连通的组
        parent = list(range(len(groups)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # 合并存在共同实体的组
        for entity, group_indices in entity_to_groups.items():
            if len(group_indices) > 1:
                for i in range(1, len(group_indices)):
                    union(group_indices[0], group_indices[i])
        
        # 收集合并后的组
        merged_groups = {}
        for i, group in enumerate(groups):
            root = find(i)
            if root not in merged_groups:
                merged_groups[root] = set()
            merged_groups[root].update(group)
        
        # 转换回列表格式
        return [list(entities) for entities in merged_groups.values()]
    
    def _process_candidate_group(self, candidates: List[str]) -> Optional[str]:
        """
        处理单个候选实体组
        
        Args:
            candidates: 候选实体列表
            
        Returns:
            str: LLM的分析结果
        """
        if not candidates or len(candidates) < 2:
            return None
            
        chat_history = []
        max_retries = 2
        for retry in range(max_retries + 1):
            try:
                # 调用LLM进行分析
                answer = self.chain.invoke({
                    "chat_history": chat_history,
                    "entities": candidates
                })
                return answer.content
            except Exception as e:
                if retry < max_retries:
                    print(f"LLM调用异常，尝试重试 ({retry+1}/{max_retries}): {e}")
                    time.sleep(1)  # 短暂延迟
                else:
                    print(f"LLM调用失败，最大重试次数已用尽: {e}")
                    return None

    def execute_merges(self, merge_groups: List[List[str]]) -> int:
        """
        执行实体合并操作 - 批处理优化版本
        
        Args:
            merge_groups: 要合并的实体分组列表
            
        Returns:
            int: 合并操作影响的节点数量
        """
        if not merge_groups:
            return 0
        
        db_start_time = time.time()
            
        # 动态批处理大小
        group_count = len(merge_groups)
        optimal_batch_size = min(self.batch_size, max(5, group_count // 10))
        total_batches = (group_count + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"开始执行 {group_count} 组实体合并，批次大小: {optimal_batch_size}")
        
        total_merged = 0
        batch_times = []
        
        # 批量处理合并操作
        for batch_index in range(total_batches):
            batch_start = time.time()
            
            start_idx = batch_index * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, group_count)
            batch = merge_groups[start_idx:end_idx]
            
            try:
                # 执行Neo4j合并操作
                result = self.graph.query("""
                UNWIND $data AS candidates
                CALL {
                  WITH candidates
                  MATCH (e:__Entity__) WHERE e.id IN candidates
                  RETURN collect(e) AS nodes
                }
                CALL apoc.refactor.mergeNodes(nodes, {properties: {
                    `.*`: 'discard'
                }})
                YIELD node
                RETURN count(*) as merged_count
                """, params={"data": batch})
                
                if result:
                    batch_merged = result[0]["merged_count"]
                    total_merged += batch_merged
                    
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)
                    
                    # 计算平均时间和剩余时间
                    avg_time = sum(batch_times) / len(batch_times)
                    remaining_batches = total_batches - (batch_index + 1)
                    estimated_remaining = avg_time * remaining_batches
                    
                    print(f"已处理合并批次 {batch_index+1}/{total_batches}, "
                          f"批次合并: {batch_merged} 实体, "
                          f"批次耗时: {batch_time:.2f}秒, "
                          f"预计剩余: {estimated_remaining:.2f}秒")
            except Exception as e:
                print(f"批量合并出错，尝试单个处理: {e}")
                batch_merged = 0
                
                # 如果批处理失败，尝试逐个合并
                for group in batch:
                    try:
                        single_result = self.graph.query("""
                        MATCH (e:__Entity__) WHERE e.id IN $candidates
                        WITH collect(e) AS nodes
                        CALL apoc.refactor.mergeNodes(nodes, {properties: {
                            `.*`: 'discard'
                        }})
                        YIELD node
                        RETURN count(*) as merged_count
                        """, params={"candidates": group})
                        
                        if single_result:
                            group_merged = single_result[0]["merged_count"]
                            total_merged += group_merged
                            batch_merged += group_merged
                    except Exception as e2:
                        print(f"单个组合并失败: {e2}")
                
                print(f"单个处理完成，本批次合并: {batch_merged} 实体")
        
        self.db_time += time.time() - db_start_time
        
        return total_merged

    def clean_duplicate_relationships(self):
        """
        清除重复关系，包括：
        1. 相同方向的重复关系
        2. SIMILAR关系的双向冗余（保留一个方向）
        """
        print("开始清除重复关系...")
        
        # 第一步：清除相同方向的重复关系
        result1 = self.graph.query("""
        MATCH (a)-[r]->(b)
        WITH a, b, type(r) as type, collect(r) as rels
        WHERE size(rels) > 1
        WITH a, b, type, rels[0] as kept, rels[1..] as rels
        UNWIND rels as rel
        DELETE rel
        RETURN count(*) as deleted
        """)
        
        deleted_count1 = result1[0]["deleted"] if result1 else 0
        print(f"已删除 {deleted_count1} 个相同方向的重复关系")
        
        # 第二步：清除SIMILAR关系的双向冗余（保留一个方向）
        result2 = self.graph.query("""
        // 找出所有双向的SIMILAR关系
        MATCH (a)-[r1:SIMILAR]->(b)
        MATCH (b)-[r2:SIMILAR]->(a)
        WHERE a.id < b.id  // 确保每对节点只处理一次
        
        // 随机选择一个方向删除（这里选择删除b->a方向）
        DELETE r2
        
        RETURN count(*) as deleted_bidirectional
        """)
        
        deleted_count2 = result2[0]["deleted_bidirectional"] if result2 else 0
        print(f"已删除 {deleted_count2} 个双向SIMILAR关系的冗余方向")
        
        total_deleted = deleted_count1 + deleted_count2
        print(f"总共删除了 {total_deleted} 个重复关系")
        
        return total_deleted

    @timer
    def process_duplicates(self, duplicate_candidates: List[Any]) -> Tuple[int, Dict[str, Any]]:
        """
        处理重复实体的完整流程，包括获取合并建议和执行合并 - 性能优化版本
        
        Args:
            duplicate_candidates: 潜在的重复实体候选列表
            
        Returns:
            Tuple[int, Dict[str, Any]]: 合并的实体数量和性能统计
        """
        start_time = time.time()
        
        # 确保duplicate_candidates是列表的列表，处理不同的数据结构
        fixed_candidates = []
        for candidates in duplicate_candidates:
            # 检查候选组是否是字典或类似对象
            if isinstance(candidates, dict) and "combinedResult" in candidates:
                candidate_list = candidates["combinedResult"]
                if isinstance(candidate_list, list) and len(candidate_list) > 1:
                    fixed_candidates.append(candidate_list)
            # 检查候选组是否已经是列表
            elif isinstance(candidates, list) and len(candidates) > 1:
                fixed_candidates.append(candidates)
        
        # 过滤处理数量过少的候选组
        filtered_candidates = [
            candidates for candidates in fixed_candidates
            if len(candidates) > 1
        ]
        
        print(f"处理后候选实体组数: {len(filtered_candidates)}")
        print(f"开始处理 {len(filtered_candidates)} 组有效重复实体候选...")
        
        # 获取合并建议
        merge_groups = self.get_merge_suggestions(filtered_candidates)
        
        suggestion_time = time.time()
        suggestion_elapsed = suggestion_time - start_time
        print(f"生成合并建议完成，用时 {suggestion_elapsed:.2f} 秒, "
            f"找到 {len(merge_groups)} 组可合并实体")
        print(f"其中: LLM处理时间: {self.llm_time:.2f}秒, 解析时间: {self.parse_time:.2f}秒")
        
        # 如果有建议的合并组，执行合并
        merged_count = 0
        if merge_groups:
            merged_count = self.execute_merges(merge_groups)

        # 合并重复关系
        self.clean_duplicate_relationships()
                
        end_time = time.time()
        merge_elapsed = end_time - suggestion_time
        total_elapsed = end_time - start_time
        
        print(f"实体合并完成，用时 {merge_elapsed:.2f} 秒, 合并了 {merged_count} 个实体")
        print(f"数据库操作时间: {self.db_time:.2f}秒")
        print(f"总耗时: {total_elapsed:.2f} 秒")
        
        # 返回性能统计摘要
        time_records = {
            "LLM处理时间": self.llm_time,
            "解析时间": self.parse_time,
            "数据库时间": self.db_time
        }
        
        performance_stats = get_performance_stats(total_elapsed, time_records)
        performance_stats.update({
            "候选实体组数": len(filtered_candidates),
            "识别出的合并组数": len(merge_groups),
            "合并的实体数": merged_count
        })
        
        print_performance_stats(performance_stats)
        
        return merged_count, performance_stats