from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graphrag_agent.models.get_models import get_llm_model
import concurrent.futures
import time

from graphrag_agent.config.settings import MAX_WORKERS
from graphrag_agent.config.prompts import COMMUNITY_SUMMARY_PROMPT

class BaseCommunityDescriber:
    """社区信息格式化工具"""
    
    @staticmethod
    def prepare_string(data: Dict) -> str:
        """转换社区信息为可读字符串"""
        try:
            nodes_str = "Nodes are:\n"
            for node in data.get('nodes', []):
                node_id = node.get('id', 'unknown_id')
                node_type = node.get('type', 'unknown_type')
                node_description = (
                    f", description: {node['description']}"
                    if 'description' in node and node['description']
                    else ""
                )
                nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

            rels_str = "Relationships are:\n"
            for rel in data.get('rels', []):
                start = rel.get('start', 'unknown_start')
                end = rel.get('end', 'unknown_end')
                rel_type = rel.get('type', 'unknown_type')
                description = (
                    f", description: {rel['description']}"
                    if 'description' in rel and rel['description']
                    else ""
                )
                rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

            return nodes_str + "\n" + rels_str
        except Exception as e:
            print(f"格式化社区信息时出错: {e}")
            return f"Error: {str(e)}\nData: {str(data)}"

class BaseCommunityRanker:
    """社区权重计算工具"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def calculate_ranks(self) -> None:
        """计算社区权重"""
        start_time = time.time()
        print("计算社区权重...")
        
        try:
            result = self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
            WITH c, count(distinct d) AS rank
            SET c.community_rank = rank
            RETURN count(c) AS processed_count
            """)
            
            processed_count = result[0]['processed_count'] if result else 0
            print(f"社区权重计算完成，处理了 {processed_count} 个社区，"
                  f"耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            print(f"计算社区权重时出错: {e}")
            self._calculate_ranks_fallback()
    
    def _calculate_ranks_fallback(self):
        """备用的权重计算方法"""
        try:
            self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e:`__Entity__`)
            WITH c, count(e) AS entity_count
            SET c.community_rank = entity_count
            """)
            print("使用实体计数作为社区权重")
        except Exception as e:
            print(f"备用权重计算也失败: {e}")

class BaseCommunityStorer:
    """社区信息存储工具"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def store_summaries(self, summaries: List[Dict]) -> None:
        """存储社区摘要"""
        if not summaries:
            print("没有社区摘要需要存储")
            return
            
        start_time = time.time()
        print(f"开始存储 {len(summaries)} 个社区摘要...")
        
        batch_size = min(100, max(10, len(summaries) // 5))
        total_batches = (len(summaries) + batch_size - 1) // batch_size
        
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_start = time.time()
            
            try:
                self.graph.query("""
                UNWIND $data AS row
                MERGE (c:__Community__ {id:row.community})
                SET c.summary = row.summary, 
                    c.full_content = row.full_content,
                    c.summary_created_at = datetime()
                """, params={"data": batch})
                
                print(f"已存储批次 {i//batch_size + 1}/{total_batches}, "
                      f"耗时: {time.time() - batch_start:.2f}秒")
                
            except Exception as e:
                print(f"存储社区摘要批次时出错: {e}")
                self._store_summaries_one_by_one(batch)
    
    def _store_summaries_one_by_one(self, summaries: List[Dict]):
        """逐个存储社区摘要"""
        for summary in summaries:
            try:
                self.graph.query("""
                MERGE (c:__Community__ {id:$community})
                SET c.summary = $summary, 
                    c.full_content = $full_content,
                    c.summary_created_at = datetime()
                """, params=summary)
            except Exception as e:
                print(f"存储单个社区摘要时出错: {e}")

class BaseSummarizer(ABC):
    """社区摘要生成器基类"""
    
    def __init__(self, graph: Neo4jGraph):
        """初始化社区摘要生成器基类"""
        self.graph = graph
        self.llm = get_llm_model()
        self.describer = BaseCommunityDescriber()
        self.ranker = BaseCommunityRanker(graph)
        self.storer = BaseCommunityStorer(graph)
        self._setup_llm_chain()
        
        # 性能监控
        self.llm_time = 0
        self.query_time = 0
        self.store_time = 0
        
        self.max_workers = MAX_WORKERS
        print(f"社区摘要生成器初始化，并行线程数: {self.max_workers}")

    def _setup_llm_chain(self) -> None:
        """设置LLM处理链"""
        try:
            community_prompt = ChatPromptTemplate.from_messages([
                ("system", COMMUNITY_SUMMARY_PROMPT),
                ("human", "{community_info}"),
            ])
            self.community_chain = community_prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"设置LLM处理链时出错: {e}")
            raise

    @abstractmethod
    def collect_community_info(self) -> List[Dict]:
        """收集社区信息的抽象方法"""
        pass

    def process_communities(self) -> List[Dict]:
        """处理所有社区"""
        total_start_time = time.time()
        print("开始处理社区摘要...")
        
        try:
            # 计算社区权重
            rank_start = time.time()
            self.ranker.calculate_ranks()
            rank_time = time.time() - rank_start
            
            # 收集社区信息
            query_start = time.time()
            community_info = self.collect_community_info()
            self.query_time = time.time() - query_start
            
            if not community_info:
                print("没有找到需要处理的社区")
                return []
            
            # 并行生成摘要
            llm_start = time.time()
            optimal_workers = min(self.max_workers, max(1, len(community_info) // 2))
            print(f"开始并行生成 {len(community_info)} 个社区摘要，"
                  f"使用 {optimal_workers} 个线程...")
            
            summaries = self._process_communities_parallel(
                community_info, 
                optimal_workers
            )
            
            self.llm_time = time.time() - llm_start
            
            # 保存摘要
            store_start = time.time()
            self.storer.store_summaries(summaries)
            self.store_time = time.time() - store_start
            
            # 输出性能统计
            total_time = time.time() - total_start_time
            self._print_performance_stats(
                total_time, rank_time, 
                self.query_time, self.llm_time, 
                self.store_time
            )
            
            return summaries
            
        except Exception as e:
            print(f"处理社区摘要时出错: {str(e)}")
            raise
    
    def _process_communities_parallel(
        self, 
        community_info: List[Dict], 
        workers: int
    ) -> List[Dict]:
        """并行处理社区摘要"""
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_community = {
                executor.submit(self._process_single_community, info): i 
                for i, info in enumerate(community_info)
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_community)):
                try:
                    result = future.result()
                    summaries.append(result)
                    
                    if (i+1) % 10 == 0 or (i+1) == len(community_info):
                        print(f"已处理 {i+1}/{len(community_info)} "
                              f"({(i+1)/len(community_info)*100:.1f}%)")
                        
                except Exception as e:
                    print(f"处理社区摘要时出错: {e}")
        
        return summaries
    
    def _process_single_community(self, community: Dict) -> Dict:
        """处理单个社区摘要"""
        community_id = community.get('communityId', 'unknown')
        
        try:
            stringify_info = self.describer.prepare_string(community)
            
            if len(stringify_info) < 10:
                print(f"社区 {community_id} 的信息太少，跳过摘要生成")
                return {
                    "community": community_id,
                    "summary": "此社区没有足够的信息生成摘要。",
                    "full_content": stringify_info
                }
            
            summary = self.community_chain.invoke({'community_info': stringify_info})
            
            return {
                "community": community_id,
                "summary": summary,
                "full_content": stringify_info
            }
        except Exception as e:
            print(f"处理社区 {community_id} 摘要时出错: {e}")
            return {
                "community": community_id,
                "summary": f"生成摘要时出错: {str(e)}",
                "full_content": str(community)
            }
    
    def _print_performance_stats(
        self, 
        total_time: float,
        rank_time: float,
        query_time: float,
        llm_time: float,
        store_time: float
    ) -> None:
        """打印性能统计信息"""
        print(f"\n社区摘要处理完成，总耗时: {total_time:.2f}秒")
        print(f"  社区权重计算: {rank_time:.2f}秒 ({rank_time/total_time*100:.1f}%)")
        print(f"  社区信息查询: {query_time:.2f}秒 ({query_time/total_time*100:.1f}%)")
        print(f"  摘要生成(LLM): {llm_time:.2f}秒 ({llm_time/total_time*100:.1f}%)")
        print(f"  结果存储: {store_time:.2f}秒 ({store_time/total_time*100:.1f}%)")
