from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from graphdatascience import GraphDataScience
from langchain_community.graphs import Neo4jGraph
import psutil
import os
import time
from contextlib import contextmanager

from graphrag_agent.config.settings import MAX_WORKERS, GDS_CONCURRENCY, GDS_MEMORY_LIMIT

class BaseCommunityDetector(ABC):
    """社区检测基类"""
    
    def __init__(self, gds: GraphDataScience, graph: Neo4jGraph):
        self.gds = gds
        self.graph = graph
        self.projection_name = "communities"
        self.G = None
        
        # 性能统计
        self.projection_time = 0
        self.detection_time = 0
        self.save_time = 0
        
        # 系统资源
        self._init_system_resources()
        
    def _init_system_resources(self):
        """初始化系统资源参数"""
        self.memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.cpu_count = os.cpu_count() or 4
        self._adjust_parameters()
        
    def _adjust_parameters(self):
        """调整运行参数"""
        # 设置并发度
        self.max_concurrency = GDS_CONCURRENCY
        
        # 使用配置的内存限制
        memory_gb = GDS_MEMORY_LIMIT
        
        # 根据配置的内存大小调整限制
        if memory_gb > 32:
            self.node_count_limit = 100000
            self.timeout_seconds = 600
        elif memory_gb > 16:
            self.node_count_limit = 50000
            self.timeout_seconds = 300
        else:
            self.node_count_limit = 20000
            self.timeout_seconds = 180
            
        print(f"社区检测参数: CPU={MAX_WORKERS}, 内存={memory_gb:.1f}GB, "
            f"并发度={self.max_concurrency}, 节点限制={self.node_count_limit}")

    @contextmanager
    def _graph_projection_context(self):
        """图投影的上下文管理器"""
        try:
            projection_start = time.time()
            self.create_projection()
            self.projection_time = time.time() - projection_start
            yield
        finally:
            cleanup_start = time.time()
            self.cleanup()
            print(f"图投影清理完成，耗时: {time.time() - cleanup_start:.2f}秒")

    def process(self) -> Dict[str, Any]:
        """执行完整的社区检测流程"""
        start_time = time.time()
        print(f"开始执行{self.__class__.__name__}社区检测...")
        
        results = {
            'status': 'success',
            'algorithm': self.__class__.__name__,
            'details': {}
        }
        
        try:
            with self._graph_projection_context():
                # 执行检测
                detection_start = time.time()
                detection_result = self.detect_communities()
                self.detection_time = time.time() - detection_start
                results['details']['detection'] = detection_result
                
                # 保存结果
                save_start = time.time()
                save_result = self.save_communities()
                self.save_time = time.time() - save_start
                results['details']['save'] = save_result
                
            # 添加性能统计
            total_time = time.time() - start_time
            results['performance'] = {
                'totalTime': total_time,
                'projectionTime': self.projection_time,
                'detectionTime': self.detection_time,
                'saveTime': self.save_time
            }
            
            return results
            
        except Exception as e:
            results.update({
                'status': 'error',
                'error': str(e),
                'elapsed': time.time() - start_time
            })
            raise

    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影"""
        pass

    @abstractmethod
    def detect_communities(self) -> Dict[str, Any]:
        """检测社区"""
        pass

    @abstractmethod
    def save_communities(self) -> Dict[str, int]:
        """保存社区结果"""
        pass

    def cleanup(self):
        """清理资源"""
        if self.G:
            try:
                self.G.drop()
                print("社区投影图已清理")
            except Exception as e:
                print(f"清理投影图出错: {e}")
            finally:
                self.G = None