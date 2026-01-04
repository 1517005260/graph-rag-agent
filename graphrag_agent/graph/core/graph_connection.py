from typing import Any, Optional
import threading
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import CHUNK_VECTOR_INDEX, ENTITY_VECTOR_INDEX, CLEAN_LEGACY_INDEXES

class GraphConnectionManager:
    """
    线程安全的图数据库连接管理器 (Singleton)

    使用双重检查锁定机制确保在多线程环境下的安全性：
    - 防止多个线程同时创建实例
    - 防止 __init__ 被重复调用导致连接重置
    - 支持显式关闭连接以优雅释放资源
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        单例模式实现，使用双重检查锁定确保线程安全

        第一重检查：性能优化，实例已存在时直接返回，避免锁开销
        第二重检查：防止多线程并发突破第一重检查
        """
        # 第一重检查：如果实例已存在，直接返回，避免锁开销
        if cls._instance is None:
            with cls._lock:
                # 第二重检查：防止多线程并发突破第一重检查
                if cls._instance is None:
                    cls._instance = super(GraphConnectionManager, cls).__new__(cls)
                    # 标记未初始化，确保 __init__ 只运行一次逻辑
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化连接管理器，只在第一次创建时执行

        使用 _initialized 标志位防止 __init__ 被重复调用导致连接重置
        """
        # 防止 __init__ 被重复调用导致连接重置
        if getattr(self, "_initialized", False):
            return

        # 初始化过程也需要加锁保护，防止并发读写 _initialized
        with self._lock:
            # 双重检查，防止在获取锁期间被其他线程初始化
            if getattr(self, "_initialized", False):
                return

            # --- 初始化逻辑开始 ---
            try:
                db_manager = get_db_manager()
                self.graph = db_manager.graph
                # 保存 driver 引用以便后续关闭
                self.driver = getattr(db_manager, 'driver', None)

                print("✅ GraphConnectionManager initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize GraphConnectionManager: {e}")
                raise e
            finally:
                # 无论成功与否都标记为已初始化，避免重复尝试
                self._initialized = True
            # --- 初始化逻辑结束 ---
    
    def get_connection(self):
        """
        获取图数据库连接
        
        Returns:
            连接到Neo4j数据库的对象
        """
        return self.graph
    
    def refresh_schema(self):
        """刷新图数据库模式"""
        self.graph.refresh_schema()
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> Any:
        """
        执行图数据库查询
        
        Args:
            query: 查询语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        return self.graph.query(query, params or {})
    
    def create_index(self, index_query: str) -> None:
        """
        创建索引
        
        Args:
            index_query: 索引创建查询
        """
        self.graph.query(index_query)
        
    def create_multiple_indexes(self, index_queries: list) -> None:
        """
        创建多个索引
        
        Args:
            index_queries: 索引创建查询列表
        """
        for query in index_queries:
            self.create_index(query)
            
    def drop_index(self, index_name: str) -> None:
        """
        删除索引

        Args:
            index_name: 索引名称
        """
        try:
            self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
            print(f"已删除索引 {index_name}（如果存在）")
        except Exception as e:
            print(f"删除索引 {index_name} 时出错 (可忽略): {e}")

    def drop_all_indexes(self) -> None:
        """
        删除所有索引（包括普通索引和向量索引）
        在开始构建流程前调用，确保清理所有旧索引
        """
        print("\n" + "="*60)
        print("开始清理所有索引...")
        print("="*60)

        try:
            # 获取所有索引
            result = self.graph.query("""
                SHOW INDEXES
                YIELD name, type
                RETURN name, type
            """)

            if result:
                print(f"发现 {len(result)} 个索引，开始删除...")

                for index_info in result:
                    index_name = index_info.get('name')
                    index_type = index_info.get('type', 'UNKNOWN')

                    if index_name:
                        try:
                            self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
                            print(f"  已删除索引: {index_name} (类型: {index_type})")
                        except Exception as e:
                            print(f"  删除索引 {index_name} 失败: {e}")

                print(f"\n索引清理完成，共删除 {len(result)} 个索引")
            else:
                print("未发现任何索引")

        except Exception as e:
            print(f"获取索引列表时出错: {e}")
            print("尝试删除常见的索引名称...")

            # 备用方案：尝试删除常见的索引（兼容清理受 CLEAN_LEGACY_INDEXES 控制）
            common_indexes = [
                CHUNK_VECTOR_INDEX,
                ENTITY_VECTOR_INDEX,
            ]
            if CLEAN_LEGACY_INDEXES:
                common_indexes.extend(
                    [
                        "chunk_embedding",
                        "chunk_vector",
                        "entity_embedding",
                        "entity_vector",
                        "vector",
                    ]
                )

            for index_name in common_indexes:
                try:
                    self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
                    print(f"  已尝试删除: {index_name}")
                except Exception as e:
                    print(f"  删除 {index_name} 失败: {e}")

        print("="*60 + "\n")

    def close(self):
        """
        显式关闭数据库连接，优雅释放资源

        在应用关闭时调用此方法以确保连接被正确释放。
        调用后允许重新创建实例。
        """
        with self._lock:
            if hasattr(self, 'graph') and self.graph:
                try:
                    # 尝试关闭 graph 连接
                    if hasattr(self.graph, 'close'):
                        self.graph.close()
                        print("✅ Graph connection closed successfully.")
                except Exception as e:
                    print(f"⚠️ Error closing graph connection: {e}")

            if hasattr(self, 'driver') and self.driver:
                try:
                    # 尝试关闭 driver 连接
                    if hasattr(self.driver, 'close'):
                        self.driver.close()
                        print("✅ Driver connection closed successfully.")
                except Exception as e:
                    print(f"⚠️ Error closing driver connection: {e}")

            # 重置状态，允许重新创建实例
            self._initialized = False
            GraphConnectionManager._instance = None

    @classmethod
    def get_instance(cls):
        """
        获取单例实例的显式方法

        Returns:
            GraphConnectionManager: 单例实例
        """
        return cls()

# 创建全局连接管理器实例
connection_manager = GraphConnectionManager()
