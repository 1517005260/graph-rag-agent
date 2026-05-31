from graphrag_agent.config.neo4jdb import get_db_manager as original_get_db_manager

class DatabaseManager:
    """Neo4j 数据库管理类"""
    def __init__(self):
        self.driver = None

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()


def get_db_manager():
    return original_get_db_manager()