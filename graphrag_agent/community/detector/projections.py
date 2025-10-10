from typing import Dict, Any, Tuple

class GraphProjectionMixin:
    """图投影功能的混入类"""
    
    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影"""
        print("开始创建社区检测的图投影...")
        
        # 检查节点数量
        node_count = self._get_node_count()
        if node_count > self.node_count_limit:
            print(f"警告: 节点数量({node_count})超过限制({self.node_count_limit})")
            return self._create_filtered_projection(node_count)
        
        # 删除已存在的投影
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"删除旧投影时出错 (可忽略): {e}")
        
        # 创建标准投影
        try:
            self.G, result = self.gds.graph.project(
                self.projection_name,
                "__Entity__",
                {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                    }
                },
            )
            print(f"图投影创建成功: {result.get('nodeCount', 0)} 节点, "
                  f"{result.get('relationshipCount', 0)} 关系")
            return self.G, result
        except Exception as e:
            print(f"标准投影创建失败: {e}")
            return self._create_conservative_projection()
    
    def _get_node_count(self) -> int:
        """获取节点数量"""
        result = self.graph.query(
            "MATCH (e:__Entity__) RETURN count(e) AS count"
        )
        return result[0]["count"] if result else 0
    
    def _create_filtered_projection(self, total_node_count: int) -> Tuple[Any, Dict]:
        """创建过滤后的投影"""
        print("创建过滤后的投影...")
        
        try:
            # 获取重要节点
            result = self.graph.query(
                """
                MATCH (e:__Entity__)-[r]-()
                WITH e, count(r) AS rel_count
                ORDER BY rel_count DESC
                LIMIT toInteger($limit)
                RETURN collect(id(e)) AS important_nodes
                """,
                params={"limit": self.node_count_limit}
            )
            
            if not result or not result[0]["important_nodes"]:
                return self._create_conservative_projection()
            
            important_nodes = result[0]["important_nodes"]
            
            # 创建过滤投影
            config = {
                "nodeProjection": {
                    "__Entity__": {
                        "properties": ["*"],
                        "filter": f"id(node) IN {important_nodes}"
                    }
                },
                "relationshipProjection": {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}}
                    }
                }
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"过滤投影创建成功: {result.get('nodeCount', 0)} 节点, "
                  f"{result.get('relationshipCount', 0)} 关系")
            return self.G, result
            
        except Exception as e:
            print(f"过滤投影创建失败: {e}")
            return self._create_conservative_projection()
    
    def _create_conservative_projection(self) -> Tuple[Any, Dict]:
        """创建保守配置的投影"""
        print("尝试使用保守配置创建投影...")
        
        try:
            # 使用最小配置
            config = {
                "nodeProjection": "__Entity__",
                "relationshipProjection": "*"
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"保守投影创建成功: {result.get('nodeCount', 0)} 节点")
            return self.G, result
            
        except Exception as e:
            print(f"保守投影创建失败: {e}")
            return self._create_minimal_projection()
    
    def _create_minimal_projection(self) -> Tuple[Any, Dict]:
        """创建最小化投影"""
        print("尝试创建最小化投影...")
        
        try:
            # 获取最重要的节点
            result = self.graph.query(
                """
                MATCH (e:__Entity__)-[r]-()
                WITH e, count(r) AS rel_count
                ORDER BY rel_count DESC
                LIMIT 1000
                RETURN collect(id(e)) AS critical_nodes
                """
            )
            
            if not result or not result[0]["critical_nodes"]:
                raise ValueError("无法获取关键节点")
            
            critical_nodes = result[0]["critical_nodes"]
            
            # 创建最小化投影
            minimal_config = {
                "nodeProjection": {
                    "__Entity__": {
                        "filter": f"id(node) IN {critical_nodes}"
                    }
                },
                "relationshipProjection": "*"
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                minimal_config
            )
            print(f"最小化投影创建成功: {result.get('nodeCount', 0)} 节点")
            return self.G, result
            
        except Exception as e:
            print(f"所有投影方法均失败: {e}")
            raise ValueError("无法创建必要的图投影")