from typing import Dict, List
import threading
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, ToolMessage


# 创建Agent管理类
class AgentManager:
    """Agent管理类"""
    
    def __init__(self):
        """初始化Agent管理器"""
        # 导入各种Agent
        from graphrag_agent.agents.graph_agent import GraphAgent
        from graphrag_agent.agents.hybrid_agent import HybridAgent
        from graphrag_agent.agents.naive_rag_agent import NaiveRagAgent
        from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
        from graphrag_agent.agents.fusion_agent import FusionGraphRAGAgent 
        
        # 初始化Agent类
        self.agent_classes = {
            "graph_agent": GraphAgent,
            "hybrid_agent": HybridAgent,
            "naive_rag_agent": NaiveRagAgent,
            "deep_research_agent": DeepResearchAgent,
            "fusion_agent": FusionGraphRAGAgent,
        }
        
        # 保留Agent实例池
        self.agent_instances = {}
        
        # 添加锁来保护实例访问
        self.agent_lock = threading.RLock()
    
    def get_agent(self, agent_type: str, session_id: str = "default"):
        """
        获取指定类型的Agent，对每个会话使用独立实例
        
        Args:
            agent_type: Agent类型名称
            session_id: 会话ID
            
        Returns:
            Agent实例
        """
        if agent_type not in self.agent_classes:
            raise ValueError(f"未知的agent类型: {agent_type}")
        
        # 为每个会话使用单独的Agent实例，避免资源争用
        instance_key = f"{agent_type}:{session_id}"
        
        with self.agent_lock:
            if instance_key not in self.agent_instances:
                # 创建新的Agent实例
                self.agent_instances[instance_key] = self.agent_classes[agent_type]()
            
            return self.agent_instances[instance_key]
    
    def clear_history(self, session_id: str) -> Dict:
        """
        清除特定会话的聊天历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 清除结果信息
        """
        remaining_text = ""
        
        try:
            # 清除对应会话的所有agent实例历史
            with self.agent_lock:
                for agent_type in self.agent_classes.keys():
                    instance_key = f"{agent_type}:{session_id}"
                    if instance_key in self.agent_instances:
                        agent = self.agent_instances[instance_key]
                        config = {"configurable": {"thread_id": session_id}}
                        
                        # 添加检查，防止None值报错
                        memory_content = agent.memory.get(config)
                        if memory_content is None or "channel_values" not in memory_content:
                            continue  # 跳过这个agent
                            
                        messages = memory_content["channel_values"]["messages"]
                        
                        # 如果消息少于2条，不进行删除操作
                        if len(messages) <= 2:
                            continue

                        i = len(messages)
                        for message in reversed(messages):
                            if isinstance(messages[2], ToolMessage) and i == 4:
                                break
                            agent.graph.update_state(config, {"messages": RemoveMessage(id=message.id)})
                            i = i - 1
                            if i == 2:  # 保留前两条消息
                                break
            
            # 获取剩余消息
            remaining_text = "已清除会话历史"
        
        except Exception as e:
            print(f"清除聊天历史时出错: {str(e)}")
        
        return {
            "status": "success",
            "remaining_messages": remaining_text
        }
    
    def close_all(self):
        """关闭所有Agent资源"""
        with self.agent_lock:
            for instance_key, agent in self.agent_instances.items():
                try:
                    agent.close()
                    print(f"已关闭 {instance_key} 资源")
                except Exception as e:
                    print(f"关闭 {instance_key} 资源时出错: {e}")
            
            # 清空实例池
            self.agent_instances.clear()


# 创建全局实例
agent_manager = AgentManager()


def format_messages_for_response(messages: List[Dict]) -> str:
    """
    将消息格式化为字符串
    
    Args:
        messages: 消息列表
    
    Returns:
        str: 格式化后的消息字符串
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            prefix = "User: " if isinstance(msg, HumanMessage) else "AI: "
            formatted.append(f"{prefix}{msg.content}")
    return "\n".join(formatted)


def format_execution_log(log: List[Dict]) -> List[Dict]:
    """
    格式化执行日志用于JSON响应
    
    Args:
        log: 原始执行日志
    
    Returns:
        List[Dict]: 格式化后的执行日志
    """
    formatted_log = []
    for entry in log:
        formatted_entry = {"node": entry["node"]}
        
        # 处理输入
        if "input" in entry:
            if isinstance(entry["input"], dict):
                input_str = {}
                for k, v in entry["input"].items():
                    if hasattr(v, "content"):
                        # 处理消息对象
                        input_str[k] = {"content": v.content}
                    elif isinstance(v, str):
                        input_str[k] = v
                    else:
                        # 安全处理其他类型
                        try:
                            import json
                            json.dumps(v)  # 测试是否可序列化
                            input_str[k] = v
                        except:
                            input_str[k] = str(v)
            elif hasattr(entry["input"], "content"):
                # 直接处理消息对象
                input_str = {"content": entry["input"].content}
            else:
                input_str = str(entry["input"])
            formatted_entry["input"] = input_str
            
        # 处理输出
        if "output" in entry:
            if isinstance(entry["output"], dict):
                output_str = {}
                for k, v in entry["output"].items():
                    if hasattr(v, "content"):
                        # 处理消息对象
                        output_str[k] = {"content": v.content}
                    elif isinstance(v, str):
                        output_str[k] = v
                    else:
                        # 安全处理其他类型
                        try:
                            import json
                            json.dumps(v)  # 测试是否可序列化
                            output_str[k] = v
                        except:
                            output_str[k] = str(v)
            elif hasattr(entry["output"], "content"):
                # 直接处理消息对象
                output_str = {"content": entry["output"].content}
            else:
                output_str = str(entry["output"])
            formatted_entry["output"] = output_str
        
        formatted_log.append(formatted_entry)
    return formatted_log