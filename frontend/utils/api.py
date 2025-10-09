import time
import uuid
import requests
import queue
import json
import threading
import time
import streamlit as st
from typing import Dict, Callable
from frontend_config.settings import API_URL
from utils.performance import monitor_performance
from graphrag_agent.config.settings import community_algorithm

@monitor_performance(endpoint="send_message")
def send_message(message: str) -> Dict:
    """发送聊天消息到 FastAPI 后端，带性能监控"""
    start_time = time.time()
    try:
        # 构建请求参数
        params = {
            "message": message,
            "session_id": st.session_state.session_id,
            "debug": st.session_state.debug_mode,
            "agent_type": st.session_state.agent_type
        }
        
        # 如果是深度研究Agent，添加是否使用增强版工具的参数
        if st.session_state.agent_type == "deep_research_agent":
            params["use_deeper_tool"] = st.session_state.get("use_deeper_tool", True)
            params["show_thinking"] = st.session_state.get("show_thinking", False)
        
        # 如果是融合Agent，添加特定参数
        if st.session_state.agent_type == "fusion_agent":
            params["use_chain_exploration"] = st.session_state.get("use_chain_exploration", True)
        
        response = requests.post(
            f"{API_URL}/chat",
            json=params,
            # timeout=120  # 增加超时时间
        )
        
        # 记录性能
        duration = time.time() - start_time
        print(f"前端API调用耗时: {duration:.4f}s")
        
        # 在会话中保存性能数据
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_message",
            "duration": duration,
            "timestamp": time.time(),
            "message_length": len(message)
        })
        
        return response.json()
    except requests.exceptions.RequestException as e:
        # 记录错误性能
        duration = time.time() - start_time
        print(f"前端API调用错误: {str(e)} ({duration:.4f}s)")
        
        st.error(f"服务器连接错误: {str(e)}")
        return None

def send_message_stream(message: str, on_token: Callable[[str, bool], None]) -> str:
    """
    向 FastAPI 后端发送聊天消息，获取流式响应
    
    Args:
        message: 要发送的消息
        on_token: 处理令牌的回调函数
        
    Returns:
        str: 收集的思考内容（如果有）
    """
    # 如果调试模式启用，直接回退到非流式API
    if st.session_state.debug_mode:
        print("调试模式已启用，使用非流式API")
        response = send_message(message)
        if response and "answer" in response:
            on_token(response["answer"])
            # 如果有思考内容，返回它
            return response.get("raw_thinking", "")
        return ""
        
    try:
        # 构建请求参数
        params = {
            "message": message,
            "session_id": st.session_state.session_id,
            "debug": st.session_state.debug_mode,
            "agent_type": st.session_state.agent_type
        }
        
        # 如果是深度研究Agent，添加特定参数
        if st.session_state.agent_type == "deep_research_agent":
            params["use_deeper_tool"] = st.session_state.get("use_deeper_tool", True)
            params["show_thinking"] = st.session_state.get("show_thinking", False)
        
        # 如果是融合Agent，添加特定参数
        if st.session_state.agent_type == "fusion_agent":
            params["use_chain_exploration"] = st.session_state.get("use_chain_exploration", True)
        
        # 设置 SSE 连接
        import sseclient
        import requests
        import json
        
        # 非阻塞模式发起请求
        response = requests.post(
            f"{API_URL}/chat/stream",
            json=params,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        # 设置 SSE 客户端
        client = sseclient.SSEClient(response)
        
        # 处理每个事件
        thinking_content = ""
        
        for event in client.events():
            try:
                # 确保解析 JSON 时捕获所有可能的异常
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}, 原始数据: {event.data[:100]}")
                    continue
                
                # 处理不同事件类型
                if data.get("status") == "token":
                    # 模型输出的令牌
                    on_token(data.get("content", ""))
                elif data.get("status") == "thinking":
                    # 思考过程块
                    chunk = data.get("content", "")
                    thinking_content += chunk
                    on_token(chunk, is_thinking=True)
                elif data.get("status") == "execution_log" and st.session_state.debug_mode:
                    # 处理执行日志
                    if "execution_log" not in st.session_state:
                        st.session_state.execution_log = []
                    st.session_state.execution_log.append(data.get("content", {}))
                elif data.get("status") == "done":
                    # 完成通知
                    break
                elif data.get("status") == "error":
                    # 错误通知
                    on_token(f"\n\n错误: {data.get('message', '未知错误')}")
                    break
                else:
                    # 其他状态类型处理
                    pass
            except Exception as e:
                # 处理任何未捕获的异常
                print(f"处理SSE事件时出错: {str(e)}")
                continue
        
        # 返回收集的思考内容用于存储
        return thinking_content
    except Exception as e:
        # 处理连接错误
        on_token(f"\n\n连接错误: {str(e)}")
        print(f"流式API连接错误: {str(e)}")
        return None

@monitor_performance(endpoint="send_feedback")
def send_feedback(message_id: str, query: str, is_positive: bool, thread_id: str, agent_type: str = "graph_agent"):
    """向后端发送用户反馈"""
    start_time = time.time()
    try:
        # 确保 agent_type 有值
        if not agent_type:
            agent_type = "graph_agent"
            
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "message_id": message_id,
                "query": query,
                "is_positive": is_positive,
                "thread_id": thread_id,
                "agent_type": agent_type  # 确保这个字段被包含在请求中
            },
            timeout=10
        )
        
        # 记录性能
        duration = time.time() - start_time
        print(f"前端反馈API调用耗时: {duration:.4f}s")
        
        # 在会话中保存性能数据
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_feedback",
            "duration": duration,
            "timestamp": time.time(),
            "is_positive": is_positive
        })
        
        # 记录和返回响应
        try:
            return response.json()
        except:
            return {"status": "error", "action": "解析响应失败"}
    except requests.exceptions.RequestException as e:
        # 记录错误性能
        duration = time.time() - start_time
        print(f"前端反馈API调用错误: {str(e)} ({duration:.4f}s)")
        
        st.error(f"发送反馈时出错: {str(e)}")
        return {"status": "error", "action": str(e)}

@monitor_performance(endpoint="get_knowledge_graph")
def get_knowledge_graph(limit: int = 100, query: str = None) -> Dict:
    """获取知识图谱数据"""
    # 生成缓存键
    cache_key = f"kg:limit={limit}:query={query}"
    
    # 检查缓存
    if cache_key in st.session_state.cache.get('knowledge_graphs', {}):
        return st.session_state.cache['knowledge_graphs'][cache_key]
    
    try:
        params = {"limit": limit}
        if query:
            params["query"] = query
            
        response = requests.get(
            f"{API_URL}/knowledge_graph",
            params=params,
            timeout=30
        )
        result = response.json()
        
        # 缓存结果
        if 'knowledge_graphs' not in st.session_state.cache:
            st.session_state.cache['knowledge_graphs'] = {}
        st.session_state.cache['knowledge_graphs'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"获取知识图谱时出错: {str(e)}")
        return {"nodes": [], "links": []}

def get_knowledge_graph_from_message(message: str, query: str = None):
    """从AI响应中提取知识图谱数据"""
    # 生成缓存键 - 使用消息哈希和查询组合
    import hashlib
    message_hash = hashlib.md5(message.encode()).hexdigest()
    cache_key = f"kg_msg:{message_hash}:query={query}"
    
    # 检查缓存
    if cache_key in st.session_state.cache.get('knowledge_graphs', {}):
        return st.session_state.cache['knowledge_graphs'][cache_key]
    
    try:
        params = {"message": message}
        if query:
            params["query"] = query
            
        response = requests.get(
            f"{API_URL}/knowledge_graph_from_message",
            params=params,
            timeout=30
        )
        result = response.json()
        
        # 缓存结果
        if 'knowledge_graphs' not in st.session_state.cache:
            st.session_state.cache['knowledge_graphs'] = {}
        st.session_state.cache['knowledge_graphs'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"从响应提取知识图谱时出错: {str(e)}")
        return {"nodes": [], "links": []}

@monitor_performance(endpoint="get_source_content")
def get_source_content(source_id: str) -> Dict:
    """获取源内容"""
    # 检查缓存
    cache_key = f"content:{source_id}"
    if cache_key in st.session_state.cache.get('api_responses', {}):
        return st.session_state.cache['api_responses'][cache_key]
    
    try:
        response = requests.post(
            f"{API_URL}/source",
            json={"source_id": source_id},
            timeout=30
        )
        result = response.json()
        
        # 缓存结果
        if 'api_responses' not in st.session_state.cache:
            st.session_state.cache['api_responses'] = {}
        st.session_state.cache['api_responses'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"获取源内容时出错: {str(e)}")
        return None

def get_source_file_info(source_id: str) -> dict:
    """获取源ID对应的文件信息"""
    # 检查缓存
    if source_id in st.session_state.cache.get('source_info', {}):
        return st.session_state.cache['source_info'][source_id]
    
    try:
        response = requests.post(
            f"{API_URL}/source_info",
            json={"source_id": source_id},
            timeout=10
        )
        result = response.json()
        
        # 缓存结果
        if 'source_info' not in st.session_state.cache:
            st.session_state.cache['source_info'] = {}
        st.session_state.cache['source_info'][source_id] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"获取源文件信息时出错: {str(e)}")
        default_info = {"file_name": f"源文本 {source_id}"}
        
        # 缓存默认结果
        if 'source_info' not in st.session_state.cache:
            st.session_state.cache['source_info'] = {}
        st.session_state.cache['source_info'][source_id] = default_info
        
        return default_info

def get_source_file_info_batch(source_ids: list) -> dict:
    """获取多个源ID对应的文件信息
    
    Args:
        source_ids: 源ID列表
        
    Returns:
        Dict: ID到文件信息的映射字典
    """
    try:
        response = requests.post(
            f"{API_URL}/source_info_batch",
            json={"source_ids": source_ids},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"批量获取源文件信息时出错: {str(e)}")
        return {sid: {"file_name": f"源文本 {sid}"} for sid in source_ids}

@monitor_performance(endpoint="kg_reasoning")
def get_kg_reasoning(reasoning_type, entity_a, entity_b=None, max_depth=3, algorithm=community_algorithm):
    """知识图谱推理API调用"""
    try:
        params = {
            "reasoning_type": reasoning_type,
            "entity_a": entity_a.strip() if entity_a else "",
            "max_depth": min(max(1, max_depth), 5),  # 确保在1-5的范围内
            "algorithm": algorithm
        }
        
        if entity_b:
            params["entity_b"] = entity_b.strip()
        
        # print(f"发送知识图谱推理请求: {params}")
        
        # 使用JSON格式发送请求
        response = requests.post(
            f"{API_URL}/kg_reasoning",
            json=params,
            timeout=60  # 社区检测可能需要更长时间
        )
        
        if response.status_code != 200:
            st.error(f"API请求失败: HTTP {response.status_code}")
            try:
                error_details = response.json()
                return {"error": f"API错误: {error_details.get('detail', '未知错误')}", "nodes": [], "links": []}
            except:
                return {"error": f"API错误: HTTP {response.status_code}", "nodes": [], "links": []}
        
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"知识图谱推理请求失败: {str(e)}")
        return {"error": str(e), "nodes": [], "links": []}

def get_entity_types():
    """获取所有实体类型"""
    try:
        response = requests.get(
            f"{API_URL}/entity_types",
            timeout=10
        )
        result = response.json()
        return result.get("entity_types", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取实体类型失败: {str(e)}")
        return []

def get_relation_types():
    """获取所有关系类型"""
    try:
        response = requests.get(
            f"{API_URL}/relation_types",
            timeout=10
        )
        result = response.json()
        return result.get("relation_types", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取关系类型失败: {str(e)}")
        return []

def get_entities(filters=None):
    """获取实体列表，支持筛选"""
    try:
        if not filters:
            filters = {}
            
        response = requests.post(
            f"{API_URL}/entities/search",
            json=filters,
            timeout=20
        )
        result = response.json()
        return result.get("entities", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取实体列表失败: {str(e)}")
        return []

def get_relations(filters=None):
    """获取关系列表，支持筛选"""
    try:
        if not filters:
            filters = {}
            
        response = requests.post(
            f"{API_URL}/relations/search",
            json=filters,
            timeout=20
        )
        result = response.json()
        return result.get("relations", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取关系列表失败: {str(e)}")
        return []

def create_entity(entity_data):
    """创建新实体"""
    try:
        response = requests.post(
            f"{API_URL}/entity/create",
            json=entity_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"创建实体失败: {str(e)}")
        return {"success": False, "message": str(e)}

def update_entity(entity_data):
    """更新实体"""
    try:
        response = requests.post(
            f"{API_URL}/entity/update",
            json=entity_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"更新实体失败: {str(e)}")
        return {"success": False, "message": str(e)}

def delete_entity(entity_id):
    """删除实体"""
    try:
        response = requests.post(
            f"{API_URL}/entity/delete",
            json={"id": entity_id},
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"删除实体失败: {str(e)}")
        return {"success": False, "message": str(e)}

def create_relation(relation_data):
    """创建新关系"""
    try:
        response = requests.post(
            f"{API_URL}/relation/create",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"创建关系失败: {str(e)}")
        return {"success": False, "message": str(e)}

def update_relation(relation_data):
    """更新关系"""
    try:
        response = requests.post(
            f"{API_URL}/relation/update",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"更新关系失败: {str(e)}")
        return {"success": False, "message": str(e)}

def delete_relation(relation_data):
    """删除关系"""
    try:
        response = requests.post(
            f"{API_URL}/relation/delete",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"删除关系失败: {str(e)}")
        return {"success": False, "message": str(e)}

def clear_chat():
    """清除聊天历史"""
    try:
        # 清除前端状态
        st.session_state.processing_lock = False
        st.session_state.messages = []
        st.session_state.execution_log = None
        st.session_state.kg_data = None
        st.session_state.source_content = None
        
        # 重要：也要清除current_kg_message
        if 'current_kg_message' in st.session_state:
            del st.session_state.current_kg_message
        
        # 清除后端状态
        response = requests.post(
            f"{API_URL}/clear",
            json={"session_id": st.session_state.session_id}
        )
        
        if response.status_code != 200:
            st.error("清除后端对话历史失败")
            return
            
        # 重新生成会话ID
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
        
    except Exception as e:
        st.session_state.processing_lock = False
        st.error(f"清除对话时发生错误: {str(e)}")

def clear_cache(cache_type=None):
    """清除指定类型或所有缓存"""
    if cache_type and cache_type in st.session_state.cache:
        st.session_state.cache[cache_type] = {}
    elif not cache_type:
        st.session_state.cache = {
            'source_info': {},
            'knowledge_graphs': {},
            'vector_search_results': {},
            'api_responses': {},
        }



class ApiBatchProcessor:
    """API请求批处理器，合并短时间内的相似请求"""
    
    def __init__(self, batch_window=0.5, max_batch_size=10):
        """
        初始化批处理器
        
        Args:
            batch_window: 批处理窗口时间(秒)
            max_batch_size: 最大批量大小
        """
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.queues = {}  # 每种请求类型的队列
        self.locks = {}   # 每种队列的锁
        self.threads = {} # 处理线程
        self.running = True
    
    def add_request(self, request_type, request_data, callback):
        """
        添加请求到队列
        
        Args:
            request_type: 请求类型(例如'source_info', 'kg_data')
            request_data: 请求数据
            callback: 回调函数，处理结果返回
        """
        # 如果是第一次使用这种请求类型，初始化
        if request_type not in self.queues:
            self.queues[request_type] = queue.Queue()
            self.locks[request_type] = threading.Lock()
            # 启动处理线程
            self.threads[request_type] = threading.Thread(
                target=self._process_queue,
                args=(request_type,),
                daemon=True
            )
            self.threads[request_type].start()
        
        # 添加到队列
        self.queues[request_type].put((request_data, callback))
    
    def _process_queue(self, request_type):
        """
        处理特定类型的请求队列
        
        Args:
            request_type: 请求类型
        """
        while self.running:
            batch = []
            callbacks = []
            
            # 尝试在窗口时间内收集请求
            try:
                # 获取第一个请求，阻塞等待
                first_request, first_callback = self.queues[request_type].get(block=True)
                batch.append(first_request)
                callbacks.append(first_callback)
                
                # 设置批处理结束时间
                end_time = time.time() + self.batch_window
                
                # 收集更多请求直到窗口结束或达到最大批量
                while time.time() < end_time and len(batch) < self.max_batch_size:
                    try:
                        request, callback = self.queues[request_type].get(block=False)
                        batch.append(request)
                        callbacks.append(callback)
                    except queue.Empty:
                        break
                
                # 处理批量请求
                if len(batch) > 1:
                    # 执行批量处理
                    self._execute_batch(request_type, batch, callbacks)
                else:
                    # 单个请求，直接处理
                    self._execute_single(request_type, batch[0], callbacks[0])
                    
            except Exception as e:
                print(f"批处理错误({request_type}): {e}")
                time.sleep(0.1)  # 避免CPU占用过高
    
    def _execute_batch(self, request_type, batch, callbacks):
        """执行批量请求"""
        try:
            if request_type == 'source_info':
                # 批量获取源信息
                source_ids = batch
                results = self._batch_get_source_info(source_ids)
                
                # 处理回调
                for i, callback in enumerate(callbacks):
                    source_id = source_ids[i]
                    if source_id in results:
                        callback(results[source_id])
                    else:
                        # 默认结果
                        callback({"file_name": f"源文本 {source_id}"})
                        
            elif request_type == 'content':
                # 批量获取内容
                chunk_ids = batch
                results = self._batch_get_content(chunk_ids)
                
                # 处理回调
                for i, callback in enumerate(callbacks):
                    chunk_id = chunk_ids[i]
                    if chunk_id in results:
                        callback(results[chunk_id])
                    else:
                        callback(None)
                        
            # 可以添加其他批处理类型...
            
        except Exception as e:
            print(f"执行批量请求错误({request_type}): {e}")
            # 出错时单独执行每个请求
            for i, request in enumerate(batch):
                try:
                    self._execute_single(request_type, request, callbacks[i])
                except Exception as single_err:
                    print(f"单个请求错误({request_type}): {single_err}")
    
    def _execute_single(self, request_type, request, callback):
        """执行单个请求"""
        try:
            if request_type == 'source_info':
                result = get_source_file_info(request)
                callback(result)
            elif request_type == 'content':
                result = get_source_content(request)
                callback(result)
            # 可以添加其他请求类型...
        except Exception as e:
            print(f"执行单个请求错误({request_type}): {e}")
            callback(None)
    
    def _batch_get_source_info(self, source_ids):
        """批量获取源信息"""
        try:
            response = requests.post(
                f"{API_URL}/source_info_batch",
                json={"source_ids": source_ids},
                timeout=10
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"批量获取源信息错误: {e}")
            return {}
    
    def _batch_get_content(self, chunk_ids):
        """批量获取内容"""
        try:
            response = requests.post(
                f"{API_URL}/content_batch",
                json={"chunk_ids": chunk_ids},
                timeout=30
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"批量获取内容错误: {e}")
            return {}
    
    def shutdown(self):
        """关闭批处理器"""
        self.running = False
        # 等待所有线程完成
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)

# 初始化批处理器
def get_batch_processor():
    if 'api_batch_processor' not in st.session_state:
        st.session_state.api_batch_processor = ApiBatchProcessor()
    return st.session_state.api_batch_processor

# 使用批处理器的API函数示例
def get_source_info_async(source_id, callback):
    """异步获取源信息，使用批处理器"""
    processor = get_batch_processor()
    processor.add_request('source_info', source_id, callback)

def get_content_async(chunk_id, callback):
    """异步获取内容，使用批处理器"""
    processor = get_batch_processor()
    processor.add_request('content', chunk_id, callback)

# 在应用退出时关闭批处理器
def shutdown_batch_processor():
    if 'api_batch_processor' in st.session_state:
        st.session_state.api_batch_processor.shutdown()