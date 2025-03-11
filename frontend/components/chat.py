import time
import streamlit as st
import uuid
from utils.api import send_message, send_feedback, get_source_content, get_knowledge_graph_from_message, clear_chat
from utils.helpers import extract_source_ids

def display_chat_interface():
    """显示主聊天界面"""
    st.title("GraphRAG 对话系统")
    
    # 设置栏
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 使用不同的key: header_agent_type
            agent_type = st.selectbox(
                "选择 Agent 类型",
                options=["graph_agent", "hybrid_agent", "naive_rag_agent"],
                key="header_agent_type",
                help="选择不同的Agent以体验不同的检索策略",
                index=0 if st.session_state.agent_type == "graph_agent" 
                        else (1 if st.session_state.agent_type == "hybrid_agent" else 2)
            )
            st.session_state.agent_type = agent_type
    
        with col2:
            st.button("🗑️ 清除聊天", on_click=clear_chat)
    
    # 分隔线
    st.markdown("---")
    
    # 聊天区域
    chat_container = st.container()
    with chat_container:
        # 显示现有消息
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # 为AI回答添加反馈按钮和源内容引用
                if msg["role"] == "assistant":
                    # 生成一个唯一的消息ID (如果之前没有)
                    if "message_id" not in msg:
                        msg["message_id"] = str(uuid.uuid4())
                        
                    # 查找对应的用户问题
                    user_query = ""
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        user_query = st.session_state.messages[i-1]["content"]
                        
                    # 检查是否已经提供过反馈
                    feedback_key = f"{msg['message_id']}"
                    feedback_type_key = f"feedback_type_{feedback_key}"
                    
                    if feedback_key not in st.session_state.feedback_given:
                        # 添加反馈按钮
                        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                        
                        with col1:
                            if st.button("👍", key=f"thumbs_up_{msg['message_id']}"):
                                # 检查是否有正在处理的请求
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                if st.session_state.feedback_in_progress:
                                    st.warning("请等待当前操作完成...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    with st.spinner("正在提交反馈..."):
                                        response = send_feedback(
                                            msg["message_id"], 
                                            user_query, 
                                            True, 
                                            st.session_state.session_id,
                                            st.session_state.agent_type
                                        )
                                        # 短暂延迟确保请求完成
                                        time.sleep(0.5)
                                    
                                    st.session_state.feedback_given.add(feedback_key)
                                    st.session_state[feedback_type_key] = "positive"
                                    
                                    # 根据响应显示不同的消息
                                    if response and "action" in response:
                                        if "高质量" in response["action"]:
                                            st.success("感谢您的肯定！此回答已被标记为高质量。", icon="🙂")
                                        else:
                                            st.success("感谢您的反馈！", icon="👍")
                                    else:
                                        st.info("已收到您的反馈。", icon="ℹ️")
                                        
                                    st.session_state.feedback_in_progress = False
                                    st.rerun()
                                
                        with col2:
                            if st.button("👎", key=f"thumbs_down_{msg['message_id']}"):
                                # 检查是否有正在处理的请求
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                if st.session_state.feedback_in_progress:
                                    st.warning("请等待当前操作完成...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    with st.spinner("正在提交反馈..."):
                                        response = send_feedback(
                                            msg["message_id"], 
                                            user_query, 
                                            False, 
                                            st.session_state.session_id,
                                            st.session_state.agent_type
                                        )
                                        # 短暂延迟确保请求完成
                                        time.sleep(0.5)
                                    
                                    st.session_state.feedback_given.add(feedback_key)
                                    st.session_state[feedback_type_key] = "negative"
                                    
                                    # 根据响应显示不同的消息
                                    if response and "action" in response:
                                        if "清除" in response["action"]:
                                            st.error("已收到您的反馈，此回答将不再使用。", icon="🔄")
                                        else:
                                            st.error("已收到您的反馈，我们会改进。", icon="👎")
                                    else:
                                        st.info("已收到您的反馈。", icon="ℹ️")
                                        
                                    st.session_state.feedback_in_progress = False
                                    st.rerun()
                    else:
                        # 显示已提供的反馈类型
                        feedback_type = st.session_state.get(feedback_type_key, None)
                        if feedback_type == "positive":
                            st.success("您已对此回答给予肯定！", icon="👍")
                        elif feedback_type == "negative":
                            st.error("您已对此回答提出改进建议。", icon="👎")
                        else:
                            st.info("已收到您的反馈。", icon="ℹ️")
                
                    # 如果是AI回答且有源内容引用，显示查看源内容按钮
                    if st.session_state.debug_mode:
                        source_ids = extract_source_ids(msg["content"])
                        if source_ids:
                            with st.expander("查看引用源文本", expanded=False):
                                for source_id in source_ids:
                                    if st.button(f"加载源文本 {source_id}", key=f"src_{source_id}_{i}"):
                                        with st.spinner(f"加载源文本 {source_id}..."):
                                            source_data = get_source_content(source_id)
                                            if source_data and "content" in source_data:
                                                st.session_state.source_content = source_data["content"]
                                                st.session_state.current_tab = "源内容"  # 自动切换到源内容标签
                                                st.rerun()
                        
                        # 如果是最后一条AI消息，添加自动提取图谱按钮
                        if st.button("提取知识图谱", key=f"extract_kg_{i}"):
                            with st.spinner("提取知识图谱数据..."):
                                # 获取对应的用户查询
                                user_query = ""
                                if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                                    user_query = st.session_state.messages[i-1]["content"]
                                    
                                # 使用用户查询来过滤知识图谱
                                kg_data = get_knowledge_graph_from_message(msg["content"], user_query)
                                if kg_data and len(kg_data.get("nodes", [])) > 0:
                                    # 确保当前消息有正确的kg_data
                                    st.session_state.messages[i]["kg_data"] = kg_data
                                    # 更新当前的图谱消息索引为当前处理的消息索引
                                    st.session_state.current_kg_message = i
                                    st.session_state.current_tab = "知识图谱"  # 自动切换到知识图谱标签
                                    st.rerun()
        
        # 处理新消息
        if prompt := st.chat_input("请输入您的问题...", key="chat_input"):
            # 检查是否有正在处理的请求
            if "processing_lock" not in st.session_state:
                st.session_state.processing_lock = False
                
            if st.session_state.processing_lock:
                st.warning("请等待当前操作完成...")
                return
                
            st.session_state.processing_lock = True
            
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    response = send_message(prompt)
                if response:
                    st.write(response["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "message_id": str(uuid.uuid4())  # 为新消息生成ID
                    })
                    if response.get("execution_log"):
                        st.session_state.execution_log = response["execution_log"]
                        
                    # 从回答中提取知识图谱数据
                    if st.session_state.debug_mode:
                        try:
                            with st.spinner("提取知识图谱数据..."):
                                # 优先使用后端返回的kg_data
                                kg_data = response.get("kg_data")
                                
                                # 如果后端没有返回kg_data，尝试从回答中提取，并传递用户查询
                                if not kg_data or len(kg_data.get("nodes", [])) == 0:
                                    kg_data = get_knowledge_graph_from_message(response["answer"], prompt)  # 传递当前查询
                                
                                if kg_data and len(kg_data.get("nodes", [])) > 0:
                                    # 获取当前新消息的索引，即最后一条消息
                                    current_msg_index = len(st.session_state.messages) - 1
                                    
                                    # 更新该消息的kg_data
                                    st.session_state.messages[current_msg_index]["kg_data"] = kg_data
                                    
                                    # 更新当前处理的图谱消息索引为最新消息的索引
                                    st.session_state.current_kg_message = current_msg_index
                                    
                                    # 自动切换到知识图谱标签
                                    st.session_state.current_tab = "知识图谱"
                        except Exception as e:
                            print(f"提取知识图谱失败: {e}")
            
            # 确保请求处理完成后释放锁
            st.session_state.processing_lock = False
            st.rerun()