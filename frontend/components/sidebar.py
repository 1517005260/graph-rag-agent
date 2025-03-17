import streamlit as st
from utils.api import clear_chat
from frontend_config.settings import examples

def display_sidebar():
    """显示应用侧边栏"""
    with st.sidebar:
        st.title("📚 GraphRAG")
        st.markdown("---")
        
        # Agent选择部分
        st.header("Agent 选择")
        agent_type = st.radio(
            "选择检索策略:",
            ["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent"],
            index=0 if st.session_state.agent_type == "graph_agent" 
                    else (1 if st.session_state.agent_type == "hybrid_agent" 
                         else (2 if st.session_state.agent_type == "naive_rag_agent" 
                              else 3)),
            help="graph_agent：使用知识图谱的局部与全局搜索；hybrid_agent：使用混合搜索方式；naive_rag_agent：使用朴素RAG；deep_research_agent：私域深度研究",
            key="sidebar_agent_type"
        )
        # 更新全局agent_type
        st.session_state.agent_type = agent_type

        # 思考过程选项 - 仅当选择 deep_research_agent 时显示
        if agent_type == "deep_research_agent":
            show_thinking = st.checkbox("显示推理过程", 
                                      value=st.session_state.get("show_thinking", False), 
                                      key="sidebar_show_thinking",
                                      help="显示AI的思考过程")
            # 更新全局 show_thinking
            st.session_state.show_thinking = show_thinking
        elif "show_thinking" in st.session_state:
            # 如果切换到其他Agent类型，重置show_thinking为False
            st.session_state.show_thinking = False
        
        st.markdown("---")
        
        # 调试选项
        st.header("调试选项")
        debug_mode = st.checkbox("启用调试模式", 
                               value=st.session_state.debug_mode, 
                               key="sidebar_debug_mode",
                               help="显示执行轨迹、知识图谱和源内容")
        # 更新全局debug_mode
        st.session_state.debug_mode = debug_mode
        
        st.markdown("---")
        
        # 示例问题部分
        st.header("示例问题")
        example_questions = examples
        
        for question in example_questions:
            st.markdown(f"""
            <div style="background-color: #f7f7f7; padding: 8px; 
                 border-radius: 4px; margin: 5px 0; font-size: 14px;">
                {question}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 项目信息
        st.markdown("""
        ### 关于
        这个 GraphRAG 演示基于本地文档建立的知识图谱，可以使用不同的Agent策略回答问题。
        
        **调试模式**可查看:
        - 执行轨迹
        - 知识图谱可视化
        - 原始文本内容
        - 性能监控
        """)
        
        # 重置按钮
        if st.button("🗑️ 清除对话历史", key="clear_chat"):
            clear_chat()