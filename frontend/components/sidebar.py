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
            ["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent", "fusion_agent"],
            index=0 if st.session_state.agent_type == "graph_agent" 
                    else (1 if st.session_state.agent_type == "hybrid_agent" 
                         else (2 if st.session_state.agent_type == "naive_rag_agent" 
                              else (3 if st.session_state.agent_type == "deep_research_agent"
                                   else 4))),
            help="graph_agent：使用知识图谱的局部与全局搜索；hybrid_agent：使用混合搜索方式；naive_rag_agent：使用朴素RAG；deep_research_agent：私域深度研究；fusion_agent：融合式图谱Agent",
            key="sidebar_agent_type"
        )
        # 更新全局agent_type
        st.session_state.agent_type = agent_type

        # 思考过程选项 - 仅当选择 deep_research_agent 时显示
        if agent_type == "deep_research_agent":
            # 思考过程选项
            show_thinking = st.checkbox("显示推理过程", 
                                    value=st.session_state.get("show_thinking", False), 
                                    key="sidebar_show_thinking",
                                    help="显示AI的思考过程")
            # 更新全局 show_thinking
            st.session_state.show_thinking = show_thinking
            
            # 添加增强版工具选择
            use_deeper = st.checkbox("使用增强版研究工具", 
                                value=st.session_state.get("use_deeper_tool", True), 
                                key="sidebar_use_deeper",
                                help="启用社区感知和知识图谱增强功能")
            # 更新全局 use_deeper_tool
            st.session_state.use_deeper_tool = use_deeper
            
            # 添加工具说明
            if use_deeper:
                st.info("增强版研究工具：整合社区感知和知识图谱增强，实现更深度的多级推理")
            else:
                st.info("标准版研究工具：实现基础的多轮推理和搜索")
                
        elif "show_thinking" in st.session_state:
            # 如果切换到其他Agent类型，重置show_thinking为False
            st.session_state.show_thinking = False
        
        st.markdown("---")
        
        # 系统设置部分 - 组合调试模式和响应设置
        st.header("系统设置")
        
        # 调试选项
        debug_mode = st.checkbox("启用调试模式", 
                               value=st.session_state.debug_mode, 
                               key="sidebar_debug_mode",
                               help="显示执行轨迹、知识图谱和源内容")
        
        # 当调试模式切换时，处理流式响应状态
        previous_debug_mode = st.session_state.debug_mode
        if debug_mode != previous_debug_mode:
            if debug_mode:
                # 启用调试模式时，禁用流式响应
                st.session_state.use_stream = False
        
        # 更新全局debug_mode
        st.session_state.debug_mode = debug_mode
        
        # 添加流式响应选项（仅当调试模式未启用时显示）
        if not debug_mode:
            use_stream = st.checkbox("使用流式响应", 
                                   value=st.session_state.get("use_stream", True), 
                                   key="sidebar_use_stream",
                                   help="启用流式响应，实时显示生成结果")
            # 更新全局 use_stream
            st.session_state.use_stream = use_stream
        else:
            # 在调试模式下显示提示
            st.info("调试模式下已禁用流式响应")
        
        st.markdown("---")
        
        # 示例问题部分
        st.header("示例问题")
        example_questions = examples

        for idx, question in enumerate(example_questions):
            if st.button(
                question,
                key=f"example_btn_{idx}",
                use_container_width=True,
                help="点击填入此问题"
            ):
                # 将示例问题设置到session state中，供主页面使用
                st.session_state.example_question = question
                st.rerun()
        
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