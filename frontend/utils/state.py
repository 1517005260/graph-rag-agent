import streamlit as st
import uuid
from frontend_config.settings import (
    DEFAULT_KG_SETTINGS,
    DEFAULT_AGENT_TYPE,
    DEFAULT_DEBUG_MODE,
    DEFAULT_SHOW_THINKING,
    DEFAULT_USE_DEEPER_TOOL,
    DEFAULT_USE_STREAM,
    DEFAULT_CHAIN_EXPLORATION,
)

def init_session_state():
    """初始化会话状态变量"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = DEFAULT_DEBUG_MODE
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = []
    if 'agent_type' not in st.session_state:
        st.session_state.agent_type = DEFAULT_AGENT_TYPE
    if 'show_thinking' not in st.session_state:
        st.session_state.show_thinking = DEFAULT_SHOW_THINKING
    if 'use_deeper_tool' not in st.session_state:
        st.session_state.use_deeper_tool = DEFAULT_USE_DEEPER_TOOL
    
    # 流式响应设置 - 默认启用，但调试模式下自动禁用
    if 'use_stream' not in st.session_state:
        st.session_state.use_stream = DEFAULT_USE_STREAM
    if st.session_state.debug_mode:
        # 确保调试模式下禁用流式响应
        st.session_state.use_stream = False
        
    if 'kg_data' not in st.session_state:
        st.session_state.kg_data = None
    if 'source_content' not in st.session_state:
        st.session_state.source_content = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "执行轨迹"
    if 'kg_display_settings' not in st.session_state:
        st.session_state.kg_display_settings = DEFAULT_KG_SETTINGS
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()
    if 'feedback_in_progress' not in st.session_state:
        st.session_state.feedback_in_progress = False
    if 'processing_lock' not in st.session_state:
        st.session_state.processing_lock = False
    if 'current_kg_message' not in st.session_state:
        st.session_state.current_kg_message = None
    
    # 知识图谱管理相关状态
    if 'entity_to_update' not in st.session_state:
        st.session_state.entity_to_update = None
    if 'found_relations' not in st.session_state:
        st.session_state.found_relations = None
    if 'relation_to_update' not in st.session_state:
        st.session_state.relation_to_update = None
    if 'use_chain_exploration' not in st.session_state:
        st.session_state.use_chain_exploration = DEFAULT_CHAIN_EXPLORATION

    if 'cache' not in st.session_state:
        st.session_state.cache = {
            'source_info': {},  # 源文件信息缓存
            'knowledge_graphs': {},  # 知识图谱缓存
            'vector_search_results': {},  # 向量搜索结果缓存
            'api_responses': {},  # API响应缓存
        }
