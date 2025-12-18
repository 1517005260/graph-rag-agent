"""
文档管理页面
提供文件上传、列表展示、删除等功能
"""

import streamlit as st
import requests
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from frontend_config.settings import API_URL, FILES_DIR


def get_file_info(filepath: Path) -> Dict:
    """获取文件信息"""
    stat = filepath.stat()
    return {
        'name': filepath.name,
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'type': filepath.suffix[1:].upper() if filepath.suffix else 'Unknown'
    }


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def list_documents() -> List[Dict]:
    """列出所有已上传文档"""
    files_path = Path(FILES_DIR)
    if not files_path.exists():
        return []

    documents = []
    for file in files_path.iterdir():
        if file.is_file() and not file.name.startswith('.'):
            documents.append(get_file_info(file))

    # 按修改时间降序排序
    documents.sort(key=lambda x: x['modified'], reverse=True)
    return documents


def delete_document(filename: str) -> bool:
    """删除文档"""
    try:
        file_path = Path(FILES_DIR) / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"删除失败: {e}")
        return False


def trigger_incremental_build():
    """触发增量构建"""
    try:
        response = requests.post(
            f"{API_URL}/admin/build/incremental",
            timeout=5  # 快速返回，构建在后台进行
        )
        if response.status_code == 200:
            return True, "增量构建已启动"
        else:
            return False, f"启动失败: {response.text}"
    except requests.exceptions.Timeout:
        return True, "增量构建已在后台启动"
    except Exception as e:
        return False, f"请求失败: {e}"


def document_manager_page():
    """文档管理主页面"""
    st.title("📚 文档管理")
    st.markdown("---")

    # 文件上传区域
    st.subheader("📤 上传文档")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "选择要上传的文件",
            type=['pdf', 'txt', 'md', 'docx', 'doc', 'csv', 'json', 'yaml', 'yml'],
            accept_multiple_files=True,
            help="支持格式: PDF, TXT, MD, DOCX, DOC, CSV, JSON, YAML"
        )

    with col2:
        auto_build = st.checkbox(
            "上传后自动构建",
            value=True,
            help="上传完成后自动触发增量构建"
        )

    if uploaded_files:
        if st.button("💾 保存文件", type="primary"):
            files_dir = Path(FILES_DIR)
            files_dir.mkdir(parents=True, exist_ok=True)

            success_count = 0
            error_files = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # 保存文件
                    file_path = files_dir / uploaded_file.name

                    # 检查文件是否已存在
                    if file_path.exists():
                        status_text.warning(f"⚠️ {uploaded_file.name} 已存在，将被覆盖")

                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    success_count += 1
                    status_text.success(f"✅ {uploaded_file.name} 保存成功")

                except Exception as e:
                    error_files.append(f"{uploaded_file.name}: {e}")

                # 更新进度
                progress_bar.progress((i + 1) / len(uploaded_files))

            # 显示结果
            if success_count > 0:
                st.success(f"🎉 成功上传 {success_count} 个文件！")

                # 触发增量构建
                if auto_build:
                    st.info("🔨 正在触发增量构建，请稍候...")
                    with st.spinner("正在触发增量构建..."):
                        success, message = trigger_incremental_build()
                        if success:
                            st.success(f"🎉 {message}")

                            # 显示构建状态链接
                            st.info("💡 你可以在「🏗️ 构建管理」页面查看构建进度")

                            # 提供跳转提示
                            if st.button("📊 前往构建管理页面"):
                                st.session_state.page_switch = "🏗️ 构建管理"
                                st.rerun()
                        else:
                            st.warning(f"⚠️ {message}")
                            st.info("💡 你可以稍后在「🏗️ 构建管理」页面手动触发构建")

            if error_files:
                st.error("❌ 以下文件上传失败:")
                for error in error_files:
                    st.text(f"  • {error}")

            # 清空上传列表
            st.rerun()

    st.markdown("---")

    # 已有文档列表
    st.subheader("📋 已导入文档")

    documents = list_documents()

    if not documents:
        st.info("📭 暂无文档，请上传文件")
    else:
        # 显示统计信息
        total_size = sum(doc['size'] for doc in documents)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("文档数量", len(documents))
        with col2:
            st.metric("总大小", format_file_size(total_size))
        with col3:
            # 统计文件类型
            types = {}
            for doc in documents:
                doc_type = doc['type']
                types[doc_type] = types.get(doc_type, 0) + 1
            st.metric("文件类型", len(types))

        st.markdown("---")

        # 搜索过滤
        search_query = st.text_input("🔍 搜索文件名", "")

        # 过滤文档
        if search_query:
            documents = [doc for doc in documents if search_query.lower() in doc['name'].lower()]

        # 文档表格
        for idx, doc in enumerate(documents):
            col1, col2, col3, col4, col5 = st.columns([4, 1, 2, 2, 1])

            with col1:
                # 文件图标
                icon = {
                    'PDF': '📄',
                    'TXT': '📝',
                    'MD': '📘',
                    'DOCX': '📘',
                    'DOC': '📘',
                    'CSV': '📊',
                    'JSON': '📋',
                    'YAML': '⚙️',
                    'YML': '⚙️',
                }.get(doc['type'], '📎')
                st.text(f"{icon} {doc['name']}")

            with col2:
                st.text(doc['type'])

            with col3:
                st.text(format_file_size(doc['size']))

            with col4:
                st.text(doc['modified'].strftime('%Y-%m-%d %H:%M'))

            with col5:
                delete_key = f"delete_{idx}"
                confirm_key = f"confirm_delete_{idx}"

                # 如果还没有点击删除按钮
                if not st.session_state.get(confirm_key, False):
                    if st.button("🗑️", key=delete_key, help=f"删除 {doc['name']}"):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    # 显示确认按钮
                    col_confirm1, col_confirm2 = st.columns(2)
                    with col_confirm1:
                        if st.button("✅", key=f"yes_{idx}", help="确认删除"):
                            if delete_document(doc['name']):
                                st.success(f"✅ {doc['name']} 已删除")
                                st.session_state[confirm_key] = False
                                st.rerun()
                            else:
                                st.error(f"❌ 删除失败")
                    with col_confirm2:
                        if st.button("❌", key=f"no_{idx}", help="取消"):
                            st.session_state[confirm_key] = False
                            st.rerun()

            # 分隔线
            if idx < len(documents) - 1:
                st.markdown("<hr style='margin: 5px 0; opacity: 0.3;'>", unsafe_allow_html=True)

        # 批量操作
        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🗑️ 清空所有文档", type="secondary"):
                st.session_state.confirm_delete_all = True

        # 确认删除对话框
        if st.session_state.get('confirm_delete_all', False):
            st.warning("⚠️ 确定要删除所有文档吗？此操作不可恢复！")
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("✅ 确认删除"):
                    for doc in documents:
                        delete_document(doc['name'])
                    st.success("所有文档已删除")
                    st.session_state.confirm_delete_all = False
                    st.rerun()
            with col2:
                if st.button("❌ 取消"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
