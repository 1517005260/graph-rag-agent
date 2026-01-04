"""
构建管理页面
提供知识图谱构建、增量更新等功能
"""

import streamlit as st
import requests
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from frontend_config.settings import API_URL


def get_build_status() -> Optional[Dict]:
    """获取构建状态"""
    try:
        response = requests.get(f"{API_URL}/admin/build/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API 返回错误: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到后端服务，请确保后端已启动"}
    except requests.exceptions.Timeout:
        return {"error": "请求超时"}
    except Exception as e:
        return {"error": f"未知错误: {str(e)}"}


def trigger_full_build() -> tuple[bool, str]:
    """触发完整构建"""
    try:
        response = requests.post(
            f"{API_URL}/admin/build/full",
            timeout=10
        )
        if response.status_code == 200:
            return True, "完整构建已启动"
        else:
            return False, f"启动失败: {response.text}"
    except requests.exceptions.Timeout:
        return True, "完整构建已在后台启动（可能需要较长时间）"
    except Exception as e:
        return False, f"请求失败: {e}"


def trigger_incremental_build() -> tuple[bool, str]:
    """触发增量构建"""
    try:
        response = requests.post(
            f"{API_URL}/admin/build/incremental",
            timeout=10
        )
        if response.status_code == 200:
            return True, "增量构建已启动"
        else:
            return False, f"启动失败: {response.text}"
    except requests.exceptions.Timeout:
        return True, "增量构建已在后台启动"
    except Exception as e:
        return False, f"请求失败: {e}"


def stop_build() -> tuple[bool, str]:
    """停止构建"""
    try:
        response = requests.post(f"{API_URL}/admin/build/stop", timeout=5)
        if response.status_code == 200:
            return True, "构建已停止"
        else:
            return False, f"停止失败: {response.text}"
    except Exception as e:
        return False, f"请求失败: {e}"


def get_graph_stats() -> Optional[Dict]:
    """获取图谱统计信息"""
    try:
        response = requests.get(f"{API_URL}/admin/graph/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def build_manager_page():
    """构建管理主页面"""
    st.title("🏗️ 知识图谱构建管理")
    st.markdown("---")

    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🚀 构建操作", "📊 构建状态", "📈 图谱统计"])

    # ===== 构建操作 =====
    with tab1:
        st.subheader("🔨 构建类型")

        # 显示当前构建状态概览
        current_status = get_build_status()
        if current_status and current_status.get('status') == 'running':
            st.info(f"🔄 构建进行中：{current_status.get('current_stage', '...')} - "
                   f"进度 {current_status.get('progress', 0)}%")
            st.caption("💡 切换到「📊 构建状态」标签查看详细进度")
            st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔄 增量构建")
            st.markdown("""
            **适用场景:**
            - 添加了新文档
            - 修改了现有文档
            - 需要更新部分内容

            **特点:**
            - ✅ 速度快
            - ✅ 只处理变化
            - ✅ 保留现有数据

            **耗时:** 几分钟到十几分钟
            """)

            if st.button("🔄 开始增量构建", type="primary", use_container_width=True):
                with st.spinner("正在启动增量构建..."):
                    success, message = trigger_incremental_build()
                    if success:
                        st.success(f"✅ {message}")
                        st.info("💡 请切换到「📊 构建状态」标签查看进度")
                        st.session_state.build_just_started = True
                        st.balloons()
                        time.sleep(2)
                    else:
                        st.error(f"❌ {message}")

        with col2:
            st.markdown("### 🔃 完整构建")
            st.markdown("""
            **适用场景:**
            - 首次使用系统
            - 更改了实体/关系配置
            - 需要重建整个图谱

            **特点:**
            - ⚠️ 耗时较长
            - ⚠️ 清空现有数据
            - ✅ 完全重建

            **耗时:** 几十分钟到几小时
            """)

            if st.button("🔃 开始完整构建", use_container_width=True):
                st.session_state.confirm_full_build = True

        # 完整构建确认对话框
        if st.session_state.get('confirm_full_build', False):
            st.warning("⚠️ 完整构建将清空现有知识图谱！确定要继续吗？")
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("✅ 确认构建", type="primary"):
                    with st.spinner("正在启动完整构建..."):
                        success, message = trigger_full_build()
                        if success:
                            st.success(f"✅ {message}")
                            st.info("💡 请切换到「📊 构建状态」标签查看进度")
                            # 设置标记，用于在构建状态tab显示提示
                            st.session_state.build_just_started = True
                            st.balloons()
                            time.sleep(2)  # 给用户2秒时间看到消息
                        else:
                            st.error(f"❌ {message}")
                    st.session_state.confirm_full_build = False
                    st.rerun()
            with col2:
                if st.button("❌ 取消"):
                    st.session_state.confirm_full_build = False
                    st.rerun()

        st.markdown("---")

        # 高级选项
        with st.expander("⚙️ 高级选项"):
            st.markdown("### 构建参数配置")

            col1, col2 = st.columns(2)

            with col1:
                max_workers = st.number_input(
                    "并行线程数",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="控制实体提取的并行度，建议设为 CPU 核心数"
                )

                batch_size = st.number_input(
                    "批处理大小",
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="每批处理的文本块数量，较大值可能提高速度但占用更多内存"
                )

            with col2:
                enable_cache = st.checkbox(
                    "启用缓存",
                    value=True,
                    help="缓存LLM响应以避免重复计算"
                )

                auto_align = st.checkbox(
                    "自动实体对齐",
                    value=True,
                    help="自动识别和合并重复实体"
                )

            st.info("💡 修改这些参数需要重新启动构建")

        st.markdown("---")

        # 操作按钮
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("⏸️ 暂停构建", use_container_width=True):
                success, message = stop_build()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")

        with col2:
            if st.button("🔄 刷新状态", use_container_width=True):
                st.rerun()

    # ===== 构建状态 =====
    with tab2:
        st.subheader("📊 实时构建状态")

        # 检查是否刚刚启动构建
        if st.session_state.get('build_just_started', False):
            st.success("✅ 构建已成功启动！正在后台运行...")
            st.info("💡 构建过程可能需要几分钟到几十分钟，请耐心等待")
            st.session_state.build_just_started = False

        # 自动刷新选项
        col1, col2 = st.columns([3, 1])
        with col1:
            auto_refresh = st.checkbox("🔄 自动刷新 (每5秒)", value=False,
                                      help="启用后将自动刷新构建状态，构建期间建议开启")
        with col2:
            if st.button("🔄 立即刷新"):
                st.rerun()

        if auto_refresh:
            # 获取当前状态，如果不是running就停止自动刷新
            current_status = get_build_status()
            if current_status and current_status.get('status') == 'running':
                # 添加自动刷新逻辑
                placeholder = st.empty()
                for i in range(5, 0, -1):
                    placeholder.info(f"⏱️ 构建进行中... {i} 秒后自动刷新")
                    time.sleep(1)
                placeholder.empty()
                st.rerun()
            else:
                st.info("💡 构建已完成或未运行，已停止自动刷新")

        # 获取构建状态
        status = get_build_status()

        if not status:
            st.error("❌ 无法获取构建状态")
            st.info("💡 请确保后端服务正在运行")
            st.code(f"后端地址: {API_URL}")
            return

        if 'error' in status:
            st.error(f"❌ 获取构建状态失败: {status['error']}")
            st.info("💡 请检查后端服务是否正常运行")

            # 提供诊断命令
            with st.expander("🔍 诊断建议"):
                st.markdown(f"""
                请在终端执行以下命令检查后端状态：

                ```bash
                # 测试后端健康检查
                curl {API_URL}/admin/health

                # 测试构建状态API
                curl {API_URL}/admin/build/status

                # 检查后端是否运行
                lsof -i :8000
                ```
                """)
            return

        # 状态正常，显示详细信息
        if status:
            # 检测构建完成（状态从 running 变为 completed）
            previous_status = st.session_state.get('previous_build_status', 'unknown')
            current_status = status.get('status', 'unknown')

            # 保存当前状态供下次比较
            st.session_state.previous_build_status = current_status

            # 如果刚刚完成构建，显示醒目提示
            if previous_status == 'running' and current_status == 'completed':
                st.balloons()
                st.success("🎉 构建已完成！知识图谱构建成功！")
                st.info("💡 现在可以在「💬 智能问答」中开始提问了")
                # 播放提示音（使用浏览器API）
                st.markdown("""
                <script>
                // 播放系统提示音
                if (window.speechSynthesis) {
                    const utterance = new SpeechSynthesisUtterance("构建已完成");
                    utterance.lang = 'zh-CN';
                    utterance.rate = 1.5;
                    window.speechSynthesis.speak(utterance);
                }
                </script>
                """, unsafe_allow_html=True)

            # 显示状态指示器
            status_text = status.get('status', 'unknown')
            if status_text == 'running':
                st.success("🟢 构建进行中")
                st.caption("💡 提示：可以勾选「自动刷新」来实时监控进度")
            elif status_text == 'completed':
                st.success("✅ 构建已完成")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("💡 知识图谱已成功构建，可以开始查询了")
                with col2:
                    if st.button("💬 去提问", type="primary"):
                        st.session_state.page_switch = "💬 智能问答"
                        st.rerun()
            elif status_text == 'idle':
                st.info("⚪ 空闲状态")
            elif status_text == 'failed':
                st.error("🔴 构建失败")
                error_msg = status.get('error', '未知错误')
                with st.expander("📋 查看错误详情"):
                    st.code(error_msg, language='text')

            # 进度条
            progress = status.get('progress', 0)
            st.progress(progress / 100 if progress else 0)
            st.caption(f"总体进度: {progress}%")

            # 详细信息
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("当前阶段", status.get('current_stage', 'N/A'))
            with col2:
                st.metric("已处理", f"{status.get('processed', 0)}/{status.get('total', 0)}")
            with col3:
                elapsed = status.get('elapsed_time', 0)
                st.metric("已用时间", f"{elapsed // 60}分{elapsed % 60}秒")

            # 日志输出
            if 'logs' in status:
                with st.expander("📜 构建日志", expanded=True):
                    st.code("\n".join(status['logs'][-50:]), language='text')

        elif status and 'error' in status:
            st.error(f"❌ 无法获取构建状态: {status['error']}")
            st.info("💡 请确保后端服务正在运行")
        else:
            st.warning("⚠️ 无法连接到构建服务")
            st.info("💡 构建功能需要后端 API 支持，目前此功能处于开发中")

            # 显示模拟状态
            st.markdown("---")
            st.markdown("### 📋 构建流程说明")
            st.markdown("""
            完整构建流程包括以下阶段:

            1. **文档读取与分块** (10%)
               - 读取 files/ 目录中的文档
               - 使用 HanLP 或简单分词器进行文本分块

            2. **实体关系提取** (40%)
               - 使用 LLM 从文本中提取实体和关系
               - 这是最耗时的阶段

            3. **实体消歧与对齐** (20%)
               - 识别和合并重复实体
               - 解决命名冲突

            4. **图谱构建** (15%)
               - 将实体和关系写入 Neo4j 数据库
               - 建立图结构

            5. **向量索引构建** (10%)
               - 为实体和文本块生成嵌入向量
               - 建立向量索引

            6. **社区检测** (5%)
               - 使用 Leiden 或 SLLPA 算法检测社区
               - 生成社区摘要
            """)

    # ===== 图谱统计 =====
    with tab3:
        st.subheader("📈 知识图谱统计")

        # 获取统计信息
        stats = get_graph_stats()

        if not stats:
            st.warning("⚠️ 无法获取图谱统计信息")
            st.info("💡 可能的原因：")
            st.markdown("""
            1. 后端服务未运行
            2. Neo4j 数据库未连接
            3. 还未构建知识图谱
            """)

            # 提供测试按钮
            if st.button("🔍 测试连接"):
                with st.spinner("测试中..."):
                    try:
                        response = requests.get(f"{API_URL}/admin/health", timeout=5)
                        if response.status_code == 200:
                            health = response.json()
                            if health.get("neo4j") == "healthy":
                                st.success("✅ Neo4j 连接正常")
                                st.info("💡 可能还未构建知识图谱，请先上传文档并构建")
                            else:
                                st.error("❌ Neo4j 连接失败")
                        else:
                            st.error("❌ 后端服务异常")
                    except Exception as e:
                        st.error(f"❌ 连接测试失败: {e}")
            return

        if 'error' in stats:
            st.error(f"❌ {stats['error']}")
            return

        # 检查是否有数据
        if stats.get('entity_count', 0) == 0:
            st.info("📭 知识图谱为空")
            st.markdown("""
            **请先构建知识图谱：**
            1. 前往「📚 文档管理」上传文档
            2. 在「🏗️ 构建管理」中触发构建
            3. 等待构建完成后再查看统计信息
            """)
            return

        # 总览
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "实体数量",
                stats.get('entity_count', 0),
                help="知识图谱中的实体总数"
            )

        with col2:
            st.metric(
                "关系数量",
                stats.get('relationship_count', 0),
                help="实体间的关系总数"
            )

        with col3:
            st.metric(
                "社区数量",
                stats.get('community_count', 0),
                help="检测到的社区数量"
            )

        with col4:
            st.metric(
                "文档数量",
                stats.get('document_count', 0),
                help="已导入的文档数量"
            )

        st.markdown("---")

        # 详细统计
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 实体类型分布")
            if 'entity_type_distribution' in stats:
                import pandas as pd
                df = pd.DataFrame(
                    list(stats['entity_type_distribution'].items()),
                    columns=['类型', '数量']
                )
                st.bar_chart(df.set_index('类型'))
            else:
                st.info("暂无数据")

        with col2:
            st.markdown("### 🔗 关系类型分布")
            if 'relationship_type_distribution' in stats:
                import pandas as pd
                df = pd.DataFrame(
                    list(stats['relationship_type_distribution'].items()),
                    columns=['类型', '数量']
                )
                st.bar_chart(df.set_index('类型'))
            else:
                st.info("暂无数据")

        st.markdown("---")

        # 最近更新
        st.markdown("### 🕒 最近更新")
        if 'last_build_time' in stats:
            last_build = datetime.fromisoformat(stats['last_build_time'])
            st.info(f"📅 最后构建时间: {last_build.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("暂无构建记录")

    # 帮助信息
    with st.expander("❓ 使用说明"):
        st.markdown("""
        ### 构建流程建议

        1. **首次使用**
           - 上传文档到「文档管理」
           - 配置实体和关系类型（如有需要）
           - 执行「完整构建」

        2. **日常更新**
           - 添加新文档后，使用「增量构建」
           - 无需清空现有数据

        3. **配置变更后**
           - 修改实体/关系类型后
           - 必须执行「完整构建」

        ### 性能优化建议

        - **并行线程数**: 设为 CPU 核心数的 50-100%
        - **批处理大小**: API 速度快时可增大 (10-20)
        - **启用缓存**: 避免重复计算，强烈建议开启

        ### 故障排查

        - 构建失败: 检查 API 配置和网络连接
        - 进度卡住: 可能是 API 速率限制，请等待
        - 内存不足: 减小批处理大小

        ### 注意事项

        - 完整构建会清空所有现有数据
        - 构建过程中不要关闭程序
        - 建议在低峰期进行大规模构建
        """)
