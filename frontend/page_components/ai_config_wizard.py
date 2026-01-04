"""
AI 配置向导页面
提供智能化的配置推荐和优化功能
"""

import streamlit as st
import requests
from typing import Dict, Optional
from frontend.frontend_config.settings import API_URL


def analyze_documents_with_ai(industry_hint: Optional[str] = None) -> Optional[Dict]:
    """
    使用 AI 分析文档并生成推荐

    Args:
        industry_hint: 行业提示

    Returns:
        分析结果和推荐
    """
    try:
        params = {}
        if industry_hint:
            params["industry_hint"] = industry_hint

        response = requests.post(
            f"{API_URL}/admin/ai-copilot/analyze-documents",
            params=params,
            timeout=300  # 5分钟超时
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        st.error("⏱️ 请求超时，文档分析可能需要较长时间，请稍后重试")
        return None
    except Exception as e:
        st.error(f"分析失败: {e}")
        return None


def apply_ai_recommendations(recommendations: Dict, project_name: str,
                            industry: Optional[str] = None,
                            description: Optional[str] = None) -> bool:
    """
    应用 AI 推荐创建配置

    Args:
        recommendations: AI 推荐结果
        project_name: 项目名称
        industry: 行业
        description: 描述

    Returns:
        是否成功
    """
    try:
        payload = {
            "recommendations": recommendations,
            "project_name": project_name,
        }
        if industry:
            payload["industry"] = industry
        if description:
            payload["description"] = description

        response = requests.post(
            f"{API_URL}/admin/ai-copilot/apply-recommendations",
            json=payload
        )
        response.raise_for_status()
        return True

    except Exception as e:
        st.error(f"应用推荐失败: {e}")
        return False


def refine_with_ai(user_feedback: str, current_config: Optional[Dict] = None) -> Optional[Dict]:
    """
    使用 AI 优化配置

    Args:
        user_feedback: 用户反馈
        current_config: 当前配置

    Returns:
        优化后的配置
    """
    try:
        payload = {
            "user_feedback": user_feedback
        }
        if current_config:
            payload["current_config"] = current_config

        response = requests.post(
            f"{API_URL}/admin/ai-copilot/refine-config",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"优化失败: {e}")
        return None


def render_analysis_results(analysis: Dict):
    """渲染文档分析结果"""
    st.subheader("📊 文档分析结果")

    # 统计信息
    stats = analysis.get("statistics", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("文档数量", stats.get("total_documents", 0))

    with col2:
        st.metric("总字符数", f"{stats.get('total_characters', 0):,}")

    with col3:
        st.metric("平均长度", stats.get("avg_doc_length", 0))

    with col4:
        st.metric("聚类数量", stats.get("num_clusters", 0))

    # 文档聚类
    st.markdown("---")
    st.markdown("### 📁 文档聚类")
    st.caption("根据内容相似度自动分组")

    clusters = analysis.get("clusters", [])
    for cluster in clusters:
        with st.expander(f"聚类 {cluster['cluster_id'] + 1} - {cluster['document_count']} 个文档"):
            st.markdown(f"**代表性词汇**：")
            terms = cluster.get("representative_terms", [])[:10]
            if terms:
                st.write(", ".join(terms))
            else:
                st.caption("（无代表性词汇）")

            st.markdown(f"**包含文档**：")
            docs = cluster.get("documents", [])
            for doc in docs[:10]:
                st.caption(f"- {doc}")
            if len(docs) > 10:
                st.caption(f"... 还有 {len(docs) - 10} 个文档")

    # 关键概念
    st.markdown("---")
    st.markdown("### 💡 关键概念")
    st.caption("从所有文档中提取的高频关键词")

    key_concepts = analysis.get("key_concepts", [])
    if key_concepts:
        # 显示为标签云样式
        concept_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>"
        for concept in key_concepts[:20]:
            score = concept.get("score", 0)
            # 根据分数计算字体大小
            font_size = 12 + int(score * 100)
            font_size = min(font_size, 24)  # 最大24px
            concept_html += f"<span style='background-color: #e8f4f8; padding: 4px 12px; border-radius: 16px; font-size: {font_size}px;'>{concept['concept']}</span>"
        concept_html += "</div>"
        st.markdown(concept_html, unsafe_allow_html=True)
    else:
        st.caption("（未提取到关键概念）")


def render_recommendations(recommendations: Dict):
    """渲染 AI 推荐结果"""
    st.subheader("🤖 AI 配置推荐")

    # 整体推荐理由
    reasoning = recommendations.get("reasoning", "")
    if reasoning:
        st.info(f"**推荐理由**：{reasoning}")

    # 桥接点推荐
    st.markdown("---")
    st.markdown("### 🔗 推荐的桥接点")

    bridges = recommendations.get("recommended_bridges", [])
    if bridges:
        for idx, bridge in enumerate(bridges):
            with st.expander(f"🔗 {bridge['name']}", expanded=False):
                st.markdown(f"**Key**: `{bridge['key']}`")
                st.markdown(f"**描述**: {bridge['description']}")

                if bridge.get("examples"):
                    st.markdown(f"**示例**: {', '.join(bridge['examples'])}")

                if bridge.get("is_enum_restricted"):
                    st.warning("⚠️ 建议限制为枚举值")

                if bridge.get("reasoning"):
                    st.caption(f"💡 {bridge['reasoning']}")
    else:
        st.caption("（AI 未推荐桥接点）")

    # 领域推荐
    st.markdown("---")
    st.markdown("### 📦 推荐的领域")

    domains = recommendations.get("recommended_domains", [])
    if domains:
        for idx, domain in enumerate(domains):
            with st.expander(f"📦 {domain['domain_name']}", expanded=False):
                st.markdown(f"**描述**: {domain['description']}")
                st.markdown(f"**触发条件**: {domain['trigger_condition']}")

                schema = domain.get("schema", {})
                st.markdown(f"**实体类型**: {', '.join(schema.get('entities', []))}")
                st.markdown(f"**关系类型**: {', '.join(schema.get('relations', []))}")

                mappings = domain.get("bridge_mappings", [])
                if mappings:
                    st.markdown("**桥接点映射**:")
                    for mapping in mappings:
                        st.caption(f"  - {mapping['bridge_key']} → {mapping['role']}")

                if domain.get("reasoning"):
                    st.caption(f"💡 {domain['reasoning']}")
    else:
        st.caption("（AI 未推荐领域）")


def ai_config_wizard_page():
    """AI 配置向导主页面"""
    st.title("🤖 AI 配置向导")
    st.markdown("---")

    st.info("💡 **AI 向导可以帮助你**：分析已上传的文档，自动推荐合适的领域和桥接点配置")

    # 步骤指示器
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1

    step = st.session_state.wizard_step

    # 显示步骤进度
    progress_cols = st.columns(4)
    steps = [
        ("1️⃣", "上传文档"),
        ("2️⃣", "AI 分析"),
        ("3️⃣", "查看推荐"),
        ("4️⃣", "应用配置")
    ]

    for i, (icon, label) in enumerate(steps):
        with progress_cols[i]:
            if i + 1 < step:
                st.success(f"{icon} {label} ✓")
            elif i + 1 == step:
                st.info(f"**{icon} {label}**")
            else:
                st.caption(f"{icon} {label}")

    st.markdown("---")

    # 步骤 1: 上传文档
    if step == 1:
        st.subheader("📚 步骤 1: 上传文档")
        st.write("请先在「📚 文档管理」页面上传你的文档，然后返回此页面")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 检查文档", type="primary"):
                # 检查是否有文档
                try:
                    response = requests.get(f"{API_URL}/admin/files/list")
                    response.raise_for_status()
                    data = response.json()
                    files = data.get("files", [])

                    if files:
                        st.success(f"✅ 检测到 {len(files)} 个文档")
                        st.session_state.wizard_step = 2
                        st.rerun()
                    else:
                        st.warning("⚠️ 未检测到文档，请先上传")
                except Exception as e:
                    st.error(f"检查失败: {e}")

        with col2:
            if st.button("📚 前往文档管理"):
                st.session_state.page_switch = "📚 文档管理"
                st.rerun()

    # 步骤 2: AI 分析
    elif step == 2:
        st.subheader("🔍 步骤 2: AI 文档分析")

        industry_hint = st.text_input(
            "行业提示（可选）",
            placeholder="例如：法务、电商、医疗、教育等",
            help="提供行业提示可以帮助 AI 生成更准确的推荐"
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                with st.spinner("🤖 AI 正在分析文档，这可能需要几分钟..."):
                    result = analyze_documents_with_ai(industry_hint if industry_hint else None)

                    if result:
                        st.session_state.ai_analysis_result = result
                        st.session_state.wizard_step = 3
                        st.success("✅ 分析完成！")
                        st.rerun()

        with col2:
            if st.button("⬅️ 返回", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()

    # 步骤 3: 查看推荐
    elif step == 3:
        st.subheader("📋 步骤 3: 查看 AI 推荐")

        result = st.session_state.get("ai_analysis_result")

        if result:
            # 创建标签页
            tab1, tab2 = st.tabs(["📊 文档分析", "🤖 AI 推荐"])

            with tab1:
                render_analysis_results(result.get("analysis", {}))

            with tab2:
                render_recommendations(result.get("recommendations", {}))

            st.markdown("---")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("✅ 应用推荐", type="primary", use_container_width=True):
                    st.session_state.wizard_step = 4
                    st.rerun()

            with col2:
                if st.button("🔄 重新分析", use_container_width=True):
                    del st.session_state.ai_analysis_result
                    st.session_state.wizard_step = 2
                    st.rerun()

        else:
            st.error("未找到分析结果，请重新分析")
            if st.button("🔄 重新分析"):
                st.session_state.wizard_step = 2
                st.rerun()

    # 步骤 4: 应用配置
    elif step == 4:
        st.subheader("💾 步骤 4: 应用配置")

        result = st.session_state.get("ai_analysis_result")

        if result:
            recommendations = result.get("recommendations", {})

            st.write("请为你的配置命名：")

            project_name = st.text_input(
                "项目名称 *",
                placeholder="例如：我的知识图谱项目",
                key="wizard_project_name"
            )

            col1, col2 = st.columns(2)

            with col1:
                industry = st.text_input(
                    "所属行业",
                    placeholder="例如：法务、电商、医疗",
                    key="wizard_industry"
                )

            with col2:
                description = st.text_area(
                    "项目描述",
                    placeholder="简要描述此知识图谱的用途",
                    key="wizard_description",
                    height=100
                )

            st.markdown("---")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("💾 创建配置", type="primary", use_container_width=True):
                    if not project_name:
                        st.error("请输入项目名称")
                    else:
                        with st.spinner("正在创建配置..."):
                            success = apply_ai_recommendations(
                                recommendations,
                                project_name,
                                industry,
                                description
                            )

                            if success:
                                st.success("🎉 配置创建成功！")
                                st.info("💡 你可以在「⚙️ 配置管理」页面查看和编辑配置")

                                # 清理会话状态
                                if "ai_analysis_result" in st.session_state:
                                    del st.session_state.ai_analysis_result
                                st.session_state.wizard_step = 1

                                # 提供导航按钮
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button("⚙️ 前往配置管理"):
                                        st.session_state.page_switch = "⚙️ 配置管理"
                                        st.rerun()
                                with col_b:
                                    if st.button("🏗️ 开始构建图谱"):
                                        st.session_state.page_switch = "🏗️ 构建管理"
                                        st.rerun()

            with col2:
                if st.button("⬅️ 返回", use_container_width=True):
                    st.session_state.wizard_step = 3
                    st.rerun()

        else:
            st.error("未找到推荐结果")
            if st.button("🔄 重新开始"):
                st.session_state.wizard_step = 1
                st.rerun()

    # 帮助信息
    with st.expander("❓ 使用说明"):
        st.markdown("""
        ### AI 配置向导工作流程

        1. **上传文档**
           - 在文档管理页面上传你的业务文档
           - 文档数量越多，AI 分析越准确

        2. **AI 分析**
           - AI 会自动分析文档内容和结构
           - 识别文档聚类和关键概念
           - 提供行业提示可以提高准确性

        3. **查看推荐**
           - AI 会推荐合适的桥接点和领域
           - 每个推荐都附带详细的理由
           - 你可以重新分析以获得不同的推荐

        4. **应用配置**
           - 为配置命名并添加描述
           - 一键应用 AI 推荐创建配置
           - 后续可以在配置管理页面进行调整

        ### 注意事项

        - AI 分析可能需要几分钟时间
        - 推荐结果仅供参考，你可以随后手动调整
        - 如果对推荐不满意，可以选择「切换模板」或手动创建配置
        """)
