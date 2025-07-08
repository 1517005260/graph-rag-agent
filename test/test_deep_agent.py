import asyncio
import json
import pytest

pytest.importorskip("langchain_core")
from agent.deep_research_agent import DeepResearchAgent

# DeepResearchAgent 综合测试
async def test_deep_research_agent_advanced():
    print("\n======= DeepResearchAgent 高级功能测试 =======")
    agent = DeepResearchAgent(use_deeper_tool=True)
    
    # 1. 基本流式输出测试 (不显示思考过程)
    print("\n--- 基本流式输出测试 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请", show_thinking=False):
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            print(chunk, end="", flush=True)
    
    # 2. 显示思考过程的流式输出
    print("\n\n--- 思考过程可见模式 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请", show_thinking=True):
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            print(chunk, end="", flush=True)
            
    # 3. 知识图谱探索功能测试
    print("\n\n--- 知识图谱探索功能 ---")
    exploration_result = agent.explore_knowledge("华东理工大学奖学金体系")
    print("\n知识图谱探索结果摘要:")
    if isinstance(exploration_result, dict):
        if "error" in exploration_result:
            print(f"错误: {exploration_result['error']}")
        else:
            print(f"探索路径: {len(exploration_result.get('exploration_path', []))} 步")
            print(f"发现实体: {exploration_result.get('discovered_entities', [])[:3]}")
            print(f"摘要: {exploration_result.get('summary', '')[:200]}...")
    else:
        print(exploration_result)
    
    # 4. 执行深度思考后获取推理链分析
    print("\n\n--- 推理链分析功能 ---")
    # 先执行一次深度思考
    thinking_result = agent.ask_with_thinking("各类奖学金申请条件和评审流程")
    print(f"思考过程包含 {len(thinking_result.get('execution_logs', []))} 条执行日志")
    
    # 然后分析推理链
    analysis = agent.analyze_reasoning_chain()
    print("\n推理链分析:")
    if isinstance(analysis, dict):
        if "error" in analysis:
            print(f"错误: {analysis['error']}")
        else:
            print(f"推理步骤: {len(analysis.get('reasoning_chain', {}).get('steps', []))} 步")
            contradictions = analysis.get('contradictions', [])
            print(f"发现 {len(contradictions)} 个信息矛盾")
            evidence_stats = analysis.get('evidence_stats', {})
            print(f"证据统计: {json.dumps(evidence_stats, indent=2)}")
    else:
        print(analysis)
    
    # 5. 矛盾检测功能
    print("\n\n--- 矛盾检测功能 ---")
    complex_query = "不同类型奖学金之间的冲突和兼容情况分析"
    contradictions = agent.detect_contradictions(complex_query)
    print("\n矛盾分析:")
    if isinstance(contradictions, dict):
        if contradictions.get("has_contradictions", False):
            print(f"发现 {contradictions.get('count', 0)} 个矛盾:")
            for i, c in enumerate(contradictions.get('contradictions', [])[:2]):
                print(f"  {i+1}. 类型: {c.get('type', '未知')}, 内容: {c.get('analysis', '')[:100]}...")
            print(f"\n影响分析: {contradictions.get('impact_analysis', '')[:200]}...")
        else:
            print("未发现明显矛盾，信息来源一致性良好")
    else:
        print(contradictions)
    
    # 6. 社区感知增强测试
    print("\n\n--- 社区感知增强测试 ---")
    community_result = agent.ask_with_thinking("奖学金申请常见问题", community_aware=True)
    community_info = community_result.get("community_context", {})
    print(f"发现 {len(community_info.get('summaries', []))} 个相关知识社区")
    
    # 7. 不同配置参数测试
    print("\n\n--- 不同配置参数测试 ---")
    # 修改最大迭代次数
    agent.research_tool.max_iterations = 2  # 减少迭代以加快测试
    print(f"设置最大迭代次数为: {agent.research_tool.max_iterations}")
    # 执行流式查询
    async for chunk in agent.ask_stream("奖学金申请流程简介", show_thinking=False):
        if isinstance(chunk, dict):
            print("\n[配置参数测试最终答案]")
            if "answer" in chunk:
                print(chunk["answer"][:300] + "...")
        else:
            # 为了节省输出，只显示进度指示
            if chunk.startswith("**"):
                print(chunk)
    
    # 8. 标准工具与增强工具比较
    print("\n\n--- 工具比较测试 ---")
    # 切换到标准模式
    agent.is_deeper_tool(False)
    print("标准研究工具:")
    standard_result = agent.ask("如何提高奖学金申请成功率", thread_id="standard_test")
    print(f"标准工具结果: {standard_result[:150]}...")
    
    # 切换回增强模式
    agent.is_deeper_tool(True)
    print("\n增强研究工具:")
    enhanced_result = agent.ask("如何提高奖学金申请成功率", thread_id="enhanced_test")
    print(f"增强工具结果: {enhanced_result[:150]}...")

async def run_advanced_tests():
    await test_deep_research_agent_advanced()

if __name__ == "__main__":
    asyncio.run(run_advanced_tests())