"""
集成测试：验证代码优化的效果

测试内容：
1. Orchestrator 快速通道性能
2. Agent 多态接口正确性
3. AgentManager 内存管理
4. ResearchExecutor 引用提取稳定性
"""
import time
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_fast_path_latency():
    """
    测试 1: Orchestrator 快速通道性能

    验证：简单查询走快速通道，耗时显著减少（跳过 Planner 的 3 次 LLM 调用）
    """
    print("\n" + "="*60)
    print("测试 1: Orchestrator 快速通道性能")
    print("="*60)

    try:
        from graphrag_agent.agents.multi_agent.orchestrator import (
            MultiAgentOrchestrator,
            OrchestratorConfig,
        )
        from graphrag_agent.agents.multi_agent.core.state import PlanExecuteState
        from graphrag_agent.agents.multi_agent.planner.core import PlannerCoordinator
        from graphrag_agent.agents.multi_agent.executor.worker_coordinator import WorkerCoordinator
        from graphrag_agent.agents.multi_agent.reporter.map_reduce_reporter import MapReduceReporter

        # 创建 Orchestrator（启用快速通道）
        config = OrchestratorConfig(
            enable_fast_path=True,
            fast_path_max_length=30,
        )

        # 注意：这里需要真实的组件，或者使用 Mock
        # 为了演示，这里假设你已经有了这些组件
        planner = PlannerCoordinator()
        worker = WorkerCoordinator()
        reporter = MapReduceReporter()

        orchestrator = MultiAgentOrchestrator(
            planner=planner,
            worker_coordinator=worker,
            reporter=reporter,
            config=config,
        )

        # 测试简单查询（应走快速通道）
        simple_queries = [
            "李白是谁",
            "北京的天气",
            "什么是Python",
        ]

        fast_path_times = []
        for query in simple_queries:
            state = PlanExecuteState(input=query)

            start_time = time.time()
            result = orchestrator.run(state)
            duration = time.time() - start_time

            fast_path_times.append(duration)

            print(f"✓ 简单查询: '{query}'")
            print(f"  耗时: {duration:.2f}s")
            print(f"  是否跳过 Planner: {result.planner is None}")

            # 断言：快速通道应该跳过 Planner
            assert result.planner is None, f"简单查询应该走快速通道，但检测到 Planner: {query}"
            # 断言：应该有执行记录
            assert len(result.execution_records) > 0, f"快速通道应该有执行记录: {query}"

        avg_fast_time = sum(fast_path_times) / len(fast_path_times)
        print(f"\n📊 简单查询平均耗时: {avg_fast_time:.2f}s")

        # 测试复杂查询（应走完整流程）
        complex_queries = [
            "分析苹果公司2024年财报并对比微软",
            "详细解释量子计算的工作原理",
        ]

        complex_path_times = []
        for query in complex_queries:
            state = PlanExecuteState(input=query)

            start_time = time.time()
            result = orchestrator.run(state)
            duration = time.time() - start_time

            complex_path_times.append(duration)

            print(f"\n✓ 复杂查询: '{query}'")
            print(f"  耗时: {duration:.2f}s")
            print(f"  是否使用 Planner: {result.planner is not None}")

            # 断言：复杂查询应该走完整流程
            assert result.planner is not None, f"复杂查询应该使用 Planner: {query}"

        avg_complex_time = sum(complex_path_times) / len(complex_path_times)
        print(f"\n📊 复杂查询平均耗时: {avg_complex_time:.2f}s")

        # 性能对比
        speedup = ((avg_complex_time - avg_fast_time) / avg_complex_time) * 100
        print(f"\n🚀 快速通道性能提升: {speedup:.1f}%")

        print("\n✅ 测试 1 通过：快速通道工作正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_polymorphism():
    """
    测试 2: Agent 多态接口

    验证：所有 Agent 都支持统一接口，无需类型判断
    """
    print("\n" + "="*60)
    print("测试 2: Agent 多态接口")
    print("="*60)

    try:
        from graphrag_agent.agents.graph_agent import GraphAgent
        from graphrag_agent.agents.hybrid_agent import HybridAgent
        from graphrag_agent.agents.deep_research_agent import DeepResearchAgent

        agents = {
            "GraphAgent": GraphAgent(),
            "HybridAgent": HybridAgent(),
            "DeepResearchAgent": DeepResearchAgent(),
        }

        # 测试 1: 所有 Agent 都支持 configure()
        print("\n检查 configure() 方法:")
        for name, agent in agents.items():
            assert hasattr(agent, 'configure'), f"{name} 缺少 configure() 方法"
            # 调用 configure() 不应报错
            agent.configure({"use_deeper_tool": True, "show_thinking": False})
            print(f"  ✓ {name}.configure() 正常")

        # 测试 2: 所有 Agent 都支持 ask_with_thinking()
        print("\n检查 ask_with_thinking() 方法:")
        for name, agent in agents.items():
            assert hasattr(agent, 'ask_with_thinking'), f"{name} 缺少 ask_with_thinking() 方法"
            print(f"  ✓ {name}.ask_with_thinking() 存在")

        # 测试 3: 所有 Agent 都支持 supports_kg_extraction()
        print("\n检查 supports_kg_extraction() 方法:")
        for name, agent in agents.items():
            assert hasattr(agent, 'supports_kg_extraction'), f"{name} 缺少 supports_kg_extraction() 方法"
            supports_kg = agent.supports_kg_extraction()
            print(f"  ✓ {name}.supports_kg_extraction() = {supports_kg}")

            # DeepResearchAgent 应该返回 False
            if name == "DeepResearchAgent":
                assert supports_kg is False, "DeepResearchAgent 应该禁用 KG 提取"
            else:
                assert supports_kg is True, f"{name} 应该支持 KG 提取"

        # 测试 4: configure() 对 DeepResearchAgent 生效
        print("\n检查 DeepResearchAgent 配置:")
        deep_agent = agents["DeepResearchAgent"]

        # 配置 show_thinking
        deep_agent.configure({"show_thinking": True})
        assert deep_agent.show_thinking is True, "show_thinking 配置未生效"
        print(f"  ✓ show_thinking 配置生效")

        # 配置 use_deeper_tool
        initial_tool = deep_agent.use_deeper_tool
        deep_agent.configure({"use_deeper_tool": not initial_tool})
        assert deep_agent.use_deeper_tool == (not initial_tool), "use_deeper_tool 配置未生效"
        print(f"  ✓ use_deeper_tool 配置生效")

        # 清理资源
        for agent in agents.values():
            if hasattr(agent, 'close'):
                agent.close()

        print("\n✅ 测试 2 通过：Agent 多态接口正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_manager_memory():
    """
    测试 3: AgentManager 内存管理

    验证：TTL 机制能够自动清理过期的 Agent 实例
    """
    print("\n" + "="*60)
    print("测试 3: AgentManager 内存管理")
    print("="*60)

    try:
        from server.services.agent_service import AgentManager

        # 创建一个短 TTL 的 AgentManager（5 秒）
        manager = AgentManager(ttl=5, enable_background_cleanup=False)

        # 创建多个 Agent 实例
        print("\n创建 Agent 实例:")
        for i in range(3):
            session_id = f"test_session_{i}"
            agent = manager.get_agent("hybrid_agent", session_id)
            print(f"  ✓ 创建 session {session_id}")

        # 检查实例数量
        stats = manager.get_stats()
        initial_count = stats["total_instances"]
        print(f"\n📊 当前实例数量: {initial_count}")
        assert initial_count == 3, f"应该有 3 个实例，实际: {initial_count}"

        # 等待超过 TTL
        print(f"\n⏳ 等待 {manager.ttl + 1} 秒让实例过期...")
        time.sleep(manager.ttl + 1)

        # 手动触发清理
        manager._cleanup_expired()

        # 检查清理后的实例数量
        stats = manager.get_stats()
        remaining_count = stats["total_instances"]
        print(f"\n📊 清理后实例数量: {remaining_count}")
        assert remaining_count == 0, f"过期实例应该被清理，实际剩余: {remaining_count}"

        # 测试手动清理特定会话
        print("\n测试手动清理特定会话:")
        agent1 = manager.get_agent("hybrid_agent", "session_a")
        agent2 = manager.get_agent("graph_agent", "session_a")
        agent3 = manager.get_agent("hybrid_agent", "session_b")

        stats = manager.get_stats()
        print(f"  创建 3 个实例: {stats['total_instances']}")
        assert stats["total_instances"] == 3

        # 清理 session_a
        manager.cleanup_session("session_a")
        stats = manager.get_stats()
        print(f"  清理 session_a 后: {stats['total_instances']}")
        assert stats["total_instances"] == 1, "应该剩余 1 个实例（session_b）"

        # 清理所有
        manager.close_all()
        stats = manager.get_stats()
        print(f"  清理所有后: {stats['total_instances']}")
        assert stats["total_instances"] == 0, "应该没有剩余实例"

        print("\n✅ 测试 3 通过：AgentManager 内存管理正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_executor_references():
    """
    测试 4: ResearchExecutor 引用提取

    验证：优先使用结构化数据，正则表达式仅作兜底
    """
    print("\n" + "="*60)
    print("测试 4: ResearchExecutor 引用提取")
    print("="*60)

    try:
        from graphrag_agent.agents.multi_agent.executor.research_executor import ResearchExecutor

        executor = ResearchExecutor()

        # 测试 1: 结构化 references 字段（最稳健）
        print("\n测试结构化 'references' 字段:")
        payload1 = {
            "content": "这是答案内容",
            "references": ["doc_001", "doc_002", "doc_003"]
        }
        refs1 = executor._extract_reference_ids(payload1, "")
        print(f"  提取结果: {refs1}")
        assert refs1 == ["doc_001", "doc_002", "doc_003"], "结构化提取失败"
        print(f"  ✓ 成功提取 {len(refs1)} 个引用")

        # 测试 2: 字典格式的 references
        print("\n测试字典格式的 references:")
        payload2 = {
            "content": "答案",
            "references": [
                {"id": "chunk_001", "score": 0.9},
                {"doc_id": "chunk_002"},
                {"chunk_id": "chunk_003"}
            ]
        }
        refs2 = executor._extract_reference_ids(payload2, "")
        print(f"  提取结果: {refs2}")
        assert "chunk_001" in refs2 and "chunk_002" in refs2 and "chunk_003" in refs2
        print(f"  ✓ 成功提取 {len(refs2)} 个引用")

        # 测试 3: 旧格式 reference 结构
        print("\n测试旧格式 'reference' 结构:")
        payload3 = {
            "content": "答案",
            "reference": {
                "chunks": [
                    {"chunk_id": "old_001"},
                    {"chunk_id": "old_002"}
                ]
            }
        }
        refs3 = executor._extract_reference_ids(payload3, "")
        print(f"  提取结果: {refs3}")
        assert "old_001" in refs3 and "old_002" in refs3
        print(f"  ✓ 成功提取 {len(refs3)} 个引用")

        # 测试 4: 正则表达式兜底
        print("\n测试正则表达式兜底:")
        payload4 = "普通字符串，没有结构化数据"
        answer_text4 = "答案包含 [证据ID: regex_001] 和 [证据ID：regex_002]"
        refs4 = executor._extract_reference_ids(payload4, answer_text4)
        print(f"  提取结果: {refs4}")
        assert "regex_001" in refs4 and "regex_002" in refs4
        print(f"  ✓ 正则兜底成功，提取 {len(refs4)} 个引用")

        # 测试 5: 去重
        print("\n测试去重:")
        payload5 = {
            "content": "答案",
            "references": ["dup_001", "dup_002", "dup_001", "dup_002"]
        }
        refs5 = executor._extract_reference_ids(payload5, "")
        print(f"  提取结果: {refs5}")
        assert len(refs5) == 2, f"应该去重为 2 个，实际: {len(refs5)}"
        print(f"  ✓ 去重正常")

        print("\n✅ 测试 4 通过：ResearchExecutor 引用提取正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("开始运行优化集成测试")
    print("🧪" * 30)

    results = {
        "快速通道性能": test_fast_path_latency(),
        "Agent多态接口": test_agent_polymorphism(),
        "内存管理": test_agent_manager_memory(),
        "引用提取": test_research_executor_references(),
    }

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}  {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有优化测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
