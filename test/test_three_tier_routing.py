"""
测试三层自适应路由逻辑

验证 Orchestrator 的三层路由策略：
- FAST: 事实性/简单查询 -> Local Search
- SLOW: 分析性/综合查询 -> Global Search
- HEAVY: 研究性/复杂任务 -> Plan-Execute-Report
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_routing_logic():
    """
    测试路由逻辑的正确性
    """
    from graphrag_agent.agents.multi_agent.orchestrator import (
        MultiAgentOrchestrator,
        OrchestratorConfig,
    )

    # 创建配置
    config = OrchestratorConfig(enable_fast_path=True)

    # 创建 Orchestrator（使用 None 作为占位，仅测试路由逻辑）
    orchestrator = MultiAgentOrchestrator(
        planner=None,  # type: ignore
        worker_coordinator=None,  # type: ignore
        reporter=None,  # type: ignore
        config=config,
    )

    # 测试用例
    test_cases = [
        # FAST 通道（事实性/简单查询）
        ("旷课多少学时会被退学？", "FAST"),
        ("什么是国家奖学金？", "FAST"),
        ("学生处的电话是多少？", "FAST"),
        ("违纪处分有哪些类型？", "FAST"),
        # SLOW 通道（分析性/综合查询）
        ("总结一下奖学金政策的变化趋势", "SLOW"),
        ("分析学生违纪的主要原因", "SLOW"),
        ("概括学生资助体系的全貌", "SLOW"),
        ("对比不同奖学金的申请条件", "SLOW"),
        # HEAVY 通道（研究性/复杂任务）
        ("深度研究学生资助政策的演变", "HEAVY"),
        ("撰写一份详细调查报告分析学生违纪情况", "HEAVY"),
        ("请深入分析奖学金评选的公平性问题", "HEAVY"),
        ("编写长文总结学生管理制度的改革方向", "HEAVY"),
        # 长查询（默认 HEAVY）
        ("请详细说明国家奖学金、国家励志奖学金和国家助学金的区别，包括申请条件、金额和发放流程", "HEAVY"),
    ]

    print("=" * 70)
    print("三层路由逻辑测试")
    print("=" * 70)

    passed = 0
    failed = 0

    for query, expected_route in test_cases:
        actual_route = orchestrator._determine_route(query)
        status = "✅" if actual_route == expected_route else "❌"

        if actual_route == expected_route:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} 查询: {query}")
        print(f"   预期路由: {expected_route}")
        print(f"   实际路由: {actual_route}")

    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


def test_signal_construction():
    """
    测试执行信号构造
    """
    from graphrag_agent.agents.multi_agent.orchestrator import (
        MultiAgentOrchestrator,
        OrchestratorConfig,
    )

    config = OrchestratorConfig(enable_fast_path=True)
    orchestrator = MultiAgentOrchestrator(
        planner=None,  # type: ignore
        worker_coordinator=None,  # type: ignore
        reporter=None,  # type: ignore
        config=config,
    )

    print("\n" + "=" * 70)
    print("执行信号构造测试")
    print("=" * 70)

    # 测试 FAST 通道信号
    query = "什么是国家奖学金？"
    signal = orchestrator._create_direct_signal(query, "FAST")

    print("\n[FAST 通道] 查询:", query)
    print("任务数量:", len(signal.tasks))
    print("任务类型:", signal.tasks[0]["task_type"])
    assert signal.tasks[0]["task_type"] == "local_search", "FAST 通道应使用 local_search"
    print("✅ FAST 通道信号构造正确")

    # 测试 SLOW 通道信号
    query = "总结奖学金政策的变化趋势"
    signal = orchestrator._create_direct_signal(query, "SLOW")

    print("\n[SLOW 通道] 查询:", query)
    print("任务数量:", len(signal.tasks))
    print("任务类型:", signal.tasks[0]["task_type"])
    assert signal.tasks[0]["task_type"] == "global_search", "SLOW 通道应使用 global_search"
    print("✅ SLOW 通道信号构造正确")

    print("\n" + "=" * 70)
    print("执行信号构造测试通过")
    print("=" * 70)

    return True


def test_dummy_plan():
    """
    测试占位 PlannerResult 构造
    """
    from graphrag_agent.agents.multi_agent.orchestrator import (
        MultiAgentOrchestrator,
        OrchestratorConfig,
    )

    config = OrchestratorConfig(enable_fast_path=True)
    orchestrator = MultiAgentOrchestrator(
        planner=None,  # type: ignore
        worker_coordinator=None,  # type: ignore
        reporter=None,  # type: ignore
        config=config,
    )

    print("\n" + "=" * 70)
    print("占位 PlannerResult 测试")
    print("=" * 70)

    dummy_plan = orchestrator._create_dummy_plan()

    assert dummy_plan is not None, "应返回占位 PlannerResult"
    assert dummy_plan.plan_spec is None, "直接通道应跳过规划"
    assert dummy_plan.executor_signal is None, "执行信号由 _create_direct_signal 单独构造"
    assert not dummy_plan.clarification.needs_clarification, "不需要澄清"

    print("✅ plan_spec: None (跳过规划)")
    print("✅ executor_signal: None (单独构造)")
    print("✅ needs_clarification: False")

    print("\n" + "=" * 70)
    print("占位 PlannerResult 测试通过")
    print("=" * 70)

    return True


def run_all_tests():
    """
    运行所有测试
    """
    print("\n" + "🧪" * 35)
    print("开始运行三层路由测试")
    print("🧪" * 35)

    results = {}

    # 测试 1: 路由逻辑
    try:
        results["路由逻辑"] = test_routing_logic()
    except Exception as e:
        print(f"\n❌ 路由逻辑测试失败: {e}")
        import traceback

        traceback.print_exc()
        results["路由逻辑"] = False

    # 测试 2: 信号构造
    try:
        results["信号构造"] = test_signal_construction()
    except Exception as e:
        print(f"\n❌ 信号构造测试失败: {e}")
        import traceback

        traceback.print_exc()
        results["信号构造"] = False

    # 测试 3: 占位 Plan
    try:
        results["占位 Plan"] = test_dummy_plan()
    except Exception as e:
        print(f"\n❌ 占位 Plan 测试失败: {e}")
        import traceback

        traceback.print_exc()
        results["占位 Plan"] = False

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}  {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        print("\n三层自适应路由已成功实现：")
        print("  ✓ FAST 通道：事实性查询 -> Local Search")
        print("  ✓ SLOW 通道：分析性查询 -> Global Search")
        print("  ✓ HEAVY 通道：研究性任务 -> Plan-Execute-Report")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
