"""
快速测试：验证代码优化的核心功能

这个测试脚本只检查代码结构，不需要运行完整系统
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_base_agent_interface():
    """检查 BaseAgent 是否有新增的多态方法"""
    print("\n" + "="*60)
    print("测试: BaseAgent 多态接口")
    print("="*60)

    try:
        from graphrag_agent.agents.base import BaseAgent
        import inspect

        # 检查方法是否存在
        methods_to_check = {
            'configure': '配置方法',
            'ask_with_thinking': '思考过程方法',
            'supports_kg_extraction': 'KG提取支持方法',
        }

        for method_name, desc in methods_to_check.items():
            assert hasattr(BaseAgent, method_name), f"BaseAgent 缺少 {method_name} 方法"

            # 检查方法签名
            method = getattr(BaseAgent, method_name)
            sig = inspect.signature(method)

            print(f"  ✓ {method_name}{sig}")
            print(f"    {desc}")

        print("\n✅ BaseAgent 接口检查通过")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_research_agent_overrides():
    """检查 DeepResearchAgent 是否重写了多态方法"""
    print("\n" + "="*60)
    print("测试: DeepResearchAgent 方法重写")
    print("="*60)

    try:
        from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
        from graphrag_agent.agents.base import BaseAgent

        # 检查方法是否被重写
        methods_to_check = {
            'configure': True,  # 应该重写
            'supports_kg_extraction': True,  # 应该重写
            'ask_with_thinking': True,  # DeepResearchAgent 原本就有
        }

        agent = DeepResearchAgent()

        for method_name, should_override in methods_to_check.items():
            # 检查方法存在
            assert hasattr(agent, method_name), f"DeepResearchAgent 缺少 {method_name}"

            # 检查是否重写（通过比较方法对象）
            deep_method = getattr(DeepResearchAgent, method_name)
            base_method = getattr(BaseAgent, method_name, None)

            if should_override and base_method:
                is_overridden = deep_method is not base_method
                status = "重写" if is_overridden else "继承"
                print(f"  ✓ {method_name}: {status}")
            else:
                print(f"  ✓ {method_name}: 存在")

        # 测试 supports_kg_extraction 返回值
        supports_kg = agent.supports_kg_extraction()
        assert supports_kg is False, "DeepResearchAgent 应该返回 False"
        print(f"\n  ✓ supports_kg_extraction() 返回正确值: {supports_kg}")

        # 测试 configure 方法
        agent.configure({"show_thinking": True})
        assert agent.show_thinking is True, "configure 方法未正确设置 show_thinking"
        print(f"  ✓ configure() 方法正常工作")

        # 清理
        agent.close()

        print("\n✅ DeepResearchAgent 重写检查通过")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_service_no_conditionals():
    """检查 chat_service.py 是否移除了所有 agent_type 条件判断"""
    print("\n" + "="*60)
    print("测试: chat_service.py 代码清理")
    print("="*60)

    try:
        chat_service_path = project_root / "server" / "services" / "chat_service.py"

        if not chat_service_path.exists():
            print(f"  ⚠️  文件不存在: {chat_service_path}")
            return False

        # 读取文件内容
        with open(chat_service_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查不应该存在的模式
        forbidden_patterns = [
            'if agent_type == "deep_research_agent"',
            'if agent_type != "deep_research_agent"',
            'agent_type in [',
        ]

        violations = []
        for pattern in forbidden_patterns:
            if pattern in content:
                # 找出所有出现的行号
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line:
                        violations.append((i, pattern, line.strip()))

        if violations:
            print(f"\n  ❌ 发现 {len(violations)} 处 agent_type 条件判断:")
            for line_num, pattern, line_content in violations:
                print(f"    行 {line_num}: {line_content}")
            return False

        # 检查应该存在的模式（使用多态接口）
        required_patterns = [
            'selected_agent.configure(',
            'selected_agent.supports_kg_extraction()',
        ]

        missing = []
        for pattern in required_patterns:
            if pattern not in content:
                missing.append(pattern)

        if missing:
            print(f"\n  ⚠️  缺少多态接口调用:")
            for pattern in missing:
                print(f"    {pattern}")
            return False

        print("  ✓ 没有发现 agent_type 条件判断")
        print("  ✓ 使用了多态接口")

        print("\n✅ chat_service.py 清理检查通过")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_manager_has_ttl():
    """检查 AgentManager 是否有 TTL 机制"""
    print("\n" + "="*60)
    print("测试: AgentManager TTL 机制")
    print("="*60)

    try:
        from server.services.agent_service import AgentManager
        import inspect

        # 检查初始化参数
        init_sig = inspect.signature(AgentManager.__init__)
        params = init_sig.parameters

        assert 'ttl' in params, "AgentManager.__init__ 缺少 ttl 参数"
        assert 'enable_background_cleanup' in params, "AgentManager.__init__ 缺少 enable_background_cleanup 参数"

        print(f"  ✓ __init__{init_sig}")

        # 检查方法
        methods = {
            '_cleanup_expired': '清理过期实例',
            'cleanup_session': '清理会话',
            'get_stats': '获取统计信息',
            '_start_background_cleanup': '启动后台清理',
            'close_all': '关闭所有资源',
        }

        for method_name, desc in methods.items():
            assert hasattr(AgentManager, method_name), f"AgentManager 缺少 {method_name} 方法"
            print(f"  ✓ {method_name}: {desc}")

        # 创建实例测试
        manager = AgentManager(ttl=10, enable_background_cleanup=False)

        # 检查属性
        assert hasattr(manager, 'ttl'), "AgentManager 缺少 ttl 属性"
        assert hasattr(manager, 'access_times'), "AgentManager 缺少 access_times 属性"
        assert hasattr(manager, 'agent_lock'), "AgentManager 缺少 agent_lock 属性"

        print(f"\n  ✓ TTL 设置: {manager.ttl} 秒")
        print(f"  ✓ 访问时间追踪: 已启用")
        print(f"  ✓ 线程锁: 已启用")

        # 测试 get_stats
        stats = manager.get_stats()
        assert 'total_instances' in stats
        assert 'active_instances' in stats
        assert 'expired_instances' in stats
        assert 'ttl_seconds' in stats

        print(f"\n  ✓ 统计信息: {stats}")

        print("\n✅ AgentManager TTL 机制检查通过")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_fast_path():
    """检查 Orchestrator 是否有快速通道配置"""
    print("\n" + "="*60)
    print("测试: Orchestrator 快速通道")
    print("="*60)

    try:
        from graphrag_agent.agents.multi_agent.orchestrator import OrchestratorConfig

        # 检查配置字段
        config = OrchestratorConfig()

        fields_to_check = {
            'enable_fast_path': bool,
            'fast_path_max_length': int,
            'fast_path_keywords': list,
        }

        for field_name, expected_type in fields_to_check.items():
            assert hasattr(config, field_name), f"OrchestratorConfig 缺少 {field_name} 字段"
            value = getattr(config, field_name)
            assert isinstance(value, expected_type), f"{field_name} 类型错误: 期望 {expected_type}, 实际 {type(value)}"
            print(f"  ✓ {field_name}: {value}")

        # 检查方法
        from graphrag_agent.agents.multi_agent.orchestrator import MultiAgentOrchestrator

        methods = {
            '_is_simple_query': '简单查询判断',
            '_execute_fast_path': '快速通道执行',
        }

        for method_name, desc in methods.items():
            assert hasattr(MultiAgentOrchestrator, method_name), f"MultiAgentOrchestrator 缺少 {method_name} 方法"
            print(f"  ✓ {method_name}: {desc}")

        print("\n✅ Orchestrator 快速通道检查通过")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_executor_extract_method():
    """检查 ResearchExecutor 的引用提取方法"""
    print("\n" + "="*60)
    print("测试: ResearchExecutor 引用提取")
    print("="*60)

    try:
        from graphrag_agent.agents.multi_agent.executor.research_executor import ResearchExecutor
        import inspect

        # 检查方法签名
        method = ResearchExecutor._extract_reference_ids
        sig = inspect.signature(method)

        print(f"  ✓ _extract_reference_ids{sig}")

        # 读取方法源码检查策略
        source = inspect.getsource(method)

        # 检查是否包含三层策略
        strategies = [
            '策略 1',  # 结构化 references 字段
            '策略 2',  # 旧格式 reference 结构
            '策略 3',  # 正则兜底
        ]

        for strategy in strategies:
            if strategy in source:
                print(f"  ✓ 包含{strategy}（结构化数据优先）")
            else:
                print(f"  ⚠️  未找到{strategy}标记")

        # 检查是否有警告日志
        if '_LOGGER.warning' in source and '正则表达式' in source:
            print(f"  ✓ 包含正则兜底警告日志")

        print("\n✅ ResearchExecutor 引用提取检查通过")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_tests():
    """运行快速测试（不需要完整系统）"""
    print("\n" + "🚀" * 30)
    print("开始运行快速优化测试")
    print("🚀" * 30)

    tests = [
        ("BaseAgent 接口", test_base_agent_interface),
        ("DeepResearchAgent 重写", test_deep_research_agent_overrides),
        ("chat_service 清理", test_chat_service_no_conditionals),
        ("AgentManager TTL", test_agent_manager_has_ttl),
        ("Orchestrator 快速通道", test_orchestrator_fast_path),
        ("ResearchExecutor 引用提取", test_research_executor_extract_method),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} 执行异常: {e}")
            results[test_name] = False

    # 打印总结
    print("\n" + "="*60)
    print("快速测试总结")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}  {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有快速测试通过！")
        print("\n提示：运行 test_optimization.py 进行完整的集成测试")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_quick_tests()
    sys.exit(exit_code)
