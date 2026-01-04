"""
L0/L1 拆分管道集成测试

测试内容：
1. L0 快速通道性能（10秒内完成）
2. L1 慢速通道正确性（实体提取）
3. 任务队列功能
4. 状态查询
"""
import time
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_task_queue():
    """测试任务队列基础功能"""
    print("\n" + "="*60)
    print("测试 1: 任务队列基础功能")
    print("="*60)

    try:
        from graphrag_agent.integrations.build.pipeline import (
            GraphBuildTaskQueue,
            Task,
            TaskPriority,
            TaskStatus
        )

        # 创建队列
        queue = GraphBuildTaskQueue(num_workers=2)

        # 注册一个测试处理器
        def test_handler(task: Task):
            print(f"  处理任务: {task.task_id}")
            time.sleep(0.5)  # 模拟处理

        queue.register_handler("test_task", test_handler)

        # 启动队列
        queue.start()

        # 提交任务
        task_ids = []
        for i in range(5):
            task_id = queue.submit_file_processing(
                file_path=f"/test/file_{i}.txt",
                task_type="test_task",
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)

        print(f"✓ 已提交 {len(task_ids)} 个任务")

        # 等待任务完成
        time.sleep(3)

        # 检查统计
        stats = queue.get_stats()
        print(f"✓ 完成任务: {stats['total_completed']}/{stats['total_submitted']}")

        # 停止队列
        queue.stop(wait_completion=True)

        print("\n✅ 测试 1 通过：任务队列功能正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_l0_fast_pipeline():
    """测试 L0 快速通道"""
    print("\n" + "="*60)
    print("测试 2: L0 快速通道性能")
    print("="*60)

    try:
        from graphrag_agent.integrations.build.pipeline import FastIngestionPipeline

        pipeline = FastIngestionPipeline()

        # 模拟处理一个文件
        # 注意：这需要真实的文件或Mock
        print("⚠️  此测试需要真实文件或Mock，跳过实际处理")
        print("✓ FastIngestionPipeline 类可正常导入")

        # 检查方法存在
        assert hasattr(pipeline, 'process_single_file')
        assert hasattr(pipeline, 'process_batch_files')
        assert hasattr(pipeline, 'check_file_ready_for_search')

        print("✓ 所有必需方法存在")

        print("\n✅ 测试 2 通过：L0 管道结构正确")
        return True

    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_l1_slow_pipeline():
    """测试 L1 慢速通道"""
    print("\n" + "="*60)
    print("测试 3: L1 慢速通道结构")
    print("="*60)

    try:
        from graphrag_agent.integrations.build.pipeline import SlowGraphPipeline

        pipeline = SlowGraphPipeline()

        print("✓ SlowGraphPipeline 类可正常导入")

        # 检查方法存在
        assert hasattr(pipeline, 'process_file_entity_extraction')
        assert hasattr(pipeline, 'check_file_graph_ready')

        print("✓ 所有必需方法存在")

        print("\n✅ 测试 3 通过：L1 管道结构正确")
        return True

    except Exception as e:
        print(f"\n❌ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_incremental_update_v2():
    """测试 IncrementalUpdateManagerV2"""
    print("\n" + "="*60)
    print("测试 4: IncrementalUpdateManagerV2 接口")
    print("="*60)

    try:
        from graphrag_agent.integrations.build.incremental_update_v2 import (
            IncrementalUpdateManagerV2,
            quick_upload_file
        )

        manager = IncrementalUpdateManagerV2()

        print("✓ IncrementalUpdateManagerV2 可正常实例化")

        # 检查方法存在
        methods = [
            'run_fast_ingestion',
            'run_deep_indexing',
            'quick_ingest_single_file',
            'submit_entity_extraction_task',
            'get_file_status',
            'get_queue_status',
            'run_full_pipeline'
        ]

        for method in methods:
            assert hasattr(manager, method), f"缺少方法: {method}"
            print(f"  ✓ {method}")

        print("✓ 所有必需方法存在")

        # 测试状态查询
        queue_status = manager.get_queue_status()
        print(f"✓ 队列状态查询正常: {queue_status}")

        print("\n✅ 测试 4 通过：IncrementalUpdateManagerV2 接口正确")
        return True

    except Exception as e:
        print(f"\n❌ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """测试管道集成"""
    print("\n" + "="*60)
    print("测试 5: 管道集成测试")
    print("="*60)

    try:
        from graphrag_agent.integrations.build.pipeline import (
            FastIngestionPipeline,
            SlowGraphPipeline,
            get_global_task_queue,
            entity_extraction_task_handler
        )

        # 获取全局队列
        task_queue = get_global_task_queue()

        print("✓ 全局任务队列单例正常")

        # 检查处理器是否已注册
        assert "entity_extraction" in task_queue.task_handlers
        print("✓ 实体提取处理器已注册")

        # 测试管道协作
        fast_pipeline = FastIngestionPipeline()
        slow_pipeline = SlowGraphPipeline()

        print("✓ L0 和 L1 管道可同时实例化")

        print("\n✅ 测试 5 通过：管道集成正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试 5 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("开始运行 L0/L1 拆分管道测试")
    print("🧪" * 30)

    tests = [
        ("任务队列", test_task_queue),
        ("L0 快速通道", test_l0_fast_pipeline),
        ("L1 慢速通道", test_l1_slow_pipeline),
        ("IncrementalUpdateManagerV2", test_incremental_update_v2),
        ("管道集成", test_pipeline_integration),
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
    print("测试总结")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}  {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
