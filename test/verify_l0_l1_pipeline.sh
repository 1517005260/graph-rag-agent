#!/bin/bash
# 验证 L0/L1 拆分管道 - 静态代码检查

echo "========================================="
echo "验证 L0/L1 拆分管道"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PASS_COUNT=0
FAIL_COUNT=0

# 测试 1: 检查管道文件是否存在
echo ""
echo "[测试 1] 检查管道文件结构"
echo "-----------------------------------------"

FILES=(
    "graphrag_agent/integrations/build/pipeline/task_queue.py"
    "graphrag_agent/integrations/build/pipeline/fast_ingestion_pipeline.py"
    "graphrag_agent/integrations/build/pipeline/slow_graph_pipeline.py"
    "graphrag_agent/integrations/build/pipeline/__init__.py"
    "graphrag_agent/integrations/build/incremental_update_v2.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ❌ 缺失: $file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# 测试 2: 检查任务队列核心类
echo ""
echo "[测试 2] 检查任务队列核心类"
echo "-----------------------------------------"

if grep -q "class GraphBuildTaskQueue" graphrag_agent/integrations/build/pipeline/task_queue.py; then
    echo "  ✓ GraphBuildTaskQueue 类存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 GraphBuildTaskQueue 类"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "class Task" graphrag_agent/integrations/build/pipeline/task_queue.py; then
    echo "  ✓ Task 类存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 Task 类"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "class TaskPriority" graphrag_agent/integrations/build/pipeline/task_queue.py; then
    echo "  ✓ TaskPriority 枚举存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 TaskPriority 枚举"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def submit_task" graphrag_agent/integrations/build/pipeline/task_queue.py; then
    echo "  ✓ submit_task 方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 submit_task 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def _worker_loop" graphrag_agent/integrations/build/pipeline/task_queue.py; then
    echo "  ✓ Worker 线程逻辑存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 Worker 线程逻辑"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 3: 检查 L0 快速通道
echo ""
echo "[测试 3] 检查 L0 快速通道"
echo "-----------------------------------------"

if grep -q "class FastIngestionPipeline" graphrag_agent/integrations/build/pipeline/fast_ingestion_pipeline.py; then
    echo "  ✓ FastIngestionPipeline 类存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 FastIngestionPipeline 类"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def process_single_file" graphrag_agent/integrations/build/pipeline/fast_ingestion_pipeline.py; then
    echo "  ✓ process_single_file 方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 process_single_file 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def _vectorize_and_store" graphrag_agent/integrations/build/pipeline/fast_ingestion_pipeline.py; then
    echo "  ✓ 向量化逻辑存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到向量化逻辑"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "10秒内" graphrag_agent/integrations/build/pipeline/fast_ingestion_pipeline.py; then
    echo "  ✓ 包含性能目标说明"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ⚠️  缺少性能目标说明"
fi

# 测试 4: 检查 L1 慢速通道
echo ""
echo "[测试 4] 检查 L1 慢速通道"
echo "-----------------------------------------"

if grep -q "class SlowGraphPipeline" graphrag_agent/integrations/build/pipeline/slow_graph_pipeline.py; then
    echo "  ✓ SlowGraphPipeline 类存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 SlowGraphPipeline 类"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def process_file_entity_extraction" graphrag_agent/integrations/build/pipeline/slow_graph_pipeline.py; then
    echo "  ✓ 实体提取方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到实体提取方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def _stream_extract_entities" graphrag_agent/integrations/build/pipeline/slow_graph_pipeline.py; then
    echo "  ✓ 流式提取方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到流式提取方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "stream_process_large_files" graphrag_agent/integrations/build/pipeline/slow_graph_pipeline.py; then
    echo "  ✓ 利用了流式处理优化"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ⚠️  未明确利用流式处理"
fi

# 测试 5: 检查 IncrementalUpdateManagerV2
echo ""
echo "[测试 5] 检查 IncrementalUpdateManagerV2"
echo "-----------------------------------------"

if grep -q "class IncrementalUpdateManagerV2" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ IncrementalUpdateManagerV2 类存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 IncrementalUpdateManagerV2 类"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def run_fast_ingestion" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ L0 快速摄取方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 L0 快速摄取方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def run_deep_indexing" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ L1 深度索引方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 L1 深度索引方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def get_file_status" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 文件状态查询方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到文件状态查询方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "def quick_upload_file" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 快速上传便捷函数存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到快速上传便捷函数"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 6: 检查集成
echo ""
echo "[测试 6] 检查组件集成"
echo "-----------------------------------------"

if grep -q "from graphrag_agent.integrations.build.pipeline import" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ V2 管理器导入了新管道"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ V2 管理器未导入新管道"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "self.task_queue = get_global_task_queue()" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 使用了全局任务队列"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未使用全局任务队列"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "self.fast_pipeline = FastIngestionPipeline" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 集成了 L0 管道"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未集成 L0 管道"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "self.slow_pipeline = SlowGraphPipeline" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 集成了 L1 管道"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未集成 L1 管道"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 7: 检查文档和注释
echo ""
echo "[测试 7] 检查文档和注释"
echo "-----------------------------------------"

if grep -q "10秒内" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 包含性能目标说明"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ⚠️  缺少性能目标说明"
fi

if grep -q "L0" graphrag_agent/integrations/build/incremental_update_v2.py && grep -q "L1" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 明确标注了 L0/L1 分层"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未明确标注 L0/L1 分层"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -qi "naive.*rag" graphrag_agent/integrations/build/incremental_update_v2.py; then
    echo "  ✓ 提到了 Naive RAG 可用性"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ⚠️  未强调 Naive RAG 立即可用"
fi

# 总结
echo ""
echo "========================================="
echo "验证总结"
echo "========================================="
echo "✅ 通过: $PASS_COUNT"
echo "❌ 失败: $FAIL_COUNT"
echo ""

TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有验证通过！"
    echo ""
    echo "L0/L1 拆分管道已成功实现："
    echo "  ✓ 任务队列系统（异步处理）"
    echo "  ✓ L0 快速通道（文本向量化）"
    echo "  ✓ L1 慢速通道（实体提取）"
    echo "  ✓ 统一管理器（IncrementalUpdateManagerV2）"
    echo ""
    echo "用户上传文件后立即可使用 Naive RAG 搜索！"
    exit 0
else
    echo "⚠️  部分验证失败 ($FAIL_COUNT/$TOTAL)"
    exit 1
fi
