#!/bin/bash
# 验证三层自适应路由 - 静态代码检查

echo "========================================="
echo "验证三层自适应路由"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PASS_COUNT=0
FAIL_COUNT=0

ORCHESTRATOR_FILE="graphrag_agent/agents/multi_agent/orchestrator.py"

# 测试 1: 检查 _determine_route 方法
echo ""
echo "[测试 1] 检查路由网关方法"
echo "-----------------------------------------"

if grep -q "def _determine_route" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ _determine_route 方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 _determine_route 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q '"FAST".*"SLOW".*"HEAVY"' "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 三层路由类型定义完整"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 三层路由类型定义不完整"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "heavy_keywords" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ HEAVY 通道关键词检测存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ HEAVY 通道关键词检测缺失"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "slow_keywords" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ SLOW 通道关键词检测存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ SLOW 通道关键词检测缺失"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 2: 检查 _create_direct_signal 方法
echo ""
echo "[测试 2] 检查执行信号构造方法"
echo "-----------------------------------------"

if grep -q "def _create_direct_signal" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ _create_direct_signal 方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 _create_direct_signal 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'task_type="local_search"' "$ORCHESTRATOR_FILE"; then
    echo "  ✓ FAST 通道使用 local_search"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ FAST 通道未配置 local_search"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'task_type="global_search"' "$ORCHESTRATOR_FILE"; then
    echo "  ✓ SLOW 通道使用 global_search"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ SLOW 通道未配置 global_search"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 3: 检查 _create_dummy_plan 方法
echo ""
echo "[测试 3] 检查占位 PlannerResult 方法"
echo "-----------------------------------------"

if grep -q "def _create_dummy_plan" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ _create_dummy_plan 方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 _create_dummy_plan 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "plan_spec=None" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 占位 Plan 跳过规划（plan_spec=None）"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 占位 Plan 配置不正确"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 4: 检查 _execute_direct_lane 方法
echo ""
echo "[测试 4] 检查直接通道执行方法"
echo "-----------------------------------------"

if grep -q "def _execute_direct_lane" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ _execute_direct_lane 方法存在"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未找到 _execute_direct_lane 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "FAST.*SLOW.*route" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 支持 FAST 和 SLOW 路由参数"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 路由参数支持不完整"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "self._worker.execute_plan" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 复用 WorkerCoordinator 执行"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 未复用 WorkerCoordinator"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 5: 检查 run 方法集成
echo ""
echo "[测试 5] 检查 run 方法集成"
echo "-----------------------------------------"

if grep -q "route = self._determine_route" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ run 方法调用路由网关"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ run 方法未调用路由网关"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'if route in \["FAST", "SLOW"\]' "$ORCHESTRATOR_FILE"; then
    echo "  ✓ run 方法处理直接通道路由"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ run 方法未处理直接通道路由"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "return self._execute_direct_lane" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ run 方法调用直接通道执行"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ run 方法未调用直接通道执行"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q "HEAVY.*Plan-Execute-Report" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ HEAVY 通道保留完整流程"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ HEAVY 通道配置缺失"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 6: 检查文档和注释
echo ""
echo "[测试 6] 检查文档和注释"
echo "-----------------------------------------"

if grep -q "三层自适应路由" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 明确标注三层路由架构"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ⚠️  缺少三层路由说明"
fi

if grep -q "跳过 Planner" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 说明了性能优化策略"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ⚠️  缺少性能优化说明"
fi

if grep -q "Local Search" "$ORCHESTRATOR_FILE" && grep -q "Global Search" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 明确标注了搜索策略"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 搜索策略说明不完整"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 7: 检查关键词配置
echo ""
echo "[测试 7] 检查路由关键词配置"
echo "-----------------------------------------"

# HEAVY 关键词
if grep -q "深度研究\|调研报告\|长文\|详细调查" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ HEAVY 通道关键词配置完整"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ HEAVY 通道关键词配置缺失"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# SLOW 关键词
if grep -q "总结\|概括\|全貌\|趋势\|对比\|分析" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ SLOW 通道关键词配置完整"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ SLOW 通道关键词配置缺失"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 长度检查
if grep -q "len(query) < 50" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ FAST 通道长度限制正确"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ FAST 通道长度限制缺失"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 8: 检查向后兼容性
echo ""
echo "[测试 8] 检查向后兼容性"
echo "-----------------------------------------"

if grep -q "enable_fast_path" "$ORCHESTRATOR_FILE"; then
    echo "  ✓ 保留了 enable_fast_path 配置"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ❌ 缺少 enable_fast_path 配置"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 检查是否删除了旧方法（应该被替换）
if grep -q "def _is_simple_query" "$ORCHESTRATOR_FILE"; then
    echo "  ⚠️  旧的 _is_simple_query 方法未删除（可能冗余）"
else
    echo "  ✓ 旧的 _is_simple_query 方法已替换"
    PASS_COUNT=$((PASS_COUNT + 1))
fi

if grep -q "def _execute_fast_path" "$ORCHESTRATOR_FILE"; then
    echo "  ⚠️  旧的 _execute_fast_path 方法未删除（可能冗余）"
else
    echo "  ✓ 旧的 _execute_fast_path 方法已替换"
    PASS_COUNT=$((PASS_COUNT + 1))
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
    echo "三层自适应路由已成功实现："
    echo "  ✓ FAST 通道: 事实性查询 -> Local Search（跳过 Planner）"
    echo "  ✓ SLOW 通道: 分析性查询 -> Global Search（跳过 Planner）"
    echo "  ✓ HEAVY 通道: 研究性任务 -> Plan-Execute-Report（完整流程）"
    echo ""
    echo "性能优化："
    echo "  ✓ FAST/SLOW 通道跳过 Planner 的 3 次 LLM 调用"
    echo "  ✓ 复用现有 WorkerCoordinator 执行器"
    echo "  ✓ 保留向后兼容性"
    exit 0
else
    echo "⚠️  部分验证失败 ($FAIL_COUNT/$TOTAL)"
    exit 1
fi
