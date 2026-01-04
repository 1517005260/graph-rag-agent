#!/bin/bash
# 验证代码优化 - 不需要运行系统，只检查代码

echo "========================================="
echo "验证代码优化"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PASS_COUNT=0
FAIL_COUNT=0

# 测试 1: chat_service.py 清理
echo ""
echo "[测试 1] 检查 chat_service.py 是否移除了 agent_type 判断"
echo "-----------------------------------------"

# 检查不应该存在的模式
if grep -n 'if agent_type == "deep_research_agent"' server/services/chat_service.py; then
    echo "❌ 发现 if agent_type == 判断"
    FAIL_COUNT=$((FAIL_COUNT + 1))
elif grep -n 'if agent_type !=' server/services/chat_service.py; then
    echo "❌ 发现 if agent_type != 判断"
    FAIL_COUNT=$((FAIL_COUNT + 1))
elif grep -n 'agent_type in \[' server/services/chat_service.py; then
    echo "❌ 发现 agent_type in [ 判断"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo "✓ 没有发现 agent_type 条件判断"
    PASS_COUNT=$((PASS_COUNT + 1))
fi

# 检查应该存在的模式
if grep -q 'selected_agent.configure(' server/services/chat_service.py && \
   grep -q 'selected_agent.supports_kg_extraction()' server/services/chat_service.py; then
    echo "✓ 使用了多态接口"
else
    echo "❌ 缺少多态接口调用"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 2: BaseAgent 新增方法
echo ""
echo "[测试 2] 检查 BaseAgent 是否有新增方法"
echo "-----------------------------------------"

if grep -q 'def configure(' graphrag_agent/agents/base.py; then
    echo "✓ 找到 configure() 方法"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ 未找到 configure() 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def ask_with_thinking(' graphrag_agent/agents/base.py; then
    echo "✓ 找到 ask_with_thinking() 方法"
else
    echo "❌ 未找到 ask_with_thinking() 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def supports_kg_extraction(' graphrag_agent/agents/base.py; then
    echo "✓ 找到 supports_kg_extraction() 方法"
else
    echo "❌ 未找到 supports_kg_extraction() 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 3: DeepResearchAgent 重写方法
echo ""
echo "[测试 3] 检查 DeepResearchAgent 重写了多态方法"
echo "-----------------------------------------"

if grep -q 'def configure(' graphrag_agent/agents/deep_research_agent.py; then
    echo "✓ DeepResearchAgent 重写了 configure()"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ DeepResearchAgent 未重写 configure()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def supports_kg_extraction(' graphrag_agent/agents/deep_research_agent.py; then
    echo "✓ DeepResearchAgent 重写了 supports_kg_extraction()"
else
    echo "❌ DeepResearchAgent 未重写 supports_kg_extraction()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'return False' graphrag_agent/agents/deep_research_agent.py | grep -A 5 'supports_kg_extraction' > /dev/null; then
    echo "✓ supports_kg_extraction() 返回 False"
else
    echo "⚠️  无法确认 supports_kg_extraction() 返回值"
fi

# 测试 4: AgentManager TTL 机制
echo ""
echo "[测试 4] 检查 AgentManager TTL 机制"
echo "-----------------------------------------"

if grep -q 'def __init__.*ttl:' server/services/agent_service.py; then
    echo "✓ AgentManager 支持 ttl 参数"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ AgentManager 缺少 ttl 参数"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def _cleanup_expired' server/services/agent_service.py; then
    echo "✓ 找到 _cleanup_expired() 方法"
else
    echo "❌ 未找到 _cleanup_expired() 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'self.access_times' server/services/agent_service.py; then
    echo "✓ 有访问时间追踪"
else
    echo "❌ 缺少访问时间追踪"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def _start_background_cleanup' server/services/agent_service.py; then
    echo "✓ 找到后台清理任务"
else
    echo "❌ 未找到后台清理任务"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 5: Orchestrator 快速通道
echo ""
echo "[测试 5] 检查 Orchestrator 快速通道"
echo "-----------------------------------------"

if grep -q 'enable_fast_path' graphrag_agent/agents/multi_agent/orchestrator.py; then
    echo "✓ 找到 enable_fast_path 配置"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ 未找到 enable_fast_path 配置"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def _is_simple_query' graphrag_agent/agents/multi_agent/orchestrator.py; then
    echo "✓ 找到 _is_simple_query() 方法"
else
    echo "❌ 未找到 _is_simple_query() 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q 'def _execute_fast_path' graphrag_agent/agents/multi_agent/orchestrator.py; then
    echo "✓ 找到 _execute_fast_path() 方法"
else
    echo "❌ 未找到 _execute_fast_path() 方法"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q '进入快速通道' graphrag_agent/agents/multi_agent/orchestrator.py; then
    echo "✓ 找到快速通道日志"
else
    echo "❌ 未找到快速通道日志"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# 测试 6: ResearchExecutor 引用提取
echo ""
echo "[测试 6] 检查 ResearchExecutor 引用提取优化"
echo "-----------------------------------------"

if grep -q '策略 1' graphrag_agent/agents/multi_agent/executor/research_executor.py; then
    echo "✓ 找到策略 1（结构化数据优先）"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ 未找到策略 1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q '策略 2' graphrag_agent/agents/multi_agent/executor/research_executor.py; then
    echo "✓ 找到策略 2（兼容旧格式）"
else
    echo "❌ 未找到策略 2"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q '策略 3' graphrag_agent/agents/multi_agent/executor/research_executor.py; then
    echo "✓ 找到策略 3（正则兜底）"
else
    echo "❌ 未找到策略 3"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if grep -q '正则表达式解析（不稳定）' graphrag_agent/agents/multi_agent/executor/research_executor.py; then
    echo "✓ 找到正则兜底警告"
else
    echo "❌ 未找到正则兜底警告"
    FAIL_COUNT=$((FAIL_COUNT + 1))
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
    exit 0
else
    echo "⚠️  部分验证失败 ($FAIL_COUNT/$TOTAL)"
    exit 1
fi
