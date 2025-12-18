#!/bin/bash
#
# RAGAS 评估流水线快速开始脚本
#
# 使用方式：
#   chmod +x quick_start.sh
#   ./quick_start.sh

set -e

echo "======================================"
echo "  RAGAS 评估流水线快速开始"
echo "======================================"
echo

# 1. 检查Python环境
echo "📋 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python 3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION"
echo

# 2. 安装依赖
echo "📦 安装评估依赖..."
pip install -q ragas datasets rich pandas numpy
echo "✅ 依赖安装完成"
echo

# 3. 检查测试集
echo "📊 检查测试集..."
if [ ! -f "golden_set.json" ]; then
    echo "❌ 错误：未找到 golden_set.json"
    exit 1
fi

TEST_COUNT=$(python3 -c "import json; data=json.load(open('golden_set.json')); print(len(data['test_cases']))")
echo "✅ 找到 $TEST_COUNT 个测试用例"
echo

# 4. 运行评估
echo "🚀 开始评估..."
echo

# 选择要评估的 Agent
echo "请选择要评估的 Agent:"
echo "  1) Naive RAG Agent"
echo "  2) Graph Agent"
echo "  3) Hybrid Agent（推荐）"
echo "  4) Deep Research Agent"
echo

read -p "请输入选项 (1-4, 默认 3): " choice
choice=${choice:-3}

case $choice in
    1) AGENT="naive_rag" ;;
    2) AGENT="graph_agent" ;;
    3) AGENT="hybrid_agent" ;;
    4) AGENT="deep_research" ;;
    *) echo "无效选项，使用默认值：hybrid_agent"; AGENT="hybrid_agent" ;;
esac

echo
echo "开始评估 $AGENT..."
echo

# 运行评估
python3 run_eval.py --agent "$AGENT" --metrics all

# 5. 显示结果
echo
echo "======================================"
echo "  评估完成！"
echo "======================================"
echo

# 查找最新的报告文件
LATEST_CSV=$(ls -t results/eval_results_*.csv 2>/dev/null | head -1)
LATEST_JSON=$(ls -t results/eval_report_*.json 2>/dev/null | head -1)

if [ -n "$LATEST_CSV" ]; then
    echo "📄 详细结果: $LATEST_CSV"
fi

if [ -n "$LATEST_JSON" ]; then
    echo "📊 汇总报告: $LATEST_JSON"
    echo
    echo "主要指标:"
    python3 -c "
import json
with open('$LATEST_JSON') as f:
    data = json.load(f)
    for metric, stats in data['summary'].items():
        print(f\"  {metric:20s}: {stats['mean']:.4f} (平均值)\")
"
fi

echo
echo "💡 提示："
echo "  - 查看详细结果: cat $LATEST_CSV"
echo "  - 查看完整报告: cat $LATEST_JSON"
echo "  - 重新运行评估: python3 run_eval.py --agent $AGENT"
echo
