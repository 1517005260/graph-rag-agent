#!/bin/bash
# 启动脚本 - 自动处理端口占用问题

echo "🚀 GraphRAG 系统启动脚本"
echo "========================"

# 检查并杀掉占用 8000 端口的进程
PORT=8000
echo "检查端口 $PORT ..."

# macOS 和 Linux 通用的方法
PID=$(lsof -ti:$PORT 2>/dev/null)

if [ ! -z "$PID" ]; then
    echo "⚠️  端口 $PORT 被进程 $PID 占用，正在停止..."
    kill -9 $PID 2>/dev/null
    sleep 1
    echo "✅ 已停止旧进程"
else
    echo "✅ 端口 $PORT 可用"
fi

# 启动后端服务
echo ""
echo "📡 启动后端服务..."
python server/main.py
