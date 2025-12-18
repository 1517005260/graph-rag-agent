#!/bin/bash
# 完整系统启动脚本

echo "🤖 GraphRAG 完整系统启动"
echo "========================"
echo ""

# 1. 检查 Docker
echo "🐳 检查 Docker 状态..."
if ! docker info &> /dev/null; then
    echo "❌ Docker 未运行"
    echo ""
    echo "请先启动 Docker Desktop："
    echo "  方法1: 在应用程序中找到 Docker 并双击启动"
    echo "  方法2: 执行命令 'open -a Docker'"
    echo ""
    read -p "启动 Docker 后按回车继续..."

    # 等待 Docker 启动
    echo "等待 Docker 启动..."
    for i in {1..30}; do
        if docker info &> /dev/null; then
            echo "✅ Docker 已启动"
            break
        fi
        sleep 2
        echo -n "."
    done

    if ! docker info &> /dev/null; then
        echo ""
        echo "❌ Docker 启动超时，请手动启动 Docker Desktop 后重试"
        exit 1
    fi
fi

echo "✅ Docker 运行正常"
echo ""

# 2. 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "⚠️  未找到 conda，请确保已安装 Anaconda/Miniconda"
    exit 1
fi

# 检查是否在 graphrag 环境中
if [[ "$CONDA_DEFAULT_ENV" != "graphrag" ]]; then
    echo "⚠️  请先激活 graphrag 环境: conda activate graphrag"
    exit 1
fi

echo "✅ conda 环境检查通过"
echo ""

# 3. 启动 Neo4j
echo "📊 启动 Neo4j 数据库..."
docker compose up -d

sleep 3

# 检查 Neo4j 是否启动成功
if docker ps | grep -q neo4j; then
    echo "✅ Neo4j 启动成功"
else
    echo "❌ Neo4j 启动失败"
    exit 1
fi

echo ""

# 3. 处理端口占用
echo "🔍 检查端口占用..."

# 检查后端端口 8000
BACKEND_PORT=8000
BACKEND_PID=$(lsof -ti:$BACKEND_PORT 2>/dev/null)
if [ ! -z "$BACKEND_PID" ]; then
    echo "⚠️  端口 $BACKEND_PORT 被占用，停止旧进程 $BACKEND_PID ..."
    kill -9 $BACKEND_PID 2>/dev/null
    sleep 1
fi

# 检查前端端口 8501
FRONTEND_PORT=8501
FRONTEND_PID=$(lsof -ti:$FRONTEND_PORT 2>/dev/null)
if [ ! -z "$FRONTEND_PID" ]; then
    echo "⚠️  端口 $FRONTEND_PORT 被占用，停止旧进程 $FRONTEND_PID ..."
    kill -9 $FRONTEND_PID 2>/dev/null
    sleep 1
fi

echo "✅ 端口检查完成"
echo ""

# 4. 启动后端（后台）
echo "📡 启动后端服务（后台运行）..."
nohup python server/main.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ 后端服务已启动 (PID: $BACKEND_PID)"
echo "   日志文件: logs/backend.log"
echo ""

sleep 3

# 5. 启动前端
echo "🎨 启动前端界面..."
echo "   访问地址: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止前端服务"
echo ""

streamlit run frontend/app.py

# 当前端停止时，询问是否停止后端
echo ""
read -p "前端已停止，是否同时停止后端服务？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "停止后端服务..."
    kill $BACKEND_PID 2>/dev/null
    echo "✅ 后端服务已停止"
fi
