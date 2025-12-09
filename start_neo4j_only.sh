#!/bin/bash
# 仅启动 Neo4j 容器的脚本

echo "🚀 启动 Neo4j 数据库（仅 Neo4j）"
echo "==================================="
echo ""

# 1. 检查 Docker 是否运行
echo "🐳 检查 Docker 状态..."
if ! docker info &> /dev/null; then
    echo "❌ Docker 未运行"
    echo ""
    echo "正在尝试启动 Docker Desktop..."
    open -a Docker

    echo "等待 Docker 启动（最多 60 秒）..."
    for i in {1..30}; do
        sleep 2
        if docker info &> /dev/null; then
            echo "✅ Docker 已启动"
            break
        fi
        echo -n "."
    done
    echo ""

    if ! docker info &> /dev/null; then
        echo "❌ Docker 启动超时"
        echo "请手动打开 Docker Desktop，等待其完全启动后重新运行此脚本"
        exit 1
    fi
fi

echo "✅ Docker 运行正常"
echo ""

# 2. 停止所有现有容器
echo "🛑 停止现有容器..."
docker compose down 2>/dev/null

sleep 2

# 3. 只启动 Neo4j
echo "📊 启动 Neo4j 容器..."
docker compose up -d neo4j

sleep 5

# 4. 验证 Neo4j 是否启动成功
if docker ps | grep -q neo4j; then
    echo ""
    echo "🎉 Neo4j 启动成功！"
    echo ""
    echo "Neo4j 信息："
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - Bolt 端口: 7687"
    echo "  - 用户名: neo4j"
    echo "  - 密码: 12345678"
    echo ""
    echo "下一步："
    echo "  1. 打开新终端，运行: ./start_backend.sh"
    echo "  2. 再打开新终端，运行: streamlit run frontend/app.py"
    echo ""
else
    echo ""
    echo "❌ Neo4j 启动失败"
    echo ""
    echo "查看日志："
    docker compose logs neo4j
    exit 1
fi
