#!/bin/bash
# 快速启动脚本 - 仅启动 Neo4j

echo "🐳 启动 Neo4j 数据库"
echo "===================="
echo ""

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo "❌ Docker 未运行"
    echo ""
    echo "请先启动 Docker Desktop："
    echo "  方法1: 在「应用程序」中找到 Docker 并双击启动"
    echo "  方法2: 执行命令: open -a Docker"
    echo "  方法3: 按 Command+Space 搜索 'Docker' 并打开"
    echo ""

    read -p "是否现在尝试自动启动 Docker? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在启动 Docker Desktop..."
        open -a Docker

        echo "等待 Docker 启动（最多60秒）..."
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
    else
        echo "请手动启动 Docker Desktop 后重新运行此脚本"
        exit 1
    fi
fi

echo "✅ Docker 运行正常"
echo ""

# 启动 Neo4j
echo "启动 Neo4j 容器..."
docker compose up -d

sleep 3

# 检查 Neo4j 是否启动成功
if docker ps | grep -q neo4j; then
    echo ""
    echo "🎉 Neo4j 启动成功！"
    echo ""
    echo "访问信息："
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - Bolt 端口: 7687"
    echo "  - 默认用户名: neo4j"
    echo "  - 默认密码: 12345678"
    echo ""
else
    echo ""
    echo "❌ Neo4j 启动失败"
    echo ""
    echo "请检查 Docker 日志："
    echo "  docker compose logs neo4j"
    exit 1
fi
