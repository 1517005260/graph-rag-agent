#!/bin/bash
# 在 Docker 容器中构建知识图谱

set -e

echo "🏗️  使用 Docker 构建知识图谱..."

# 确保 Neo4j 在运行
echo "📊 检查 Neo4j 状态..."
docker compose up -d neo4j
sleep 10

# 运行构建
echo "🔨 开始构建..."
docker compose run --rm \
  -e NEO4J_URI=neo4j://neo4j:7687 \
  backend \
  python graphrag_agent/integrations/build/main.py

echo "✅ 知识图谱构建完成！"
