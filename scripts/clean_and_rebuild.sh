#!/bin/bash

# ========================================
# 知识图谱清理和重建脚本
# ========================================

set -e  # 遇到错误立即退出

echo "🗑️  开始清理知识图谱和缓存..."
echo ""

# 1. 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误：请在项目根目录执行此脚本"
    exit 1
fi

# 2. 停止并重置 Neo4j
echo "📦 步骤 1/4: 重置 Neo4j 数据库..."
docker compose down -v
echo "   ✅ Neo4j 容器已停止并清理"

echo "   🔄 重新启动 Neo4j..."
docker compose up -d
echo "   ⏳ 等待 Neo4j 启动（20秒）..."
sleep 20
echo "   ✅ Neo4j 已重新启动"
echo ""

# 3. 清空缓存
echo "🧹 步骤 2/4: 清空缓存..."
rm -rf cache/
rm -rf cache/graph/
rm -rf cache/global/
echo "   ✅ 缓存已清空"
echo ""

# 4. 清空文件注册表
echo "📋 步骤 3/4: 清空文件注册表..."
rm -f file_registry.json
echo "   ✅ 文件注册表已清空"
echo ""

# 5. 清理 Python 缓存（可选）
echo "🐍 步骤 4/4: 清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "   ✅ Python 缓存已清理"
echo ""

echo "✅ 清理完成！"
echo ""
echo "📝 接下来的步骤："
echo "   1. 确保你的文档文件在 files/ 目录中"
echo "   2. 运行以下命令重新构建图谱："
echo ""
echo "   # 方式一：完整构建（推荐）"
echo "   python graphrag_agent/integrations/build/main.py"
echo ""
echo "   # 方式二：V2 增量构建（L0+L1）"
echo "   python graphrag_agent/integrations/build/incremental_update_v2.py --mode full"
echo ""
