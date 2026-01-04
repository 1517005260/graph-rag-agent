#!/bin/bash
# 修复 file_registry.json 问题

echo "🔧 修复 file_registry.json"
echo "========================="
echo ""

# 检查 file_registry.json 是否存在
if [ -e "file_registry.json" ]; then
    # 检查是否是目录
    if [ -d "file_registry.json" ]; then
        echo "⚠️  发现 file_registry.json 是目录，正在删除..."
        rm -rf file_registry.json
        echo "✅ 已删除错误的目录"
    elif [ -f "file_registry.json" ]; then
        echo "✅ file_registry.json 已经是正确的文件"
    fi
else
    echo "ℹ️  file_registry.json 不存在"
fi

# 创建正确的文件
echo "{}" > file_registry.json
echo "✅ 已创建正确的 file_registry.json 文件"
echo ""

# 验证
echo "验证文件内容："
cat file_registry.json
echo ""
echo "✅ 修复完成！"
