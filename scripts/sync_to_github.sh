#!/bin/bash

# ========================================
# Git 同步到 GitHub 脚本
# ========================================

set -e

echo "🔍 检查当前状态..."
echo ""

# 检查当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "📌 当前分支: $CURRENT_BRANCH"
echo ""

# 检查是否有未提交的修改
if [[ -n $(git status -s) ]]; then
    echo "📝 发现未提交的修改："
    git status -s
    echo ""

    # 询问是否提交
    read -p "是否要提交这些修改？(y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 显示修改详情
        echo ""
        echo "📋 修改详情："
        git diff --stat
        echo ""

        # 添加所有修改
        git add .
        echo "✅ 已暂存所有修改"
        echo ""

        # 输入提交信息
        read -p "请输入提交信息: " COMMIT_MSG

        if [[ -z "$COMMIT_MSG" ]]; then
            echo "❌ 提交信息不能为空"
            exit 1
        fi

        # 提交
        git commit -m "$COMMIT_MSG"
        echo "✅ 已提交修改"
        echo ""
    else
        echo "❌ 已取消提交"
        exit 0
    fi
else
    echo "✅ 工作区干净，没有需要提交的修改"
    echo ""
fi

# 检查是否有未推送的提交
UNPUSHED=$(git log origin/$CURRENT_BRANCH..$CURRENT_BRANCH --oneline)

if [[ -n "$UNPUSHED" ]]; then
    echo "📤 发现未推送的提交："
    echo "$UNPUSHED"
    echo ""

    # 推送
    echo "🚀 正在推送到 GitHub..."
    git push -u origin $CURRENT_BRANCH

    # 重试逻辑（如果失败）
    RETRY_COUNT=0
    MAX_RETRIES=4
    RETRY_DELAY=2

    while [ $? -ne 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "⚠️  推送失败，等待 ${RETRY_DELAY} 秒后重试..."
        sleep $RETRY_DELAY

        RETRY_COUNT=$((RETRY_COUNT + 1))
        RETRY_DELAY=$((RETRY_DELAY * 2))

        echo "🔄 重试 $RETRY_COUNT/$MAX_RETRIES..."
        git push -u origin $CURRENT_BRANCH
    done

    if [ $? -eq 0 ]; then
        echo "✅ 成功推送到 GitHub！"
    else
        echo "❌ 推送失败，请检查网络连接或手动推送"
        exit 1
    fi
else
    echo "✅ 所有提交已同步到 GitHub"
fi

echo ""
echo "🎉 同步完成！"
echo ""
echo "📊 最近的提交："
git log --oneline -5
