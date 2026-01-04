#!/usr/bin/env python3
"""
语义缓存测试脚本

测试基于向量相似度的查询缓存功能。

使用方法：
    python test/test_semantic_cache.py

功能：
    1. 单元测试 - 测试 SemanticCache 类的基本功能
    2. 相似度测试 - 测试语义相似问题的缓存命中
    3. 性能测试 - 测试缓存性能提升
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# 导入语义缓存
from server.utils.semantic_cache import SemanticCache, get_semantic_cache, reset_semantic_cache

# 导入 embeddings 模型
from graphrag_agent.models.get_models import get_embeddings_model

console = Console()


def test_basic_functionality():
    """测试基本功能"""
    console.print("\n[bold cyan]测试 1: 基本功能[/bold cyan]\n")

    # 创建缓存实例
    cache = SemanticCache(threshold=0.95, max_size=100)

    # 测试数据
    query1_vec = np.random.rand(1536).tolist()
    query2_vec = np.random.rand(1536).tolist()
    response1 = "这是第一个查询的响应"
    response2 = "这是第二个查询的响应"

    # 测试添加
    cache.add(query1_vec, response1, metadata={"query": "查询1"})
    cache.add(query2_vec, response2, metadata={"query": "查询2"})

    console.print(f"✅ 已添加 2 个缓存条目")
    console.print(f"缓存状态: {cache}")

    # 测试精确匹配
    result = cache.get(query1_vec)
    assert result == response1, "精确匹配失败"
    console.print(f"✅ 精确匹配测试通过")

    # 测试不相似的查询
    random_vec = np.random.rand(1536).tolist()
    result = cache.get(random_vec)
    assert result is None, "不应该命中缓存"
    console.print(f"✅ 未命中测试通过")

    # 测试统计信息
    stats = cache.get_stats()
    console.print(f"\n[bold]缓存统计:[/bold]")
    console.print(f"  大小: {stats['size']}/{stats['max_size']}")
    console.print(f"  命中: {stats['hits']}")
    console.print(f"  未命中: {stats['misses']}")
    console.print(f"  命中率: {stats['hit_rate']}")

    console.print("\n[green]✅ 基本功能测试通过[/green]")


def test_semantic_similarity():
    """测试语义相似度"""
    console.print("\n[bold cyan]测试 2: 语义相似度[/bold cyan]\n")

    # 重置并获取全局缓存
    reset_semantic_cache()
    cache = get_semantic_cache(threshold=0.90, max_size=100)

    # 获取 embeddings 模型
    console.print("正在加载 embeddings 模型...")
    embeddings_model = get_embeddings_model()
    console.print("✅ Embeddings 模型加载完成\n")

    # 准备测试查询
    queries = [
        ("旷课多少学时会被退学？", "旷课达到一定学时会被退学"),
        ("请问旷课多少小时会被开除？", None),  # 相似问题，应该命中缓存
        ("国家奖学金的申请条件是什么？", "国家奖学金需要满足多个条件"),
        ("完全不相关的问题", None),  # 不相似，不应该命中
    ]

    # 创建测试表格
    table = Table(title="语义相似度测试结果")
    table.add_column("查询", style="cyan")
    table.add_column("缓存状态", style="magenta")
    table.add_column("响应来源", style="green")

    for query, expected_response in queries:
        # 计算向量
        query_vec = embeddings_model.embed_query(query)

        # 如果有预期响应，先添加到缓存
        if expected_response:
            cache.add(query_vec, expected_response, metadata={"query": query})
            table.add_row(query, "已缓存", "原始查询")
        else:
            # 检查缓存
            cached_result = cache.get(query_vec)
            if cached_result:
                table.add_row(query, "命中", f"缓存: {cached_result[:30]}...")
            else:
                table.add_row(query, "未命中", "需要执行查询")

    console.print(table)

    # 显示统计信息
    stats = cache.get_stats()
    console.print(f"\n[bold]缓存统计:[/bold]")
    console.print(f"  命中率: {stats['hit_rate']}")
    console.print(f"  命中次数: {stats['hits']}")
    console.print(f"  未命中次数: {stats['misses']}")

    console.print("\n[green]✅ 语义相似度测试完成[/green]")


def test_performance():
    """测试性能提升"""
    console.print("\n[bold cyan]测试 3: 性能对比[/bold cyan]\n")

    # 重置缓存
    reset_semantic_cache()
    cache = get_semantic_cache(threshold=0.95, max_size=1000)

    # 获取 embeddings 模型
    embeddings_model = get_embeddings_model()

    # 准备测试数据
    num_queries = 100
    queries = [f"测试查询 {i}" for i in range(num_queries)]

    console.print(f"准备测试 {num_queries} 个查询...\n")

    # 第一轮：全部未命中（需要计算 embedding）
    console.print("[bold]第一轮：冷启动（全部未命中）[/bold]")
    start_time = time.time()

    for query in track(queries, description="执行查询"):
        query_vec = embeddings_model.embed_query(query)
        result = cache.get(query_vec)

        # 模拟 Agent 处理（未命中时）
        if result is None:
            response = f"这是 {query} 的响应"
            cache.add(query_vec, response)

    first_round_time = time.time() - start_time

    # 第二轮：全部命中（直接从缓存返回）
    console.print("\n[bold]第二轮：缓存命中（全部命中）[/bold]")
    start_time = time.time()

    hits = 0
    for query in track(queries, description="执行查询"):
        query_vec = embeddings_model.embed_query(query)
        result = cache.get(query_vec)

        if result:
            hits += 1

    second_round_time = time.time() - start_time

    # 显示结果
    console.print("\n[bold]性能对比结果:[/bold]\n")

    comparison_table = Table()
    comparison_table.add_column("轮次", style="cyan")
    comparison_table.add_column("总耗时", style="magenta")
    comparison_table.add_column("平均耗时", style="green")
    comparison_table.add_column("命中率", style="yellow")

    comparison_table.add_row(
        "第一轮（未命中）",
        f"{first_round_time:.2f}s",
        f"{first_round_time/num_queries*1000:.2f}ms",
        "0%"
    )
    comparison_table.add_row(
        "第二轮（命中）",
        f"{second_round_time:.2f}s",
        f"{second_round_time/num_queries*1000:.2f}ms",
        f"{hits/num_queries*100:.1f}%"
    )

    console.print(comparison_table)

    # 计算性能提升
    if second_round_time > 0:
        speedup = first_round_time / second_round_time
        console.print(f"\n[bold green]性能提升: {speedup:.2f}x[/bold green]")

    # 显示缓存统计
    stats = cache.get_stats()
    console.print(f"\n[bold]最终缓存统计:[/bold]")
    console.print(f"  缓存大小: {stats['size']}/{stats['max_size']}")
    console.print(f"  总请求数: {stats['total_requests']}")
    console.print(f"  命中率: {stats['hit_rate']}")

    console.print("\n[green]✅ 性能测试完成[/green]")


def test_threshold_sensitivity():
    """测试阈值敏感度"""
    console.print("\n[bold cyan]测试 4: 阈值敏感度[/bold cyan]\n")

    embeddings_model = get_embeddings_model()

    # 测试不同阈值
    thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]

    # 准备两个相似的查询
    query1 = "旷课多少学时会被退学？"
    query2 = "请问旷课多少小时会被开除？"

    vec1 = embeddings_model.embed_query(query1)
    vec2 = embeddings_model.embed_query(query2)

    # 计算实际相似度
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    actual_similarity = np.dot(vec1_np, vec2_np) / (
        np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
    )

    console.print(f"查询 1: {query1}")
    console.print(f"查询 2: {query2}")
    console.print(f"实际相似度: {actual_similarity:.4f}\n")

    # 测试不同阈值
    table = Table(title="不同阈值下的缓存命中情况")
    table.add_column("阈值", style="cyan")
    table.add_column("是否命中", style="magenta")
    table.add_column("说明", style="green")

    for threshold in thresholds:
        cache = SemanticCache(threshold=threshold, max_size=100)
        cache.add(vec1, "答案 1")
        result = cache.get(vec2)

        hit = "✅ 命中" if result else "❌ 未命中"
        explanation = "阈值过高，不匹配" if not result and actual_similarity < threshold else \
                      "相似度满足要求" if result else "阈值合适"

        table.add_row(f"{threshold:.2f}", hit, explanation)

    console.print(table)

    console.print(f"\n[bold]推荐阈值:[/bold]")
    console.print(f"  严格模式（高精度）: 0.95-0.99")
    console.print(f"  平衡模式（推荐）: 0.90-0.95")
    console.print(f"  宽松模式（高召回）: 0.85-0.90")

    console.print("\n[green]✅ 阈值敏感度测试完成[/green]")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold cyan]语义缓存测试套件[/bold cyan]\n\n"
        "测试基于向量相似度的查询缓存系统",
        border_style="cyan"
    ))

    try:
        # 运行所有测试
        test_basic_functionality()
        test_semantic_similarity()
        test_performance()
        test_threshold_sensitivity()

        console.print("\n" + "="*60)
        console.print("[bold green]🎉 所有测试通过！[/bold green]")
        console.print("="*60 + "\n")

        # 显示使用建议
        console.print(Panel.fit(
            "[bold]使用建议:[/bold]\n\n"
            "1. 生产环境推荐阈值: 0.95\n"
            "2. 建议最大缓存大小: 1000-5000\n"
            "3. 定期监控缓存命中率\n"
            "4. 生产环境可替换为 Redis Vector",
            title="💡 提示",
            border_style="yellow"
        ))

    except Exception as e:
        console.print(f"\n[bold red]❌ 测试失败: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
