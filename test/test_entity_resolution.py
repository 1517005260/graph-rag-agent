"""
实体对齐功能测试脚本

功能：
- 测试实体相似度计算
- 预览可能重复的实体
- 执行实体合并
- 验证合并结果

使用方法：
    python test/test_entity_resolution.py --preview      # 预览重复实体
    python test/test_entity_resolution.py --resolve      # 执行对齐
    python test/test_entity_resolution.py --threshold 0.85  # 自定义阈值
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from graphrag_agent.graph.processing import EntityResolver
from graphrag_agent.config.neo4jdb import get_db_manager

console = Console()


def test_similarity():
    """测试相似度计算"""
    console.print("\n[bold cyan]测试 1: 字符串相似度计算[/bold cyan]")

    try:
        from Levenshtein import ratio
        similarity_func = ratio
        console.print("[green]✓ 使用 python-Levenshtein 库[/green]")
    except ImportError:
        from difflib import SequenceMatcher

        def similarity_func(s1, s2):
            return SequenceMatcher(None, s1, s2).ratio()

        console.print("[yellow]⚠ 使用 difflib.SequenceMatcher（较慢）[/yellow]")
        console.print("[yellow]建议安装：pip install python-Levenshtein[/yellow]")

    # 测试示例
    test_cases = [
        ("国家奖学金", "国家奖学金", 1.0),
        ("国家奖学金", "国家励志奖学金", 0.67),
        ("优秀学生", "优秀学生干部", 0.67),
        ("学生会", "学生会主席", 0.6),
        ("计算机科学", "计算机科学与技术", 0.67),
    ]

    table = Table(title="相似度测试结果")
    table.add_column("实体1", style="cyan")
    table.add_column("实体2", style="magenta")
    table.add_column("计算相似度", style="green")
    table.add_column("预期相似度", style="yellow")

    for s1, s2, expected in test_cases:
        similarity = similarity_func(s1, s2)
        table.add_row(s1, s2, f"{similarity:.2f}", f"{expected:.2f}")

    console.print(table)


def preview_duplicates(threshold: float = 0.9, limit: int = 10):
    """预览可能重复的实体"""
    console.print(f"\n[bold cyan]测试 2: 预览可能重复的实体（阈值: {threshold}）[/bold cyan]")

    try:
        resolver = EntityResolver(threshold=threshold)

        # 获取统计信息
        stats = resolver.get_statistics()
        console.print(f"[blue]图谱统计：{stats['total_entities']} 个实体[/blue]")

        # 预览重复实体
        duplicates = resolver.preview_duplicates(limit=limit)

        if not duplicates:
            console.print("[green]✅ 未发现重复实体[/green]")
            return

        console.print(f"[yellow]发现 {len(duplicates)} 组可能重复的实体：[/yellow]")

        for idx, group in enumerate(duplicates, 1):
            panel_content = []
            panel_content.append(f"[bold]合并目标：[/bold] {group['merge_to']}")
            panel_content.append(f"[bold]实体列表：[/bold]")
            for entity in group['entities']:
                panel_content.append(f"  - {entity}")

            console.print(
                Panel(
                    "\n".join(panel_content),
                    title=f"[bold]组 {idx}[/bold]",
                    border_style="yellow"
                )
            )

        return len(duplicates)

    except Exception as e:
        console.print(f"[red]❌ 预览失败: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 0


def resolve_entities(threshold: float = 0.9, entity_type: str = None):
    """执行实体对齐"""
    console.print(f"\n[bold cyan]测试 3: 执行实体对齐（阈值: {threshold}）[/bold cyan]")

    try:
        resolver = EntityResolver(threshold=threshold)

        # 执行对齐
        result = resolver.resolve_entities(entity_type=entity_type)

        if result["status"] == "success":
            console.print("\n[bold green]✅ 实体对齐成功[/bold green]")

            # 显示结果
            table = Table(title="对齐结果")
            table.add_column("指标", style="cyan")
            table.add_column("数值", style="green")

            table.add_row("总实体数", str(result["total_entities"]))
            table.add_row("合并组数", str(result["merge_groups"]))
            table.add_row("成功合并", str(result["merged_count"]))
            table.add_row("相似度阈值", str(result["threshold"]))

            console.print(table)

            # 显示统计变化
            stats_after = resolver.get_statistics()
            console.print(f"\n[blue]对齐后实体数：{stats_after['total_entities']}[/blue]")

            return result
        else:
            console.print(f"[red]❌ 实体对齐失败: {result}[/red]")
            return None

    except Exception as e:
        console.print(f"[red]❌ 执行失败: {e}[/red]")
        import traceback

        traceback.print_exc()
        return None


def verify_neo4j_connection():
    """验证 Neo4j 连接"""
    console.print("\n[bold cyan]验证 Neo4j 连接[/bold cyan]")

    try:
        db_manager = get_db_manager()
        query = "RETURN 1 as test"
        result = db_manager.graph.query(query)

        if result:
            console.print("[green]✅ Neo4j 连接成功[/green]")
            return True
        else:
            console.print("[red]❌ Neo4j 连接失败[/red]")
            return False

    except Exception as e:
        console.print(f"[red]❌ Neo4j 连接错误: {e}[/red]")
        return False


def check_apoc_installation():
    """检查 APOC 插件是否安装"""
    console.print("\n[bold cyan]检查 APOC 插件[/bold cyan]")

    try:
        db_manager = get_db_manager()
        query = "CALL apoc.help('refactor.mergeNodes')"
        result = db_manager.graph.query(query)

        if result:
            console.print("[green]✅ APOC 插件已安装[/green]")
            return True
        else:
            console.print("[red]❌ APOC 插件未安装或不可用[/red]")
            console.print(
                "[yellow]请安装 APOC 插件：https://neo4j.com/labs/apoc/[/yellow]"
            )
            return False

    except Exception as e:
        console.print(f"[red]❌ APOC 检查失败: {e}[/red]")
        console.print(
            "[yellow]请确保 Neo4j 已安装 APOC 插件[/yellow]"
        )
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实体对齐功能测试")

    parser.add_argument(
        "--preview",
        action="store_true",
        help="预览可能重复的实体（不执行合并）"
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="执行实体对齐（合并重复实体）"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="相似度阈值（0-1），默认 0.9"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="预览时显示的组数限制，默认 10"
    )
    parser.add_argument(
        "--entity-type",
        type=str,
        default=None,
        help="指定实体类型（可选）"
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="运行所有测试"
    )

    args = parser.parse_args()

    # 显示标题
    console.print(
        Panel(
            "[bold cyan]实体对齐功能测试[/bold cyan]\n"
            "测试基于 Levenshtein 距离的实体去重功能",
            border_style="cyan"
        )
    )

    # 验证环境
    if not verify_neo4j_connection():
        console.print("\n[red]❌ 无法连接到 Neo4j，请检查配置[/red]")
        return 1

    if not check_apoc_installation():
        console.print("\n[red]❌ 需要安装 APOC 插件才能使用实体合并功能[/red]")
        return 1

    # 执行测试
    if args.all_tests or (not args.preview and not args.resolve):
        # 默认运行所有测试
        test_similarity()
        dup_count = preview_duplicates(args.threshold, args.limit)

        if dup_count > 0:
            console.print("\n[yellow]⚠️  发现重复实体，使用 --resolve 执行对齐[/yellow]")
    else:
        if args.preview:
            dup_count = preview_duplicates(args.threshold, args.limit)

            if dup_count > 0:
                console.print(
                    "\n[yellow]⚠️  发现重复实体，使用 --resolve 执行对齐[/yellow]"
                )

        if args.resolve:
            console.print(
                "\n[yellow]⚠️  即将执行实体合并，此操作不可逆！[/yellow]"
            )
            confirm = input("确认继续？(yes/no): ")

            if confirm.lower() in ["yes", "y"]:
                result = resolve_entities(args.threshold, args.entity_type)

                if result:
                    console.print("\n[bold green]✅ 实体对齐完成！[/bold green]")
                else:
                    console.print("\n[red]❌ 实体对齐失败[/red]")
                    return 1
            else:
                console.print("\n[yellow]已取消操作[/yellow]")

    console.print("\n[bold green]测试完成！[/bold green]")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
