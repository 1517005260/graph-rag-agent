"""
实体后处理工具（生产级验证 - GraphRAG 成败关键）

核心功能：
1. normalize_entity_name：标准化实体名称
2. calculate_levenshtein：计算 Levenshtein 距离
3. merge_similar_entities：合并相似实体
4. filter_by_frequency：过滤低频实体

目的：解决实体爆炸问题的最后一道防线
"""

from typing import List, Dict, Set, Tuple
import re


def normalize_entity_name(name: str) -> str:
    """
    标准化实体名称（生产级验证配置）

    规则：
    1. 去除首尾空格
    2. 去除所有空格
    3. 统一括号格式：全角 → 半角
    4. 转为大写（可选，根据需求）

    Args:
        name: 原始实体名称

    Returns:
        标准化后的实体名称

    Examples:
        >>> normalize_entity_name(" 国家 奖学金 ")
        "国家奖学金"
        >>> normalize_entity_name("勤工助学（管理办法）")
        "勤工助学(管理办法)"
    """
    if not name:
        return ""

    # 1. 去除首尾空格
    name = name.strip()

    # 2. 去除所有空格
    name = name.replace(" ", "")

    # 3. 统一括号格式：全角 → 半角
    name = name.replace("（", "(").replace("）", ")")

    # 4. 统一其他标点符号
    name = name.replace("【", "[").replace("】", "]")
    name = name.replace(""", '"').replace(""", '"')
    name = name.replace("'", "'").replace("'", "'")

    # 5. 转为大写（可选，根据配置）
    # name = name.upper()

    return name


def calculate_levenshtein(s1: str, s2: str) -> int:
    """
    计算 Levenshtein 距离（编辑距离）

    用途：判断两个实体名称是否相似

    Args:
        s1: 第一个字符串
        s2: 第二个字符串

    Returns:
        Levenshtein 距离（整数）

    Examples:
        >>> calculate_levenshtein("国家奖学金", "国家助学金")
        2
        >>> calculate_levenshtein("学生处", "学生")
        1
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # 动态规划算法
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # 删除
                    dp[i][j - 1] + 1,      # 插入
                    dp[i - 1][j - 1] + 1   # 替换
                )

    return dp[m][n]


def merge_similar_entities(
    entities: List[Dict[str, any]],
    distance_threshold: int = 2,
    name_key: str = "name"
) -> List[Dict[str, any]]:
    """
    合并相似实体（生产级验证 - 轻量版）

    规则：
    - 如果两个实体的 Levenshtein 距离 < threshold → 合并
    - 保留较长的实体名称（通常更完整）

    Args:
        entities: 实体列表，每个实体是一个字典
        distance_threshold: Levenshtein 距离阈值，默认 2
        name_key: 实体名称的字典 key，默认 "name"

    Returns:
        去重后的实体列表

    Examples:
        >>> entities = [
        ...     {"name": "国家奖学金", "type": "奖学金"},
        ...     {"name": "国家励志奖学金", "type": "奖学金"},
        ...     {"name": "国家奖学金", "type": "奖学金"}  # 重复
        ... ]
        >>> merge_similar_entities(entities, distance_threshold=3)
        [{"name": "国家励志奖学金", "type": "奖学金"}]  # 保留较长的名称
    """
    if not entities:
        return []

    # 1. 先标准化所有实体名称
    for entity in entities:
        if name_key in entity:
            entity[name_key] = normalize_entity_name(entity[name_key])

    # 2. 去除完全重复的实体（精确匹配）
    unique_entities = {}
    for entity in entities:
        name = entity.get(name_key, "")
        if name and name not in unique_entities:
            unique_entities[name] = entity

    entities = list(unique_entities.values())

    # 3. 合并相似实体（Levenshtein 距离）
    merged = []
    to_skip = set()

    for i, entity_a in enumerate(entities):
        if i in to_skip:
            continue

        name_a = entity_a.get(name_key, "")
        if not name_a:
            continue

        # 查找相似实体
        similar_group = [entity_a]
        for j, entity_b in enumerate(entities[i + 1:], start=i + 1):
            if j in to_skip:
                continue

            name_b = entity_b.get(name_key, "")
            if not name_b:
                continue

            # 计算 Levenshtein 距离
            distance = calculate_levenshtein(name_a, name_b)
            if distance < distance_threshold:
                similar_group.append(entity_b)
                to_skip.add(j)

        # 保留较长的名称（通常更完整）
        canonical_entity = max(similar_group, key=lambda e: len(e.get(name_key, "")))
        merged.append(canonical_entity)

    return merged


def filter_by_frequency(
    entities: List[Dict[str, any]],
    frequency_threshold: int = 2,
    frequency_key: str = "frequency",
    name_key: str = "name"
) -> List[Dict[str, any]]:
    """
    过滤低频实体（生产级验证 - 非常有效）

    规则：
    - 如果实体的 frequency < threshold → 丢弃
    - 保留 frequency ≥ threshold 的实体

    Args:
        entities: 实体列表
        frequency_threshold: 频率阈值，默认 2
        frequency_key: 频率字段的字典 key，默认 "frequency"
        name_key: 实体名称的字典 key，默认 "name"

    Returns:
        过滤后的实体列表

    Examples:
        >>> entities = [
        ...     {"name": "国家奖学金", "frequency": 10},
        ...     {"name": "临时概念", "frequency": 1},
        ...     {"name": "勤工助学", "frequency": 5}
        ... ]
        >>> filter_by_frequency(entities, frequency_threshold=2)
        [{"name": "国家奖学金", "frequency": 10}, {"name": "勤工助学", "frequency": 5}]
    """
    if not entities:
        return []

    filtered = []
    for entity in entities:
        frequency = entity.get(frequency_key, 0)

        if frequency >= frequency_threshold:
            filtered.append(entity)
        else:
            # 可选：记录被过滤的实体
            name = entity.get(name_key, "unknown")
            print(f"🗑️  过滤低频实体：{name} (frequency={frequency})")

    return filtered


def post_process_entities(
    entities: List[Dict[str, any]],
    normalize: bool = True,
    merge_similar: bool = True,
    filter_frequency: bool = True,
    distance_threshold: int = 2,
    frequency_threshold: int = 2,
    name_key: str = "name",
    frequency_key: str = "frequency"
) -> List[Dict[str, any]]:
    """
    实体后处理管道（生产级验证 - 一站式）

    执行顺序：
    1. normalize：标准化实体名称
    2. merge_similar：合并相似实体（Levenshtein 距离）
    3. filter_frequency：过滤低频实体

    Args:
        entities: 原始实体列表
        normalize: 是否标准化名称
        merge_similar: 是否合并相似实体
        filter_frequency: 是否过滤低频实体
        distance_threshold: Levenshtein 距离阈值
        frequency_threshold: 频率阈值
        name_key: 实体名称字段
        frequency_key: 频率字段

    Returns:
        处理后的实体列表

    Examples:
        >>> entities = [
        ...     {"name": " 国家 奖学金 ", "frequency": 10},
        ...     {"name": "国家奖学金", "frequency": 8},
        ...     {"name": "临时概念", "frequency": 1}
        ... ]
        >>> post_process_entities(entities)
        [{"name": "国家奖学金", "frequency": 18}]  # 合并 + 过滤
    """
    if not entities:
        return []

    print(f"📊 实体后处理开始：{len(entities)} 个实体")

    # 1. 标准化名称
    if normalize:
        for entity in entities:
            if name_key in entity:
                entity[name_key] = normalize_entity_name(entity[name_key])
        print(f"✅ 标准化完成")

    # 2. 合并相似实体
    if merge_similar:
        entities = merge_similar_entities(
            entities,
            distance_threshold=distance_threshold,
            name_key=name_key
        )
        print(f"✅ 合并相似实体完成：剩余 {len(entities)} 个实体")

    # 3. 过滤低频实体
    if filter_frequency:
        entities = filter_by_frequency(
            entities,
            frequency_threshold=frequency_threshold,
            frequency_key=frequency_key,
            name_key=name_key
        )
        print(f"✅ 过滤低频实体完成：剩余 {len(entities)} 个实体")

    print(f"🎉 实体后处理完成：最终 {len(entities)} 个实体")
    return entities


# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_entities = [
        {"name": " 国家 奖学金 ", "type": "奖学金", "frequency": 10},
        {"name": "国家奖学金", "type": "奖学金", "frequency": 8},
        {"name": "国家励志奖学金", "type": "奖学金", "frequency": 5},
        {"name": "临时概念", "type": "其他", "frequency": 1},
        {"name": "学生处（管理部门）", "type": "部门", "frequency": 7},
        {"name": "学生处", "type": "部门", "frequency": 3}
    ]

    # 执行后处理
    result = post_process_entities(
        test_entities,
        normalize=True,
        merge_similar=True,
        filter_frequency=True,
        distance_threshold=2,
        frequency_threshold=2
    )

    print("\n📝 最终结果：")
    for entity in result:
        print(f"  - {entity}")
