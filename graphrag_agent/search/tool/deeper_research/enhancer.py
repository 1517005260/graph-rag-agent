from typing import Dict, List


def enhance_search_with_coe(tool, query: str, keywords: Dict[str, List[str]]):
    """
    使用社区感知搜索结合 Chain-of-Exploration 结果对查询进行增强。

    该函数与旧版 `_enhance_search_with_coe` 逻辑一致，通过接收 `tool` 实例
    来复用其依赖对象，避免主类体积过大。
    """
    cache_key = f"coe_search:{query}"
    if hasattr(tool, "_coe_cache") and cache_key in tool._coe_cache:
        return tool._coe_cache[cache_key]

    community_context = tool.community_search.enhance_search(query, keywords)
    search_strategy = community_context.get("search_strategy", {})

    focus_entities = search_strategy.get("focus_entities", [])
    if not focus_entities:
        focus_entities = keywords.get("high_level", []) + keywords.get("low_level", [])

    if focus_entities:
        coe_cache_key = f"coe:{query}:{','.join(focus_entities[:3])}"
        if hasattr(tool, "_specific_coe_cache") and coe_cache_key in tool._specific_coe_cache:
            exploration_results = tool._specific_coe_cache[coe_cache_key]
        else:
            exploration_results = tool.chain_explorer.explore(
                query,
                focus_entities[:3],
                max_steps=3,
            )
            if not hasattr(tool, "_specific_coe_cache"):
                tool._specific_coe_cache = {}
            tool._specific_coe_cache[coe_cache_key] = exploration_results

        community_context["exploration_results"] = exploration_results

        discovered_entities = []
        for step in exploration_results.get("exploration_path", []):
            if step["step"] > 0:
                discovered_entities.append(step["node_id"])

        if discovered_entities:
            search_strategy["discovered_entities"] = discovered_entities
            community_context["search_strategy"] = search_strategy

    if not hasattr(tool, "_coe_cache"):
        tool._coe_cache = {}
    tool._coe_cache[cache_key] = community_context

    return community_context
