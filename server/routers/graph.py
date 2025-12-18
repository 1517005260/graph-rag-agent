"""
图谱可视化交互 API

提供知识图谱探索、节点关系查询、路径分析等功能。

API Endpoints:
    - GET  /graph/overview          - 获取图谱概览
    - GET  /graph/subgraph          - 获取节点周围的子图
    - GET  /graph/entity/{entity_id} - 获取实体详细信息
    - GET  /graph/path              - 查询两个实体之间的路径
    - GET  /graph/shortest-path     - 查询最短路径
    - GET  /graph/common-neighbors  - 查询共同邻居
    - GET  /graph/influence         - 查询实体影响范围
    - GET  /graph/community         - 查询实体所属社区
    - GET  /graph/cycles            - 查询实体的环路
    - GET  /source/chunk            - 获取原文片段
    - GET  /source/file-info        - 获取文件信息
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from server_config.database import get_db_manager
import traceback

# 导入 kg_service 中的功能
from services.kg_service import (
    get_knowledge_graph,
    get_knowledge_graph_for_ids,
    get_source_content,
    get_source_file_info,
    get_shortest_path,
    get_all_paths,
    get_common_neighbors,
    get_entity_influence,
    get_simplified_community,
    get_entity_cycles
)

router = APIRouter(prefix="/graph", tags=["图谱探索"])

# 获取数据库连接
db_manager = get_db_manager()
driver = db_manager.driver


@router.get("/overview", summary="获取图谱概览")
async def get_graph_overview(
    limit: int = Query(100, description="节点数量限制", ge=1, le=1000),
    query: Optional[str] = Query(None, description="搜索关键词（可选）")
) -> Dict[str, Any]:
    """
    获取知识图谱概览

    参数:
        - limit: 返回的最大节点数（1-1000）
        - query: 可选的搜索关键词，用于过滤节点

    返回:
        - nodes: 节点列表
        - links: 边列表
    """
    try:
        result = get_knowledge_graph(limit=limit, query=query)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "node_count": len(result.get("nodes", [])),
                "link_count": len(result.get("links", []))
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取图谱概览失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取图谱概览失败: {str(e)}")


@router.get("/subgraph", summary="获取实体子图")
async def get_subgraph(
    entity_id: str = Query(..., description="中心实体ID"),
    hops: int = Query(1, description="扩展跳数（1-3）", ge=1, le=3)
) -> Dict[str, Any]:
    """
    获取指定实体周围的子图

    参数:
        - entity_id: 中心实体的ID
        - hops: 扩展的跳数（1-3跳）

    返回:
        - nodes: 子图节点列表
        - links: 子图边列表
        - center: 中心实体ID
    """
    try:
        # 使用 get_entity_influence 获取实体周围的子图
        result = get_entity_influence(driver, entity_id, max_depth=hops)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "center_entity": entity_id,
                "hops": hops,
                "node_count": len(result.get("nodes", [])),
                "link_count": len(result.get("links", []))
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取子图失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取子图失败: {str(e)}")


@router.get("/entity/{entity_id}", summary="获取实体详细信息")
async def get_entity_details(
    entity_id: str,
    include_neighbors: bool = Query(False, description="是否包含邻居节点")
) -> Dict[str, Any]:
    """
    获取实体的详细信息

    参数:
        - entity_id: 实体ID
        - include_neighbors: 是否包含一跳邻居

    返回:
        实体详细信息及其邻居（如果请求）
    """
    try:
        # 查询实体基本信息
        query = """
        MATCH (e:__Entity__)
        WHERE e.id = $entity_id
        RETURN e.id AS id,
               e.description AS description,
               labels(e) AS labels,
               properties(e) AS properties
        """

        result = driver.execute_query(query, {"entity_id": entity_id})

        if not result.records or len(result.records) == 0:
            raise HTTPException(status_code=404, detail=f"实体 '{entity_id}' 不存在")

        record = result.records[0]
        entity_info = {
            "id": record.get("id"),
            "description": record.get("description", ""),
            "labels": [lbl for lbl in record.get("labels", []) if lbl != "__Entity__"],
            "properties": dict(record.get("properties", {}))
        }

        # 如果需要邻居信息
        neighbors = None
        if include_neighbors:
            neighbor_result = get_entity_influence(driver, entity_id, max_depth=1)
            neighbors = {
                "nodes": neighbor_result.get("nodes", []),
                "links": neighbor_result.get("links", []),
                "stats": neighbor_result.get("influence_stats", {})
            }

        return {
            "status": "success",
            "data": {
                "entity": entity_info,
                "neighbors": neighbors
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取实体详情失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取实体详情失败: {str(e)}")


@router.get("/shortest-path", summary="查询最短路径")
async def query_shortest_path(
    source: str = Query(..., description="起始实体ID"),
    target: str = Query(..., description="目标实体ID"),
    max_hops: int = Query(3, description="最大跳数（1-5）", ge=1, le=5)
) -> Dict[str, Any]:
    """
    查询两个实体之间的最短路径

    参数:
        - source: 起始实体ID
        - target: 目标实体ID
        - max_hops: 最大搜索深度（1-5跳）

    返回:
        最短路径的节点和边
    """
    try:
        result = get_shortest_path(driver, source, target, max_hops=max_hops)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "source": source,
                "target": target,
                "path_length": result.get("path_length", 0),
                "max_hops": max_hops
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询最短路径失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询最短路径失败: {str(e)}")


@router.get("/path", summary="查询所有路径")
async def query_all_paths(
    source: str = Query(..., description="起始实体ID"),
    target: str = Query(..., description="目标实体ID"),
    max_depth: int = Query(3, description="最大路径深度（1-5）", ge=1, le=5)
) -> Dict[str, Any]:
    """
    查询两个实体之间的所有路径（限制最多10条）

    参数:
        - source: 起始实体ID
        - target: 目标实体ID
        - max_depth: 最大路径深度（1-5跳）

    返回:
        所有路径的节点、边和路径描述
    """
    try:
        result = get_all_paths(driver, source, target, max_depth=max_depth)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "source": source,
                "target": target,
                "path_count": result.get("path_count", 0),
                "max_depth": max_depth
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询所有路径失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询所有路径失败: {str(e)}")


@router.get("/common-neighbors", summary="查询共同邻居")
async def query_common_neighbors(
    entity_a: str = Query(..., description="实体A的ID"),
    entity_b: str = Query(..., description="实体B的ID")
) -> Dict[str, Any]:
    """
    查询两个实体的共同邻居

    参数:
        - entity_a: 实体A的ID
        - entity_b: 实体B的ID

    返回:
        共同邻居节点及其连接关系
    """
    try:
        result = get_common_neighbors(driver, entity_a, entity_b)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "entity_a": entity_a,
                "entity_b": entity_b,
                "neighbor_count": result.get("neighbor_count", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询共同邻居失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询共同邻居失败: {str(e)}")


@router.get("/influence", summary="查询实体影响范围")
async def query_entity_influence(
    entity_id: str = Query(..., description="实体ID"),
    max_depth: int = Query(2, description="扩展深度（1-3）", ge=1, le=3)
) -> Dict[str, Any]:
    """
    分析实体的影响范围（周围N跳的所有实体）

    参数:
        - entity_id: 实体ID
        - max_depth: 扩展深度（1-3跳）

    返回:
        影响范围内的所有节点、边及统计信息
    """
    try:
        result = get_entity_influence(driver, entity_id, max_depth=max_depth)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "entity_id": entity_id,
                "max_depth": max_depth
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询实体影响范围失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询实体影响范围失败: {str(e)}")


@router.get("/community", summary="查询实体社区")
async def query_entity_community(
    entity_id: str = Query(..., description="实体ID"),
    max_depth: int = Query(2, description="社区扩展深度（1-3）", ge=1, le=3)
) -> Dict[str, Any]:
    """
    查询实体所属的社区

    参数:
        - entity_id: 实体ID
        - max_depth: 社区扩展深度（1-3跳）

    返回:
        社区内的节点、边及社区统计信息
    """
    try:
        result = get_simplified_community(driver, entity_id, max_depth=max_depth)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "entity_id": entity_id,
                "max_depth": max_depth,
                "community_count": result.get("community_count", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询实体社区失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询实体社区失败: {str(e)}")


@router.get("/cycles", summary="查询实体环路")
async def query_entity_cycles(
    entity_id: str = Query(..., description="实体ID"),
    max_depth: int = Query(4, description="环路最大深度（1-4）", ge=1, le=4)
) -> Dict[str, Any]:
    """
    查询实体的环路（从实体出发又回到自身的路径）

    参数:
        - entity_id: 实体ID
        - max_depth: 环路最大深度（1-4跳）

    返回:
        环路的节点、边及环路描述
    """
    try:
        result = get_entity_cycles(driver, entity_id, max_depth=max_depth)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "data": result,
            "meta": {
                "entity_id": entity_id,
                "max_depth": max_depth,
                "cycle_count": result.get("cycle_count", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询实体环路失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询实体环路失败: {str(e)}")


@router.get("/source/chunk", summary="获取原文片段")
async def get_chunk_content(
    chunk_id: str = Query(..., description="文本块ID")
) -> Dict[str, Any]:
    """
    获取文本块的原文内容

    参数:
        - chunk_id: 文本块ID

    返回:
        原文内容及文件信息
    """
    try:
        content = get_source_content(chunk_id)
        file_info = get_source_file_info(chunk_id)

        return {
            "status": "success",
            "data": {
                "chunk_id": chunk_id,
                "content": content,
                "file_name": file_info.get("file_name", "未知文件")
            }
        }
    except Exception as e:
        print(f"获取原文片段失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取原文片段失败: {str(e)}")


@router.get("/source/file-info", summary="获取文件信息")
async def get_file_information(
    source_id: str = Query(..., description="源ID（Chunk ID或Community ID）")
) -> Dict[str, Any]:
    """
    获取源ID对应的文件信息

    参数:
        - source_id: 源ID

    返回:
        文件名等信息
    """
    try:
        file_info = get_source_file_info(source_id)

        return {
            "status": "success",
            "data": file_info
        }
    except Exception as e:
        print(f"获取文件信息失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取文件信息失败: {str(e)}")


@router.get("/stats", summary="获取图谱统计信息")
async def get_graph_stats() -> Dict[str, Any]:
    """
    获取知识图谱的统计信息

    返回:
        - 实体总数
        - 关系总数
        - 实体类型分布
        - 关系类型分布
    """
    try:
        # 查询实体统计
        entity_query = """
        MATCH (e:__Entity__)
        WITH labels(e) AS labels
        UNWIND labels AS label
        WITH label
        WHERE label <> '__Entity__'
        RETURN label AS type, count(*) AS count
        ORDER BY count DESC
        """

        entity_result = driver.execute_query(entity_query)
        entity_types = [
            {"type": r.get("type"), "count": r.get("count")}
            for r in entity_result.records
        ]

        # 查询关系统计
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
        """

        rel_result = driver.execute_query(rel_query)
        rel_types = [
            {"type": r.get("type"), "count": r.get("count")}
            for r in rel_result.records
        ]

        # 查询总数
        total_entities_query = "MATCH (e:__Entity__) RETURN count(e) AS total"
        total_entities_result = driver.execute_query(total_entities_query)
        total_entities = total_entities_result.records[0].get("total", 0) if total_entities_result.records else 0

        total_rels_query = "MATCH ()-[r]->() RETURN count(r) AS total"
        total_rels_result = driver.execute_query(total_rels_query)
        total_rels = total_rels_result.records[0].get("total", 0) if total_rels_result.records else 0

        return {
            "status": "success",
            "data": {
                "total_entities": total_entities,
                "total_relationships": total_rels,
                "entity_types": entity_types,
                "relationship_types": rel_types
            }
        }
    except Exception as e:
        print(f"获取图谱统计信息失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取图谱统计信息失败: {str(e)}")
