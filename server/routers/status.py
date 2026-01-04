from fastapi import APIRouter, HTTPException
from graphrag_agent.config.settings import (
    CHUNK_VECTOR_INDEX,
    ENTITY_VECTOR_INDEX,
)
from server_config.database import get_db_manager


router = APIRouter()


def _index_exists(graph, index_name: str) -> bool:
    """Check whether a vector index exists by name."""
    try:
        result = graph.query(
            """
            SHOW INDEXES
            YIELD name, type
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN count(*) > 0 AS exists
            """,
            {"index_name": index_name},
        )
        return bool(result and result[0].get("exists"))
    except Exception:
        return False


@router.get("/status")
async def get_system_status():
    """
    Return minimal system readiness for vector search and node counts.

    This is used by the frontend to gate chat requests until indexes are ready.
    """
    db_manager = get_db_manager()
    graph = db_manager.get_graph()

    try:
        chunk_index_exists = _index_exists(graph, CHUNK_VECTOR_INDEX)
        entity_index_exists = _index_exists(graph, ENTITY_VECTOR_INDEX)

        chunk_count_res = graph.query("MATCH (c:`__Chunk__`) RETURN count(c) AS count")
        entity_count_res = graph.query("MATCH (e:`__Entity__`) RETURN count(e) AS count")

        return {
            "chunk_vector_index_exists": chunk_index_exists,
            "entity_vector_index_exists": entity_index_exists,
            "chunk_count": chunk_count_res[0]["count"] if chunk_count_res else 0,
            "entity_count": entity_count_res[0]["count"] if entity_count_res else 0,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {exc}") from exc
