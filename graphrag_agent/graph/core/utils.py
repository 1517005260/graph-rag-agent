import time
import hashlib
from typing import Any, Callable, Dict, List, Optional

from graphrag_agent.config.settings import EMBEDDING_DIM, VECTOR_SIMILARITY_FUNCTION


def ensure_vector_index(
    graph: Any,
    index_name: str,
    label: str,
    property_name: str,
    dim: Optional[int] = None,
    similarity: Optional[str] = None,
    drop_on_mismatch: bool = True,
) -> None:
    """Create a Neo4j vector index if it does not already exist.

    Args:
        graph: Neo4j connection object exposing ``query``.
        index_name: Name of the vector index.
        label: Node label to index.
        property_name: Embedding property name.
        dim: Embedding dimension. Defaults to ``EMBEDDING_DIM`` from settings.
        similarity: Similarity function (e.g., ``cosine``). Defaults to
            ``VECTOR_SIMILARITY_FUNCTION`` from settings.
        drop_on_mismatch: Whether to drop and recreate the index when an
            existing index has different dimensions or similarity settings.
    """
    dim = dim or EMBEDDING_DIM
    similarity = (similarity or VECTOR_SIMILARITY_FUNCTION).lower()

    def _extract_index_config(index_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract indexConfig map from SHOW INDEXES output."""
        options = index_info.get("options") or {}
        candidates = [
            options,
            index_info,
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("indexConfig", "indexconfig", "index-config"):
                config = candidate.get(key)
                if isinstance(config, dict):
                    return config
        return {}

    def _get_existing_index_config() -> tuple[Dict[str, Any], bool]:
        try:
            existing_indexes = graph.query(
                """
                SHOW INDEXES
                YIELD name, type, options
                WHERE name = $index_name
                RETURN name, type, options
                """,
                {"index_name": index_name},
            )
        except Exception:
            return {}, False

        if not existing_indexes:
            return {}, False

        return _extract_index_config(existing_indexes[0]), True

    def _drop_index_if_needed(
        existing_config: Dict[str, Any],
        has_index: bool,
    ) -> None:
        if not drop_on_mismatch or not has_index:
            return

        existing_dim = existing_config.get("vector.dimensions")
        existing_similarity = existing_config.get("vector.similarity_function")

        mismatch_dimension = existing_dim is not None and int(existing_dim) != int(dim)
        mismatch_similarity = (
            existing_similarity
            and isinstance(existing_similarity, str)
            and existing_similarity.lower() != similarity
        )

        if mismatch_dimension or mismatch_similarity or not existing_config:
            print(
                f"重新创建向量索引 {index_name} 以应用"
                f" dim={dim} similarity={similarity} 配置"
            )
            graph.query(f"DROP INDEX {index_name} IF EXISTS")

    current_config, has_index = _get_existing_index_config()
    _drop_index_if_needed(current_config, has_index)

    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:`{label}`)
    ON (n.{property_name})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: '{similarity}'
      }}
    }}
    """
    try:
        graph.query(query)
    except Exception:
        # 索引已存在或后端忽略重复创建时不视为错误
        pass


def timer(func):
    """
    计时装饰器，用于测量函数执行时间
    
    Args:
        func: 要测量的函数
        
    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"函数 {func.__name__} 执行耗时: {elapsed:.2f}秒")
        return result
    return wrapper

def generate_hash(text: str) -> str:
    """
    生成文本的哈希值
    
    Args:
        text: 输入文本
        
    Returns:
        str: 哈希字符串
    """
    return hashlib.sha1(text.encode()).hexdigest()

def batch_process(items: List[Any], 
                 process_func: Callable, 
                 batch_size: int = 100, 
                 show_progress: bool = True) -> List[Any]:
    """
    批量处理项目
    
    Args:
        items: 待处理项目列表
        process_func: 处理单个批次的函数，接收一个批次作为参数
        batch_size: 批处理大小
        show_progress: 是否显示进度
        
    Returns:
        List[Any]: 所有处理结果
    """
    if not items:
        return []
        
    results = []
    total = len(items)
    batches = (total + batch_size - 1) // batch_size
    
    if show_progress:
        print(f"开始批处理：共{total}项，分{batches}批进行")
    
    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        batch_results = process_func(batch)
        
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
            
        if show_progress:
            progress = (i + len(batch)) / total * 100
            print(f"进度: {progress:.1f}% ({i + len(batch)}/{total})")
    
    return results

def retry(times: int = 3, exceptions: tuple = (Exception,), delay: float = 1.0):
    """
    重试装饰器
    
    Args:
        times: 最大重试次数
        exceptions: 捕获的异常类型
        delay: 重试延迟秒数
        
    Returns:
        包装后的函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= times:
                        raise
                    print(f"函数 {func.__name__} 执行失败: {e}，尝试重试 ({attempt}/{times})")
                    time.sleep(delay)
        return wrapper
    return decorator

def get_performance_stats(total_time: float, 
                         time_records: Dict[str, float]) -> Dict[str, str]:
    """
    生成性能统计摘要
    
    Args:
        total_time: 总耗时
        time_records: 各阶段耗时记录
        
    Returns:
        Dict[str, str]: 性能统计摘要
    """
    stats = {"总耗时": f"{total_time:.2f}秒"}
    
    for name, t in time_records.items():
        percentage = (t/total_time*100) if total_time > 0 else 0
        stats[name] = f"{t:.2f}秒 ({percentage:.1f}%)"
    
    return stats

def print_performance_stats(stats: Dict[str, str], title: str = "性能统计摘要") -> None:
    """
    打印性能统计摘要
    
    Args:
        stats: 性能统计摘要
        title: 标题
    """
    print(f"\n{title}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
