"""
图谱构建管道 - L0/L1 拆分架构

L0（快速通道）: 文本分块与向量化，10秒内完成
L1（慢速通道）: 实体提取与图谱构建，后台异步执行
"""
from graphrag_agent.integrations.build.pipeline.task_queue import (
    GraphBuildTaskQueue,
    Task,
    TaskPriority,
    TaskStatus,
    get_global_task_queue
)
from graphrag_agent.integrations.build.pipeline.fast_ingestion_pipeline import (
    FastIngestionPipeline,
    quick_ingest_file,
    quick_ingest_directory
)
from graphrag_agent.integrations.build.pipeline.slow_graph_pipeline import (
    SlowGraphPipeline,
    extract_entities_from_file,
    entity_extraction_task_handler
)

__all__ = [
    # 任务队列
    "GraphBuildTaskQueue",
    "Task",
    "TaskPriority",
    "TaskStatus",
    "get_global_task_queue",
    # L0 快速通道
    "FastIngestionPipeline",
    "quick_ingest_file",
    "quick_ingest_directory",
    # L1 慢速通道
    "SlowGraphPipeline",
    "extract_entities_from_file",
    "entity_extraction_task_handler",
]
