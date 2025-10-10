from typing import Dict
from fastapi import APIRouter, HTTPException
from models.schemas import SourceRequest, SourceResponse, SourceInfoResponse, SourceInfoBatchRequest, ContentBatchRequest
from services.kg_service import get_source_content, get_source_file_info
from utils.neo4j_batch import BatchProcessor
from graphrag_agent.config.neo4jdb import get_db_manager

# 创建路由器
router = APIRouter()


@router.post("/source", response_model=SourceResponse)
async def source(request: SourceRequest):
    """
    处理源内容请求
    
    Args:
        request: 源内容请求
        
    Returns:
        SourceResponse: 源内容响应
    """
    content = get_source_content(request.source_id)
    return SourceResponse(content=content)

@router.post("/source_info")
async def source_info(request: SourceRequest):
    """
    处理源文件信息请求
    
    Args:
        request: 源内容请求
        
    Returns:
        Dict: 包含文件名等信息的响应
    """
    info = get_source_file_info(request.source_id)
    return info

@router.post("/content_batch", response_model=Dict)
async def get_content_batch(request: ContentBatchRequest):
    """批量获取内容"""
    try:
        # 获取数据库驱动
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        
        # 使用BatchProcessor批量处理
        result = BatchProcessor.get_content_batch(request.chunk_ids, driver)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量获取内容失败: {str(e)}")


@router.post("/source_info_batch", response_model=Dict)
async def get_source_info_batch(request: SourceInfoBatchRequest):
    """批量获取源信息"""
    try:
        # 获取数据库驱动
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        
        # 使用BatchProcessor批量处理
        result = BatchProcessor.get_source_info_batch(request.source_ids, driver)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量获取源信息失败: {str(e)}")