from fastapi import APIRouter
from . import chat, feedback, knowledge_graph, source, admin, build, graph, status

# 创建总路由器
api_router = APIRouter()

# 包含各个子路由器
api_router.include_router(chat.router, tags=["聊天"])
api_router.include_router(feedback.router, tags=["反馈"])
api_router.include_router(knowledge_graph.router, tags=["知识图谱"])
api_router.include_router(source.router, tags=["源内容"])
api_router.include_router(admin.router, tags=["管理"])
api_router.include_router(build.router, tags=["图谱构建"])
api_router.include_router(graph.router, tags=["图谱探索"])
api_router.include_router(status.router, tags=["系统状态"])

__all__ = ['api_router']
