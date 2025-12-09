"""
管理路由
提供文档管理、配置管理、构建管理等管理 API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import subprocess
import threading
import time
import logging
from datetime import datetime
from pathlib import Path

from graphrag_agent.config.settings import FILES_DIR, BASE_DIR
from graphrag_agent.graph.core import connection_manager
from graphrag_agent.config.graph_config_model import GraphConfig
from graphrag_agent.config.graph_config_storage import get_storage

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

router = APIRouter(prefix="/admin", tags=["admin"])

# 全局构建状态
build_status = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,
    "current_stage": "",
    "processed": 0,
    "total": 0,
    "elapsed_time": 0,
    "start_time": None,
    "logs": [],
    "error": None
}

build_lock = threading.Lock()


def run_build_command(command: list, build_type: str):
    """在后台运行构建命令"""
    global build_status

    with build_lock:
        build_status["status"] = "running"
        build_status["progress"] = 0
        build_status["current_stage"] = "初始化"
        build_status["processed"] = 0
        build_status["total"] = 100
        build_status["start_time"] = time.time()
        build_status["logs"] = [f"开始{build_type}构建..."]
        build_status["error"] = None

    try:
        # 执行构建命令
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 读取输出
        for line in iter(process.stdout.readline, ''):
            if line:
                with build_lock:
                    build_status["logs"].append(line.strip())
                    # 简单的进度估算（可以根据实际日志解析更准确的进度）
                    if "文档读取" in line or "读取文件" in line:
                        build_status["current_stage"] = "文档读取"
                        build_status["progress"] = 10
                    elif "实体提取" in line or "提取实体" in line:
                        build_status["current_stage"] = "实体提取"
                        build_status["progress"] = 40
                    elif "实体消歧" in line or "对齐" in line:
                        build_status["current_stage"] = "实体消歧"
                        build_status["progress"] = 60
                    elif "图谱构建" in line or "构建图谱" in line:
                        build_status["current_stage"] = "图谱构建"
                        build_status["progress"] = 75
                    elif "向量索引" in line or "索引" in line:
                        build_status["current_stage"] = "向量索引"
                        build_status["progress"] = 85
                    elif "社区检测" in line:
                        build_status["current_stage"] = "社区检测"
                        build_status["progress"] = 95
                    elif "完成" in line:
                        build_status["progress"] = 100

                    build_status["elapsed_time"] = int(time.time() - build_status["start_time"])

        process.wait()

        with build_lock:
            if process.returncode == 0:
                build_status["status"] = "completed"
                build_status["progress"] = 100
                build_status["current_stage"] = "完成"
                build_status["logs"].append(f"{build_type}构建成功完成")
            else:
                build_status["status"] = "failed"
                build_status["error"] = f"构建失败，退出码: {process.returncode}"
                build_status["logs"].append(f"{build_type}构建失败")

    except Exception as e:
        with build_lock:
            build_status["status"] = "failed"
            build_status["error"] = str(e)
            build_status["logs"].append(f"构建异常: {e}")


@router.post("/build/full")
async def trigger_full_build(background_tasks: BackgroundTasks):
    """触发完整构建"""
    logger.info("收到完整构建请求")

    if build_status["status"] == "running":
        logger.warning("已有构建任务正在运行，拒绝新请求")
        raise HTTPException(status_code=400, detail="已有构建任务正在运行")

    # 在后台启动构建
    command = ["python", "graphrag_agent/integrations/build/main.py"]
    thread = threading.Thread(target=run_build_command, args=(command, "完整"))
    thread.daemon = True
    thread.start()

    logger.info("完整构建已在后台启动")
    return {"message": "完整构建已启动", "status": "running"}


@router.post("/build/incremental")
async def trigger_incremental_build(background_tasks: BackgroundTasks):
    """触发增量构建"""
    logger.info("收到增量构建请求")

    if build_status["status"] == "running":
        logger.warning("已有构建任务正在运行，拒绝新请求")
        raise HTTPException(status_code=400, detail="已有构建任务正在运行")

    # 在后台启动增量构建
    command = ["python", "graphrag_agent/integrations/build/incremental_update.py", "--once"]
    thread = threading.Thread(target=run_build_command, args=(command, "增量"))
    thread.daemon = True
    thread.start()

    logger.info("增量构建已在后台启动")
    return {"message": "增量构建已启动", "status": "running"}


@router.post("/build/stop")
async def stop_build():
    """停止构建"""
    with build_lock:
        if build_status["status"] == "running":
            build_status["status"] = "idle"
            build_status["logs"].append("构建已被用户停止")
            return {"message": "构建已停止"}
        else:
            return {"message": "没有正在运行的构建任务"}


@router.get("/build/status")
async def get_build_status():
    """获取构建状态"""
    with build_lock:
        return build_status.copy()


@router.get("/graph/stats")
async def get_graph_stats():
    """获取图谱统计信息"""
    logger.info("收到图谱统计请求")

    try:
        # 查询实体数量
        entity_count_query = "MATCH (n) RETURN count(n) as count"
        entity_result = connection_manager.execute_query(entity_count_query)
        entity_count = entity_result[0]['count'] if entity_result else 0
        logger.info(f"实体数量: {entity_count}")

        # 查询关系数量
        relationship_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        relationship_result = connection_manager.execute_query(relationship_count_query)
        relationship_count = relationship_result[0]['count'] if relationship_result else 0
        logger.info(f"关系数量: {relationship_count}")

        # 查询实体类型分布
        entity_type_query = """
        MATCH (n)
        WHERE n.entity_type IS NOT NULL
        RETURN n.entity_type as type, count(*) as count
        ORDER BY count DESC
        """
        entity_type_result = connection_manager.execute_query(entity_type_query)
        entity_type_distribution = {
            item['type']: item['count'] for item in entity_type_result
        } if entity_type_result else {}

        # 查询关系类型分布
        relationship_type_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        ORDER BY count DESC
        """
        relationship_type_result = connection_manager.execute_query(relationship_type_query)
        relationship_type_distribution = {
            item['type']: item['count'] for item in relationship_type_result
        } if relationship_type_result else {}

        # 查询社区数量
        community_count_query = """
        MATCH (n)
        WHERE n.community_id IS NOT NULL
        RETURN count(DISTINCT n.community_id) as count
        """
        community_result = connection_manager.execute_query(community_count_query)
        community_count = community_result[0]['count'] if community_result else 0

        # 统计文档数量
        files_dir = Path(FILES_DIR)
        document_count = len([f for f in files_dir.iterdir() if f.is_file()]) if files_dir.exists() else 0

        stats = {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "community_count": community_count,
            "document_count": document_count,
            "entity_type_distribution": entity_type_distribution,
            "relationship_type_distribution": relationship_type_distribution,
            "last_build_time": datetime.now().isoformat() if entity_count > 0 else None
        }

        logger.info(f"返回图谱统计: {entity_count} 实体, {relationship_count} 关系")
        return stats

    except Exception as e:
        logger.error(f"获取图谱统计失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取图谱统计失败: {str(e)}")


@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查 Neo4j 连接
        connection_manager.execute_query("RETURN 1")
        neo4j_status = "healthy"
    except:
        neo4j_status = "unhealthy"

    return {
        "status": "ok",
        "neo4j": neo4j_status,
        "timestamp": datetime.now().isoformat()
    }


# ==================== 图谱配置管理 API ====================

@router.get("/graph/config")
async def get_graph_config():
    """获取当前图谱配置"""
    logger.info("收到获取图谱配置请求")

    try:
        storage = get_storage()
        config = storage.load()

        if config is None:
            logger.info("当前无配置，返回空")
            return {"exists": False, "config": None}

        logger.info(f"返回配置: {config.project_name}")
        return {
            "exists": True,
            "config": config.model_dump(mode='json')
        }

    except Exception as e:
        logger.error(f"获取图谱配置失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.post("/graph/config")
async def save_graph_config(config: GraphConfig):
    """保存图谱配置"""
    logger.info(f"收到保存图谱配置请求: {config.project_name}")

    try:
        storage = get_storage()
        success = storage.save(config)

        if success:
            logger.info(f"配置保存成功: {config.project_name}")
            return {"message": "配置保存成功", "project_name": config.project_name}
        else:
            logger.error("配置保存失败")
            raise HTTPException(status_code=500, detail="配置保存失败")

    except Exception as e:
        logger.error(f"保存图谱配置失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@router.delete("/graph/config")
async def delete_graph_config():
    """删除图谱配置"""
    logger.info("收到删除图谱配置请求")

    try:
        storage = get_storage()
        success = storage.delete()

        if success:
            logger.info("配置删除成功")
            return {"message": "配置删除成功"}
        else:
            logger.error("配置删除失败")
            raise HTTPException(status_code=500, detail="配置删除失败")

    except Exception as e:
        logger.error(f"删除图谱配置失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除配置失败: {str(e)}")


@router.get("/graph/templates")
async def list_graph_templates():
    """列出所有可用的行业模板"""
    logger.info("收到列出模板请求")

    try:
        storage = get_storage()
        templates = storage.list_templates()

        logger.info(f"返回 {len(templates)} 个模板")
        return {"templates": templates}

    except Exception as e:
        logger.error(f"列出模板失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"列出模板失败: {str(e)}")


@router.get("/graph/templates/{template_name}")
async def get_graph_template(template_name: str):
    """获取指定模板的详细信息"""
    logger.info(f"收到获取模板请求: {template_name}")

    try:
        storage = get_storage()
        config = storage.load_template(template_name)

        if config is None:
            logger.warning(f"模板不存在: {template_name}")
            raise HTTPException(status_code=404, detail=f"模板 '{template_name}' 不存在")

        logger.info(f"返回模板: {template_name}")
        return {"config": config.model_dump(mode='json')}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模板失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模板失败: {str(e)}")


@router.post("/graph/config/from-template")
async def create_config_from_template(
    template_name: str,
    project_name: Optional[str] = None,
    created_by: Optional[str] = None
):
    """从模板创建并保存配置"""
    logger.info(f"收到从模板创建配置请求: {template_name}")

    try:
        storage = get_storage()
        success = storage.save_from_template(
            template_name=template_name,
            project_name=project_name,
            created_by=created_by
        )

        if success:
            logger.info(f"从模板创建配置成功: {template_name}")
            return {
                "message": "配置创建成功",
                "template_name": template_name,
                "project_name": project_name or f"{template_name}_project"
            }
        else:
            logger.error(f"模板不存在或保存失败: {template_name}")
            raise HTTPException(
                status_code=400,
                detail=f"模板 '{template_name}' 不存在或保存失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"从模板创建配置失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"从模板创建配置失败: {str(e)}")


# ==================== AI Copilot API ====================

@router.post("/ai-copilot/analyze-documents")
async def analyze_documents_for_config(
    industry_hint: Optional[str] = None,
    num_clusters: Optional[int] = None
):
    """
    分析已上传的文档，生成配置推荐

    Args:
        industry_hint: 行业提示（可选）
        num_clusters: 聚类数量（可选，默认自动确定）

    Returns:
        文档分析结果和配置推荐
    """
    logger.info("收到 AI Copilot 文档分析请求")

    try:
        from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
        from graphrag_agent.ai_copilot import DocumentAnalyzer
        from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
        from graphrag_agent.config.settings import FILES_DIR, CHUNK_SIZE, OVERLAP

        # 初始化模型
        llm = get_llm_model()
        embeddings = get_embeddings_model()

        # 创建文档分析器
        analyzer = DocumentAnalyzer(llm, embeddings)

        # 读取文档
        doc_processor = DocumentProcessor(FILES_DIR, CHUNK_SIZE, OVERLAP)
        processed_docs = doc_processor.process_directory()

        if not processed_docs:
            logger.warning("没有找到可分析的文档")
            raise HTTPException(status_code=400, detail="没有找到可分析的文档，请先上传文档")

        # 准备文档数据
        documents = [
            {
                "filename": doc["filename"],
                "content": doc["content"]
            }
            for doc in processed_docs
        ]

        logger.info(f"开始分析 {len(documents)} 个文档")

        # 分析文档
        analysis_result = analyzer.analyze_documents(documents, num_clusters)

        logger.info(f"文档分析完成，聚类数: {len(analysis_result['clusters'])}")

        # 生成推荐
        logger.info("开始生成配置推荐")
        recommendations = analyzer.recommend_domains_and_bridges(analysis_result, industry_hint)

        logger.info("配置推荐生成完成")

        return {
            "analysis": analysis_result,
            "recommendations": recommendations
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI Copilot 分析失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档分析失败: {str(e)}")


@router.post("/ai-copilot/refine-config")
async def refine_configuration(
    user_feedback: str,
    current_config: Optional[Dict] = None
):
    """
    基于用户反馈优化配置

    Args:
        user_feedback: 用户反馈
        current_config: 当前配置（可选，如果为空则从存储加载）

    Returns:
        优化后的配置
    """
    logger.info("收到 AI Copilot 配置优化请求")

    try:
        from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
        from graphrag_agent.ai_copilot import DocumentAnalyzer

        # 初始化模型
        llm = get_llm_model()
        embeddings = get_embeddings_model()

        # 创建文档分析器
        analyzer = DocumentAnalyzer(llm, embeddings)

        # 获取当前配置
        if current_config is None:
            storage = get_storage()
            config_obj = storage.load()
            if config_obj is None:
                raise HTTPException(status_code=400, detail="当前无配置，请先创建配置")
            current_config = config_obj.model_dump(mode='json')

        logger.info(f"开始优化配置，用户反馈: {user_feedback[:100]}...")

        # 优化配置
        refined_config = analyzer.refine_recommendations(current_config, user_feedback)

        logger.info("配置优化完成")

        return refined_config

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI Copilot 配置优化失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"配置优化失败: {str(e)}")


@router.post("/ai-copilot/apply-recommendations")
async def apply_ai_recommendations(
    recommendations: Dict,
    project_name: str,
    industry: Optional[str] = None,
    description: Optional[str] = None
):
    """
    将 AI 推荐应用为新配置

    Args:
        recommendations: AI 推荐结果
        project_name: 项目名称
        industry: 行业
        description: 项目描述

    Returns:
        创建的配置
    """
    logger.info(f"收到应用 AI 推荐请求: {project_name}")

    try:
        # 构建配置
        config_dict = {
            "project_name": project_name,
            "version": "1.0",
            "description": description or "基于 AI 推荐创建",
            "industry": industry or "通用",
            "bridge_definitions": recommendations.get("recommended_bridges", []),
            "domain_definitions": recommendations.get("recommended_domains", [])
        }

        # 验证并保存
        config = GraphConfig(**config_dict)
        storage = get_storage()
        success = storage.save(config)

        if success:
            logger.info(f"AI 推荐配置已保存: {project_name}")
            return {
                "message": "配置创建成功",
                "config": config.model_dump(mode='json')
            }
        else:
            raise HTTPException(status_code=500, detail="配置保存失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"应用 AI 推荐失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"应用推荐失败: {str(e)}")


@router.get("/files/list")
async def list_files():
    """列出所有已上传的文档"""
    try:
        files_dir = Path(FILES_DIR)
        if not files_dir.exists():
            files_dir.mkdir(parents=True, exist_ok=True)
            return {"files": [], "count": 0}

        files = []
        for file_path in files_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # 按修改时间倒序排序
        files.sort(key=lambda x: x["modified_at"], reverse=True)

        logger.info(f"返回文件列表，共 {len(files)} 个文件")
        return {
            "files": files,
            "count": len(files),
            "directory": str(files_dir)
        }

    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")
