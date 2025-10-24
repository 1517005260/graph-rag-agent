import os

from dotenv import load_dotenv

from graphrag_agent.config.settings import examples as eg

load_dotenv()


def _get_env_bool(key: str, default: bool) -> bool:
    """读取布尔型环境变量"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(key: str, default: int) -> int:
    """读取整型环境变量"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要整数值，但实际为 {raw}") from exc


# ===== 前端接口与会话配置 =====

API_URL = os.getenv("FRONTEND_API_URL", "http://localhost:8000")  # 后端接口地址

DEFAULT_AGENT_TYPE = os.getenv("FRONTEND_DEFAULT_AGENT", "naive_rag_agent")  # 默认Agent
DEFAULT_DEBUG_MODE = _get_env_bool("FRONTEND_DEFAULT_DEBUG", False)  # 默认是否开启调试
DEFAULT_SHOW_THINKING = _get_env_bool("FRONTEND_SHOW_THINKING", True)  # 是否默认展示思考过程
DEFAULT_USE_DEEPER_TOOL = _get_env_bool("FRONTEND_USE_DEEPER_TOOL", True)  # 深度研究工具默认开关
DEFAULT_USE_STREAM = _get_env_bool("FRONTEND_USE_STREAM", True)  # 是否默认使用流式输出
DEFAULT_CHAIN_EXPLORATION = _get_env_bool("FRONTEND_USE_CHAIN_EXPLORATION", True)  # 链式探索开关

# 示例问题直接复用核心配置，保持前后端一致
examples = eg

# ===== 知识图谱展示参数 =====

KG_COLOR_PALETTE = [
    "#4285F4",  # 谷歌蓝
    "#EA4335",  # 谷歌红
    "#FBBC05",  # 谷歌黄
    "#34A853",  # 谷歌绿
    "#7B1FA2",  # 紫色
    "#0097A7",  # 青色
    "#FF6D00",  # 橙色
    "#757575",  # 灰色
    "#607D8B",  # 蓝灰色
    "#C2185B"   # 粉色
]

NODE_TYPE_COLORS = {
    "Center": "#F0B2F4",     # 中心/源节点 - 紫色
    "Source": "#4285F4",     # 源节点 - 蓝色
    "Target": "#EA4335",     # 目标节点 - 红色
    "Common": "#34A853",     # 共同邻居 - 绿色
    "Level1": "#0097A7",     # 一级关联 - 青色
    "Level2": "#FF6D00",     # 二级关联 - 橙色
}

DEFAULT_KG_SETTINGS = {
    "physics_enabled": _get_env_bool("KG_PHYSICS_ENABLED", True),
    "node_size": _get_env_int("KG_NODE_SIZE", 25),
    "edge_width": _get_env_int("KG_EDGE_WIDTH", 2),
    "spring_length": _get_env_int("KG_SPRING_LENGTH", 150),
    "gravity": _get_env_int("KG_GRAVITY", -5000),
}
