import os

from dotenv import load_dotenv

from graphrag_agent.config.settings import workers as core_workers

load_dotenv()


def _get_env_int(key: str, default: int) -> int:
    """读取整型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要整数值，但实际为 {raw}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    """读取布尔型环境变量，支持 true/false/1/0 等表达"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "y", "yes", "on"}


# ===== FastAPI / Uvicorn 运行参数 =====

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")  # 服务监听地址
SERVER_PORT = _get_env_int("SERVER_PORT", 8000)  # 服务端口
SERVER_RELOAD = _get_env_bool("SERVER_RELOAD", False)  # 热重载开关
SERVER_LOG_LEVEL = os.getenv("SERVER_LOG_LEVEL", "info")  # 日志等级

# Worker 数量优先使用 SERVER_WORKERS，否则回落到核心配置
SERVER_WORKERS = _get_env_int("SERVER_WORKERS", core_workers) or core_workers

# 统一封装 uvicorn.run 可用参数
UVICORN_CONFIG = {
    "host": SERVER_HOST,
    "port": SERVER_PORT,
    "reload": SERVER_RELOAD,
    "log_level": SERVER_LOG_LEVEL,
    "workers": SERVER_WORKERS,
}
