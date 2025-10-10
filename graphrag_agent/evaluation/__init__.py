# 全局调试模式开关
DEBUG_MODE = True

def debug_print(*args, **kwargs):
    """调试日志打印函数，仅在DEBUG_MODE为True时打印"""
    if DEBUG_MODE:
        print(*args, **kwargs)

def set_debug_mode(enabled=True):
    """设置全局调试模式
    
    Args:
        enabled (bool): 是否启用调试模式
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled