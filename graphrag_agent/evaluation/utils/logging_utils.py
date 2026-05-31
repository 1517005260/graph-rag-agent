import os
import logging
from typing import Optional

# 全局日志字典
_loggers = {}

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logging.Logger: 日志记录器
    """
    # 检查是否已经有此名称的记录器
    if name in _loggers:
        return _loggers[name]
    
    # 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 保存记录器
    _loggers[name] = logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    if name not in _loggers:
        # 如果没有找到记录器，创建一个默认的
        return setup_logger(name)
    
    return _loggers[name]