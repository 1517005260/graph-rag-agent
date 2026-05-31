import time
import hashlib
from typing import Callable, Dict, List, Any

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