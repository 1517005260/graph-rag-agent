import os
import time
import json
import threading
from typing import Any, Optional, List, Tuple, Dict
from collections import OrderedDict
from .base import CacheStorageBackend


class DiskCacheBackend(CacheStorageBackend):
    """磁盘缓存后端实现"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000, 
                 batch_size: int = 10, flush_interval: float = 30.0):
        """初始化磁盘缓存后端"""
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # 使用OrderedDict维护访问顺序
        self.metadata: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.write_queue: List[Tuple[str, Any]] = []
        self.last_flush_time = time.time()
        self._lock = threading.RLock()
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载索引
        self._load_index()
    
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 使用子目录避免单个目录文件过多
        subdir = key[:2]
        subdir_path = os.path.join(self.cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(subdir_path, f"{key}.json")
    
    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.cache_dir, "index.json")
    
    def _load_index(self) -> None:
        """加载缓存索引"""
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 保持访问顺序
                    for key in sorted(data.keys(), key=lambda k: data[k].get('last_accessed', 0)):
                        self.metadata[key] = data[key]
            except Exception as e:
                print(f"加载缓存索引失败: {e}")
                self.metadata = OrderedDict()
        
        # 验证磁盘上的文件并同步索引
        self._sync_index_with_filesystem()
    
    def _sync_index_with_filesystem(self) -> None:
        """同步索引与文件系统"""
        existing_files = set()
        
        # 扫描所有子目录
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if os.path.isdir(item_path) and len(item) == 2:
                for filename in os.listdir(item_path):
                    if filename.endswith(".json"):
                        key = filename[:-5]  # 移除.json后缀
                        existing_files.add(key)
        
        # 移除索引中不存在的文件
        keys_to_remove = [key for key in self.metadata if key not in existing_files]
        for key in keys_to_remove:
            del self.metadata[key]
        
        # 添加文件系统中存在但索引中没有的文件
        for key in existing_files:
            if key not in self.metadata:
                file_path = self._get_cache_path(key)
                try:
                    stat = os.stat(file_path)
                    self.metadata[key] = {
                        "created_at": stat.st_ctime,
                        "last_accessed": stat.st_atime,
                        "access_count": 0,
                        "file_size": stat.st_size
                    }
                except OSError:
                    continue
    
    def _save_index(self) -> None:
        """保存缓存索引"""
        try:
            with open(self._get_index_path(), 'w', encoding='utf-8') as f:
                json.dump(dict(self.metadata), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存索引失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            cache_path = self._get_cache_path(key)
            
            if key in self.metadata and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        value = json.load(f)
                    
                    # 更新访问信息并移到末尾（LRU）
                    self.metadata[key]["last_accessed"] = time.time()
                    self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
                    
                    # 移动到OrderedDict末尾
                    self.metadata.move_to_end(key)
                    
                    # 异步保存索引
                    self._schedule_index_save()
                    
                    return value
                except Exception as e:
                    print(f"读取缓存文件失败 ({key}): {e}")
                    # 如果文件损坏，从索引中删除
                    if key in self.metadata:
                        del self.metadata[key]
            
            return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        with self._lock:
            # 如果缓存已满且是新键，执行淘汰
            if len(self.metadata) >= self.max_size and key not in self.metadata:
                self._evict_items()
            
            # 更新元数据
            current_time = time.time()
            if key in self.metadata:
                # 更新现有项
                self.metadata[key].update({
                    "last_accessed": current_time,
                    "access_count": self.metadata[key].get("access_count", 0)
                })
                self.metadata.move_to_end(key)
            else:
                # 新增项
                self.metadata[key] = {
                    "created_at": current_time,
                    "last_accessed": current_time,
                    "access_count": 0
                }
            
            # 添加到写入队列
            self.write_queue.append((key, value))
            
            # 检查是否需要刷新
            if (len(self.write_queue) >= self.batch_size or 
                (time.time() - self.last_flush_time) > self.flush_interval):
                self._flush_write_queue()
    
    def _flush_write_queue(self) -> None:
        """刷新写入队列"""
        if not self.write_queue:
            return
        
        successful_writes = []
        failed_writes = []
        
        for key, value in self.write_queue:
            try:
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2, default=str)
                
                # 更新文件大小信息
                if key in self.metadata:
                    self.metadata[key]["file_size"] = os.path.getsize(cache_path)
                
                successful_writes.append(key)
            except Exception as e:
                print(f"写入缓存文件失败 ({key}): {e}")
                failed_writes.append((key, value))
        
        # 只保留失败的写入操作
        self.write_queue = failed_writes
        self.last_flush_time = time.time()
        
        # 如果有成功的写入，保存索引
        if successful_writes:
            self._save_index()
    
    def _schedule_index_save(self) -> None:
        """调度索引保存"""
        current_time = time.time()
        if current_time - self.last_flush_time > 60:  # 每分钟最多保存一次
            self._save_index()
            self.last_flush_time = current_time
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self._lock:
            if key not in self.metadata:
                return False
            
            # 从元数据中删除
            del self.metadata[key]
            
            # 删除文件
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    print(f"删除缓存文件失败 ({key}): {e}")
                    return False
            
            # 从写入队列中移除
            self.write_queue = [(k, v) for k, v in self.write_queue if k != key]
            
            # 保存索引
            self._save_index()
            return True
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            # 清空写入队列
            self.write_queue.clear()
            
            # 清空元数据
            self.metadata.clear()
            
            # 删除所有缓存文件
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith(".json"):
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception as e:
                            print(f"删除缓存文件失败: {e}")
            
            # 保存空索引
            self._save_index()
    
    def flush(self) -> None:
        """强制刷新所有待写入的数据"""
        with self._lock:
            self._flush_write_queue()
    
    def _evict_items(self, num_to_evict: int = None) -> None:
        """淘汰缓存项"""
        if not self.metadata:
            return
        
        if num_to_evict is None:
            num_to_evict = max(1, len(self.metadata) // 10)  # 淘汰10%
        
        # 使用复合评分策略：访问频率 + 新近度 + 文件大小
        current_time = time.time()
        scores = {}
        
        for key, meta in self.metadata.items():
            age = current_time - meta.get("created_at", current_time)
            access_count = meta.get("access_count", 0)
            last_accessed = meta.get("last_accessed", meta.get("created_at", current_time))
            recency = current_time - last_accessed
            file_size = meta.get("file_size", 1000)  # 默认1KB
            
            # 计算复合分数（分数越低越容易被淘汰）
            frequency_score = access_count / max(age / 3600, 1)  # 每小时访问频率
            recency_score = 1 / max(recency / 3600, 1)  # 新近度分数
            size_penalty = file_size / 1024  # 大文件惩罚
            
            scores[key] = frequency_score + recency_score - size_penalty * 0.1
        
        # 选择分数最低的项目进行淘汰
        keys_to_evict = sorted(scores.keys(), key=lambda k: scores[k])[:num_to_evict]
        
        for key in keys_to_evict:
            self.delete(key)