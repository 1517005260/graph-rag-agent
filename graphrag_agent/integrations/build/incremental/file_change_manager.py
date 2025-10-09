import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class FileChangeManager:
    """
    文件变更管理器，负责追踪文件的变更状态。
    
    主要功能：
    1. 扫描文件目录，计算文件哈希值
    2. 与历史记录比较，识别变更的文件
    3. 更新文件注册表
    """
    
    def __init__(self, files_dir: str, registry_path: str = "./file_registry.json"):
        """
        初始化文件变更管理器
        
        Args:
            files_dir: 要监控的文件目录
            registry_path: 文件注册表保存路径
        """
        self.files_dir = Path(files_dir)
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        加载文件注册表
        
        Returns:
            Dict: 文件注册表，键为文件路径，值为文件元数据
        """
        if not self.registry_path.exists():
            return {}
            
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"无法加载文件注册表，将创建新的注册表")
            return {}
    
    def _save_registry(self):
        """保存文件注册表到磁盘"""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        计算文件的SHA256哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件哈希值
        """
        hash_obj = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"计算文件哈希值失败: {file_path}, 错误: {e}")
            return ""
    
    def _scan_current_files(self) -> Dict[str, Dict[str, Any]]:
        """
        扫描当前文件目录中的所有文件
        
        Returns:
            Dict: 当前文件状态，键为文件路径，值为文件元数据
        """
        current_files = {}
        
        # 遍历文件目录
        for root, _, files in os.walk(self.files_dir):
            for filename in files:
                file_path = Path(root) / filename
                rel_path = str(file_path.relative_to(self.files_dir))
                
                # 计算文件哈希值
                file_hash = self._compute_file_hash(file_path)
                if not file_hash:
                    continue
                
                # 记录文件元数据
                file_info = {
                    "hash": file_hash,
                    "size": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime,
                    "last_scanned": time.time()
                }
                current_files[rel_path] = file_info
        
        return current_files
    
    def detect_changes(self) -> Dict[str, List[str]]:
        """
        检测文件变更
        
        Returns:
            Dict: 包含三种变更类型的文件列表：added, modified, deleted
        """
        current_files = self._scan_current_files()
        
        # 构建变更列表
        added_files = []
        modified_files = []
        deleted_files = []
        
        # 检测新增和修改的文件
        for file_path, file_info in current_files.items():
            if file_path not in self.registry:
                added_files.append(file_path)
            elif file_info["hash"] != self.registry[file_path]["hash"]:
                modified_files.append(file_path)
        
        # 检测删除的文件
        for file_path in self.registry:
            if file_path not in current_files:
                deleted_files.append(file_path)
        
        return {
            "added": added_files,
            "modified": modified_files,
            "deleted": deleted_files
        }
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件元数据
        
        Args:
            file_path: 文件相对路径
            
        Returns:
            Dict: 文件元数据
        """
        return self.registry.get(file_path, {})
    
    def update_registry(self):
        """更新文件注册表，记录当前所有文件的状态"""
        self.registry = self._scan_current_files()
        self._save_registry()
        print(f"文件注册表已更新，共记录 {len(self.registry)} 个文件")
    
    def update_file_status(self, file_path: str, status: Dict[str, Any]):
        """
        更新单个文件的状态
        
        Args:
            file_path: 文件相对路径
            status: 文件状态
        """
        if file_path in self.registry:
            self.registry[file_path].update(status)
            self._save_registry()
    
    def register_file_processing(self, file_path: str, processing_info: Dict[str, Any]):
        """
        记录文件处理信息
        
        Args:
            file_path: 文件相对路径
            processing_info: 处理信息（如处理时间、节点数等）
        """
        if file_path in self.registry:
            if "processing_history" not in self.registry[file_path]:
                self.registry[file_path]["processing_history"] = []
            
            processing_record = {
                "timestamp": datetime.now().isoformat(),
                **processing_info
            }
            
            self.registry[file_path]["processing_history"].append(processing_record)
            self.registry[file_path]["last_processed"] = processing_record["timestamp"]
            self._save_registry()