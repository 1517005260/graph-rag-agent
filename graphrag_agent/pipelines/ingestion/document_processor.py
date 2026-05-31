import os
from typing import List, Dict, Optional, Any

from graphrag_agent.pipelines.ingestion.file_reader import FileReader
from graphrag_agent.pipelines.ingestion.text_chunker import ChineseTextChunker
from graphrag_agent.config.settings import FILES_DIR, CHUNK_SIZE, OVERLAP


class DocumentProcessor:
    """
    文档处理器，用于整合文件读取、文本分块和向量操作等功能
    """
    
    def __init__(self, directory_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        """
        初始化文档处理器
        
        Args:
            directory_path: 文件目录路径
            chunk_size: 分块大小
            overlap: 分块重叠大小
        """
        self.directory_path = directory_path
        self.file_reader = FileReader(directory_path)
        self.chunker = ChineseTextChunker(chunk_size, overlap)
        
    def process_directory(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        处理目录中的所有支持文件
        
        Args:
            file_extensions: 指定要处理的文件扩展名，如不指定则处理所有支持的类型
            recursive: 是否递归处理子目录，默认为True
            
        Returns:
            List[Dict]: 处理结果，每个文件一个字典，包含文件名、内容、分块等信息
        """
        # 读取文件
        file_contents = self.file_reader.read_files(file_extensions, recursive=recursive)
        
        # 打印调试信息
        print(f"DocumentProcessor找到的文件数量: {len(file_contents)}")
        if len(file_contents) > 0:
            print(f"文件类型: {[os.path.splitext(f[0])[1] for f in file_contents]}")
        
        # 处理每个文件
        results = []
        for filepath, content in file_contents:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # 创建文件处理结果字典
            file_result = {
                "filepath": filepath,  # 相对路径
                "filename": os.path.basename(filepath),  # 仅文件名
                "extension": file_ext,
                "content": content,
                "content_length": len(content),
                "chunks": None
            }
            
            # 对文本内容进行分块
            try:
                chunks = self.chunker.chunk_text(content)
                file_result["chunks"] = chunks
                file_result["chunk_count"] = len(chunks)
                
                # 计算每个块的长度
                chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
                file_result["chunk_lengths"] = chunk_lengths
                file_result["average_chunk_length"] = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                
            except Exception as e:
                file_result["chunk_error"] = str(e)
                print(f"分块错误 ({filepath}): {str(e)}")
                
            results.append(file_result)
            
        return results
        
    def get_file_stats(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> Dict[str, Any]:
        """
        获取目录中文件的统计信息
        
        Args:
            file_extensions: 指定要统计的文件扩展名，如不指定则处理所有支持的类型
            recursive: 是否递归统计子目录，默认为True
            
        Returns:
            Dict: 文件统计信息
        """
        # 读取文件
        file_contents = self.file_reader.read_files(file_extensions, recursive=recursive)
        
        # 统计每种扩展名的文件数量
        extension_counts = {}
        total_content_length = 0
        
        # 统计子目录数量
        directories = set()
        
        for filepath, content in file_contents:
            ext = os.path.splitext(filepath)[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # 记录文件所在的子目录
            dirpath = os.path.dirname(filepath)
            if dirpath:  # 非空表示在子目录中
                directories.add(dirpath)
                
            if content is not None:
                total_content_length += len(content)
            else:
                print(f"警告: 文件 {filepath} 的内容为None")
            
        return {
            "total_files": len(file_contents),
            "extension_counts": extension_counts,
            "total_content_length": total_content_length,
            "average_file_length": total_content_length / len(file_contents) if file_contents else 0,
            "directories": list(directories),
            "directory_count": len(directories)
        }
        
    def get_extension_type(self, extension: str) -> str:
        """
        获取文件扩展名对应的文档类型
        
        Args:
            extension: 文件扩展名（包括'.'，如'.pdf'）
            
        Returns:
            str: 文档类型描述
        """
        extension_types = {
            '.txt': '文本文件',
            '.pdf': 'PDF文档',
            '.md': 'Markdown文档',
            '.doc': 'Word文档',
            '.docx': 'Word文档',
            '.csv': 'CSV数据文件',
            '.json': 'JSON数据文件',
            '.yaml': 'YAML配置文件',
            '.yml': 'YAML配置文件',
        }
        
        return extension_types.get(extension.lower(), '未知类型')
        
        
if __name__ == "__main__":
    # 创建文档处理器
    processor = DocumentProcessor(FILES_DIR)
    
    # 列出目录中的所有文件
    print(f"目录 {FILES_DIR} 及其子目录中的所有文件:")
    all_files = processor.file_reader.list_all_files(recursive=True)
    for filepath in all_files:
        print(f"  {filepath}")
    
    # 获取文件统计信息
    stats = processor.get_file_stats(recursive=True)
    print("目录文件统计:")
    print(f"总文件数: {stats['total_files']}")
    print(f"子目录数: {stats['directory_count']}")
    if stats['directory_count'] > 0:
        print("子目录列表:")
        for directory in stats['directories']:
            print(f"  {directory}")
    
    print("文件类型分布:")
    for ext, count in stats["extension_counts"].items():
        print(f"  {ext} ({processor.get_extension_type(ext)}): {count}文件")
    print(f"总文本长度: {stats['total_content_length']}字符")
    print(f"平均文件长度: {stats['average_file_length']:.2f}字符")
    
    # 处理所有文件
    print("\n开始处理所有文件...")
    results = processor.process_directory(recursive=True)
    
    # 打印处理结果摘要
    for result in results:
        print(f"\n文件: {result['filepath']}")
        print(f"类型: {processor.get_extension_type(result['extension'])}")
        print(f"内容长度: {result['content_length']}字符")
        
        if result.get("chunks"):
            print(f"分块数量: {result['chunk_count']}")
            print(f"平均分块长度: {result['average_chunk_length']:.2f}字符")
        else:
            print(f"分块失败: {result.get('chunk_error', '未知错误')}")