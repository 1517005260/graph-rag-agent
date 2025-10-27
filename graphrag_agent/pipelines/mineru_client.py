"""
MinerU Client Library for GraphRAG Agent
用于与 MinerU API 服务器交互的客户端库

使用示例:
    from graphrag_agent.integrations.mineru_client import MinerUClient

    client = MinerUClient("http://localhost:8899")

    # 解析单个文件
    result = client.parse_file("document.pdf", formula_enable=True, table_enable=True)
    print(result.markdown_path)

    # 批量解析
    results = client.parse_batch(["doc1.pdf", "doc2.pdf"], lang="en")
    for r in results.results:
        print(r.filename, r.status)
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class ParseOptions:
    """文档解析选项"""

    backend: str = "pipeline"
    parse_method: str = "auto"
    lang: str = "ch"
    formula_enable: bool = True
    table_enable: bool = True
    start_page: int = 0
    end_page: Optional[int] = None
    output_format: str = "mm_md"


@dataclass
class ParseResult:
    """解析结果"""

    task_id: str
    filename: str
    status: str
    original_filename: Optional[str] = None
    output_dir: Optional[str] = None
    markdown_path: Optional[str] = None
    content_list_path: Optional[str] = None
    images_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ParseResult":
        """从字典创建 ParseResult"""
        return cls(**data)

    def is_success(self) -> bool:
        """是否解析成功"""
        return self.status == "success"

    def get_normalized_filename(self) -> str:
        """获取 MinerU 使用的标准化文件名"""
        return self.metadata.get("normalized_filename", self.filename)

    def get_markdown_content(self) -> Optional[str]:
        """读取 Markdown 内容"""
        if self.markdown_path and Path(self.markdown_path).exists():
            with open(self.markdown_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def get_content_list(self) -> Optional[List[Dict]]:
        """读取内容列表"""
        if self.content_list_path and Path(self.content_list_path).exists():
            with open(self.content_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_images(self) -> List[Path]:
        """获取所有图像文件路径"""
        if self.images_dir and Path(self.images_dir).exists():
            images_path = Path(self.images_dir)
            return list(images_path.glob("*"))
        return []


@dataclass
class BatchParseResult:
    """批量解析结果"""

    total: int
    success: int
    failed: int
    results: List[ParseResult]

    @classmethod
    def from_dict(cls, data: dict) -> "BatchParseResult":
        """从字典创建 BatchParseResult"""
        results = [ParseResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            total=data["total"],
            success=data["success"],
            failed=data["failed"],
            results=results
        )

    def get_success_results(self) -> List[ParseResult]:
        """获取所有成功的结果"""
        return [r for r in self.results if r.is_success()]

    def get_failed_results(self) -> List[ParseResult]:
        """获取所有失败的结果"""
        return [r for r in self.results if not r.is_success()]


class MinerUClient:
    """MinerU API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8899", timeout: int = 300):
        """
        初始化客户端

        Args:
            base_url: API 服务器地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            服务器健康状态信息
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_config(self) -> Dict[str, Any]:
        """
        获取服务器配置

        Returns:
            服务器配置信息
        """
        response = requests.get(f"{self.base_url}/config", timeout=10)
        response.raise_for_status()
        return response.json()

    def parse_file(
        self,
        file_path: Union[str, Path],
        backend: str = "pipeline",
        parse_method: str = "auto",
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        start_page: int = 0,
        end_page: Optional[int] = None,
        output_format: str = "mm_md"
    ) -> ParseResult:
        """
        解析单个文件

        Args:
            file_path: 文件路径
            backend: 解析后端 (pipeline, vlm-*)
            parse_method: 解析方法 (auto, ocr, txt)
            lang: 语言 (ch, en)
            formula_enable: 是否识别公式
            table_enable: 是否识别表格
            start_page: 起始页码
            end_page: 结束页码
            output_format: 输出格式 (mm_md, md_only, content_list)

        Returns:
            ParseResult: 解析结果
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 准备表单数据
        data = {
            "backend": backend,
            "parse_method": parse_method,
            "lang": lang,
            "formula_enable": str(formula_enable).lower(),
            "table_enable": str(table_enable).lower(),
            "start_page": str(start_page),
            "output_format": output_format
        }

        if end_page is not None:
            data["end_page"] = str(end_page)

        # 准备文件
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}

            # 发送请求
            response = requests.post(
                f"{self.base_url}/parse",
                files=files,
                data=data,
                timeout=self.timeout
            )

        response.raise_for_status()
        return ParseResult.from_dict(response.json())

    def parse_batch(
        self,
        file_paths: List[Union[str, Path]],
        backend: str = "pipeline",
        parse_method: str = "auto",
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        start_page: int = 0,
        end_page: Optional[int] = None,
        output_format: str = "mm_md"
    ) -> BatchParseResult:
        """
        批量解析文件

        Args:
            file_paths: 文件路径列表
            其他参数同 parse_file

        Returns:
            BatchParseResult: 批量解析结果
        """
        # 验证所有文件存在
        valid_paths = []
        for fp in file_paths:
            path = Path(fp)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            valid_paths.append(path)

        # 准备表单数据
        data = {
            "backend": backend,
            "parse_method": parse_method,
            "lang": lang,
            "formula_enable": str(formula_enable).lower(),
            "table_enable": str(table_enable).lower(),
            "start_page": str(start_page),
            "output_format": output_format
        }

        if end_page is not None:
            data["end_page"] = str(end_page)

        # 准备文件
        files = []
        file_handles = []
        try:
            for path in valid_paths:
                f = open(path, "rb")
                file_handles.append(f)
                files.append(("files", (path.name, f, "application/octet-stream")))

            # 发送请求
            response = requests.post(
                f"{self.base_url}/parse/batch",
                files=files,
                data=data,
                timeout=self.timeout
            )

            response.raise_for_status()
            return BatchParseResult.from_dict(response.json())

        finally:
            # 关闭所有文件句柄
            for f in file_handles:
                f.close()

    def download_file(self, task_id: str, filename: str, save_path: Optional[Union[str, Path]] = None) -> Path:
        """
        下载处理后的文件

        Args:
            task_id: 任务ID
            filename: 文件名
            save_path: 保存路径，默认为当前目录

        Returns:
            保存的文件路径
        """
        response = requests.get(
            f"{self.base_url}/download/{task_id}/{filename}",
            timeout=self.timeout
        )
        response.raise_for_status()

        # 确定保存路径
        if save_path is None:
            save_path = Path(filename)
        else:
            save_path = Path(save_path)
            if save_path.is_dir():
                save_path = save_path / filename

        # 保存文件
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)

        return save_path

    def cleanup_task(self, task_id: str) -> Dict[str, Any]:
        """
        清理指定任务的输出文件

        Args:
            task_id: 任务ID

        Returns:
            清理结果
        """
        response = requests.delete(
            f"{self.base_url}/cleanup/{task_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def cleanup_old_outputs(self, hours: int = 24) -> Dict[str, Any]:
        """
        清理超过指定时间的输出文件

        Args:
            hours: 保留时间（小时）

        Returns:
            清理结果
        """
        response = requests.post(
            f"{self.base_url}/cleanup/old",
            params={"hours": hours},
            timeout=30
        )
        response.raise_for_status()
        return response.json()


# ============== 便捷函数 ==============
def parse_document(
    file_path: Union[str, Path],
    server_url: str = "http://localhost:8899",
    **kwargs
) -> ParseResult:
    """
    便捷函数：解析单个文档

    Args:
        file_path: 文件路径
        server_url: MinerU 服务器地址
        **kwargs: 其他解析选项

    Returns:
        ParseResult: 解析结果
    """
    client = MinerUClient(server_url)
    return client.parse_file(file_path, **kwargs)


def parse_documents(
    file_paths: List[Union[str, Path]],
    server_url: str = "http://localhost:8899",
    **kwargs
) -> BatchParseResult:
    """
    便捷函数：批量解析文档

    Args:
        file_paths: 文件路径列表
        server_url: MinerU 服务器地址
        **kwargs: 其他解析选项

    Returns:
        BatchParseResult: 批量解析结果
    """
    client = MinerUClient(server_url)
    return client.parse_batch(file_paths, **kwargs)


# ============== 示例用法 ==============
if __name__ == "__main__":
    # 创建客户端
    client = MinerUClient("http://localhost:8899")

    # 健康检查
    print("Health Check:")
    print(json.dumps(client.health_check(), indent=2, ensure_ascii=False))

    # 获取配置
    print("\nServer Config:")
    print(json.dumps(client.get_config(), indent=2, ensure_ascii=False))

    # 解析示例文件（如果存在）
    example_pdf = Path("example.pdf")
    if example_pdf.exists():
        print(f"\nParsing {example_pdf}...")
        result = client.parse_file(
            example_pdf,
            formula_enable=True,
            table_enable=True,
            lang="ch"
        )

        print(f"Status: {result.status}")
        print(f"Task ID: {result.task_id}")
        print(f"Markdown Path: {result.markdown_path}")
        print(f"Images Dir: {result.images_dir}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        if result.is_success():
            # 读取 Markdown 内容
            md_content = result.get_markdown_content()
            if md_content:
                print(f"\nMarkdown Preview (first 500 chars):")
                print(md_content[:500])

            # 获取图像
            images = result.get_images()
            print(f"\nExtracted Images: {len(images)}")
            for img in images[:5]:  # 只显示前5个
                print(f"  - {img.name}")
