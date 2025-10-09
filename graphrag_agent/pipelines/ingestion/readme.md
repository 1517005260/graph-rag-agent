# 文档处理器模块

## 项目结构
```
.
├── config/             # 配置文件目录
│   └── settings.py     # 全局配置参数
├── processor/          # 文档处理模块
│   ├── __init__.py     # 包初始化文件
│   ├── document_processor.py  # 文档处理器核心类 
│   ├── file_reader.py  # 多格式文件读取器
│   └── text_chunker.py # 中文文本分块器
```

## 模块简介

文档处理器(Document Processor)是一个用于读取、处理和分块各种格式文档的模块。该模块支持多种文件格式的读取，能够将文档内容按照语义进行智能分块，为后续的文本向量化和检索提供基础。

## 核心功能与实现思路

### 1. 文件读取 (FileReader)

`FileReader` 类提供了丰富的文件读取功能，支持多种常见文档格式：

- 文本文件 (TXT)
- PDF文档 (PDF)
- Markdown文档 (MD)
- Word文档 (DOCX, DOC)
- 数据文件 (CSV, JSON, YAML/YML)

读取器采用多种策略确保文件内容能够被正确读取：
- 支持多种编码格式，包括自动编码检测
- 对于旧版Word文档(.doc)采用多种备选读取方法
- 提供专用方法返回结构化数据（如CSV转为字典列表）

```python
# 使用示例
reader = FileReader(directory_path)
file_contents = reader.read_files(['.txt', '.pdf'])  # 指定读取的文件类型
```

### 2. 中文文本分块 (ChineseTextChunker)

`ChineseTextChunker` 类专门用于处理中文文本的分块需求，它能够：

- 使用HanLP进行准确的中文分词
- 根据语义界限（如句号、问号等）智能切分文本
- 支持块间重叠，保证上下文连贯性
- 处理异常情况和超长文本

```python
# 使用示例
chunker = ChineseTextChunker(chunk_size=500, overlap=100)
chunks = chunker.chunk_text(text_content)
```

### 3. 文档处理器 (DocumentProcessor)

`DocumentProcessor` 类整合文件读取和文本分块功能，提供完整的文档处理流程：

- 批量处理指定目录下的文件
- 生成文件统计信息（类型分布、内容长度等）
- 对每个文件进行分块处理
- 收集处理结果和错误信息

```python
# 使用示例
processor = DocumentProcessor(directory_path)
stats = processor.get_file_stats()  # 获取文件统计信息
results = processor.process_directory()  # 处理目录下所有支持的文件
```

## 核心函数

1. **`FileReader.read_files()`**: 根据指定的文件扩展名读取文件内容，返回文件名和内容的元组列表。

2. **`ChineseTextChunker.chunk_text()`**: 将单个文本智能分割成块，考虑语义边界和上下文重叠。

3. **`DocumentProcessor.process_directory()`**: 处理指定目录中的所有支持文件，返回包含文件信息、内容和分块结果的详细列表。

4. **`DocumentProcessor.get_file_stats()`**: 获取目录中文件的统计信息，包括文件类型分布、总内容长度和平均文件长度等。

## 使用场景

该模块适用于需要处理大量文档并进行后续语义分析的场景，如：

- 构建文档搜索系统
- 知识库问答应用
- 文档智能处理和归类
- 语料库建设与分析