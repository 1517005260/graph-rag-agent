# 图谱构建模块

## 目录结构

```
graphrag_agent/graph/
├── __init__.py                # 模块入口，导出主要类和函数
├── core/                      # 核心功能组件
│   ├── __init__.py            # 导出核心组件
│   ├── base_indexer.py        # 基础索引器类
│   ├── graph_connection.py    # 图数据库连接管理
│   └── utils.py               # 工具函数(定时器、哈希生成等)
├── extraction/                # 实体关系提取组件
│   ├── __init__.py            # 导出提取组件
│   ├── entity_extractor.py    # 实体关系提取器
│   └── graph_writer.py        # 图数据写入器
├── graph_consistency_validator.py  # 图谱一致性验证工具
├── indexing/                  # 索引管理组件
│   ├── __init__.py            # 导出索引组件
│   ├── chunk_indexer.py       # 文本块索引管理
│   ├── embedding_manager.py   # 嵌入向量管理
│   └── entity_indexer.py      # 实体索引管理
├── processing/                # 实体处理组件
│   ├── __init__.py            # 导出处理组件
│   ├── entity_merger.py       # 实体合并管理
│   ├── similar_entity.py      # 相似实体检测
│   ├── entity_disambiguation.py # 实体消歧器
│   ├── entity_alignment.py    # 实体对齐器
│   └── entity_quality.py      # 实体质量处理器
└── structure/                 # 图结构构建组件
    ├── __init__.py            # 导出结构组件
    └── struct_builder.py      # 图结构构建器
```

## 模块概述

本模块是一个完整的图谱构建与查询系统，基于Neo4j图数据库实现。主要功能包括文档解析、实体关系提取、嵌入向量索引建立、相似实体检测与合并等。模块采用高度模块化设计，支持大规模数据处理和优化的查询性能。

## 核心实现思路

### 1. 图谱数据结构设计

系统基于以下核心节点类型构建图结构：
- `__Document__`：文档节点，代表一个完整的文档
- `__Chunk__`：文本块节点，文档的片段
- `__Entity__`：实体节点，从文本中提取的概念、对象等
- `__ModalSegment__`：多模态段落节点，承载 MinerU 抽取出的文本、图片、表格、公式等片段

节点之间的关系包括：
- `PART_OF`：Chunk与Document间的从属关系
- `NEXT_CHUNK`：文本块之间的顺序关系
- `MENTIONS`：文本块与实体间的提及关系
- `SIMILAR`：实体之间的相似关系
- `HAS_MODAL_SEGMENT`：Document 与多模态段落之间的归属关系
- `HAS_MODAL`：Chunk 与多模态段落之间的内容映射关系

### 2. 图谱构建流程

1. **文档结构化**：通过`GraphStructureBuilder`将文档拆分为Chunk并建立结构
2. **实体关系提取**：`EntityRelationExtractor`使用LLM从文本中提取实体和关系
3. **图谱写入**：`GraphWriter`将提取的实体和关系写入Neo4j
4. **向量索引建立**：`ChunkIndexManager`和`EntityIndexManager`为节点创建嵌入向量索引
5. **相似实体检测**：`SimilarEntityDetector`使用向量相似度和GDS算法检测重复实体
6. **实体合并**：`EntityMerger`基于LLM决策合并相似实体
7. **实体质量提升**：`EntityQualityProcessor`通过消歧和对齐进一步优化实体质量

### 3. 性能优化策略

- **批处理**：所有模块实现批量操作，减少数据库交互
- **并行处理**：利用线程池并行处理数据
- **缓存机制**：实体提取过程中使用缓存避免重复计算
- **高效索引**：合理的索引策略提升查询性能
- **错误恢复**：实现重试机制和错误恢复

## 核心功能与类

### 图数据库连接

`GraphConnectionManager`提供对Neo4j的连接管理，实现单例模式确保连接复用，统一管理查询和索引创建。

```python
# 示例使用
graph = connection_manager.get_connection()
result = graph.query("MATCH (n) RETURN count(n) as count")
```

### 图结构构建

`GraphStructureBuilder`负责创建文档和文本块节点，并建立它们之间的结构关系：

```python
processor_result = document_processor.process_directory()[0]
builder = GraphStructureBuilder()
builder.create_document(type="text", uri="path/to/doc", file_name=processor_result["filename"], domain="test")
chunks_with_hash = builder.create_relation_between_chunks(
    processor_result["filename"],
    processor_result["chunks"],
    chunk_annotations=processor_result.get("chunk_annotations"),
    segments=processor_result.get("modal_segments"),
)
```

### 实体关系提取

`EntityRelationExtractor`通过LLM从文本块中提取实体和关系：

```python
extractor = EntityRelationExtractor(llm, system_template, human_template, entity_types, relationship_types)
processed_chunks = extractor.process_chunks(file_contents)
```

### 向量索引管理

`ChunkIndexManager`和`EntityIndexManager`计算嵌入向量并创建索引：

```python
chunk_indexer = ChunkIndexManager()
vector_store = chunk_indexer.create_chunk_index()
```

### 相似实体检测和合并

`SimilarEntityDetector`和`EntityMerger`配合完成实体去重：

```python
detector = SimilarEntityDetector()
duplicate_candidates = detector.process_entities()

merger = EntityMerger()
merged_count = merger.process_duplicates(duplicate_candidates)
```

### 图谱一致性验证

`GraphConsistencyValidator`检查和修复图谱中的一致性问题：

```python
validator = GraphConsistencyValidator()
validation_result = validator.validate_graph()
repair_result = validator.repair_graph()
```

### 实体质量提升

`EntityQualityProcessor`整合实体消歧和对齐，进一步提升实体质量：

```python
quality_processor = EntityQualityProcessor()
result = quality_processor.process()
```

**包含的子模块**：

#### 实体消歧（EntityDisambiguator）

将mention映射到知识图谱中的规范实体：

```python
disambiguator = EntityDisambiguator()

# 消歧单个mention
result = disambiguator.disambiguate("实体名称")

# 批量消歧
results = disambiguator.batch_disambiguate(["实体1", "实体2"])

# 应用到整个图谱
updated_count = disambiguator.apply_to_graph()
```

**核心流程**：
1. **字符串召回**：使用编辑距离快速找到相似实体候选
2. **向量重排**：利用语义相似度对候选进行重新排序
3. **NIL检测**：识别未登录实体（知识库中不存在的新实体）
4. **应用到图谱**：为WCC分组中的实体设置`canonical_id`

#### 实体对齐（EntityAligner）

将具有相同canonical_id的实体对齐合并：

```python
aligner = EntityAligner()

# 执行完整对齐流程
result = aligner.align_all(batch_size=100)
```

**核心流程**：
1. **按canonical_id分组**：找出所有指向同一canonical实体的实体
2. **冲突检测**：通过关系类型相似度检测语义冲突
3. **冲突解决**：使用LLM智能决策保留哪个实体
4. **实体合并**：将同组实体合并，保留所有关系和属性

**关键特性**：
- 使用CALL子查询隔离边处理，确保流程健壮性
- 保留原始关系类型，不丢失语义信息
- 支持批量处理，避免内存溢出
- 智能冲突解决，基于Jaccard相似度和LLM判断
