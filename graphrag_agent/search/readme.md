# Search 模块

这个模块是项目中的搜索功能组件，提供了多种搜索策略，包括本地搜索、全局搜索、混合搜索以及深度研究搜索等。该模块通过知识图谱、向量检索和大语言模型结合的方式，实现了高效的知识检索和问答功能。

## 目录结构

```
graphrag_agent/search/
├── __init__.py                  # 模块初始化文件，导出主要类和工具类
├── local_search.py              # 本地搜索实现，基于向量检索的社区内精确查询
├── global_search.py             # 全局搜索实现，基于Map-Reduce模式的跨社区查询
├── utils.py                     # 向量工具类，提供余弦相似度计算和向量排序等功能
├── tool_registry.py             # 工具注册表，集中管理所有搜索工具类
├── retrieval_adapter.py         # 检索结果适配器，统一转换为RetrievalResult格式
└── tool/                        # 搜索工具集合目录
    ├── __init__.py              # 工具初始化文件
    ├── base.py                  # 搜索工具基类，提供通用功能
    ├── local_search_tool.py     # 本地搜索工具实现
    ├── global_search_tool.py    # 全局搜索工具实现
    ├── hybrid_tool.py           # 混合搜索工具实现，结合局部和全局搜索
    ├── naive_search_tool.py     # 简单搜索工具实现，仅使用向量搜索
    ├── deep_research_tool.py    # 深度研究工具实现，支持多步骤思考-搜索-推理
    ├── deeper_research_tool.py  # 增强版深度研究工具，添加社区感知和知识图谱功能
    ├── chain_exploration_tool.py # 链式探索工具，封装ChainOfExplorationSearcher为LangChain工具
    ├── hypothesis_tool.py       # 假设生成工具，针对复杂问题生成多种分析假设
    ├── validation_tool.py       # 答案验证工具，基于关键词和错误模式检测评估答案质量
    ├── deeper_research/         # 增强版深度研究辅助模块目录
    │   ├── __init__.py          # 辅助模块初始化
    │   ├── enhancer.py          # 搜索增强功能，如CoE增强搜索
    │   └── branching.py         # 分支推理功能，包括多分支创建、矛盾检测和引用生成
    └── reasoning/               # 推理相关组件目录
        ├── __init__.py          # 推理组件初始化
        ├── nlp.py               # 自然语言处理工具
        ├── prompts.py           # 提示模板
        ├── thinking.py          # 思考引擎，管理多轮迭代思考过程
        ├── search.py            # 推理搜索实现，包含QueryGenerator
        ├── validator.py         # 答案验证器，评估答案长度、相关性与可用性
        ├── community_enhance.py # 社区感知搜索增强器
        ├── kg_builder.py        # 动态知识图谱构建器
        ├── evidence.py          # 证据链收集和推理跟踪
        └── chain_of_exploration.py # 链式探索搜索实现
```

## 实现思路

### 多模态检索增强（参考 LightRAG 方案）

- `DocumentProcessor` 启用 MinerU 解析时，会并行调用视觉模型生成图片描述，并将结果写入 `modal_segments` / Neo4j 的 `visionSummary` 属性。
- 所有检索工具（`Naive`, `Local`, `Global`, `Hybrid`）在调用 LLM 前会利用 `ModalAssetProcessor.prepare_enhancement` 将 `[视觉解析]` 注入上下文，确保正文即可引用二维码/图像内容；回答末尾仍保留 Markdown 与结构化引用供前端渲染。
- `ModalEnricher` 统一查询 `__ModalSegment__` 节点、构建图片 URL、聚合 `vision_summary`；结果通过 `retrieval_adapter` 写入 `RetrievalResult.metadata.extra`，便于上层 Agent 与前端消费。
- 大型 Agent 工具（`GraphAgent`, `DeepResearchTool` 等）复用上述检索器，无需额外改动即可图文并茂；若需要进一步加工图片展示，可根据 `modal_image_details` 或 Markdown 信息自定义渲染。

```mermaid
flowchart LR
    subgraph MinerU Ingest
        A["DocumentProcessor\n(MinerU 模式)"]
        A -->|生成 modal_segments & vision_summary| B["Neo4j __ModalSegment__"]
    end

    subgraph Retrieval Layer
        B --> C["ModalEnricher.fetch_modal_map()"]
        C --> D["ModalAssetProcessor.prepare_enhancement()"]
        D --> E["Local/Naive/Global/Hybrid\n搜索工具"]
        E --> F["LLM 回答\n(带视觉解析)"]
        D --> G["Markdown / modal_image_details"]
    end

    F --> H["最终回答\n含视觉解析"]
    G --> H
```

> 若需要修改图片访问地址，可在 `.env` 中配置 `MINERU_ASSET_BASE_URL`；`modal_image_details.url` 同样遵循该前缀。

该搜索模块采用分层架构，通过组合不同级别的搜索策略来满足不同场景的需求：

1. **基础搜索层**：
   - `LocalSearch`：基于向量检索，在特定社区内进行精确搜索，适合明确问题
   - `GlobalSearch`：基于Map-Reduce模式，跨社区进行广泛搜索，适合概念性问题

2. **工具封装层**：
   - `BaseSearchTool`：提供通用功能，如缓存管理、性能监控等
   - 各种具体搜索工具类（如`LocalSearchTool`，`GlobalSearchTool`等）封装底层搜索实现
   - `tool_registry.py`：集中管理所有搜索工具类，提供统一的工具注册和获取接口

3. **高级搜索策略**：
   - `HybridSearchTool`：类似LightRAG实现，结合低级实体详情和高级主题概念
   - `NaiveSearchTool`：简单的向量搜索实现，适合作为备选方案（根据微软的Graphrag实现，我们已经有了`__Chunk__`节点，为了简单，直接在Neo4j里做向量化即可，没有采用向量数据库）
   - `DeepResearchTool`：实现多步骤的思考-搜索-推理过程，适合复杂问题
   - `DeeperResearchTool`：增强版深度研究，添加社区感知、知识图谱分析和分支推理能力

4. **专用工具**：
   - `ChainOfExplorationTool`：将Chain of Exploration封装为LangChain工具，支持图谱路径探索
   - `HypothesisGeneratorTool`：针对复杂问题生成多种分析假设，辅助深度研究
   - `AnswerValidationTool`：基于关键词和错误模式检测评估答案质量

5. **推理组件**：
   - `ThinkingEngine`：管理多轮迭代的思考过程，支持分支推理
   - `QueryGenerator`：生成子查询和跟进查询，支持假设生成
   - `DualPathSearcher`：支持同时使用多种方式搜索知识库
   - `CommunityAwareSearchEnhancer`：社区感知搜索增强器
   - `DynamicKnowledgeGraphBuilder`：动态构建知识子图
   - `EvidenceChainTracker`：收集和管理证据链，追踪推理步骤
   - `AnswerValidator`：答案验证器，评估答案的长度、相关性与可用性

6. **统一数据适配**：
   - `retrieval_adapter.py`：将不同搜索工具的原始输出统一转换为`RetrievalResult`数据模型
   - 提供`results_from_documents`、`results_from_entities`、`results_from_relationships`等转换函数
   - 支持多个检索结果的合并和去重

7. **工具函数**：
   - `utils.py`：提供`VectorUtils`类，包含余弦相似度计算、向量排序、批量相似度计算等功能

## 核心功能

### 向量检索与知识图谱结合

系统将Neo4j知识图谱与向量检索相结合，既能利用语义相似性进行检索，又能利用知识图谱的结构化关系进行推理：

```python
# LocalSearch中的向量检索核心实现
def as_retriever(self, **kwargs):
    final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks))
        .replace("$topCommunities", str(self.top_communities))
        .replace("$topOutsideRels", str(self.top_outside_rels))
        .replace("$topInsideRels", str(self.top_inside_rels))

    vector_store = Neo4jVector.from_existing_index(
        self.embeddings,
        url=db_manager.neo4j_uri,
        username=db_manager.neo4j_username,
        password=db_manager.neo4j_password,
        index_name=self.index_name,
        retrieval_query=final_query
    )
    
    return vector_store.as_retriever(
        search_kwargs={"k": self.top_entities}
    )
```

### Map-Reduce模式的全局搜索

全局搜索采用Map-Reduce模式，对社区数据进行批量处理后合并结果：

```python
# GlobalSearchTool中的核心搜索实现
def search(self, query_input: Any) -> List[str]:
    # 解析输入...
    
    # 获取社区数据
    community_data = self._get_community_data(keywords)
    
    # 处理社区数据，生成中间结果
    intermediate_results = self._process_communities(query, community_data)
    
    # 缓存结果
    self.cache_manager.set(cache_key, intermediate_results)
    
    return intermediate_results
```

### Chain of Thought推理，详细见reasoning部分的[readme](./tool/reasoning/readme.md)

深度研究工具实现了多步的思考-搜索-推理过程，能够处理复杂问题：

```python
# 思考引擎的核心推理实现
def generate_next_query(self) -> Dict[str, Any]:
    # 使用LLM进行推理分析，获取下一个搜索查询
    formatted_messages = [SystemMessage(content=REASON_PROMPT)] + self.msg_history

    # 调用LLM生成查询
    msg = self.llm.invoke(formatted_messages)
    query_think = msg.content if hasattr(msg, 'content') else str(msg)

    # 从AI响应中提取搜索查询
    queries = self.extract_queries(query_think)

    # 返回结果状态和查询
    return {
        "status": "has_query",
        "content": query_think,
        "queries": queries
    }
```

### 证据链跟踪与验证

为了提高答案可靠性，系统实现了证据链跟踪和验证机制：

```python
# 证据链跟踪核心实现
def add_evidence(self, step_id: str, source_id: str, content: str, source_type: str) -> str:
    # 生成证据ID
    evidence_id = hashlib.md5(f"{source_id}:{content[:50]}".encode()).hexdigest()[:10]

    # 创建证据记录
    evidence = {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "content": content,
        "source_type": source_type,
        "timestamp": time.time()
    }

    # 存储证据并关联到步骤
    self.evidence_items[evidence_id] = evidence

    # 查找步骤并添加证据ID
    for step in self.reasoning_steps:
        if step["step_id"] == step_id:
            if evidence_id not in step["evidence_ids"]:
                step["evidence_ids"].append(evidence_id)
            break

    return evidence_id
```

### 工具注册表机制

通过`tool_registry.py`集中管理所有搜索工具，便于多Agent层统一引用：

```python
# 注册所有搜索工具
TOOL_REGISTRY: Dict[str, Type[BaseSearchTool]] = {
    "local_search": LocalSearchTool,
    "global_search": GlobalSearchTool,
    "hybrid_search": HybridSearchTool,
    "naive_search": NaiveSearchTool,
    "deep_research": DeepResearchTool,
    "deeper_research": DeeperResearchTool,
}

# 额外专用工具（不继承BaseSearchTool）
EXTRA_TOOL_FACTORIES: Dict[str, Any] = {
    "chain_exploration": ChainOfExplorationTool,
    "hypothesis_generator": HypothesisGeneratorTool,
    "answer_validator": AnswerValidationTool,
}

# 获取工具类
def get_tool_class(tool_name: str) -> Type[BaseSearchTool]:
    return TOOL_REGISTRY[tool_name]
```

### 检索结果统一适配

通过`retrieval_adapter.py`将不同搜索工具的输出统一转换为`RetrievalResult`数据模型：

```python
# 从LangChain Documents生成统一的RetrievalResult
def results_from_documents(
    docs: Iterable[Any],
    *,
    source: str,
    default_confidence: float = 0.6,
    granularity: str = "Chunk",
) -> List[RetrievalResult]:
    results: List[RetrievalResult] = []
    for doc in docs:
        # 提取元数据和内容
        metadata_dict = getattr(doc, "metadata", {}) or {}
        page_content = getattr(doc, "page_content", None) or metadata_dict.get("text") or ""

        # 构建统一的元数据结构
        metadata = create_retrieval_metadata(
            source_id=str(metadata_dict.get("id") or uuid.uuid4()),
            source_type="chunk",
            confidence=metadata_dict.get("confidence", score),
            community_id=metadata_dict.get("community_id"),
            extra={"source": metadata_dict.get("source")}
        )

        # 创建标准化的检索结果
        results.append(
            create_retrieval_result(
                evidence=page_content,
                source=source,
                granularity=granularity,
                metadata=metadata,
                score=score,
            )
        )
    return results

# 合并多个检索结果并去重
def merge_retrieval_results(*result_groups: Iterable[RetrievalResult]) -> List[RetrievalResult]:
    merged: Dict[tuple[str, str], RetrievalResult] = {}
    for group in result_groups:
        for result in group:
            key = (result.metadata.source_id, result.granularity)
            existing = merged.get(key)
            # 保留分数更高的结果
            if existing is None or result.score > existing.score:
                merged[key] = result
    return list(merged.values())
```

### 向量相似度计算

`utils.py`提供了统一的向量操作工具类：

```python
class VectorUtils:
    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray],
                         vec2: Union[List[float], np.ndarray]) -> float:
        """计算两个向量的余弦相似度"""
        # 转换为numpy数组
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def batch_cosine_similarity(query_embedding: np.ndarray,
                            embeddings: List[np.ndarray]) -> np.ndarray:
        """批量计算余弦相似度，提高效率"""
        # 使用矩阵乘法一次性计算所有相似度
        matrix = np.vstack(embeddings)
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_normalized = matrix / matrix_norm
        similarities = np.dot(matrix_normalized, query_normalized)
        return similarities
```

### 假设生成工具

`HypothesisGeneratorTool`针对复杂问题生成多种分析假设：

```python
class HypothesisGeneratorTool:
    """生成多种假设以辅助深度研究"""

    def __init__(self):
        self.llm = get_llm_model()
        self.query_generator = QueryGenerator(self.llm, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT)

    def generate(self, query: str) -> List[str]:
        """针对复杂问题生成2-3个可能的分析假设"""
        return QueryGenerator.generate_multiple_hypotheses(query, self.llm)
```

### 答案验证工具

`AnswerValidationTool`基于关键词和错误模式检测评估答案质量：

```python
class AnswerValidationTool:
    """将AnswerValidator封装成LangChain Tool，评估答案的长度、相关性与可用性"""

    def __init__(self):
        keyword_tool = HybridSearchTool()
        self.validator = AnswerValidator(keyword_tool.extract_keywords)

    def validate(
        self,
        query: str,
        answer: str,
        *,
        reference_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """验证答案质量，返回验证结果"""
        result = self.validator.validate(
            query,
            answer,
            reference_keywords=reference_keywords,
        )
        return {"query": query, "answer": answer, "validation": result}
```

### 链式探索工具

`ChainOfExplorationTool`封装了链式探索搜索器为LangChain工具：

```python
class ChainOfExplorationTool:
    """将ChainOfExplorationSearcher封装为LangChain Tool"""

    def explore(
        self,
        query: str,
        *,
        start_entities: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        exploration_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行图谱探索并返回统一结构"""
        results = self.searcher.explore(
            query=query,
            starting_entities=start_entities or [],
            max_steps=max_steps or self.max_steps,
            exploration_width=exploration_width or self.exploration_width,
        )

        # 转换为统一的RetrievalResult格式
        entity_results = results_from_entities(
            results.get("entities", []), source="chain_exploration"
        )
        relation_results = results_from_relationships(
            results.get("relationships", []), source="chain_exploration"
        )
        content_results = results_from_documents(
            results.get("content", []),
            source="chain_exploration",
            granularity="Chunk",
        )

        merged_results = merge_retrieval_results(
            entity_results, relation_results, content_results
        )

        return {
            "query": query,
            "start_entities": start_entities,
            "summary": {...},
            "retrieval_results": results_to_payload(merged_results),
        }
```

### 分支推理与矛盾检测

`deeper_research/branching.py`提供了分支推理功能：

```python
# 创建多个推理分支
def create_multiple_reasoning_branches(
    query: str,
    initial_evidence: List[Dict],
    num_branches: int = 3
) -> List[Dict]:
    """基于初始证据创建多个推理分支，从不同角度分析问题"""
    # 实现细节...
    pass

# 检测和解决分支间的矛盾
def detect_and_resolve_contradictions(branches: List[Dict]) -> Dict[str, Any]:
    """检测多个推理分支之间的矛盾并尝试解决"""
    # 实现细节...
    pass

# 生成引用
def generate_citations(answer: str, evidence_items: Dict[str, Dict]) -> str:
    """为答案生成标准化的引用格式"""
    # 实现细节...
    pass
```

### 社区感知与Chain of Exploration

增强版深度研究工具整合了社区感知和链式探索能力：

```python
# Chain of Exploration核心实现
def explore(self, query: str, starting_entities: List[str], max_steps: int = 5, exploration_width: int = 3):
    # 初始化...

    # 多步探索
    for step in range(max_steps):
        if not current_entities:
            break

        # 1. 找出邻居节点
        neighbors = self._get_neighbors(current_entities)

        # 2. 评估每个邻居与查询的相关性
        scored_neighbors = self._score_neighbors_enhanced(
            neighbors, query, query_embedding, exploration_strategy
        )

        # 3. 让LLM决定探索方向
        next_entities, reasoning = self._decide_next_step_with_memory(
            query, current_entities, scored_neighbors, current_width, step
        )

        # 4. 更新已访问节点
        new_entities = [e for e in next_entities if e not in self.visited_nodes]
        self.visited_nodes.update(new_entities)

        # 5. 获取新发现实体的内容
        entity_info = self._get_entity_info(new_entities)
        results["entities"].extend(entity_info)

        # 继续探索...
```

## 使用场景

不同的搜索工具适用于不同的使用场景：

1. **LocalSearchTool**: 适合针对明确问题的精确搜索，快速找到相关内容
2. **GlobalSearchTool**: 适合概念性问题，需要广泛整合多个社区知识
3. **HybridSearchTool**: 适合需要同时了解具体实体和高级概念的问题
4. **NaiveSearchTool**: 适合简单问题，作为快速检索的备选方案
5. **DeepResearchTool**: 适合复杂问题，需要多步推理和深入挖掘
6. **DeeperResearchTool**: 适合最复杂的问题，需要社区感知、知识图谱分析和分支推理
7. **ChainOfExplorationTool**: 适合需要沿关系链探索的问题，从起始实体开始逐步扩展
8. **HypothesisGeneratorTool**: 适合开放性问题，需要从多个角度分析
9. **AnswerValidationTool**: 适合验证答案质量，确保答案的相关性和可用性

## 工具组合使用

通过`tool_registry.py`，可以灵活组合不同的搜索工具：

```python
from graphrag_agent.search.tool_registry import get_tool_class, create_extra_tool

# 获取基础搜索工具
HybridTool = get_tool_class("hybrid_search")
hybrid_search = HybridTool()

# 获取专用工具
hypothesis_tool = create_extra_tool("hypothesis_generator")
validator_tool = create_extra_tool("answer_validator")

# 组合使用：先生成假设，再搜索，最后验证
hypotheses = hypothesis_tool.generate(query)
search_results = hybrid_search.search(query)
validation = validator_tool.validate(query, answer)
```

## 数据流转

1. **输入层**: 用户查询 → 各搜索工具
2. **检索层**: 搜索工具 → 原始检索结果（Documents、Entities、Relationships）
3. **适配层**: `retrieval_adapter` → 统一的 `RetrievalResult` 格式
4. **输出层**: 标准化的检索结果 → Agent或下游组件

这种统一的数据流转保证了不同搜索工具之间的互操作性，便于在多Agent系统中集成使用。
