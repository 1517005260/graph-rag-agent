# Search 模块

这个模块是项目中的搜索功能组件，提供了多种搜索策略，包括本地搜索、全局搜索、混合搜索以及深度研究搜索等。该模块通过知识图谱、向量检索和大语言模型结合的方式，实现了高效的知识检索和问答功能。

## 目录结构

```
search/
├── __init__.py                  # 模块初始化文件，导出主要类和工具类
├── local_search.py              # 本地搜索实现，基于向量检索的社区内精确查询
├── global_search.py             # 全局搜索实现，基于Map-Reduce模式的跨社区查询
├── utils.py                     # 工具函数，提供向量相似度计算等通用功能
└── tool/                        # 搜索工具集合目录
    ├── __init__.py              # 工具初始化文件
    ├── base.py                  # 搜索工具基类，提供通用功能
    ├── local_search_tool.py     # 本地搜索工具实现
    ├── global_search_tool.py    # 全局搜索工具实现
    ├── hybrid_tool.py           # 混合搜索工具实现，结合局部和全局搜索
    ├── naive_search_tool.py     # 简单搜索工具实现，仅使用向量搜索
    ├── deep_research_tool.py    # 深度研究工具实现，支持多步骤思考-搜索-推理
    ├── deeper_research_tool.py  # 增强版深度研究工具，添加社区感知和知识图谱功能
    └── reasoning/               # 推理相关组件目录
        ├── __init__.py          # 推理组件初始化
        ├── nlp.py               # 自然语言处理工具
        ├── prompts.py           # 提示模板
        ├── thinking.py          # 思考引擎，管理多轮迭代思考过程
        ├── search.py            # 推理搜索实现
        ├── validator.py         # 答案验证
        ├── community_enhance.py # 社区感知搜索增强器
        ├── kg_builder.py        # 动态知识图谱构建器
        ├── evidence.py          # 证据链收集和推理跟踪
        └── chain_of_exploration.py # 链式探索搜索实现
```

## 实现思路

该搜索模块采用分层架构，通过组合不同级别的搜索策略来满足不同场景的需求：

1. **基础搜索层**：
   - `LocalSearch`：基于向量检索，在特定社区内进行精确搜索，适合明确问题
   - `GlobalSearch`：基于Map-Reduce模式，跨社区进行广泛搜索，适合概念性问题

2. **工具封装层**：
   - `BaseSearchTool`：提供通用功能，如缓存管理、性能监控等
   - 各种具体搜索工具类（如`LocalSearchTool`，`GlobalSearchTool`等）封装底层搜索实现

3. **高级搜索策略**：
   - `HybridSearchTool`：类似LightRAG实现，结合低级实体详情和高级主题概念
   - `NaiveSearchTool`：简单的向量搜索实现，适合作为备选方案（根据微软的Graphrag实现，我们已经有了`__Chunk__`节点，为了简单，直接在Neo4j里做向量化即可，没有采用向量数据库）
   - `DeepResearchTool`：实现多步骤的思考-搜索-推理过程，适合复杂问题
   - `DeeperResearchTool`：增强版深度研究，添加社区感知和知识图谱分析

4. **推理组件**：
   - `ThinkingEngine`：管理多轮迭代的思考过程，支持分支推理
   - `QueryGenerator`：生成子查询和跟进查询
   - `DualPathSearcher`：支持同时使用多种方式搜索知识库
   - `CommunityAwareSearchEnhancer`：社区感知搜索增强器
   - `DynamicKnowledgeGraphBuilder`：动态构建知识子图
   - `EvidenceChainTracker`：收集和管理证据链，追踪推理步骤

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
6. **DeeperResearchTool**: 适合最复杂的问题，需要社区感知和知识图谱分析