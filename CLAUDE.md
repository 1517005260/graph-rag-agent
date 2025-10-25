# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GraphRAG + Deep Search implementation with multi-agent collaboration system. The project combines knowledge graph construction, entity disambiguation, community detection, and multiple agent types (NaiveRAG, GraphAgent, HybridAgent, DeepResearchAgent, FusionGraphRAGAgent) to build an explainable and reasoning-capable Q&A system.

**Language**: Primarily Chinese (comments, docs, UI) with English code structures.

## Architecture

### Core Package Structure (`graphrag_agent/`)

- **agents/**: Agent implementations with Plan-Execute-Report multi-agent orchestration
  - `base.py`: BaseAgent with LangGraph integration, cache managers, and stream/non-stream support
  - Individual agents: `naive_rag_agent.py`, `graph_agent.py`, `hybrid_agent.py`, `deep_research_agent.py`, `fusion_agent.py`
  - `multi_agent/`: Plan-Execute-Report architecture
    - `planner/`: Clarifier, TaskDecomposer, PlanReviewer → generates `PlanSpec`
    - `executor/`: RetrievalExecutor, ResearchExecutor, ReflectionExecutor
    - `reporter/`: OutlineBuilder, SectionWriter, ConsistencyChecker (Map-Reduce)
    - `integration/`: Facade for backward compatibility

- **graph/**: Knowledge graph construction
  - `extraction/`: LLM-based entity/relationship extraction
  - `processing/`: Entity disambiguation and alignment
  - `indexing/`: Vector index management
  - `core/`: Connection manager for Neo4j

- **search/**: Multi-level search strategies
  - `local_search.py`: Entity-centric search with neighborhood exploration
  - `global_search.py`: Community-level search
  - `tool/`: NaiveSearchTool, DeepResearchTool, reasoning components

- **cache_manager/**: Two-tier caching (session-aware + global)
  - `backends/`: Hybrid memory/disk storage
  - `strategies/`: Context-aware and global key strategies
  - `vector_similarity/`: Semantic cache matching

- **community/**: Graph community detection and summarization
  - `detector/`: Leiden and SLLPA algorithms
  - `summary/`: LLM-based community summary generation

- **pipelines/ingestion/**: Multi-format document processing (TXT, PDF, MD, DOCX, DOC, CSV, JSON, YAML)

- **evaluation/**: 20+ evaluation metrics for answer quality, retrieval performance, graph quality

### Services Layer

- **server/**: FastAPI backend (`main.py`)
  - `routers/`: API endpoints
  - `services/agent_service.py`: Agent lifecycle management
  - `server_config/`: Auto-inherits from root config

- **frontend/**: Streamlit UI (`app.py`)
  - Debug mode with trace visualization, graph interaction
  - Knowledge graph visualization (Neo4j-style)

### Integration Entry Points

- **`graphrag_agent/integrations/build/main.py`**: Full pipeline orchestration
  - Calls `KnowledgeGraphBuilder` → `IndexCommunityBuilder` → `ChunkIndexBuilder`
  - Must run in this order; chunk index depends on entity index

- **`graphrag_agent/integrations/build/incremental_update.py`**: Incremental updates
  - `--once`: Single incremental build
  - `--daemon`: Background daemon for periodic updates

## Configuration

### Three-Tier Config System

1. **`.env`**: Runtime parameters, API keys, performance tuning (see `.env.example`)
2. **`graphrag_agent/config/settings.py`**: Knowledge graph schema (entity_types, relationship_types, theme, examples)
3. **Service configs**: Auto-inherit from layers 1-2

**Critical `.env` settings**:
```env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=http://localhost:13000/v1  # One-API or compatible proxy
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=12345678
```

**Editing entity/relationship schema**: Modify `graphrag_agent/config/settings.py`:
```python
entity_types = ["学生类型", "奖学金类型", "处分类型", "部门", "学生职责", "管理规定"]
relationship_types = ["申请", "评选", "违纪", "资助", "申诉", "管理", "权利义务", "互斥"]
```

## Common Commands

### Environment Setup
```bash
# Create environment
conda create -n graphrag python==3.10
conda activate graphrag

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Neo4j & One-API Setup
```bash
# Start Neo4j
cd graph-rag-agent/
docker compose up -d

# Start One-API (optional, for API proxy)
docker run --name one-api -d --restart always \
  -p 13000:3000 \
  -e TZ=Asia/Shanghai \
  -v /home/ubuntu/data/one-api:/data \
  justsong/one-api
```

### Knowledge Graph Construction
```bash
# Place source files in files/ directory first

# Full build (must run in this order)
python graphrag_agent/integrations/build/main.py

# Incremental update (single run)
python graphrag_agent/integrations/build/incremental_update.py --once

# Incremental update (daemon mode)
python graphrag_agent/integrations/build/incremental_update.py --daemon
```

**IMPORTANT**: Entity index must exist before chunk index. If running individual steps, complete entity indexing before chunk indexing to avoid errors.

### Testing
```bash
cd test/

# Non-streaming test
python search_without_stream.py

# Streaming test
python search_with_stream.py

# Evaluation
cd graphrag_agent/evaluation/test/
# See README in that directory
```

### Running Services
```bash
# Backend (FastAPI)
python server/main.py

# Frontend (Streamlit)
streamlit run frontend/app.py
```

## Agent System

### BaseAgent Architecture
All agents inherit from `graphrag_agent/agents/base.py`:
- Uses LangGraph for workflow orchestration (`StateGraph`, `ToolNode`)
- Dual LLM instances: `self.llm` (standard) and `self.stream_llm` (streaming)
- Two-tier caching: `cache_manager` (session context-aware) + `global_cache_manager` (cross-session)
- `MemorySaver` for conversation state

### Agent Types
- **NaiveRagAgent**: Basic vector retrieval
- **GraphAgent**: Graph-structure reasoning
- **HybridAgent**: Multi-strategy search
- **DeepResearchAgent**: Multi-step think-search-reasoning
- **FusionGraphRAGAgent**: Plan-Execute-Report multi-agent orchestration

### Streaming Implementation
**Note**: Current streaming is pseudo-streaming due to LangChain version constraints (generates full answer, then chunks it). True streaming awaits framework updates.

To test agents, comment out unwanted agents in test scripts to avoid long runtimes.

## Search Strategies

- **Local Search**: Entity-centric with neighborhood expansion (`graphrag_agent/search/local_search.py`)
- **Global Search**: Community-level aggregation (`graphrag_agent/search/global_search.py`)
- **Hybrid Search**: Combines multiple search modes
- **Deep Research**: Chain of Exploration on knowledge graph with evidence tracking

Search tools are registered via `graphrag_agent/search/tool_registry.py` and consumed by agents.

## Known Issues & Compatibility

### Model Compatibility
- **Tested & Working**: DeepSeek (20241226), GPT-4o
- **Known Issues**:
  - DeepSeek (20250324): Severe hallucination, entity extraction failures
  - Qwen series: LangChain/LangGraph compatibility issues; use [Qwen-Agent](https://qwen.readthedocs.io/zh-cn/latest/framework/qwen_agent.html) instead

### Embedding Disambiguation Limitation
Due to embedding similarity, "优秀学生" (honor title) may be confused with "国家奖学金" (scholarship). Future work: Fine-tune embeddings for domain-specific distinctions.

### Frontend Timeout for Deep Search
For deep research queries, disable timeout in `frontend/utils/api.py`:
```python
response = requests.post(
    f"{API_URL}/chat",
    json={...},
    # timeout=120  # Comment this out
)
```

## Development Guidelines

### File Registry
`file_registry.json` tracks ingested documents for incremental updates. Do not manually edit.

### Cache Management
- Session cache: `./cache/` (context-aware, conversation-scoped)
- Global cache: `./cache/global/` (persistent across sessions)
- Cache embedding provider: Configurable via `CACHE_EMBEDDING_PROVIDER` (openai / sentence_transformer)

### Entity Quality Mechanisms
- **Entity Disambiguation**: Maps mentions to canonical entities via string recall + vector reranking + NIL detection
- **Entity Alignment**: Detects and resolves conflicts within canonical entities, preserving all relationships

### Performance Tuning
Key `.env` parameters:
```env
MAX_WORKERS=4                # Thread pool size
BATCH_SIZE=100               # General batch size
ENTITY_BATCH_SIZE=50         # Entity operations
CHUNK_BATCH_SIZE=100         # Text chunks
EMBEDDING_BATCH_SIZE=64      # Vector generation
GDS_MEMORY_LIMIT=6           # Neo4j GDS memory (GB)
GDS_CONCURRENCY=4            # GDS parallelism
```

### Adding New Agents
1. Inherit from `BaseAgent`
2. Implement `_setup_tools()` to return tool list
3. Graph setup is handled by base class via `_setup_graph()`
4. Override `ask()` and `ask_stream()` for custom behavior

### Graph Consistency
Use `graphrag_agent/graph/graph_consistency_validator.py` to check and fix graph inconsistencies after bulk operations.

## Testing & Evaluation

Evaluation framework in `graphrag_agent/evaluation/`:
- **Metrics**: Answer quality, retrieval precision/recall, graph structure quality, deep research evaluation
- **Preprocessing**: Question-answer pair generation
- **Test harness**: See `graphrag_agent/evaluation/test/README.md`

Example test config in `test/search_with_stream.py`:
```python
TEST_CONFIG = {
    "queries": ["旷课多少学时会被退学？", ...],
    "max_wait_time": 300
}
```

## Multi-Agent (Plan-Execute-Report) Details

Located in `graphrag_agent/agents/multi_agent/`:

**Plan Phase** (`planner/`):
- Clarifier: Disambiguates user intent
- TaskDecomposer: Breaks query into subtasks
- PlanReviewer: Validates and optimizes plan
- Output: `PlanSpec` (task graph with dependencies)

**Execute Phase** (`executor/`):
- WorkerCoordinator: Dispatches tasks based on signals (retrieval/research/reflection)
- Records evidence and execution metadata
- Output: `ExecutionRecord` list

**Report Phase** (`reporter/`):
- OutlineBuilder: Generates document structure
- SectionWriter: Map-Reduce for long document generation
- ConsistencyChecker: Validates evidence citations and logical coherence
- Output: Final report with references

**Integration** (`integration/`):
- `LegacyCoordinatorFacade`: Provides same `process_query` interface as old coordinator for smooth migration

## Incremental Updates

`graphrag_agent/integrations/build/incremental/`:
- Detects file changes via `file_registry.json`
- Supports additions, deletions, modifications
- Conflict resolution strategies: `manual_first`, `auto_first`, `merge` (set in `settings.py` or `GRAPH_CONFLICT_STRATEGY` env var)

## Community Detection

Two algorithms supported (set via `settings.py` or `GRAPH_COMMUNITY_ALGORITHM`):
- **leiden**: Standard Leiden algorithm
- **sllpa**: Speaker-Listener Label Propagation (fallback to Leiden if no communities detected)

Community summaries generated via LLM for global search context.

## Documentation Standards

Each module has a `readme.md` explaining functionality. When adding features, update relevant readme.
