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
  - `extraction/`: LLM-based entity/relationship extraction (Production-grade refactoring)
    - `entity_extractor.py`: Final merged version with production quality controls + Schema-aware routing
    - `entity_extractor_production.py`: Production variant with GraphConfig integration
    - `extractor_factory.py`: Factory pattern supporting dynamic/traditional modes
  - `processing/`: Entity disambiguation and alignment
  - `indexing/`: Vector index management
  - `core/`: Connection manager for Neo4j

- **search/**: Multi-level search strategies
  - `local_search.py`: Entity-centric search with neighborhood exploration
  - `global_search.py`: Community-level search
  - `tool/`: NaiveSearchTool, DeepResearchTool, reasoning components
  - `response_models.py`: Unified return structure (SearchResponse, ResponseBuilder)
  - `neo4j_vector_search.py`: Neo4j native vector search (engineering-level practice)

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

- **`graphrag_agent/integrations/build/incremental_update.py`**: Incremental updates (V1)
  - `--once`: Single incremental build
  - `--daemon`: Background daemon for periodic updates

- **`graphrag_agent/integrations/build/incremental_update_v2.py`**: Incremental updates (V2 - Recommended)
  - **L0/L1 Split Architecture**: Fast ingestion + background graph building
  - **L0 Fast Lane**: File searchable within 10 seconds (text chunking + vectorization)
  - **L1 Slow Lane**: Background entity extraction + graph construction via task queue
  - **Real-time Progress**: WebSocket broadcasting of build progress and file status
  - **Zero Wait Experience**: Users can search immediately after upload
  - Usage:
    - `--mode full`: Complete pipeline (L0 + L1)
    - `--mode l0`: Fast ingestion only
    - `--mode l1`: Submit graph building tasks only
    - `--file <path>`: Process single file
    - `--status`: Display queue status

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

# Incremental update V1 (single run)
python graphrag_agent/integrations/build/incremental_update.py --once

# Incremental update V1 (daemon mode)
python graphrag_agent/integrations/build/incremental_update.py --daemon

# Incremental update V2 (recommended - L0/L1 split architecture)
# Complete pipeline (L0 + L1)
python graphrag_agent/integrations/build/incremental_update_v2.py --mode full

# Fast ingestion only (L0)
python graphrag_agent/integrations/build/incremental_update_v2.py --mode l0

# Graph building only (L1)
python graphrag_agent/integrations/build/incremental_update_v2.py --mode l1

# Process single file
python graphrag_agent/integrations/build/incremental_update_v2.py --file /path/to/file.pdf

# Check queue status
python graphrag_agent/integrations/build/incremental_update_v2.py --status
```

**IMPORTANT**: Entity index must exist before chunk index. If running individual steps, complete entity indexing before chunk indexing to avoid errors.

**V2 vs V1**: V2 uses L0/L1 split - users can search immediately after L0 (< 10s), while graph construction happens in background (L1). V1 requires waiting for complete build.

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

### Unified Return Structure (Engineering-level Practice)

All search tools now follow a standardized return structure defined in `graphrag_agent/search/response_models.py`:

**Architecture:**
```
Tool (returns dict) → Agent (dict→prompt→LLM) → Frontend (shows answer)
```

**Return Format:**
```python
{
  "answer": str,                    # LLM generated answer
  "references": {                   # Referenced resources
    "chunks": List[str],            # Chunk IDs
    "entities": List[str],          # Entity IDs
    "communities": List[str],       # Community IDs
    "relationships": List[str]      # Relationship IDs
  },
  "meta": {                         # Metadata
    "retriever": str,               # naive | graph | hybrid | deep_research
    "search_time": float,           # Search time (seconds)
    "llm_time": float,              # LLM generation time
    "total_time": float,            # Total time
    "scores": List[float],          # Similarity scores
    "top_k": int,                   # Number of retrieved docs
    "cache_hit": bool,              # Cache hit status
    "timestamp": str                # ISO timestamp
  }
}
```

**Key Components:**
- `SearchResponse`: Pydantic model for type-safe responses
- `ResponseBuilder`: Utility class to construct standardized responses
- `create_error_response()`: Unified error handling

**Benefits:**
- Type-safe return values with Pydantic validation
- No more manual JSON string concatenation
- Frontend doesn't need to parse unreliable LLM-generated JSON
- Performance metrics automatically tracked
- Consistent error handling across all retrievers

**Usage Example:**
```python
from graphrag_agent.search.response_models import ResponseBuilder

builder = ResponseBuilder(retriever_name="naive")
builder.add_chunks(chunk_ids)
builder.add_scores(scores)
builder.set_timing(search_time=0.5, llm_time=1.2)
return builder.build_dict(answer=answer)
```

See `graphrag_agent/search/RESPONSE_FORMAT.md` for detailed documentation and migration guide.

### Neo4j Native Vector Search (Engineering-level Practice)

The system uses Neo4j's native vector search API (`db.index.vector.queryNodes`) instead of client-side similarity computation:

**Key Components:**
- `Neo4jVectorSearch` class in `graphrag_agent/search/neo4j_vector_search.py`
- Explicit vector index creation via `CREATE VECTOR INDEX`
- Global index configuration in `graphrag_agent/config/settings.py`

**Configuration:**
```python
# In settings.py or .env
CHUNK_VECTOR_INDEX = "chunk_embedding_index"
ENTITY_VECTOR_INDEX = "entity_embedding_index"
EMBEDDING_DIM = 1536  # Must match embedding model
VECTOR_SIMILARITY_FUNCTION = "cosine"  # cosine | euclidean | dot_product
```

**Benefits:**
- 10-100x performance improvement (server-side computation)
- Searches entire dataset (not limited to LIMIT 100)
- Reduced network transfer
- Leverages Neo4j's optimized vector indexing

**Before (❌ Client-side):**
```python
# Fetch limited candidates
chunks = graph.query("MATCH (c:__Chunk__) ... LIMIT 100")
# Compute similarity in Python
scored = VectorUtils.rank_by_similarity(query_embedding, chunks)
```

**After (✅ Server-side):**
```python
# Neo4j native vector search
results = vector_search.search_chunks(
    query_embedding=query_embedding,
    top_k=10
)
```

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

## Entity Extraction Refactoring (Production-Grade)

### Overview

The entity extraction system has been completely refactored to production-grade quality with **Three-Pillar Quality Control (三板斧)** + **Schema-aware Domain Routing**.

Located in: `graphrag_agent/graph/extraction/`

### Three-Pillar Quality Control (三板斧)

**1️⃣ Entity Normalization (`normalize_entity_name`)**
- Standardize punctuation: `（）` → `()`，`【】` → `[]`
- Remove whitespace
- Ensure consistent naming across chunks

**2️⃣ Type Whitelist Filtering**
- Only extract entities matching `allowed_entity_types` (dynamic or global)
- Only extract relationships matching `allowed_relation_types`
- Strict schema enforcement

**3️⃣ Frequency Filtering (≥ 2 occurrences)**
- Entity must appear at least `MIN_ENTITY_FREQUENCY` times (default: 2)
- Filters noise entities that appear only once
- Focuses on core concepts

**4️⃣ Similarity Deduplication**
- Uses `SequenceMatcher` with threshold 0.85 (configurable)
- Merges similar entity names (e.g., "学生管理办法" vs "学生管理规定")
- Reduces redundancy from chunk overlap

### Safe JSON Parsing (`safe_json_loads`)

LLM output is unreliable - this function handles:
- Extra text before/after JSON
- JSON wrapped in arrays `[{...}]`
- Malformed JSON → regex extraction fallback
- Complete parsing failure → return empty structure `{"entities": [], "relations": []}`

**Example handling:**
```python
# LLM returns: "Sure, here is the result:\n{\"entities\": [...]}"
# safe_json_loads() extracts the JSON block correctly
```

### Schema-aware Domain Routing

**Architecture:**
```
extractor_factory.py
  ↓ (if GraphConfig exists)
  Creates DynamicPromptBuilder(config)
  ↓
  extractor.prompt_builder = prompt_builder
  extractor.is_dynamic = True
  ↓
entity_extractor.py
  ↓
  _route_domain_for_document(filename, content)
    → LLM classifies document based on trigger_condition
  ↓
  _schema_for_domain(domain_name)
    → Returns (entity_types, relation_types) for that domain
  ↓
  post_process_entities(allowed_types=domain_entity_types)
  post_process_relations(allowed_relation_types=domain_relation_types)
```

**Two Modes:**

**Traditional Mode** (No GraphConfig):
- Uses global `entity_types` and `relationship_types` from `settings.py`
- Single schema for all documents
- Backward compatible

**Dynamic Mode** (With GraphConfig):
- Per-document domain classification via LLM
- Each domain has independent schema (entities/relations whitelist)
- Configured via AI Copilot or manual GraphConfig
- `extractor_factory` detects GraphConfig and enables dynamic mode automatically

### Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Entities (19 files) | 2544 | ~600 | -76% |
| Relationships | 11832 | ~2500 | -79% |
| LLM Calls | ~2000 | ~500 | -75% |
| Build Time | T | ~0.4T | -60% |

### Key Files

**`entity_extractor.py`** (Main implementation - used by build pipeline):
- Final merged version
- Contains all production quality controls
- Schema-aware domain routing with LLM-based classification
- `_route_domain_for_document()`: Domain classification
- `_schema_for_domain()`: Schema retrieval
- Compatible with both traditional and dynamic modes

**`entity_extractor_production.py`** (Alternative variant):
- Uses simpler GraphConfig.route_domain() approach
- Assumes GraphConfig has built-in routing methods
- Same quality controls

**`extractor_factory.py`**:
- Creates extractor instances
- Detects GraphConfig from storage
- Sets `extractor.prompt_builder` and `extractor.is_dynamic` in dynamic mode
- Falls back to traditional mode if no GraphConfig

### Usage Example

```python
from graphrag_agent.graph.extraction.extractor_factory import create_entity_extractor

# Factory automatically detects mode
extractor = create_entity_extractor(
    llm=llm,
    system_template=system_template,  # Used in traditional mode
    human_template=human_template,
    entity_types=entity_types,        # Fallback for traditional mode
    relationship_types=relationship_types
)

# In dynamic mode:
# - extractor.is_dynamic = True
# - extractor.prompt_builder.config = GraphConfig instance
# - Domain routing happens automatically per file

# In traditional mode:
# - extractor.is_dynamic = False
# - Uses provided entity_types/relationship_types globally
```

### Critical Bug Fixes

**`process_chunks_batch()` Empty Implementation Bug**:
- **Before**: `pass` → empty run, no extraction happening
- **After**: `return self.process_chunks(file_contents, progress_callback)`
- **Impact**: Batch processing (>100 chunks) now works correctly

## Chunking Strategy (Solving Entity Explosion)

### Problem: Entity/Relationship Explosion

**Symptoms**: 19 files producing 2544 entities and 11832 relationships - far beyond reasonable scale.

**Root Cause**:
- Token-level chunking with heavy overlap (chunk_size=500, overlap=100)
- Same entity appears in 10+ chunks → extracted 10+ times
- Insufficient deduplication → entity/relationship explosion

### Solution: Dual-Chunker Strategy

The system now supports **two specialized chunkers** for different purposes:

#### 1. GraphChunker (for Entity Extraction)
- **Config**: chunk_size=1000, overlap=50
- **Purpose**: Knowledge graph construction
- **Benefits**:
  - More context → better entity recognition
  - Fewer chunks → fewer LLM calls
  - Less redundancy → reduced deduplication pressure

**Usage**:
```python
from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor(
    directory_path="./files",
    chunker_mode='graph'  # Use GraphChunker
)
```

#### 2. RAGChunker (for Vector Search)
- **Config**: chunk_size=400, overlap=80
- **Purpose**: Semantic retrieval
- **Benefits**:
  - Fine-grained matching → more relevant results
  - Faster queries → smaller vector index
  - Better precision → focused semantic understanding

**Usage**:
```python
processor = DocumentProcessor(
    directory_path="./files",
    chunker_mode='rag'  # Use RAGChunker
)
```

### Entity Extraction Constraints

Added **hard constraint** in extraction prompt (`graph_prompts.py`):
```
⚠️ Only extract entities that appear ≥2 times in the text
- Appear 1 time → Skip (likely noise)
- Appear ≥2 times → Extract (indicates importance)
```

**Purpose**: Filter noise entities and focus on core concepts.

### Expected Improvements

| Metric | Old (500/100) | New (1000/50) | Improvement |
|--------|--------------|--------------|-------------|
| Chunks | ~2000 | ~500 | -75% |
| Entities | 2544 | ~600 (expected) | -76% |
| Relationships | 11832 | ~2500 (expected) | -79% |
| LLM Calls | ~2000 | ~500 | -75% |
| Build Time | T | ~0.4T | -60% |

See `graphrag_agent/pipelines/ingestion/CHUNKING_STRATEGY.md` for detailed documentation.

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
