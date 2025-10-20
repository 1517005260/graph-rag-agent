# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GraphRAG + DeepSearch implementation that combines knowledge graphs with private domain deep search for intelligent Q&A systems. The project integrates GraphRAG, multi-agent collaboration, and knowledge graph enhancement to build a complete RAG-based intelligent interaction solution.

**Tech Stack**: Python 3.10, LangChain, LangGraph, Neo4j, FastAPI, Streamlit

## Essential Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n graphrag python==3.10
conda activate graphrag

# Install dependencies
pip install -r requirements.txt

# Initialize project package
pip install -e .
```

### Neo4j Database
```bash
# Start Neo4j with Docker Compose
docker compose up -d

# Default credentials:
# Username: neo4j
# Password: 12345678
```

### Knowledge Graph Construction
```bash
# Full knowledge graph build (run first time or for complete rebuild)
python graphrag_agent/integrations/build/main.py

# Incremental update (single run)
python graphrag_agent/integrations/build/incremental_update.py --once

# Incremental update (daemon mode for continuous updates)
python graphrag_agent/integrations/build/incremental_update.py --daemon
```

### Testing
```bash
# Run tests from test directory
cd test/

# Non-streaming query test
python search_without_stream.py

# Streaming query test
python search_with_stream.py

# Cache system test
python test_cache_system.py
```

### Evaluation
```bash
cd graphrag_agent/evaluation/test
# See evaluation/test/readme.md for specific commands
```

### Frontend & Backend Services
```bash
# Start backend (FastAPI)
python server/main.py

# Start frontend (Streamlit) - in separate terminal
streamlit run frontend/app.py
```

## Architecture Overview

### Three-Stage Knowledge Graph Pipeline

**Stage 1: Base Graph Construction**
- Document ingestion (`pipelines/ingestion/`) processes multiple formats (TXT, PDF, MD, DOCX, CSV, JSON, YAML)
- Text chunking with configurable `CHUNK_SIZE` and `OVERLAP`
- LLM-driven entity and relationship extraction (`graph/extraction/`)
- Graph structure creation in Neo4j

**Stage 2: Entity Quality Enhancement & Community Detection**
- **Entity Disambiguation** (`graph/processing/entity_disambiguation.py`): Maps mentions to canonical entities using string recall, vector reranking, and NIL detection
- **Entity Alignment** (`graph/processing/entity_alignment.py`): Resolves conflicts between entities sharing the same canonical_id, preserves all relationship information
- Entity indexing with vector embeddings
- Community detection using Leiden or SLLPA algorithms (`community/`)

**Stage 3: Chunk Index Construction**
- Vector indexing for text chunks enabling similarity-based retrieval

### Multi-Agent System Architecture

The system implements a **Plan-Execute-Report** pattern through agent coordination:

**Base Agent** (`agents/base.py`):
- Built on LangGraph StateGraph with nodes: `agent → retrieve → generate`
- Provides caching (session-level and global), streaming support, and performance monitoring
- Subclasses customize via `_add_retrieval_edges()`

**Agent Types**:
1. **NaiveRagAgent**: Simple vector retrieval for straightforward queries
2. **GraphAgent**: Graph-based search with local/global strategies
3. **HybridAgent**: Combines multiple retrieval methods
4. **DeepResearchAgent**: Multi-step think-search-reason with thought process visualization
5. **FusionGraphRAGAgent**: Most advanced - orchestrates multiple agents via `GraphRAGAgentCoordinator`

**FusionGraphRAG Coordinator** (`agents/agent_coordinator.py`):
- **Retrieval Planner**: Analyzes query complexity and generates execution plan
- **Specialized Agents**: Local searcher, global searcher, deep explorer, chain explorer
- **Multi-path Execution**: Runs tasks by priority (local_search, global_search, exploration, chain_exploration)
- **Synthesizer**: Integrates all retrieval results into final answer
- **Thinking Engine**: Manages multi-round iterative reasoning

### Search Strategies

**LocalSearch** (`search/local_search.py`):
- Vector-based retrieval within specific communities
- Efficient for precise, entity-specific queries
- Uses Neo4jVector index with custom retrieval queries

**GlobalSearch** (`search/global_search.py`):
- Map-Reduce pattern across communities
- Aggregates community summaries for broad conceptual questions
- Two-stage: map (process communities) → reduce (synthesize)

**DeepResearch Tools** (`search/tool/`):
- **ThinkingEngine** (`reasoning/thinking.py`): Multi-round thought process with query generation
- **DualPathSearcher** (`reasoning/search.py`): Parallel search with/without knowledge base context
- **EvidenceChainTracker** (`reasoning/evidence.py`): Tracks reasoning steps and evidence sources
- **ChainOfExploration** (`reasoning/chain_of_exploration.py`): Multi-hop graph traversal starting from seed entities
- **CommunityAwareEnhancer** (`reasoning/community_enhance.py`): Leverages community structure for context
- **DynamicKGBuilder** (`reasoning/kg_builder.py`): Constructs query-specific knowledge subgraphs

### Incremental Update System

**File Change Management** (`integrations/build/incremental/`):
- `FileChangeManager`: Detects file additions, modifications, deletions
- `ManualEditManager`: Protects user edits in Neo4j from auto-update overwrites
- Conflict resolution strategies: `manual_first`, `auto_first`, `merge`

**Smart Scheduling** (`incremental_update_scheduler.py`):
- File changes: High frequency (default 5 min)
- Entity embeddings: Medium frequency (default 30 min)
- Community detection: Low frequency (default 48 hours)

**Consistency Validation**:
- Detects orphaned nodes, broken relationships
- Repairs data integrity issues

### Caching System

**Multi-Backend Architecture** (`cache_manager/`):
- Memory cache, disk cache, hybrid cache with LRU eviction
- Thread-safe wrappers for concurrent access
- Vector similarity matching for semantic cache hits

**Cache Key Strategies** (`strategies/`):
- Simple: Hash-based key generation
- Context-aware: Incorporates conversation history
- Keyword-aware: Extracts query keywords for better matching

### Evaluation Framework

**20+ Metrics** across categories (`evaluation/metrics/`):
- **Answer Quality**: EM, F1 Score, BLEU, ROUGE
- **Retrieval**: Precision, Recall, Utilization, Latency
- **Graph Coverage**: Entity Coverage, Relationship Coverage, Community Relevance
- **LLM Metrics**: Coherence, Factual Consistency, Comprehensiveness
- **Deep Search**: Reasoning Coherence, Reasoning Depth, Iterative Improvement

## Configuration

### Main Settings (`graphrag_agent/config/settings.py`)

**Graph Construction**:
```python
theme = "悟空传"  # Knowledge domain
entity_types = ["人物", "妖怪", "位置"]
relationship_types = ["师徒", "师兄弟", "对抗", ...]
similarity_threshold = 0.9  # Entity merging threshold
community_algorithm = 'leiden'  # or 'sllpa'
```

**Text Chunking**:
```python
CHUNK_SIZE = 300
OVERLAP = 50
MAX_TEXT_LENGTH = 500000
```

**Performance Tuning**:
```python
MAX_WORKERS = 4
BATCH_SIZE = 100
ENTITY_BATCH_SIZE = 50
EMBEDDING_BATCH_SIZE = 64
LLM_BATCH_SIZE = 5
```

**Entity Quality**:
```python
DISAMBIG_STRING_THRESHOLD = 0.7
DISAMBIG_VECTOR_THRESHOLD = 0.85
DISAMBIG_NIL_THRESHOLD = 0.6
ALIGNMENT_CONFLICT_THRESHOLD = 0.5
```

### Environment Variables (`.env`)
```env
OPENAI_API_KEY = 'sk-xxx'
OPENAI_BASE_URL = 'http://localhost:13000/v1'
OPENAI_EMBEDDINGS_MODEL = 'text-embedding-3-large'
OPENAI_LLM_MODEL = 'gpt-4o'

NEO4J_URI = 'neo4j://localhost:7687'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = '12345678'

CACHE_EMBEDDING_PROVIDER = 'openai'  # or 'sentence_transformer'
```

## Development Workflow

### Adding New Documents
1. Place files in `files/` directory (supports nested folders)
2. Run incremental update: `python graphrag_agent/integrations/build/incremental_update.py --once`
3. System automatically detects new files, extracts entities/relationships, updates indices

### Modifying Entity Types or Relationships
1. Update `entity_types` and `relationship_types` in `config/settings.py`
2. Full rebuild required: `python graphrag_agent/integrations/build/main.py`
3. Incremental updates only work for document changes, not schema changes

### Testing New Agent Behavior
1. Modify agent in `graphrag_agent/agents/`
2. Update test queries in `test/search_with_stream.py` or `test/search_without_stream.py`
3. Run `python test/search_with_stream.py` to see streaming output with performance metrics

### Custom Search Tools
1. Inherit from `BaseSearchTool` in `search/tool/base.py`
2. Implement `search()` method
3. Register in Agent's tool list
4. See `search/tool/deep_research_tool.py` for reference implementation

## Important Notes

### Model Compatibility
- **Tested Models**: DeepSeek (20241226), GPT-4o
- **Issues**: DeepSeek (20250324) has hallucination issues; Qwen models incompatible with LangChain/LangGraph
- Entity extraction failures often stem from model not following prompt templates

### Streaming Limitations
Due to LangChain version constraints, current streaming is "pseudo-streaming" - answer generated completely, then chunked for output.

### Index Dependencies
When running individual build stages:
- **Must** complete entity index build before chunk index build
- Chunk indexing depends on entity embeddings existing in Neo4j

### Manual Neo4j Edits
To preserve manual graph edits during incremental updates:
- Set `conflict_strategy="manual_first"` in `config/settings.py`
- System tracks `manual_edit` property on nodes/relationships

### Cache Warming
First query to new knowledge graph will be slow (no cache). Subsequent similar queries benefit from:
- Semantic cache matching via vector similarity
- Result caching at multiple levels (fast cache, session cache, global cache)

## Key File Locations

- **Entry Points**: `graphrag_agent/integrations/build/main.py` (build), `server/main.py` (backend), `frontend/app.py` (frontend)
- **Agent Orchestration**: `graphrag_agent/agents/agent_coordinator.py`
- **Core Search**: `graphrag_agent/search/local_search.py`, `global_search.py`
- **Reasoning**: `graphrag_agent/search/tool/reasoning/thinking.py`
- **Entity Quality**: `graphrag_agent/graph/processing/entity_disambiguation.py`, `entity_alignment.py`
- **Incremental Updates**: `graphrag_agent/integrations/build/incremental_update.py`
- **Evaluation**: `graphrag_agent/evaluation/core/base_evaluator.py`
