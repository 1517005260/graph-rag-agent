# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraphRAG Agent is a comprehensive Retrieval-Augmented Generation (RAG) system that combines GraphRAG with private-domain Deep Search, implementing explainable and reasoning-capable intelligent Q&A through multi-agent collaboration and knowledge graph enhancement.

The project is written primarily in Python and uses:
- **LangChain/LangGraph** for agent orchestration and workflows
- **Neo4j** for knowledge graph storage
- **FastAPI** for backend services
- **Streamlit** for frontend interface

## Development Commands

### Environment Setup

```bash
# Create conda environment
conda create -n graphrag python==3.10
conda activate graphrag

# Install dependencies
pip install -r requirements.txt

# Initialize project
pip install -e .
```

### Building Knowledge Graph

```bash
# Initial full build
python graphrag_agent/integrations/build/main.py

# Incremental update (single run)
python graphrag_agent/integrations/build/incremental_update.py --once

# Daemon mode (periodic updates)
python graphrag_agent/integrations/build/incremental_update.py --daemon
```

### Testing

```bash
cd test/

# Non-streaming query test
python search_without_stream.py

# Streaming query test
python search_with_stream.py

# Agent-specific tests
python test_deep_agent.py
python test_cache_system.py
```

### Evaluation

```bash
cd graphrag_agent/evaluation/test/
# See README in that directory for details
```

### Running Services

```bash
# Start backend (from project root)
python server/main.py

# Start frontend (from project root, in separate terminal)
streamlit run frontend/app.py

# Start Neo4j via Docker
docker compose up -d
```

## Architecture

### Core Components

#### 1. Agent System (`graphrag_agent/agents/`)

The agent system has evolved from basic GraphRAG to a sophisticated multi-agent architecture:

- **BaseAgent**: Foundation class providing caching, LangGraph workflows, streaming, and performance monitoring
- **NaiveRagAgent**: Simple vector retrieval for basic queries
- **GraphAgent**: Graph-structure based agent using local/global search
- **HybridAgent**: Combines low-level entity details with high-level topic concepts
- **DeepResearchAgent**: Multi-step think-search-reason with visible reasoning traces
- **FusionGraphRAGAgent**: Advanced agent using Plan-Execute-Report multi-agent architecture

#### 2. Multi-Agent System (`graphrag_agent/agents/multi_agent/`)

**Plan-Execute-Report Architecture**:

```
FusionGraphRAGAgent → MultiAgentFacade → Orchestrator
                                            ↓
                    ┌───────────────────────┼───────────────────────┐
                    ↓                       ↓                       ↓
                Planner                 Executor                Reporter
        (澄清、分解、审校)           (检索、研究、反思)        (纲要、章节、一致性)
                    ↓                       ↓                       ↓
                PlanSpec            ExecutionRecords          ReportResult
```

**Planner** generates structured task plans via:
- **Clarifier**: Identifies ambiguity and generates clarification questions
- **TaskDecomposer**: Breaks queries into subtasks with dependency graph
- **PlanReviewer**: Audits and optimizes task plans

**WorkerCoordinator** dispatches executors based on task types:
- **RetrievalExecutor**: Runs local_search, global_search, hybrid_search, naive_search
- **ResearchExecutor**: Executes deep_research, chain_exploration
- **Reflector**: Quality assessment and improvement suggestions
- Supports sequential, parallel, and adaptive execution modes

**Reporter** generates structured reports using Map-Reduce pattern:
- **OutlineBuilder**: Creates report structure
- **SectionWriter**: Writes sections (traditional or Map-Reduce mode)
- **EvidenceMapper**: Maps large evidence sets to summaries (parallel)
- **SectionReducer**: Reduces summaries to coherent sections
- **ReportAssembler**: Assembles final report
- **ConsistencyChecker**: Validates content against evidence
- **CitationFormatter**: Generates citations
- Two-level caching: report-level and section-level

#### 3. Graph Module (`graphrag_agent/graph/`)

Handles knowledge graph construction:
- **extraction/**: LLM-driven entity and relation extraction
- **indexing/**: Entity and chunk indexing
- **processing/**: Entity disambiguation, alignment, merging, and quality improvement
- **structure/**: Graph structure building

Key mechanisms:
- **Entity Disambiguation**: Maps mentions to canonical entities using string recall + vector re-ranking + NIL detection
- **Entity Alignment**: Resolves conflicts within canonical entities while preserving all relationships

#### 4. Search Module (`graphrag_agent/search/`)

Multiple search strategies:
- **LocalSearch**: Neighborhood search for specific details
- **GlobalSearch**: Community-level search for macro analysis
- **HybridSearch**: Combines local + global
- **DeepResearch Tool** (`tool/deeper_research/`): Multi-step reasoning with:
  - Chain of Exploration on knowledge graph
  - Evidence chain tracking
  - Thinking process visualization
  - Dual-path search (precise + enhanced queries)

#### 5. Cache Manager (`graphrag_agent/cache_manager/`)

Multi-layer caching system:
- **Backends**: Memory, Disk, Hybrid, ThreadSafe wrappers
- **Strategies**: Simple keys, context-aware keys, keyword-aware keys
- **Vector Similarity Matching**: Semantic cache lookup
- Used by agents for session-level and global caching

#### 6. Community Detection (`graphrag_agent/community/`)

- **Algorithms**: Leiden and SLLPA
- **Summary Generation**: Automatic community summarization
- Configurable via `settings.py` (`community_algorithm = 'leiden'`)

#### 7. Evaluation System (`graphrag_agent/evaluation/`)

20+ metrics across dimensions:
- **Answer Quality**: EM, F1 Score
- **Retrieval Performance**: Precision, Utilization, Latency
- **Graph Evaluation**: Entity Coverage, Graph Coverage, Community Relevance
- **LLM Evaluation**: Coherence, Factual Consistency, Comprehensiveness
- **Deep Search**: Reasoning Coherence, Reasoning Depth, Iterative Improvement

### Data Flow

1. **Graph Construction**: Documents (files/) → DocumentProcessor → EntityExtractor → EntityDisambiguator/Aligner → Neo4j
2. **Querying**: User Query → Agent (choose based on complexity) → Search Tools (local/global/deep) → LLM → Response
3. **Multi-Agent Flow**: Query → Planner (PlanSpec) → WorkerCoordinator (ExecutionRecords) → Reporter (ReportResult)

## Configuration

### Environment Variables (`.env`)

```env
OPENAI_API_KEY='sk-xxx'
OPENAI_BASE_URL='http://localhost:13000/v1'  # One-API gateway
OPENAI_EMBEDDINGS_MODEL='text-embedding-3-large'
OPENAI_LLM_MODEL='gpt-4o'

NEO4J_URI='neo4j://localhost:7687'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='12345678'

# Cache embedding provider: 'openai' or 'sentence_transformer'
CACHE_EMBEDDING_PROVIDER='openai'
CACHE_SENTENCE_TRANSFORMER_MODEL='all-MiniLM-L6-v2'
MODEL_CACHE_ROOT='./cache'

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="xxx"
LANGSMITH_PROJECT="xxx"
```

### Graph Settings (`graphrag_agent/config/settings.py`)

Key configurations:
- `theme`: Knowledge graph theme (e.g., "华东理工大学学生管理")
- `entity_types` / `relationship_types`: Entity/relation schemas
- `community_algorithm`: 'leiden' or 'sllpa'
- `conflict_strategy`: 'manual_first', 'auto_first', or 'merge' for incremental updates
- `CHUNK_SIZE` / `OVERLAP`: Text chunking parameters
- Performance tuning: `MAX_WORKERS`, `BATCH_SIZE`, `GDS_CONCURRENCY`, etc.
- Disambiguation/alignment thresholds: `DISAMBIG_STRING_THRESHOLD`, `DISAMBIG_VECTOR_THRESHOLD`, etc.

Example questions for frontend are also defined here in `examples` list.

## Key Design Patterns

### 1. LangGraph State-Based Workflows

All agents inherit from `BaseAgent` and use LangGraph's `StateGraph`:
- Nodes: `agent`, `retrieve`, `generate`, `reduce`
- Conditional edges based on document grading
- Checkpointer for memory management
- Subclasses customize via `_add_retrieval_edges()`

### 2. Tool Abstraction

Search tools (`graphrag_agent/search/tool/`) inherit from base classes:
- Unified interface: `search()`, `search_stream()`, `thinking_stream()`
- RetrievalAdapter standardizes results across different search types
- Tools are registered with agents and invoked via LangGraph

### 3. Multi-Agent Orchestration

Plan-Execute-Report pattern with clear separation of concerns:
- **Planner**: Strategy layer (what to do)
- **Executor**: Execution layer (how to do it)
- **Reporter**: Presentation layer (how to present it)
- **Orchestrator**: Coordination layer (workflow management)
- **Legacy Facade**: Compatibility layer for gradual migration

### 4. Caching Strategy

Hierarchical caching:
- **Session Cache**: Per-thread caching for conversational context
- **Global Cache**: Cross-session caching for common queries
- **Vector Similarity**: Semantic matching for cache hits on similar queries
- **Report/Section Cache**: Two-level caching in Reporter with evidence fingerprinting

### 5. Evidence Tracking

EvidenceTracker maintains provenance:
- Records all retrieval sources (chunk/entity/community/document)
- Supports querying by type, source, granularity
- Automatic deduplication and ranking
- Used for citation generation and consistency checking

## Important Patterns and Conventions

### File Placement

- Source documents: `files/` directory (supports nested structure)
- Supported formats: TXT, PDF, MD, DOCX, DOC, CSV, JSON, YAML/YML

### Agent Selection

Choose agent based on query complexity:
- Simple factual queries → **NaiveRagAgent** or **GraphAgent**
- Queries needing relationship reasoning → **GraphAgent** or **HybridAgent**
- Complex multi-step reasoning → **DeepResearchAgent**
- Long-form structured reports → **FusionGraphRAGAgent** (Plan-Execute-Report)

### Streaming Responses

All agents implement `ask_stream()` for real-time streaming:
- Checks fast cache first
- For DeepResearch: use `show_thinking=True` to expose reasoning trace
- Current implementation is "pseudo-streaming" (generate full answer, then stream chunks) due to LangChain version limitations

### Working with Neo4j

- Connection managed via `graphrag_agent/config/neo4jdb.py`
- Use `GraphConnection` class for database operations
- Entity/relation schemas defined in `settings.py`
- Cypher queries abstracted in search tools

### Model Compatibility

Tested models:
- ✅ **DeepSeek (20241226)**: Full support
- ✅ **GPT-4o**: Full support
- ⚠️ **DeepSeek (20250324)**: Hallucination issues, may fail entity extraction
- ⚠️ **Qwen**: Can extract entities but LangChain/LangGraph compatibility issues

### Incremental Updates

Use `incremental_update.py` for dynamic graph updates:
- Tracks files via `file_registry.json`
- Detects additions/modifications/deletions
- Applies `conflict_strategy` for manual vs. auto edits
- Smart entity merging via disambiguation and alignment

## Common Development Tasks

### Adding a New Search Tool

1. Create tool class in `graphrag_agent/search/tool/`
2. Inherit from `BaseSearchTool` or relevant base class
3. Implement `search()` and `search_stream()` methods
4. Register tool in agent's `tools` list
5. Update tool descriptions in `settings.py` if needed

### Adding a New Agent

1. Create agent class in `graphrag_agent/agents/`
2. Inherit from `BaseAgent`
3. Override `_setup_graph()` or `_add_retrieval_edges()` for custom workflow
4. Implement agent-specific nodes if needed
5. Test with `test/search_without_stream.py` and `test/search_with_stream.py`

### Modifying Entity Extraction Schema

1. Update `entity_types` and `relationship_types` in `settings.py`
2. Adjust prompts in `graphrag_agent/config/prompt.py` if needed
3. Rebuild graph with `python graphrag_agent/integrations/build/main.py`

### Adding New Evaluation Metrics

1. Create metric class in `graphrag_agent/evaluation/metrics/`
2. Inherit from `BaseMetric`
3. Implement `evaluate()` method
4. Register in appropriate evaluator (answer/retrieval/graph/deep_search)

### Customizing Multi-Agent Behavior

To customize Plan-Execute-Report:
1. **Planning**: Modify `PlannerConfig` or implement custom planner components
2. **Execution**: Adjust `WorkerCoordinator` execution modes or add custom executors
3. **Reporting**: Configure `ReporterConfig` (Map-Reduce thresholds, reduce strategy, etc.)
4. **Factory**: Use `MultiAgentFactory.create_default_bundle()` with custom components

## Troubleshooting

### Graph Construction Issues

- If community detection fails (sllpa), switch to `community_algorithm = 'leiden'` in `settings.py`
- For entity extraction failures: check model compatibility (use DeepSeek 20241226 or GPT-4o)
- Missing dependencies for `.doc` files: install system packages per `requirements.txt` comments

### Performance Optimization

- Increase `MAX_WORKERS` and `BATCH_SIZE` for parallel processing
- Adjust `GDS_MEMORY_LIMIT` and `GDS_CONCURRENCY` for large graphs
- Enable `enable_parallel_map = True` in Reporter for faster Map-Reduce
- Use `CACHE_EMBEDDING_PROVIDER = 'sentence_transformer'` for local embedding to reduce API calls

### Deep Search Timeout

- Disable frontend timeout in `frontend/utils/api.py` (comment out `timeout=120`)
- For backend, adjust FastAPI `workers` in `server/main.py`

## External Resources

- [GraphRAG](https://github.com/microsoft/graphrag) - Microsoft's original GraphRAG framework
- [LightRAG](https://github.com/HKUDS/LightRAG) - Lightweight knowledge-enhanced generation
- [deep-searcher](https://github.com/zilliztech/deep-searcher) - Zilliz's private-domain semantic search
- [Neo4j LLM Graph Builder](https://github.com/neo4j-labs/llm-graph-builder)

## Notes

- Current streaming is pseudo-streaming due to LangChain version constraints
- Frontend examples configured in `settings.py` → `examples`
- One-API gateway recommended for unified LLM API management
- For Chinese chart display on Linux, install Chinese fonts (see `assets/start.md`)
