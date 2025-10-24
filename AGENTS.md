# Agents

The GraphRAG agent suite lives under `graphrag_agent/agents` and assembles several retrieval strategies on top of LangGraph. Each agent combines purpose-built tools, caching layers, and streaming helpers so you can pick the right balance between speed, coverage, and reasoning depth.

## Directory Layout

```
graphrag_agent/agents/
├── base.py                # Common LangGraph plumbing, caching, streaming utilities
├── naive_rag_agent.py     # Lightweight vector-only retrieval
├── hybrid_agent.py        # Hybrid keyword + vector search
├── graph_agent.py         # Graph-enhanced retrieval with local/global tools
├── deep_research_agent.py # Iterative deep-reasoning workflows
├── fusion_agent.py        # Wrapper over the multi-agent Plan-Execute-Report stack
├── multi_agent/           # Planner / executor / reporter modules for fusion workflows
└── readme.md              # Additional design notes (Chinese)
```

Supporting search tools are defined in `graphrag_agent/search/tool/`, shared prompts in `graphrag_agent/config/prompt.py`, and cache utilities under `graphrag_agent/cache_manager/`.

## Shared Runtime Behaviour

- All class-based agents inherit from `BaseAgent`, which wires LangGraph nodes (`agent → retrieve → generate`), memory checkpointing, and cache lookups before executing.
- Two cache tiers are available by default: a thread-scoped contextual cache and a global cache persisted under `./cache/<agent_name>`.
- `ask_with_trace` exposes an execution log, while `ask_stream` yields sentence-sized chunks for UI streaming paths.
- Performance metrics (duration per stage) are logged via `_log_performance`, making it easier to profile complex queries.

## Quick Usage

```python
from graphrag_agent.agents.graph_agent import GraphAgent

agent = GraphAgent()
answer = agent.ask("Summarize the 2023 product roadmap for GraphRAG.")
print(answer)

# Streaming (async context)
# async for chunk in agent.ask_stream("...", thread_id="session-42"):
#     consume(chunk)
```

Fusion workflows use the facade directly:

```python
from graphrag_agent.agents.fusion_agent import FusionGraphRAGAgent

agent = FusionGraphRAGAgent()
report, payload = agent.ask_with_trace("Compare leading knowledge-graph RAG approaches.")
```

## Agent Matrix

| Agent | Module | Retrieval stack | When to choose | Special notes |
| --- | --- | --- | --- | --- |
| `NaiveRagAgent` | `graphrag_agent/agents/naive_rag_agent.py` | Single vector index via `NaiveSearchTool` | Fast prototyping, low-latency lookups | Minimal keywords, cheapest cache footprint |
| `HybridAgent` | `graphrag_agent/agents/hybrid_agent.py` | Hybrid vector + keyword search with auto keyword extraction | Blended entity/context questions where keyword recall matters | Supports streaming generation with hierarchical headings |
| `GraphAgent` | `graphrag_agent/agents/graph_agent.py` | Graph-local and graph-global retrievers with reduce stage | Queries that benefit from graph traversals or multi-hop aggregation | Grades documents to decide between direct answer vs. reduce |
| `DeepResearchAgent` | `graphrag_agent/agents/deep_research_agent.py` | Iterative research tools (`DeepResearchTool`, `DeeperResearchTool`) with reasoning analyzers | Open-ended investigations requiring speculative thinking and exploration | Optional “show thinking” mode, community-aware enrichment, knowledge-graph exploration |
| `FusionGraphRAGAgent` | `graphrag_agent/agents/fusion_agent.py` | Multi-agent Plan → Execute → Report stack orchestrated by `MultiAgentFacade` | Long-form reports, decomposition-heavy tasks, citation-rich deliverables | Provides structured payload (plan, execution records, metrics, report artifacts) |

## Agent Details

### NaiveRagAgent (`graphrag_agent/agents/naive_rag_agent.py`)
- Wraps `NaiveSearchTool` and skips keyword extraction for tight query latency.
- Uses the `NAIVE_PROMPT` template to summarize context; headings are intentionally simple.
- Ideal for health checks (see `python test/search_without_stream.py`) and fallback paths when broader pipelines are unavailable.

### HybridAgent (`graphrag_agent/agents/hybrid_agent.py`)
- Coordinates `HybridSearchTool.get_tool()` (vector) and `get_global_tool()` (keyword/lexical) calls.
- Automatically extracts low/high-level keywords and caches them per thread to improve reruns.
- Generates sectioned answers with `LC_SYSTEM_PROMPT` and caches both per-session and globally.

### GraphAgent (`graphrag_agent/agents/graph_agent.py`)
- Blends `LocalSearchTool` and `GlobalSearchTool`, grading document relevance to decide whether to run a reduce step.
- Maintains execution logs for each decision point (`grade_documents`, `reduce`, etc.) to aid debugging.
- Best suited when the underlying knowledge graph is healthy and graph-reduce answers add value.

### DeepResearchAgent (`graphrag_agent/agents/deep_research_agent.py`)
- Toggles between `DeepResearchTool` and `DeeperResearchTool` (enhanced reasoning, exploration, community search).
- Supports `ask_with_thinking` to surface intermediate thoughts and optional knowledge-graph exploration streams.
- Handles generator or dict responses from research tools, normalizing `<think>...</think>` outputs before caching.

### FusionGraphRAGAgent (`graphrag_agent/agents/fusion_agent.py`)
- Lightweight shim over `MultiAgentFacade` for the Plan → Execute → Report orchestration stack in `agents/multi_agent/`.
- Maintains in-memory session/global caches to avoid re-running expensive decompositions.
- `ask_with_trace` returns both the final narrative and a payload containing planner output, execution records, metrics, and report artifacts (outline, sections, references).

## Plan-Execute-Report Stack (Fusion Workflows)

- **Planner (`agents/multi_agent/planner/`)**: Clarifies input, decomposes tasks, and reviews the plan, producing a `PlanSpec`.
- **Executor (`agents/multi_agent/executor/`)**: Worker coordinator dispatches retrieval, research, and reflection executors, capturing `ExecutionRecord` data.
- **Reporter (`agents/multi_agent/reporter/`)**: Outline builders, section writers, and Map-Reduce reducers synthesize long-form reports, with consistency checks and citation formatting.
- **Integration (`agents/multi_agent/integration/legacy_facade.py`)**: `MultiAgentFacade.process_query` converts orchestrator output into the legacy payload used by `FusionGraphRAGAgent`.

## Extending the Agent Suite

1. Inherit from `BaseAgent` (or follow the fusion wrapper pattern) and implement `_setup_tools` plus `_add_retrieval_edges`.
2. Reuse cache helpers in `BaseAgent` rather than rolling custom persistence. Prefer pure retrieval functions so they can be tested independently.
3. Register any new orchestration tools under `agents/multi_agent/tools/` if they participate in fusion flows.
4. Document inputs/outputs in module docstrings and add sample prompts under `documents/` or `assets/start.md` so operators can reproduce your scenario.
5. Update API wiring or orchestration registries if the new agent needs to be exposed through `server/` or CLI entry points.

Supplements such as smoke scripts should live beside the relevant agent under `test/`, following the `test_<feature>.py` naming convention.
