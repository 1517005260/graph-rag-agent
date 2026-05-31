# Repository Guidelines

## Project Structure & Module Organization
Core runtime sits in `graphrag_agent/`: `agents/` implements GraphRAG agents (multi-agent flows under `multi_agent/`), `graph/` and `integrations/build/` own graph ingestion, and `cache_manager/` wraps persistence. The FastAPI backend lives in `server/`; the Streamlit UI in `frontend/`. Tests and evaluation scripts reside in `test/`. Data inputs (`datasets/`, `documents/`) and generated artifacts (`cache/`, `files/`) stay separated—do not mix sources and outputs.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated Python 3.10+ environment.
- `pip install -r requirements.txt` installs runtime and evaluation deps; install the OS packages noted in the file for DOC/PDF support.
- `uvicorn server.main:app --reload` starts the FastAPI service; it consumes `.env` variables for Neo4j and LLM access.
- `streamlit run frontend/app.py` launches the chat UI. Pair it with the build pipelines via `python -m graphrag_agent.integrations.build.main --help` when refreshing graph indexes.
- `python -m unittest discover test -v` runs the default regression suite prior to any PR.

## Coding Style & Naming Conventions
Adhere to PEP 8: 4-space indentation, `snake_case` modules and functions, `PascalCase` classes, and upper-snake constants. Public methods should carry type hints and concise docstrings. Keep prompt templates readable and avoid trailing-space churn. Format locally with `black` and `isort` (no repo config, but matching their defaults keeps diffs clean).

## Testing Guidelines
Tests rely on `unittest` scripts in `test/`. Name new cases `test_{feature}.py` and mirror the package layout so discovery works. Run the suite with `python -m unittest discover test -v`; exercise targeted flows via explicit modules (e.g., `python test/test_deep_agent.py`). Document any external prerequisites (Neo4j, API keys, cached embeddings) in your PR and offer fallbacks or skips when they are unavailable.

## Commit & Pull Request Guidelines
Follow the short, imperative commit style in history (`add multi-agent config`, `unify configs`). Scope each logical change to one commit and use optional prefixes (`agents:`) when clarifying impact. PRs should include: summary, linked issue/TODO, test results, and configuration changes. Attach screenshots for `frontend/` changes and describe migration steps for datasets, caches, or graph indexes.

## Configuration & Data Handling
Clone `.env.example` when adding settings; never commit secrets. Update both `.env.example` and `assets/start.md` when introducing new knobs or services. Keep raw corpora in `documents/` or `datasets/`; persist generated embeddings and caches in `cache/` or `files/` and avoid adding them to git. Note required ports or Docker services in `docker-compose.yaml` for reviewers.
