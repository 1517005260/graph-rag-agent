# Repository Guidelines

## Project Structure & Module Organization
The core library lives in `graphrag_agent/` with subpackages for `agents/`, `graph/`, `config/`, and `integrations/build/` (Neo4j ingest pipeline). Server code sits under `server/` (FastAPI) and the Streamlit frontend lives in `frontend/`. Shared datasets reside in `data/` and `files/`, while evaluation artifacts and training scripts are in `training/` and `graphrag_agent/evaluation/`. Tests and smoke scripts are collected inside `test/`.

## Build, Test, and Development Commands
Install dependencies with `python -m pip install -r requirements.txt`. Launch the API via `uvicorn server.main:app --reload`. Run the research agents end-to-end using `python -m graphrag_agent.integrations.build.main`. The Streamlit client starts with `streamlit run frontend/app.py`. For quick verification, execute `python test/search_without_stream.py`; append `--stream` or use `search_with_stream.py` to validate streaming paths. Docker users can provision services through `docker-compose up` once `.env` is prepared.

## Coding Style & Naming Conventions
Follow standard Python 3 style with 4-space indentation, type hints where practical, and descriptive module-level docstrings. Stick to snake_case for modules, functions, and variables; classes use PascalCase. Keep agent configuration files (`config/*.py`) declarative and avoid side-effect imports in `__init__` files. Use docstrings to clarify reasoning steps for new agent behaviors, and prefer pure functions in `graph/` utilities to aid reuse.

## Testing Guidelines
Pytest-style scripts live in `test/`; add new cases beside the feature under test or extend the relevant helper script. Name test modules `test_<feature>.py` and include scenario-focused functions (e.g., `test_hybrid_agent_handles_empty_context`). When agents rely on Neo4j or external APIs, provide mock adapters under `graphrag_agent/agents/tests/` or gate integration runs behind environment checks. Always run both search scripts before submitting and document any skipped coverage in the PR notes.

## Commit & Pull Request Guidelines
Commit messages follow short imperative clauses (see `git log`: “integrate multi agent ...”). Scope each commit to one feature or fix and add context in the body only when necessary. Pull requests should summarize the change scope, list test commands executed, link to tracking issues, and include screenshots or logs for UI or agent behavior shifts. Tag reviewers responsible for the touched subsystem (agents, graph, server) and confirm `.env.example` updates when configuration changes.

## Agent-Specific Practices
When introducing a new agent, wire it through `agents/__init__.py`, document expected inputs in `agents/base.py`, and register tools under `agents/multi_agent/tools/` if used by the orchestration stack. Provide sample prompts in `documents/` or update `assets/start.md` so operators can reproduce your scenario.
