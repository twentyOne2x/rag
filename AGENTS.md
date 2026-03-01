# AGENTS Map

Short navigation map for agent workflows in this repo.

## Product Surface
- `src/rag_v2/`: query engine, routing, retriever, vector-store integration, and tools.
- `configs/`: runtime channel/model and retrieval configuration.
- `tests/` and `src/rag_v2/tests/`: behavior and integration checks.

## Knowledge Base
- `docs/index.md`: docs entrypoint.
- `docs/core-beliefs.md`: operating principles.
- `ARCHITECTURE.md`: boundaries and data flow.
- `docs/plans/`: active/completed plans and debt tracker.

## Verification
- Run `python3 scripts/knowledge_check.py` before opening or updating a PR.
