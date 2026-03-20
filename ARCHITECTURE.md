# Architecture

## Overview
`icmfyi/rag` serves retrieval-augmented responses by combining routing, query processing, and vector-backed evidence resolution.

## Main Components
- App/API entrypoints in `src/rag_v2/app.py` and `src/rag_v2/app_main.py`.
- Query pipeline and routing in `src/rag_v2/query_engine_v2.py`, `src/rag_v2/router/`.
- Retrieval and vector resolution in `src/rag_v2/retriever/` and `src/rag_v2/vector_store/`.
- Tooling, post-processing, and shared helpers in `src/rag_v2/tools.py` and `src/rag_v2/postprocessors/`.

## Data and Control Flow
1. Incoming question enters app endpoint and query pipeline.
2. Router selects strategy and retrieval mode.
3. Retriever resolves candidate evidence and parent/child relationships.
4. Post-processing and tools shape final response payload.

## Ops Notes
- Keep `configs/rag_v2/` synchronized with supported channels and retrieval constraints.
- Run `python3 scripts/knowledge_check.py` after docs or workflow edits.
