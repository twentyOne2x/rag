# rag-v2 Retrieval Service

This repository hosts the production code for `rag_v2`, a FastAPI service that answers research questions over long-form video transcripts indexed in Pinecone. The service uses a parent/child retrieval strategy, optional cross-encoder reranking, and multiple research modes so callers can trade off latency for coverage.

## What Users Get

- Structured answers with an `Executive Summary` (no citations), sectioned `Detailed Evidence` laden with sources, and `Key Takeaways` that surface the decisive quote.
- Timestamped, speaker-attributed citations that link directly into clips so users can jump to the proof.
- Channel filters and entity-aware routing to keep responses relevant to the viewer’s interests.
- Consistent experience across default (`quick`) and research-heavy (`deep`) modes, with deeper dives staying readable.

## What Developers Get

- A modular FastAPI backend with swap-friendly retrieval, reranking, and prompt layers.
- Centralised configuration (`configs/rag_v2/config.yaml`) that controls models, modes, and retrieval budgets without code changes.
- Observability primitives (progress recorder, JSONL telemetry, rolling histograms) for production monitoring.
- CLI tooling and smoke tests (`src/rag_v2/agent_main.py`, `src/rag_v2/tests`) to validate changes before deployment.

## Repository Layout

- `src/rag_v2/app.py` – FastAPI entrypoint exposing `/chat`, `/chat/simple`, `/chat/stream`, `/channels`, and `/healthz`.
- `src/rag_v2/app_main.py` – Bootstraps the parent/child query engine, including Pinecone index attachment.
- `src/rag_v2/retriever/` – Parent/child retriever, entity-aware routing, and metadata boosts.
- `src/rag_v2/rerankers/` – Cross-encoder reranker with recency and stale-phrase penalties.
- `configs/rag_v2/` – Runtime configuration (`config.yaml`, channel catalogue).
- `docs/` – Design notes (research modes, telemetry, roadmap, topic summary integration).

## Prerequisites

- Python 3.10 or later
- Pinecone 3.x account and API key (serverless index configured for 3072-dim embeddings)
- OpenAI API key for GPT-4o or compatible endpoint (`Settings.llm`) and `text-embedding-3-large`
- Optional: Google Cloud Secret Manager if secrets are fetched at runtime

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Populate a `.env` file (or export the variables) with at least:

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=<your-pinecone-index>
PINECONE_NAMESPACE=<your-namespace>
INFERENCE_MODEL=gpt-4o-mini           # optional override for Settings.llm
EMBEDDING_MODEL=text-embedding-3-large
APP_ORIGINS=http://localhost:3000     # comma-separated list for CORS
```

When running locally the service automatically loads `.env`/`.env.local` if present.

## Configuration

The primary configuration file is `configs/rag_v2/config.yaml`:

- `models.*` – default LLM and embedding models.
- `retrieval.*` – global retrieval defaults (stage-1 counts, CE gating, etc.).
- `modes.default` – fallback when the request omits `mode`/`research_mode`.
- `modes.quick` / `modes.deep` – per-mode prompt and retrieval overrides.

Deep mode’s structured response and higher citation floor come from this config and the logic in `_enrich_query` inside `src/rag_v2/app.py`.

Channel filters are sourced from `configs/rag_v2/channels.json`, which maps a scope (e.g., `videos`, `streams`) to a list of allowed channels.

## Running the API

```bash
uvicorn src.rag_v2.app:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints:

- `POST /chat` – Primary endpoint (enforces enriched prompt). Accepts `mode` or `research_mode` (`quick` or `deep`).
- `POST /chat/simple` – Bypasses prompt enrichment; useful for raw model debugging.
- `POST /chat/stream` – Server-sent events streaming version (progress + final answer).
- `GET /channels` – Lists known channels per scope (`?scope=videos|streams`).
- `GET /healthz` – Recent startup profile, telemetry snapshot, and recent queries.

All chat responses include `diagnostics.mode_config` describing the resolved mode, retrieval overrides, and whether a fallback occurred.

## Research Modes at a Glance

| Mode    | Intent                         | Retrieval Budget                                      | Citation Target | Prompt Structure |
|---------|--------------------------------|--------------------------------------------------------|-----------------|------------------|
| quick   | Low-latency everyday answers   | Stage-1≈160, CE keep≈20, max_final_nodes≈12            | ≥4 quotes       | Single-section synthesis |
| deep    | Exhaustive research with proof | Stage-1≈260, CE keep≈28, broader parent/child expansion | ≥6 quotes       | Executive Summary → Detailed Evidence → Key Takeaways |

Adjust the budgets or prompt settings in `config.yaml`, then redeploy. Unknown modes fall back to `default` and note the fallback in diagnostics.

## Telemetry & Diagnostics

- `ProgressRecorder` emits step-level timing metadata per request.
- `AppDiagnostics` stores the latest startup profile, query trace, and telemetry summary (available via `/healthz`).
- Enable persistent telemetry by setting `RAG_TELEMETRY_PATH=/path/to/metrics.jsonl`; summaries and events are appended as JSONL.

## Testing

Run the lightweight unit tests and import smoke checks:

```bash
python -m pytest src/rag_v2/tests
```

The test suite covers telemetry helpers and confirms the query engine module imports cleanly.

## Troubleshooting

- **Engine unavailable** – Increase `RAG_ENGINE_POOL_SIZE` or `RAG_ENGINE_ACQUIRE_TIMEOUT` if you see pool timeout errors under load.
- **Missing Pinecone index** – Ensure `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `PINECONE_NAMESPACE` are set before starting the service.
- **Cross-encoder load failures** – The reranker is optional; failure to load the `sentence-transformers` model simply disables CE reranking and logs a warning.

## CLI Agent

The `TinyV2Agent` in `src/rag_v2/agent_main.py` routes questions to the same query engine for manual testing:

```bash
python src/rag_v2/agent_main.py "return all videos about Firedancer"
```

It applies a simple heuristic to decide when to invoke the retrieval tool versus returning a fallback response.
