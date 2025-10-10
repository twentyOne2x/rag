# RAG v2 Performance & Observability Roadmap

This document converts the recent assessment into an execution plan. The goals are:

- Improve end-to-end latency without sacrificing answer quality or existing features.
- Provide rich telemetry to diagnose issues quickly (both online and offline).
- Supply analysis tooling so optimisation decisions are data-driven.
- Prepare supporting services so Cloud Run deployments expose and consume metrics safely.

The plan below is grouped by theme, with priority levels (P0 highest), owners (default **infra** unless noted), and deliverables/tests.

---

## 1. Telemetry & Metrics Foundation

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Request metrics aggregator** – build an in-process metrics collector that records per-stage timings (retrieve, rerank, post-process, stitch, synthesize) and Pinecone round-trip stats. Persist structured events to a rolling JSONL file and optionally push to stdout for Cloud Logging. | infra | Module `src/rag_v2/telemetry/collector.py` + config knobs | unit tests for aggregation; golden JSON snapshot |
| P0 | **Startup profiler enrichment** – extend existing startup profile to include model warmup timings, token/profile counts, and environment metadata. | infra | Updated `bootstrap_query_engine_v2` | unit test verifying keys in profile |
| P0 | **Persistent diagnostics cache** – maintain an in-memory ring buffer + optional disk cache of the last N traces for debugging. | app | Module `src/rag_v2/telemetry/cache.py` | unit tests for eviction & retrieval |
| P1 | **Offline metrics sink** – produce daily JSONL/GCS export summarising p50/p95 for each stage; integrate with BigQuery schema (if available). | infra | CLI script `scripts/export_metrics.py` | integration test using temp dir |
| P1 | **Health endpoint telemetry** – expose `/healthz` extended view with last-deploy revision, recent latency summary, and cache stats. | app | Updated FastAPI endpoint | unit test checking response keys |

## 2. Retriever Optimisations

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Adaptive top-k** – start queries with lower `similarity_top_k` and increase only when needed (low scores / few relevant entities). | retrieval | Configurable strategy inside `ParentChildRetrieverV2.retrieve` | unit tests mocking retriever to confirm scaling behaviour |
| P0 | **Entity-filter gating** – only run `_entity_filtered_retrieve` when canonical entities are detected; collect before/after timing telemetry. | retrieval | Conditional trigger + telemetry fields | test ensuring filter is skipped with entity-free queries |
| P0 | **Channel filter memoisation** – cache compiled Pinecone filter expressions; invalidate on updates. | retrieval | LRU cache in retriever | unit test verifying reuse |
| P1 | **Parallel metadata boosts** – use thread pool or vectorised operations to speed metadata adjustments without changing results. | retrieval | Refactored scoring function | property test verifying identical outputs |
| P1 | **Parent metadata prefetch** – maintain async warm cache for parent metadata via background task triggered at startup. | retrieval | Background worker service | integration test stub verifying fetch scheduling |

## 3. Reranker & Scoring Layer

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Conditional CE bypass** – skip cross-encoder if candidate count already <= minimum keep or if last run indicates low benefit. | reranker | Policy in `_query` + telemetry flag | tests covering bypass + fallback |
| P0 | **Warmup request** – issue synthetic CE call during startup to avoid first-request cold latency. | infra | Startup hook | integration test with mock CE |
| P1 | **CE result cache** – optional in-memory TTL cache keyed by (query_hash, segment_id). | reranker | Cache module | unit tests for hit/miss |
| P1 | **Model benchmarking harness** – script to compare alternative CE models and produce latency/quality report. | infra | `scripts/benchmark_ce_models.py` | test verifying CLI args |

## 4. Post-processing & Synthesis

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Selective post-processors** – skip entity canonicalisation and speaker propagation when metadata absent or query doesn’t reference entities. | core | Guards around `_entity_canonicalizer`/`_speaker_propagator` | tests ensuring bypass behaviour |
| P0 | **Early exit stitching** – terminate stitching when final node count already <= `CFG.max_final_nodes`. | core | Updated `_stitch_adjacent` logic | regression test verifying same output |
| P1 | **LLM streaming support** – optional streaming responses to reduce perceived latency (feature flag). | app | Async generator endpoint or SSE | integration test w/ streaming client |

## 5. Serving Layer Enhancements (`app.py`)

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Background warmers** – after startup, warm retriever & CE caches via background task so first user sees consistent latency. | app | FastAPI background task | unit test verifying job scheduling |
| P0 | **Timeout safeguards** – add per-stage timeouts with graceful fallback (e.g., partial results). | app | Configurable timeout manager | tests for timeout path |
| P1 | **Improved diagnostics response** – include aggregated metrics, CE bypass info, and Pinecone timings in `ChatResp.diagnostics`. | app | Response schema update | schema test |

## 6. Analytics & Insight Tooling

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Telemetry service** – small FastAPI (or background thread) that exposes telemetry summaries (e.g., rolling averages, top slow queries) for dashboards. | analytics | `src/rag_v2/telemetry/service.py` | tests for API handlers |
| P0 | **Analysis notebook/CLI** – build pandas-based analyser to summarise exported JSONL metrics; optionally integrate GPT-assisted commentary (configurable). | analytics | `notebooks/telemetry_analysis.ipynb` + `scripts/analyse_metrics.py` | CLI smoke test |
| P1 | **Alert hooks** – if telemetry indicates regression (e.g., p95 > threshold), print actionable hints derived from collected metadata; optionally send Slack/Webhook in future. | analytics | Threshold evaluator | unit tests with synthetic data |

## 7. Deployment & Ops

| Priority | Item | Owner | Deliverable | Tests |
|----------|------|-------|-------------|-------|
| P0 | **Update Cloud Build deploy configs** to surface new env vars (e.g., telemetry toggles, service endpoints). | infra | Amend `cloudbuild_app.yaml`, `cloudbuild_deploy_app.yaml` | manual validation |
| P1 | **Cloud Run resource tuning** – add guidance/scripts for autoscaling & concurrency based on telemetry (future milestone). | infra | Doc section in README | n/a |

---

## Execution Order

1. Telemetry collector + diagnostics cache (enable visibility before optimisation).
2. Retriever optimisations (largest expected latency wins).
3. Reranker adjustments (conditional bypass, warmup).
4. Selective post-processing + stitching improvements.
5. Serving safeguards and background warmers.
6. Analytics tooling (CLI/notebook/service).
7. Deployment updates / alerts.

Each change must:
- Be test-driven (unit/integration as appropriate).
- Preserve behaviour (validate final responses unchanged when features disabled).
- Emit telemetry to confirm impact.

Track progress in this doc by ticking items and linking PRs/build IDs once merged.
