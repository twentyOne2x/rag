# RAG Performance Roadmap & Instrumentation Plan

## 1. Goals
- Shorten perceived and actual latency for both cold starts and steady-state queries.
- Provide actionable telemetry that attributes latency to specific pipeline stages.
- Surface live progress updates so the frontend can reassure users during long-running queries.
- Build the foundation for future automated regression tracking (dashboards, alerts).

## 2. Current State Summary
- **Startup**: `agent_main` bootstraps the query engine lazily on first request; the cost includes OpenAI client setup, index deserialization, and Pinecone metadata alignment. This sequence is opaque—only stdout logs indicate Pinecone namespace overrides.
- **Per-query pipeline**: `ParentChildQueryEngineV2` already tracks internal timers (`retrieve_ms`, `ce_ms`, etc.) but only emits them via logs and discards once the request completes. No live progress notifications exist for the frontend.
- **Observability gaps**:
  - No structured startup metrics.
  - Per-request timings are not returned to the caller.
  - No correlation IDs or step-level breakdown suitable for dashboards.
  - Lack of standard interface to plug in profilers (CPU, memory) or toggled tracing.

## 3. Instrumentation Enhancements (Phase 1 – Immediate)
1. **Startup profiler**
   - Wrap `bootstrap_query_engine_v2()` with a timer and emit a structured event containing durations for model configuration, index load, Pinecone namespace alignment, and retriever/query-engine construction.
   - Expose this data via a globally accessible cache (`AppDiagnostics.startup_profile`) and include in health/status endpoints if useful.
2. **Query progress reporter**
   - Introduce a `ProgressRecorder` helper that records ordered steps with start/end timestamps, durations, and optional metadata payloads.
   - Expand `ParentChildQueryEngineV2` to populate the recorder with events such as `retrieve`, `entity_filter`, `rerank`, `postprocess`, `stitch`, `synthesize`, `early_abort`.
   - Persist the last trace (`qe.last_trace`) for retrieval outside the query engine.
3. **API response enrichment**
   - Extend `ChatResp` with a `diagnostics` payload:
     ```json
     {
       "timings": {"total_ms": 1234, "retrieve_ms": 420, ...},
       "progress": [
         {"name": "retrieve", "status": "completed", "duration_ms": 420, "started_at": "...", "metadata": {...}},
         ...
       ],
       "models": {"llm": "...", "embed_model": "..."},
       "config": {...subset of CFG...}
     }
     ```
   - Allow the frontend to show a status timeline immediately upon response; set the stage for live streaming later.
4. **Structured logging**
   - Ensure each progress event is mirrored to logs with a consistent correlation ID (request UUID) to simplify future ingestion into log-based dashboards.

## 4. Measurement & Optimization Strategy (Phase 2 – Short Term)
1. **Per-stage optimization levers**
   - **Startup**
     - Cache `Settings` construction across workers.
     - Persist the index in memory via warmup job or reuse across workers when server reloads.
   - **Retrieval**
     - Evaluate `similarity_top_k` trade-offs; consider request-driven overrides based on query intent.
     - Introduce asynchronous fetching where Pinecone latency dominates.
   - **Cross-Encoder rerank**
     - Record queue wait and inference time separately; benchmark batching vs. single inference.
     - Evaluate optional CE models (smaller/faster) guarded by score thresholds.
   - **Post-processing/Stitching**
     - Measure time per node to highlight slow entity canonicalization or speaker propagation.
   - **Synthesis**
     - Capture prompt token count estimates to correlate with LLM latency.
     - Consider partial streaming of the LLM output to reduce perceived wait.
2. **Profiling toolchain**
   - Enable on-demand async profiling via environment toggle (e.g., `ENABLE_PROFILING=1`) to avoid constant overhead.
   - Integrate lightweight CPU sampling (e.g., `yappi` or `pyinstrument`) triggered for selected requests only.
   - Emit histograms (p50/p95) by stage using an in-process metrics registry ready for Prometheus export.
3. **Data collection plan**
   - Log queries with anonymized intent labels (scope, definition flag) and associated latencies.
   - Store aggregated metrics (e.g., JSONL) for offline analysis across deployments.

## 5. Advanced Optimization Tracks (Phase 3 – Medium Term)
1. **Concurrency & streaming**
   - Implement server-sent events (SSE) or WebSocket endpoint using the existing progress recorder to stream events and partial answers.
   - Overlap retrieval and rerank by fetching CE features while the base retriever finishes.
2. **Caching**
   - Layer query normalization + LLM output cache (e.g., Redis) to short-circuit repeated questions.
   - Cache Pinecone parent metadata lookups and cross-encoder embeddings with TTLs.
3. **Dynamic routing**
   - Train a lightweight intent classifier to skip expensive stages (e.g., skip CE on definition queries with high-confidence scores).
4. **Model & infra tuning**
   - Benchmark alternative LLMs (gpt-4o-mini vs. gpt-4o vs. gpt-4o-mini-high) under real load.
   - Fine-tune cross-encoder thresholds and `CFG` heuristics using collected telemetry.

## 6. Frontend Integration Points
- `ChatResp.diagnostics.progress` provides a ready-to-render timeline; each entry includes human-readable `label` plus machine-readable fields.
- Add a `request_id` header/field for correlating subsequent UI polling or SSE connections.
- UI can render a “Processing…” status panel updating through the steps once live streaming is implemented.

## 7. Next Steps Checklist
- [ ] Land Phase 1 instrumentation (startup timer, per-query progress, diagnostics payload).
- [ ] Aggregate latency metrics over a representative workload to establish baselines.
- [ ] Prioritize hotspots based on data (e.g., CE latency vs. Pinecone lookup).
- [ ] Decide on streaming mechanism (SSE vs. WebSocket) and roadmap its implementation using the existing progress recorder.
- [ ] Revisit thresholds and configuration knobs informed by collected telemetry.

