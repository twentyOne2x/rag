# Sources-Only Fast Path Design

> Draft owner: Codex  
> Last updated: 2025-10-13

---

## 1. Background

Today every query flows through the full RAG pipeline:

1. Parent/child retrieval (vector + metadata boosts)
2. Cross-encoder rerank (CE)
3. Post-processing (entity canonicalizer, speaker propagation)
4. Stitching
5. LLM synthesis (`_synthesize_clean`)

For prompts that only ask for clips/videos (“return all videos about…”, “list clips where…”, coverage queries), steps 2 and 5 dominate latency without adding value—the user just wants pointers. Benchmarking via `ProgressRecorder` shows typical timings:

| Stage                  | P50 ms | Notes                              |
|------------------------|--------|------------------------------------|
| retrieve               | 150–250| Acceptable                         |
| rerank_cross_encoder   | 650–950| Most expensive stage               |
| synthesize             | 700–1200| LLM writes paragraphs unnecessarily |
| downstream processing  | <200   | Negligible                         |

Goal: introduce a “sources-only” fast path that skips unnecessary work, returning curated source metadata in ≤350 ms P50.

---

## 2. Goals & Success Criteria

- **Latency**: Reduce coverage/list queries by ≥60% (P50 & P90) relative to baseline.
- **Cost**: Avoid CE + LLM usage for pure source-list intents.
- **Output quality**: Provide accurate, canonicalized metadata (clip URL, timestamps, channel, score, reasoning snippet).
- **Telemetry**: Emit aggregated metrics via new histogram to validate improvements.
- **Comparative evaluation**: Automated AI judge reports “better/same/worse” from user perspective for sample queries.

**Non-Goals**
- Replacing CE for answer-style queries.
- Changing retrieval backends or index layout.
- Rewriting frontend UX (will consume same `final_kept`/progress stream).

---

## 3. Proposed Solution

### 3.1 Intent Detection

1. **Rule-based classifier (phase 1)**  
   - Extend `_final_k` / new helper `is_sources_only_query(query: str)`.  
   - Signals: `"return all"`, `"list clips"`, `"show videos"`, query length ≤12 tokens + contains channel/entity.  
   - Router hints: `scope == "videos"` from frontend.  
2. **Telemetry feedback loop**  
   - Log decisions in progress metadata (`progress.metadata["workflow"] = "sources_only"`).  
   - Later upgrade to ML classifier (optional).

### 3.2 Pipeline Branch

```
retrieve → (optional entity gate) → [sources-only?] ─┬→ Fast Path
                                                     └→ Existing Path
```

**Fast Path Steps**
1. **Aggregation**  
   - If `return videos`: group by parent_id; select top child per video.  
   - If `return clips`: keep top N child nodes (dedupe by segment_id).  
2. **Ranking**  
   - Use stage1 score + metadata boosts (already applied).  
   - Optionally apply light rerank: recency decay, router boost (no CE).  
3. **Formatting**  
   - Compose canonicalized metadata payload:  
     ```json
     {
       "response": "Here are the best matching clips:",
       "sources": [
         {
           "title": "Solana Firedancer Milestone Update",
           "channel": "Solana Foundation",
           "clip_url": "https://youtube.com/watch?v=abc&t=482s",
           "start_hms": "00:08:02",
           "score": 0.94,
           "summary": "Firedancer progress update...",
           "matched_entities": ["Firedancer", "Anza"]
         }
       ]
     }
     ```
   - Response body uses existing Response object with `source_nodes` only; front-end already reads `final_kept`.
   - Add explicit label in progress stream: `"workflow": "sources_only"`.
4. **Skip stages**  
   - CE rerank: emit `progress.add_event(..., status="skipped", metadata={"reason": "sources_only"})`.  
   - review_docs/stitch if no modifications needed (skip or run minimal canonicalization).  
   - synthesize: add skip event and return.

### 3.3 Configuration & Flags

- `ENABLE_SOURCES_FAST_PATH` (default off → staged rollout)
- `SOURCES_FAST_PATH_MIN_RESULTS` (e.g., require ≥3 nodes)
- `SOURCES_FAST_PATH_MAX_RETURN` (cap 30 videos / 20 clips)
- Keep override via header or query param for A/B testing.

### 3.4 Caching Canonical Metadata

- Introduce `NodeMetadataCache` (LRU keyed by segment_id) holding normalized text + summaries; shared across fast & regular paths to avoid repeated normalization.
- Warm cache during retrieval stage to reuse data for synthesis when needed.

### 3.5 Telemetry & Monitoring

- Existing `TelemetryCollector.summary()` already tracks per-stage counts/timings.
- New histogram (`/healthz.telemetry_histogram`) now aggregates across requests. Add new stage labels:
  - `workflow_select` (time spent deciding branch)
  - `format_sources` (time building payload)
- Dashboard metrics: P50/P90 per workflow, count of fast path usages per minute.

### 3.6 AI Evaluator

- Build offline eval harness:
  1. Query corpus split: {coverage/list} vs {answer}.  
  2. Run baseline (full pipeline) and fast-path variant (with fallback to baseline for non-sources).  
  3. Prompt LLM evaluator: _“Compare these two responses for the user request. Which better satisfies the request? Provide verdict {baseline_better, fastpath_better, tie} with rationale.”_
  4. Record coverage of returned sources (precision/recall vs ground truth when available).
- Use results to tune thresholds before GA.

---

## 4. Detailed Plan

### 4.1 Implementation Tasks

| # | Task | Owner | Status |
|---|------|-------|--------|
| [ ] 1 | Add `is_sources_only_query` heuristic + router metadata |  |  |
| [ ] 2 | Introduce fast-path branch in `ParentChildQueryEngineV2.query` |  |  |
| [ ] 3 | Implement aggregation helpers (`_top_per_parent`, `_top_clips`) |  |  |
| [ ] 4 | Format response payload & progress events |  |  |
| [ ] 5 | Config flags & environment toggles |  |  |
| [ ] 6 | Node canonical metadata cache |  |  |
| [ ] 7 | Telemetry updates (`workflow_select`, histogram fields) |  |  |
| [ ] 8 | Unit tests + integration scenarios |  |  |
| [ ] 9 | AI evaluator harness + baseline comparison |  |  |
| [ ] 10 | Rollout script / feature flag management |  |  |

### 4.2 Expected Code Touchpoints

- `src/rag_v2/query_engine_v2.py` (branching logic, progress events)
- `src/rag_v2/retriever/parent_child_retriever.py` (optional new helper)
- `src/rag_v2/postprocessors/...` (cache integration)
- `src/rag_v2/config.py` (new feature flags)
- `src/rag_v2/app.py` (expose mode in diagnostics)
- `src/rag_v2/tests/...` (unit/integration)
- New `scripts/eval_sources_fast_path.py` for evaluator harness.

---

## 5. Test Strategy

### 5.1 Unit Tests

- `tests/query_engine/test_sources_fast_path.py`
  - [ ] `test_sources_only_branch_skips_synthesis`
  - [ ] `test_returns_top_parent_videos_with_canonical_metadata`
  - [ ] `test_falls_back_when_min_results_not_met`
  - [ ] `test_clip_mode_preserves_segment_dedupe`

- `tests/router/test_intent_detection.py`
  - [ ] Regex/keyword detection + whitelist/blacklist scenarios.

- `tests/telemetry/test_workflow_metrics.py`
  - [ ] Histogram increments for both workflows.

### 5.2 Integration Tests

- API-level tests using `/chat/simple` and `/chat/stream`:
  - [ ] Coverage query returns within 350 ms (mock CE + LLM).  
  - [ ] Progress stream shows skipped CE & synth events.  
  - [ ] `diagnostics.progress_metadata["workflow"] == "sources_only"`.

- Regression: ensure answer-style queries still go through full pipeline.

### 5.3 Load/Perf Tests

- [ ] Apply using replayed production queries (10k sample).  
- [ ] Measure median/95th latency before & after; confirm CE GPU/CPU usage drop.

### 5.4 AI Evaluation

- [ ] Build dataset of 200 coverage queries (manual labels optional).  
- [ ] Run baseline vs fast-path; feed to evaluator prompt.  
- [ ] Ensure ≥90% of verdicts are “fastpath_better” or “tie”, flag regressions.

### 5.5 Observability Checks

- [ ] `/healthz` shows increasing `telemetry_histogram.requests`.  
- [ ] Dashboard alerts if fast-path error rate > baseline (fallback engaged).  
- [ ] Ensure histogram resets gracefully on deploy.

---

## 6. Rollout Plan

1. **Behind feature flag**: default off.  
2. **Internal QA**: enable for staging + small dev group.  
3. **Canary**: enable for 10% of production requests (weighted by session).  
4. **Monitor**: track latency, abort rate, AI eval summary.  
5. **Gradual ramp**: 25% → 50% → 100% if stable.  
6. **Post-launch**: keep ability to disable quickly via env var.

---

## 7. Examples

### Example A – Video Listing

**Query**: “Return all videos about Firedancer progress in 2024”  
**Workflow**: `sources_only`  
**Expected Response** (trimmed):

```json
{
  "response": "Here are the latest Firedancer videos:",
  "source_nodes": [],
  "diagnostics": {
    "final_kept": [
      {
        "title": "Solana Firedancer Milestone Update",
        "channel_name": "Solana Foundation",
        "clip_url": "https://youtube.com/watch?v=abcd&t=482s",
        "score": 0.94,
        "start_seconds": 482,
        "summary": "Firedancer team announces validator throughput improvements."
      },
      ...
    ],
    "workflow": "sources_only"
  }
}
```

Progress events:
1. retrieve (completed)
2. rerank_cross_encoder (skipped, reason=sources_only)
3. format_sources (completed)
4. synthesize (skipped)

### Example B – Mixed Intent (fallback)

**Query**: “What is Firedancer and why does it matter?”  
**Workflow**: `default` (full pipeline).  
Fast-path not triggered; ensures no regressions.

---

## 8. AI Evaluator Outline

```
system: You are an impartial QA judge.

user: 
Request: {query}
Baseline response: {baseline}
Fast-path response: {candidate}
Instructions: Choose which response better satisfies the request.
Return JSON: {"verdict": "baseline|fastpath|tie", "rationale": "..."}
```

- Store results in BigQuery or local JSON.  
- Use aggregated verdicts to iterate on heuristics.

---

## 9. Open Questions

1. Should we preserve CE for “clips” but not “videos”? (Hybrid strategy)
2. Is there product appetite for returning structured JSON vs natural language?
3. How to signal fast-path usage back to UI (badge, note)?
4. Do we need rate limits to prevent abuse on high-volume scraping queries?

---

## 10. References

- `src/rag_v2/query_engine_v2.py` – current pipeline flow.
- `src/rag_v2/instrumentation.py` – progress recorder & telemetry histogram.
- `docs/topic_summary_integration.md` – related summary improvements.

---

## Appendix A – Checklist (TL;DR)

- [ ] Intent detection heuristics
- [ ] Fast-path branch implementation
- [ ] Aggregation helpers (videos, clips)
- [ ] Canonical metadata cache
- [ ] Progress + telemetry updates
- [ ] Config flags & rollout scripts
- [ ] Unit + integration tests
- [ ] AI evaluator harness
- [ ] Canary + monitoring dashboards

