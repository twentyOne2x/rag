# Topic Summary Integration (Parent-Level Summaries)

## Goal
Use parent-level `topic_summary` to improve retrieval precision and answer quality, especially for definition/explainer queries and sparse/noisy transcripts.

## Scope
- Read `topic_summary` from parent metadata in Pinecone (when present).
- Surface a normalized `parent_topic_summary` field on child nodes at retrieval time.
- Use the summary to (a) lightly boost retrieval, (b) blend reranker scores, and (c) prepend a compact context block for synthesis.

---

## Tasks (checklist)

- [ ] Resolver: include `parent_topic_summary` in fetched parent meta
  - File: `src/rag_v2/vector_store/parent_resolver.py`
  - Add key to mapped parent dict (e.g., `"parent_topic_summary"`)
  - Ensure missing summaries are handled safely

- [ ] Enrichment: propagate summary onto children during retrieval
  - File: `src/rag_v2/retriever/parent_child_retriever.py`
  - In parent-meta enrichment loop, set `md["parent_topic_summary"] = pm.get("parent_topic_summary")` when available

- [ ] Stage‑1 boost: add summary overlap feature
  - File: `src/rag_v2/retriever/parent_child_retriever.py`
  - Compute quick overlap between query tokens and `parent_topic_summary`
  - Apply a small multiplier when overlap > 0 (feature‑flagged)
  - Config flags in `CFG`: `enable_summary_boost` (default on), `summary_boost_mult` (~1.05)

- [ ] Rerank blend: incorporate summary signal with CE score
  - File: `src/rag_v2/rerankers/cross_encoder.py` and/or `src/rag_v2/query_engine_v2.py`
  - Compute `summary_overlap_norm ∈ [0,1]` per candidate
  - Blend: `score = (1-α) * ce + α * summary_overlap_norm` (α=0.2 for definition mode, else 0.05)
  - Log `ce_summary_blend` in `trace`

- [ ] Synthesis context: prepend parent summaries (bounded)
  - File: `src/rag_v2/query_engine_v2.py`
  - Before calling the synthesizer, build a short bullet list using final kept parents’ `parent_topic_summary`
  - Enforce global token cap (~800 tokens) and dedupe identical summaries
  - Flag: `enable_parent_summary_context` (default on)

- [ ] Diagnostics & streaming
  - Include in progress metadata:
    - `retrieve.summary_boost_used`, `retrieve.summary_overlap_stats`
    - `rerank_cross_encoder.ce_summary_blend` samples and `alpha`
    - `synthesize.parent_summaries_included`

- [ ] Configuration
  - File: `src/rag_v2/config.py`
  - Add:
    - `enable_summary_boost: bool = True`
    - `summary_boost_mult: float = 1.05`
    - `summary_rerank_alpha_def: float = 0.2`
    - `summary_rerank_alpha_default: float = 0.05`
    - `enable_parent_summary_context: bool = True`
    - `summary_max_len_chars: int = 600`
    - `summary_context_token_cap: int = 800`

---

## Tests & Acceptance Criteria

### Unit / Functional
- [ ] Resolver mapping: when a parent has `topic_summary`, `fetch_parent_meta()` returns `parent_topic_summary` (string ≤ 600 chars).
- [ ] Enrichment: child nodes enriched with `parent_topic_summary` only when present; no KeyError when missing.
- [ ] Stage‑1 boost: with a summary that shares ≥1 query token, node score increases by ~5% vs. no summary.
- [ ] Blend math: given CE score X and overlap O, blended score equals `(1-α)*X + α*O` within tolerance.
- [ ] Synthesis: final answer prepends a compact `Parent summaries` block only when summaries exist and stays within the token cap.

### Integration
- [ ] `/chat` returns richer, more specific answers for definition‑style prompts (e.g., “What are Token Extensions on Solana?”) vs. baseline.
- [ ] `/chat/stream` progress contains `ce_summary_blend` and `parent_summaries_included` metadata.

### Regression / Manual
- [ ] Non‑definition queries remain stable (no regression in kept counts or latency > 10%).
- [ ] When summaries are absent, behaviour matches current pipeline.

Acceptance is met when at least 3 eval prompts show improved specificity/grounding vs. baseline, and no measurable degradation in unrelated queries.

---

## Risks & Mitigations
- **Stale summaries**: mitigate with recency bias and modest α; keep summaries short and curated.
- **Latency**: token cap keeps synthesis overhead bounded; summary overlap is a cheap bag‑of‑words metric.
- **Over‑steering**: allow easy flagging off via config if needed.

---

## Rollout Plan
1. Implement resolver + enrichment behind flags.
2. Add retrieval boost + rerank alpha blend, log diagnostics.
3. Add synthesis context preface with token cap.
4. Evaluate on a small curated set; tune α and boost.
5. Enable by default in production config.

