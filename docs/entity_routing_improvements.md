# Entity-Aware Retrieval Improvements for CZ + Hyperliquid Queries

## Authors
- Codex agent (implementation partner)

## Last Updated
- 2025-10-12

---

## Background
Recent queries such as **“What are CZ’s thoughts on Hyperliquid?”** return generic Hyperliquid commentary with no mention of CZ. Retrieved clips include hosts like Jez or Threadguy but lack the requested speaker (CZ / Changpeng Zhao). The UI shows high-scoring sources that are unrelated to the entity combination implied by the question.

Root causes:
1. Retriever ranks mostly by lexical overlap; it is not enforcing that *all* query entities occur together.
2. Metadata (entities/speakers) is not leveraged to gate results; re-rankers see many irrelevant candidates.
3. The synthesizer proceeds even when no clip matches the constrained intent.

We need an entity-aware strategy that increases precision when users ask for content involving multiple entities (e.g., CZ ∧ Hyperliquid).

---

## Goals
1. **Precision** – Ensure retrieved clips mention all salient entities from the query.
2. **Transparency** – Provide diagnostics indicating why a response might fail (e.g., entity gate filtered everything).
3. **Minimal regression** – Preserve current behaviour for broad queries while improving targeted ones.

---

## Current Behaviour
1. **Stage 1 (vector retrieval)** – Returns children with highest semantic similarity scores; no hard entity constraints.
2. **Parent/child expansion** – Adds neighbours based on parents but ignores targeted entity presence.
3. **Cross-encoder rerank** – Scores each child individually; still lacks explicit entity features.
4. **Synthesizer** – Always attempts to compose an answer, even if evidence doesn’t satisfy query entity constraints, producing hallucinated summaries.

---

## Proposed Solution

### 1. Entity Signal Extraction
- Expand canonical entity map (`ENT_CANON_MAP`) to cover:
  - “Hyperliquid”, “CZ”, “Changpeng Zhao”, “Binance”, ticker `$BNB`, translations.
- During ingestion, ensure child metadata includes:
  - `canonical_entities`: normalised entity list.
  - `speakers`: the person speaking in the segment (if recorded).

### 2. Multi-Entity Filtering
- Parse user queries for entities (existing router already canonicalises tokens).
- When multiple distinct entities are found (e.g., `{"CZ", "Hyperliquid"}`):
  - **Stage 1 Pre-filter:** drop parents/children that do not include all required entities.
  - **Stage 2 Reweighting:** compute `entity_overlap_score = (# of required entities present / total required entities)`.
  - Combined score: `0.6 * cross_encoder_score + 0.4 * entity_overlap_score`.

### 3. Gating & Diagnostics
- Define a minimum entity satisfaction threshold (e.g., at least **N = 2** nodes mentioning all required entities).
- If gating fails:
  - Abort with “No matching sources mention CZ and Hyperliquid together.”
  - Populate diagnostics with `"entity_gate": {"required": [...], "passed": False}`.

### 4. Synthesizer Guardrails
- Pass only nodes that satisfy entity overlap to the synthesizer.
- If nothing remains after gating, return the abort message without calling the LLM.

### 5. UX Improvements
- Include entity gate status and kept sources in the progress metadata (already enriched).
- Frontend can show messages like “We couldn’t find any clips where CZ mentions Hyperliquid” instead of rendering generic content.

---

## Implementation Plan

| Step | Work Item | Owner | Notes |
|------|-----------|-------|-------|
| 1 | Ensure ingestion outputs `canonical_entities` and `speakers` for children & parents | Data pipeline | Already partially present; audit ingestion repo. |
| 2 | Extend entity parser and canonical map (Hyperliquid, CZ aliases) | Backend | Update `ENT_CANON_MAP`, router patterns. |
| 3 | Add entity pre-filter in `ParentChildRetrieverV2` | Backend | Modify `_build_channel_filter` or introduce new gating function. |
| 4 | Incorporate entity overlap into rerank phase | Backend | Adjust `CEReranker` or apply post-CE reweighting. |
| 5 | Add synthesizer gating and improved diagnostics | Backend | Update `ParentChildQueryEngineV2._query`. |
| 6 | Frontend messaging & testing | Frontend | Consume new diagnostics (e.g., `entity_gate`) to inform users. |
| 7 | Regression testing | Backend + Frontend | Manual evaluation with curated prompts (below). |

---

## Evaluation
Create a test suite of queries:

1. **Positive** – “What are CZ’s thoughts on Hyperliquid?”  
   Expect: clips mentioning CZ *and* Hyperliquid.
2. **Positive** – “What does CZ say about Hyperliquid’s founders?”  
   Expect: similar to above.
3. **Negative** – “What does Threadguy think about Hyperliquid?”  
   Expect: Threadguy clips; ensure CZ gate doesn’t trigger.
4. **Edge** – “What does CZ think about Orca?”  
   Should return “no matching sources” if data absent.

Metrics:
- Top-k precision (proportion of retrieved clips with required entities).
- Human eval of synthesised answers (at least 3 queries).
- Monitoring: entity gate counts, gating false positives.

---

## Risks & Mitigations
- **Sparse metadata** – Some clips may lack speakers/entities; gating may suppress relevant content.
  - Mitigation: fallback to approximate matching (e.g., speaker name from title).
- **Over-filtering** – Strict entity set may discard legitimate context (e.g., CZ mentions Binance while host describes Hyperliquid).
  - Mitigation: allow partial match if at least one segment fully satisfies requirement.
- **Performance** – Additional filtering logic could increase latency.
  - Mitigation: operations are simple dictionary lookups; measure after rollout.

---

## Open Questions
1. Should we expose entity weights as configuration (per entity)?  
2. Do we need per-language synonyms for CZ beyond the alias mapping?  
3. How to update ingestion automatically when new entities (e.g., “Hyperliquid founder”) appear?

---

## Next Steps
1. Confirm ingestion metadata coverage for `canonical_entities`.  
2. Implement entity gate + reweighting in retriever/reranker.  
3. Update frontend diagnostics display.  
4. Validate with the test suite and perform manual QA.  
5. Iterate on weighting/gating thresholds based on observed accuracy.

---

