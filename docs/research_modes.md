# Research Modes

This service now supports two query flows so you can choose between lower-cost answers and a deeper dive when you need more coverage.

## Modes at a Glance

| Mode | Intent | Retrieval Budget | Citation Target |
| --- | --- | --- | --- |
| `quick` *(default)* | Snappy responses for everyday questions. | Mirrors the baseline `RetrievalConfig` (stage1≈160, CE keep≈20, max_final_nodes≈12). | ≥4 quotes. |
| `deep` | Higher recall when you want exhaustive evidence. | Expands `stage1_topn`, `topk_post_rerank`, `ce_min_keep`, `max_segments_per_parent`, and `max_final_nodes`. | ≥6 quotes. |

### Picking a Mode

- `quick` is used automatically unless you send an explicit `mode` / `research_mode` field.
- `deep` widens retrieval, keeps more nodes after reranking, and asks the model for denser supporting quotes. Expect higher latency and token spend.
- If a caller requests an unknown mode, the server falls back to the configured default and flags that fallback in diagnostics.

## Calling the API

```json
POST /chat
{
  "message": "Summarise Firedancer progress in 2024.",
  "mode": "deep"
}
```

All chat endpoints (`/chat`, `/chat/simple`, `/chat/stream`) accept either `mode` or `research_mode`. Diagnostics now include `research_mode`, `mode_label`, and a `mode_config` block so observability dashboards can segment performance per mode.

## Configuration Surface

The new `modes` section in `configs/rag_v2/config.yaml` defines the behaviour of each mode:

```yaml
modes:
  default: quick            # server fallback + null-request behaviour
  quick:
    label: Quick Research
    description: Low-latency baseline retrieval for everyday prompts.
    prompt:
      quote_min_count: 4
    retrieval:
      stage1_topn: 160
      topk_post_rerank: 40
      ce_min_keep: 20
      max_final_nodes: 12
  deep:
    label: Deep Research
    description: Wider recall pass with larger synthesis budget and denser citations.
    aliases: [thorough, verbose]
    prompt:
      quote_min_count: 6
    retrieval:
      stage1_topn: 260
      topk_post_rerank: 60
      ce_min_keep: 28
      ce_keep_percentile: 0.9
      ce_abs_min: 0.28
      max_final_nodes: 18
      max_segments_per_parent: 10
```

Key points:

- `prompt.quote_min_count` controls the minimum citation target injected into the system prompt.
- `retrieval.*` entries map directly to `RetrievalConfig` fields and override them per request via a context-local config. Unsupported keys are ignored.
- `aliases` let you accept multiple spellings for the same mode.
- The `default` entry decides which mode is used when none is specified (and can be overridden with the `RAG_DEFAULT_MODE` environment variable).

## Observability Notes

- Progress metadata now carries `research_mode`, optional `research_mode_label`, and any applied `mode_overrides`.
- Each trace stored via `AppDiagnostics` includes `mode_config`, `mode_label`, and the requested alias, so dashboards or log filters can group traces by mode.

## Operational Tips

1. **Budgeting** – Track latency, token usage, and cost for each mode separately. The added diagnostics fields make this straightforward.
2. **Experimentation** – You can introduce additional modes (e.g., `factcheck`, `fast`) by adding entries under `modes` and redeploying; no code changes are required as long as keys map to `RetrievalConfig`.
3. **Governance** – If you expose the mode toggle in the UI, note the trade-offs (latency/cost vs. coverage) so users know what to expect.

