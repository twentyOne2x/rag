# rag_v2 (parent/child + router + CE rerank)

This module consumes your v2 ingestion schema (timestamped child segments), adds:
- Video/Definition router using parent metadata (`is_explainer`, `router_boost`, aliases).
- Stage-1 dense recall on children → Auto-Merging to parent → neighbor-children expansion.
- Optional BM25+RRF fusion (title/description/intro) before CE.
- Optional Stage-2 Cross-Encoder rerank.
- Entities/speakers filters and streams bias.
