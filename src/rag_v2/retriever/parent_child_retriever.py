from __future__ import annotations
from typing import List, Dict, Any
from ..config import CFG
from ..router.video_router import wants_definition
from ..utils.scoring import recency_decay, apply_multiplier
from ..logging_utils import setup_logger, node_brief

"""
This retriever assumes your vector index stores CHILD nodes.
It fetches top-N children, expands to their parent, then re-expands to neighbor children
from the same parent (Auto-Merging / ParentDocument pattern).
"""

log = setup_logger("rag_v2.retriever")

class ParentChildRetrieverV2:
    def __init__(self, base_retriever):
        """
        base_retriever: a LlamaIndex retriever (core) already pointing at your Pinecone store.
        """
        self.base = base_retriever
        self._last_debug: Dict[str, Any] = {}

    def debug_snapshot(self) -> Dict[str, Any]:
        """Return a copy of last debug dictionary for the QE to include."""
        return dict(self._last_debug) if self._last_debug else {}

    def _expand_neighbors(self, nodes, per_parent_cap=3):
        by_parent: Dict[str, List[Any]] = {}
        for nws in nodes:
            md = (nws.node.metadata or {})
            pid = md.get("parent_id") or md.get("video_id")
            by_parent.setdefault(pid, []).append(nws)
        out = []
        expanded_map: Dict[str, List[Dict[str, Any]]] = {}
        for pid, rows in by_parent.items():
            rows_sorted = sorted(rows, key=lambda r: (r.score or 0.0), reverse=True)
            kept = rows_sorted[:per_parent_cap]
            out.extend(kept)
            expanded_map[str(pid)] = [node_brief(n) for n in kept]
        return out, expanded_map

    def _apply_metadata_boosts(self, nodes, query: str):
        def_score = 1.0 if not wants_definition(query) else 1.2
        boosted_list = []
        for nws in nodes:
            md = nws.node.metadata or {}
            # recency
            decay = recency_decay(md.get("published_date") or md.get("published_at"), CFG.recency_half_life_days)
            s = (nws.score or 0.0) * decay
            # router bias (parents mirrored on child metadata via upsert)
            rb = md.get("router_boost") or 1.0
            if wants_definition(query) and md.get("is_explainer"):
                rb *= CFG.router_boost_mult
            s = apply_multiplier(s, rb * def_score)
            # streams bias
            if "stream" in (md.get("document_type") or "") and any(k in query.lower() for k in ["live","yesterday","stream","@"]):
                s = apply_multiplier(s, CFG.streams_bias_mult)
            nws.score = s
            boosted_list.append({
                "node": node_brief(nws),
                "decay": decay,
                "def_score": def_score,
                "applied_router_boost": rb,
            })
        return nodes, boosted_list

    def retrieve(self, query_bundle):
        # Stage 1: vector retrieve
        nodes = self.base.retrieve(query_bundle)

        # Stage-1 snapshot
        stage1 = [node_brief(n) for n in nodes]
        if nodes:
            s = [float(n.score or 0.0) for n in nodes]
            stage1_scores = {"count": len(nodes), "min": min(s), "max": max(s)}
            by_dtype = {}
            for n in nodes:
                d = (n.node.metadata or {}).get("document_type") or "unknown"
                by_dtype[d] = by_dtype.get(d, 0) + 1
        else:
            stage1_scores, by_dtype = {"count": 0, "min": 0.0, "max": 0.0}, {}

        # Apply metadata boosts
        nodes, boost_details = self._apply_metadata_boosts(nodes, query_bundle.query_str)

        # Neighbor expansion
        neighbors, expanded_map = self._expand_neighbors(nodes, per_parent_cap=3)

        # Truncate to stage1_topn (pre-CE)
        neighbors_sorted = sorted(neighbors, key=lambda n: (n.score or 0.0), reverse=True)
        neighbors_topn = neighbors_sorted[:CFG.stage1_topn]

        # Save debug snapshot
        self._last_debug = {
            "query": query_bundle.query_str,
            "stage1_raw": stage1,  # list[dict] of child nodes
            "stage1_scores": stage1_scores,  # {count,min,max}
            "stage1_doc_types": by_dtype,  # {"youtube_video": 12, ...}
            "after_metadata_boosts": boost_details,  # per-node decay/boosts
            "neighbor_expansion": expanded_map,  # {parent_id: [child_briefs...]}
            "pre_ce_topN": [node_brief(n) for n in neighbors_topn],
        }

        return neighbors_topn
