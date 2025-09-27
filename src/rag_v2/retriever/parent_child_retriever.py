from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os
from ..config import CFG
from ..router.video_router import wants_definition, router_bias
from ..utils.scoring import recency_decay, apply_multiplier

"""
This retriever assumes your vector index stores CHILD nodes.
It fetches top-N children, expands to their parent, then re-expands to neighbor children
from the same parent (Auto-Merging / ParentDocument pattern).
"""

class ParentChildRetrieverV2:
    def __init__(self, base_retriever):
        """
        base_retriever: a LlamaIndex retriever (core) already pointing at your Pinecone store.
        """
        self.base = base_retriever

    def _expand_neighbors(self, nodes, per_parent_cap=3):
        by_parent: Dict[str, List[Any]] = {}
        for nws in nodes:
            pid = (nws.node.metadata or {}).get("parent_id") or (nws.node.metadata or {}).get("video_id")
            by_parent.setdefault(pid, []).append(nws)
        out = []
        for pid, rows in by_parent.items():
            rows_sorted = sorted(rows, key=lambda r: (r.score or 0.0), reverse=True)
            out.extend(rows_sorted[:per_parent_cap])
        return out

    def _apply_metadata_boosts(self, nodes, query: str):
        def_score = 1.0 if not wants_definition(query) else 1.2
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
        return nodes

    def retrieve(self, query_bundle):
        nodes = self.base.retrieve(query_bundle)  # stage-1 dense
        nodes = self._apply_metadata_boosts(nodes, query_bundle.query_str)

        # neighbor expansion (keep K small after)
        neighbors = self._expand_neighbors(nodes, per_parent_cap=3)

        # truncate to stage1_topn (pre-CE)
        neighbors = sorted(neighbors, key=lambda n: (n.score or 0.0), reverse=True)[:CFG.stage1_topn]
        return neighbors
