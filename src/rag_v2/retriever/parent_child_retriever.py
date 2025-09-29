# --- DROP-IN: entity-aware retrieve (filtered merge + overlap boosts) ---
from __future__ import annotations
from typing import List, Dict, Any
from ..config import CFG
from ..router.video_router import TICKER_RE, HANDLE_RE, wants_definition
from ..utils.scoring import recency_decay, apply_multiplier
from ..logging_utils import setup_logger, node_brief

log = setup_logger("rag_v2.retriever")

class ParentChildRetrieverV2:
    def __init__(self, base_retriever):
        self.base = base_retriever
        self._last_debug: Dict[str, Any] = {}

    def debug_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_debug) if self._last_debug else {}

    # --- NEW: tiny helpers ---
    @staticmethod
    def _qid(nws) -> str:
        return getattr(getattr(nws, "node", None), "node_id", None) or getattr(getattr(nws, "node", None), "id_", None) or str(id(nws))

    @staticmethod
    def _query_entities(q: str):
        ents = set(m.group(0).strip() for m in TICKER_RE.finditer(q))
        ents |= set(m.group(0).strip() for m in HANDLE_RE.finditer(q))
        # canon like router: $UPPER / @lower
        canon = set()
        for e in ents:
            canon.add(e.upper() if e.startswith("$") else (e.lower() if e.startswith("@") else e.lower()))
        return canon

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

    def _apply_metadata_boosts(self, nodes, query: str, qents: set):
        def_score = 1.0 if not wants_definition(query) else 1.2
        ent_gain  = float(getattr(CFG, "entity_overlap_gain", 0.15))
        router_mult = float(getattr(CFG, "router_boost_mult", 1.05))
        streams_mult = float(getattr(CFG, "streams_bias_mult", 1.08))

        boosted_list = []
        for nws in nodes:
            md = nws.node.metadata or {}
            s = float(nws.score or 0.0)

            # recency decay
            decay = recency_decay(md.get("published_date") or md.get("published_at"), CFG.recency_half_life_days)
            s *= decay

            # router/explainer bias mirrored on child
            rb = float(md.get("router_boost") or 1.0)
            if wants_definition(query) and md.get("is_explainer"):
                rb *= router_mult
            s = apply_multiplier(s, rb * def_score)

            # entity overlap bonus
            ents = set(md.get("entities") or [])
            overlap = len(ents & qents)
            s = apply_multiplier(s, (1.0 + ent_gain * min(overlap, 3)))

            # streams bias
            if "stream" in (md.get("document_type") or "") and any(k in query.lower() for k in ["live","yesterday","stream","@"]):
                s = apply_multiplier(s, streams_mult)

            nws.score = s
            boosted_list.append({
                "node": node_brief(nws),
                "decay": decay,
                "def_score": def_score,
                "applied_router_boost": rb,
                "entity_overlap": overlap,
            })
        return nodes, boosted_list

    def _entity_filtered_retrieve(self, query_bundle, qents: set):
        """Best-effort second pass with metadata filter; falls back silently if base doesn't support it."""
        if not qents:
            return [], {"used": False, "filter": {}, "count": 0}
        filt = {"entities": {"$in": list(qents)}}
        nodes2 = []
        try:
            nodes2 = self.base.retrieve(query_bundle, metadata_filter=filt)
        except TypeError:
            try:
                nodes2 = self.base.retrieve(query_bundle, filters=filt)
            except Exception:
                nodes2 = []
        return nodes2, {"used": bool(nodes2), "filter": filt, "count": len(nodes2)}

    def _merge_and_boost_filtered(self, nodes_a, nodes_b, boost: float):
        """Merge two node lists by id; multiply scores of list B by boost."""
        by_id = {}
        for n in nodes_a:
            by_id[self._qid(n)] = n
        for n in nodes_b:
            qid = self._qid(n)
            if qid in by_id:
                by_id[qid].score = max(float(by_id[qid].score or 0.0), float(n.score or 0.0) * boost)
            else:
                n.score = float(n.score or 0.0) * boost
                by_id[qid] = n
        return list(by_id.values())

    def retrieve(self, query_bundle):
        q = query_bundle.query_str
        qents = self._query_entities(q)

        # Stage 1: vanilla vector retrieve
        nodes = self.base.retrieve(query_bundle)
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

        # Stage 1b: entity-filtered retrieve (optional) + merge
        filt_nodes, filt_dbg = self._entity_filtered_retrieve(query_bundle, qents)
        merged = self._merge_and_boost_filtered(nodes, filt_nodes, float(getattr(CFG, "entity_filter_boost", 1.25)))

        # Apply metadata boosts (recency/router/entity/streams)
        merged, boost_details = self._apply_metadata_boosts(merged, q, qents)

        # Neighbor expansion
        neighbors, expanded_map = self._expand_neighbors(merged, per_parent_cap=3)

        # Pre-CE shortlist
        neighbors_sorted = sorted(neighbors, key=lambda n: (n.score or 0.0), reverse=True)
        neighbors_topn = neighbors_sorted[:CFG.stage1_topn]

        # Debug snapshot
        self._last_debug = {
            "query": q,
            "query_entities": sorted(qents),
            "stage1_raw": stage1,
            "stage1_scores": stage1_scores,
            "stage1_doc_types": by_dtype,
            "entity_filtered": filt_dbg,
            "after_metadata_boosts": boost_details,
            "neighbor_expansion": expanded_map,
            "pre_ce_topN": [node_brief(n) for n in neighbors_topn],
        }
        return neighbors_topn
