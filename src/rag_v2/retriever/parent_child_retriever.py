# --- DROP-IN: entity-aware retrieve (filtered merge + overlap boosts) ---
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Set
import re

from ..config import CFG, ENT_CANON_MAP
from ..router.video_router import TICKER_RE, HANDLE_RE, wants_definition
from ..postprocessors.entity_utils import canon_entity, canon_entities
from ..utils.scoring import recency_decay, apply_multiplier
from ..logging_utils import setup_logger, node_brief, _clean_title
from ..vector_store.parent_resolver import fetch_parent_meta

log = setup_logger("rag_v2.retriever")


class ParentChildRetrieverV2:
    def __init__(self, base_retriever):
        self.base = base_retriever
        self._last_debug: Dict[str, Any] = {}
        self._channel_filter: Optional[Dict[str, List[str]]] = None
        self._entity_requirements: Optional[Set[str]] = None

    def debug_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_debug) if self._last_debug else {}

    # --- helpers ---
    @staticmethod
    def _qid(nws) -> str:
        return (
            getattr(getattr(nws, "node", None), "node_id", None)
            or getattr(getattr(nws, "node", None), "id_", None)
            or str(id(nws))
        )

    def _query_entities(self, q: str) -> Tuple[set, set, Set[str]]:
        """
        Extract $tickers/@handles plus canonical Solana nouns (Firedancer/Anza/Alpenglow/Aster).
        Returns:
          - all_forms: for metadata filter ($UPPER/@lower/Mixed + lc + title-case)
          - lc_forms: lowercase forms for overlap math
          - canonical_set: deduped canonical entity names
        """
        raw = set(m.group(0).strip() for m in TICKER_RE.finditer(q))
        raw |= set(m.group(0).strip() for m in HANDLE_RE.finditer(q))
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", q):
            ce = canon_entity(tok)
            if not ce:
                continue
            if tok.startswith("@"):  # already handled
                raw.add(ce)
                continue
            lower_tok = tok.strip().lower()
            if lower_tok in ENT_CANON_MAP or ce != tok:
                raw.add(ce)
        for tok in re.findall(r"\b[A-Z]{2,}\b", q):
            ce = canon_entity(tok)
            lower_tok = tok.lower()
            if lower_tok in ENT_CANON_MAP or ce != tok:
                raw.add(ce)
        all_forms = set()
        for e in raw:
            if e.startswith("$"):
                all_forms.add(e.upper())
            elif e.startswith("@"):
                all_forms.add(e.lower())
            else:
                all_forms.update({e, e.lower(), e.title()})
        canonical = set(canon_entities(raw))
        return all_forms, {x.lower() for x in all_forms if not x.startswith("$")}, canonical

    def set_channel_filter(self, channel_filter: Optional[Dict[str, List[str]]]) -> None:
        if not channel_filter:
            self._channel_filter = None
            return
        copied: Dict[str, List[str]] = {}
        for key, values in channel_filter.items():
            if not values:
                continue
            copied[key] = list(values)
        self._channel_filter = copied or None
    def set_entity_requirements(self, required: Optional[Set[str]]) -> None:
        if required:
            self._entity_requirements = {str(r) for r in required if r}
        else:
            self._entity_requirements = None

    @staticmethod
    def _dedupe_clean(values: Optional[List[str]]) -> List[str]:
        if not values:
            return []
        seen = set()
        out: List[str] = []
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _build_channel_filter(self, channel_filter: Optional[Dict[str, List[str]]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        debug = {
            "requested": channel_filter,
            "expr": None,
            "applied": False,
        }
        if not channel_filter:
            return None, debug

        include_clauses: List[Dict[str, Any]] = []
        exclude_clauses: List[Dict[str, Any]] = []

        inc_ids = self._dedupe_clean(channel_filter.get("include_ids"))
        inc_names = self._dedupe_clean(channel_filter.get("include_names"))
        exc_ids = self._dedupe_clean(channel_filter.get("exclude_ids"))
        exc_names = self._dedupe_clean(channel_filter.get("exclude_names"))

        if inc_ids:
            include_clauses.append({"channel_id": {"$in": inc_ids}})
        if inc_names:
            include_clauses.append({"channel_name": {"$in": inc_names}})

        if exc_ids:
            exclude_clauses.append({"channel_id": {"$nin": exc_ids}})
        if exc_names:
            exclude_clauses.append({"channel_name": {"$nin": exc_names}})

        clauses: List[Dict[str, Any]] = []
        if include_clauses:
            clauses.append({"$or": include_clauses} if len(include_clauses) > 1 else include_clauses[0])
        clauses.extend(exclude_clauses)

        if not clauses:
            return None, debug

        if len(clauses) == 1:
            expr = clauses[0]
        else:
            expr = {"$and": clauses}

        debug["expr"] = expr
        debug["applied"] = True
        return expr, debug

    @staticmethod
    def _merge_filters(filters: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        conds = [f for f in filters if f]
        if not conds:
            return None
        if len(conds) == 1:
            return conds[0]
        return {"$and": conds}

    def _base_retrieve_with_filter(self, query_bundle, filter_expr: Optional[Dict[str, Any]]):
        if not filter_expr:
            return self.base.retrieve(query_bundle)
        try:
            return self.base.retrieve(query_bundle, metadata_filter=filter_expr)
        except TypeError:
            try:
                return self.base.retrieve(query_bundle, filters=filter_expr)
            except Exception:
                return self.base.retrieve(query_bundle)

    def _expand_neighbors(self, nodes, per_parent_cap=3):
        by_parent: Dict[str, List[Any]] = {}
        for nws in nodes:
            md = (nws.node.metadata or {})
            pid = md.get("parent_id") or md.get("video_id")
            by_parent.setdefault(str(pid), []).append(nws)
        out = []
        expanded_map: Dict[str, List[Dict[str, Any]]] = {}
        for pid, rows in by_parent.items():
            rows_sorted = sorted(rows, key=lambda r: (r.score or 0.0), reverse=True)
            kept = rows_sorted[:per_parent_cap]
            out.extend(kept)
            expanded_map[pid] = [node_brief(n) for n in kept]
        return out, expanded_map

    def _apply_metadata_boosts(self, nodes, query: str, qents_lc: set):
        def_score = 1.0 if not wants_definition(query) else 1.2
        ent_gain = float(getattr(CFG, "entity_overlap_gain", 0.15))
        router_mult = float(getattr(CFG, "router_boost_mult", 1.05))
        streams_mult = float(getattr(CFG, "streams_bias_mult", 1.08))

        boosted_list = []
        q_lc = query.lower()

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

            # entity overlap bonus (use entities + canonical_entities, case-insensitive)
            ents_lc = {str(e).lower() for e in (md.get("entities") or [])}
            ents_lc |= {str(e).lower() for e in (md.get("canonical_entities") or [])}
            overlap = len(ents_lc & qents_lc)
            s = apply_multiplier(s, (1.0 + ent_gain * min(overlap, 3)))

            # streams bias
            if "stream" in (md.get("document_type") or "") and any(k in q_lc for k in ["live", "yesterday", "stream", "@"]):
                s = apply_multiplier(s, streams_mult)

            nws.score = s
            boosted_list.append(
                {
                    "node": node_brief(nws),
                    "decay": decay,
                    "def_score": def_score,
                    "applied_router_boost": rb,
                    "entity_overlap": overlap,
                }
            )
        return nodes, boosted_list

    @staticmethod
    def _canonical_entities_from_metadata(md: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for source in ("canonical_entities", "entities"):
            for ent in md.get(source) or []:
                ce = canon_entity(ent)
                if ce:
                    out.add(ce)
        speaker = md.get("speaker")
        if speaker:
            ce = canon_entity(speaker)
            if ce:
                out.add(ce)
        channel = md.get("channel_name") or md.get("parent_channel_name")
        if channel:
            ce = canon_entity(channel)
            if ce:
                out.add(ce)
        return out

    def _entity_gate_nodes(self, nodes: List[Any], required: Set[str]) -> Tuple[List[Any], Dict[str, Any]]:
        debug = {
            "required": sorted(required),
            "applied": bool(required),
            "kept": 0,
            "dropped": 0,
            "dropped_samples": [],
        }
        if not required:
            return nodes, debug
        kept: List[Any] = []
        dropped_meta: List[Dict[str, Any]] = []
        for n in nodes:
            md = n.node.metadata or {}
            ents = self._canonical_entities_from_metadata(md)
            if required.issubset(ents):
                kept.append(n)
            else:
                dropped_meta.append({
                    "node": node_brief(n),
                    "entities": sorted(ents),
                })
        debug["kept"] = len(kept)
        debug["dropped"] = len(nodes) - len(kept)
        debug["dropped_samples"] = dropped_meta[:5]
        return kept, debug

    def _entity_filtered_retrieve(self, query_bundle, qents_all: set, extra_filter: Optional[Dict[str, Any]]):
        """Second pass with metadata filter; falls back silently if base doesn't support it."""
        if not qents_all:
            return [], {"used": False, "filter": {}, "count": 0}

        forms = list(qents_all)
        filt = {
            "$or": [
                {"entities": {"$in": forms}},
                {"canonical_entities": {"$in": forms}},
            ]
        }

        nodes2 = []
        merged_filter = self._merge_filters([filt, extra_filter])
        try:
            if merged_filter:
                nodes2 = self.base.retrieve(query_bundle, metadata_filter=merged_filter)
            else:
                nodes2 = self.base.retrieve(query_bundle, metadata_filter=filt)
        except TypeError:
            try:
                if merged_filter:
                    nodes2 = self.base.retrieve(query_bundle, filters=merged_filter)
                else:
                    nodes2 = self.base.retrieve(query_bundle, filters=filt)
            except Exception:
                nodes2 = []
        return nodes2, {"used": bool(nodes2), "filter": merged_filter or filt, "count": len(nodes2)}

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
        qents_all, qents_lc, qents_canonical = self._query_entities(q)

        channel_filter_expr, channel_filter_dbg = self._build_channel_filter(self._channel_filter)
        # channel filter is single-use per request; clear after capturing
        self._channel_filter = None

        # Stage 1: vanilla vector retrieve (with optional channel filter)
        nodes = self._base_retrieve_with_filter(query_bundle, channel_filter_expr)
        stage1 = [node_brief(n) for n in nodes]
        channel_filter_dbg["stage1_applied"] = bool(channel_filter_expr)
        channel_filter_dbg["stage1_candidates"] = len(nodes)
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
        filt_nodes, filt_dbg = self._entity_filtered_retrieve(query_bundle, qents_all, channel_filter_expr)
        channel_filter_dbg["stage1b_candidates"] = len(filt_nodes)
        channel_filter_dbg["stage1b_filter"] = filt_dbg.get("filter")

        merged = self._merge_and_boost_filtered(
            nodes,
            filt_nodes,
            float(getattr(CFG, "entity_filter_boost", 1.35)),
        )

        # --- NEW: enrich children with parent metadata (title/channel/date/url) BEFORE boosts ---
        parent_ids = set()
        for n in merged:
            md = n.node.metadata or {}
            pid = md.get("parent_id") or md.get("video_id")
            if pid:
                parent_ids.add(str(pid))

        parent_meta_dbg = {"requested": len(parent_ids), "found": 0, "enriched_nodes": 0, "missed_nodes": 0,
                           "error": None}
        if parent_ids:
            try:
                # lazy import here to keep this method drop-in
                from ..vector_store.parent_resolver import fetch_parent_meta
                ns = os.getenv("PINECONE_NAMESPACE", "videos")
                pres = fetch_parent_meta(parent_ids)  # resolver has its own "videos" default
                sample = next(iter(parent_ids)) if parent_ids else None
                if sample:
                    print("[debug] sample parent_id:", sample, "->", pres.get(str(sample)))

                parent_meta_dbg["found"] = sum(1 for _k, v in pres.items() if v)

                hit, miss = 0, 0
                for n in merged:
                    md = n.node.metadata or {}
                    pid = str(md.get("parent_id") or md.get("video_id") or "")
                    pm = pres.get(pid) or {}
                    if pm:
                        md.setdefault("parent_title", _clean_title(pm.get("parent_title")))
                        md.setdefault("parent_channel_name", pm.get("parent_channel_name"))
                        md.setdefault("parent_channel_id", pm.get("parent_channel_id"))
                        md.setdefault("parent_published_at", pm.get("parent_published_at"))
                        md.setdefault("parent_url", pm.get("parent_url"))
                        if not md.get("title") and pm.get("parent_title"):
                            md["title"] = _clean_title(pm["parent_title"])
                        if not md.get("channel_name") and pm.get("parent_channel_name"):
                            md["channel_name"] = pm["parent_channel_name"]
                        if not md.get("channel_id") and pm.get("parent_channel_id"):
                            md["channel_id"] = pm["parent_channel_id"]
                        if not md.get("published_at") and pm.get("parent_published_at"):
                            md["published_at"] = pm["parent_published_at"]
                        if not md.get("url") and pm.get("parent_url"):
                            md["url"] = pm["parent_url"]
                        if md.get("title"):
                            md["title"] = _clean_title(md["title"])
                        n.node.metadata = md
                        hit += 1  # <-- add this
                    else:
                        miss += 1
                parent_meta_dbg["enriched_nodes"] = hit
                parent_meta_dbg["missed_nodes"] = miss

                log.info(
                    "retriever[parent-meta] parents=%d found=%d enriched_nodes=%d missed_nodes=%d",
                    parent_meta_dbg["requested"],
                    parent_meta_dbg["found"],
                    parent_meta_dbg["enriched_nodes"],
                    parent_meta_dbg["missed_nodes"],
                )
            except Exception as e:
                parent_meta_dbg["error"] = str(e)
                log.warning("retriever[parent-meta] enrichment skipped due to error: %s", e)

        # Apply metadata boosts (recency/router/entity/streams)
        merged, boost_details = self._apply_metadata_boosts(merged, q, qents_lc)

        # --- per-parent diversity cap BEFORE neighbor expansion ---
        by_parent: Dict[str, List[Any]] = {}
        for n in merged:
            pid = (n.node.metadata or {}).get("parent_id") or (n.node.metadata or {}).get("video_id")
            by_parent.setdefault(str(pid), []).append(n)
        merged_diverse = []
        for pid, rows in by_parent.items():
            rows.sort(key=lambda r: (r.score or 0.0), reverse=True)
            merged_diverse.extend(rows[:CFG.max_segments_per_parent])
        merged = merged_diverse

        # Neighbor expansion
        neighbors, expanded_map = self._expand_neighbors(merged, per_parent_cap=3)

        # Pre-CE shortlist
        neighbors_sorted = sorted(neighbors, key=lambda n: (n.score or 0.0), reverse=True)
        neighbors_topn = neighbors_sorted[: CFG.stage1_topn]
        required_entities = self._entity_requirements
        if not required_entities and len(qents_canonical) >= 2:
            required_entities = qents_canonical
        entity_gate_dbg = {"required": [], "applied": False, "kept": len(neighbors_topn), "dropped": 0}
        if required_entities:
            neighbors_topn, entity_gate_dbg = self._entity_gate_nodes(neighbors_topn, required_entities)
        self._entity_requirements = None
        channel_filter_dbg["final_candidates"] = len(neighbors_topn)

        # Debug snapshot
        self._last_debug = {
            "query": q,
            "query_entities": sorted(qents_all),
            "stage1_raw": stage1,
            "stage1_scores": stage1_scores,
            "stage1_doc_types": by_dtype,
            "entity_filtered": filt_dbg,
            "channel_filter": channel_filter_dbg,
            "parent_meta": parent_meta_dbg,  # <-- NEW
            "after_metadata_boosts": boost_details,
            "neighbor_expansion": expanded_map,
            "pre_ce_topN": [node_brief(n) for n in neighbors_topn],
            "entity_gate": entity_gate_dbg,
        }
        return neighbors_topn
