# File: src/rag_v2/query_engine_v2.py
from __future__ import annotations
import asyncio
import math
import re
from typing import List, Tuple, Any, Dict

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import CFG
from .postprocessors.speaker_propagator import SpeakerPropagator
from .rerankers.cross_encoder import CEReranker
from .postprocessors.entity_canonicalizer import EntityCanonicalizer  # NEW
from .postprocessors.entity_utils import normalize_text_entities  # NEW
from .router.video_router import TICKER_RE, HANDLE_RE
from .postprocessors.entity_utils import canon_entity
from .logging_utils import (
    setup_logger,
    pretty,
    node_brief,
    clean_model_refs,
    time_block,
    cfg_snapshot,
    model_snapshot,
    append_sources_block,
)
from .router.video_router import wants_definition

log = setup_logger("rag_v2.qe")


class ParentChildQueryEngineV2(BaseQueryEngine):
    """
    Query engine that:
      1) Uses a parent/child retriever,
      2) Optionally reranks with a cross-encoder,
      3) Applies entity canonicalization to fix text errors,  # NEW
      4) Synthesizes with LlamaIndex's core response synthesizer.

    Emits a single JSON 'trace' per query at DEBUG (file sink if configured) and
    concise INFO logs for humans.
    """

    def __init__(self, retriever, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self._retriever = retriever
        self._core = RetrieverQueryEngine.from_args(retriever=self._retriever)
        self._ce = (
            CEReranker(model_name=CFG.ce_model, batch_size=CFG.ce_batch_size)
            if CFG.enable_ce
            else None
        )
        # NEW: Entity canonicalizer to fix "Soul" -> "SOL" etc.
        self._entity_canonicalizer = EntityCanonicalizer()
        self._speaker_propagator = SpeakerPropagator()

    # Required for some LI versions
    def _get_prompt_modules(self):
        return {}

    # -------- helpers --------
    def _ce_pack(self, nodes: List[NodeWithScore]) -> List[Tuple[str, str, float]]:
        items: List[Tuple[str, str, float]] = []
        for n in nodes:
            md = n.node.metadata or {}
            sid = md.get("segment_id") or md.get("id") or n.node.node_id
            items.append((sid, n.node.get_content(), float(n.score or 0.0)))
        return items

    def _reinject_scores(
            self, nodes: List[NodeWithScore], rescored: List[Tuple[str, str, float]]
    ) -> List[NodeWithScore]:
        score_by_sid = {sid: sc for sid, _, sc in rescored}
        for n in nodes:
            md = n.node.metadata or {}
            sid = md.get("segment_id") or md.get("id") or n.node.node_id
            if sid in score_by_sid:
                n.score = score_by_sid[sid]
        nodes.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return nodes

    def _final_sources_view(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in nodes:
            view = node_brief(n)
            md = n.node.metadata or {}
            view.update(
                {
                    "url": md.get("clip_url") or md.get("url"),
                    "title": md.get("title"),
                    "channel_name": md.get("channel_name"),
                }
            )
            out.append(view)
        return out

    def _annotate_speakers(self, nodes):
        out = []
        for n in nodes:
            md = n.node.metadata or {}
            spk = md.get("speaker")
            if spk:
                hms = " - ".join(filter(None, [md.get("start_hms"), md.get("end_hms")]))
                # mutate text once per final node
                n.node.text = f"[{spk}{' | ' + hms if hms else ''}] " + n.node.get_content()
            out.append(n)
        return out

    def _synthesize_clean(self, query_bundle: QueryBundle, nodes: List[NodeWithScore]) -> Response:
        raw = self._core._response_synthesizer.synthesize(query=query_bundle, nodes=nodes)
        try:
            cleaned_text = clean_model_refs(str(raw))
            # NEW: Apply final entity normalization to the answer
            cleaned_text = normalize_text_entities(cleaned_text)

            # prefer whatever the synthesizer returned for source_nodes; else fall back to our `nodes`
            src_nodes = getattr(raw, "source_nodes", None) or nodes
            final_text = append_sources_block(cleaned_text, src_nodes)

            if hasattr(raw, "response"):
                # mutate in-place to preserve Response extras/metadata and sources
                raw.response = final_text
                # make sure source_nodes are present
                if not getattr(raw, "source_nodes", None):
                    raw.source_nodes = nodes
                return raw

            return Response(final_text, source_nodes=src_nodes)
        except Exception:
            # last-resort: return raw as-is
            return raw

    import math
    import re

    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except Exception:
            return float(x)

    def _percentile_cut(scores, p: float) -> float:
        if not scores:
            return float("inf")
        s = sorted(scores)
        idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return float(s[idx])

    def _rough_token_count(txt: str) -> int:
        # ~4 chars/token heuristic
        return max(1, len(txt) // 4)

    def _hms_to_seconds(hms: str) -> int:
        if not hms: return -1
        try:
            h, m, s = [int(x) for x in hms.split(":")]
            return h * 3600 + m * 60 + s
        except Exception:
            return -1

    def _stitch_adjacent(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        groups: Dict[str, List[NodeWithScore]] = {}
        for n in nodes:
            md = n.node.metadata or {}
            pid = str(md.get("parent_id") or md.get("video_id"))
            groups.setdefault(pid, []).append(n)

        out: List[NodeWithScore] = []
        gap_s = CFG.stitch_gap_seconds
        target_tokens = CFG.stitch_target_tokens
        max_merge = CFG.stitch_max_merge

        for pid, rows in groups.items():
            def _sec(node, key):
                return type(self)._hms_to_seconds((node.node.metadata or {}).get(key) or "")

            rows.sort(key=lambda r: _sec(r, "start_hms"))
            i = 0
            while i < len(rows):
                chunk = [rows[i]]
                i += 1
                # greedily merge forward while gaps small and under token budget
                while i < len(rows) and len(chunk) < max_merge:
                    prev = chunk[-1]
                    cur = rows[i]
                    end_prev = _sec(prev, "end_hms")
                    start_cur = _sec(cur, "start_hms")
                    if end_prev >= 0 and start_cur >= 0 and (start_cur - end_prev) <= gap_s:
                        merged_text = " ".join(c.node.get_content() for c in (chunk + [cur]))
                        if type(self)._rough_token_count(merged_text) <= target_tokens:
                            chunk.append(cur)
                            i += 1
                            continue
                    break

                if len(chunk) == 1:
                    out.append(chunk[0])
                else:
                    base = chunk[0]
                    md = base.node.metadata or {}
                    md["start_hms"] = (chunk[0].node.metadata or {}).get("start_hms")
                    md["end_hms"] = (chunk[-1].node.metadata or {}).get("end_hms")
                    base.node.text = "\n".join(c.node.get_content() for c in chunk)
                    base.score = max(float(c.score or 0.0) for c in chunk)
                    out.append(base)

        out.sort(key=lambda n: (n.score or 0.0), reverse=True)
        return out[: CFG.max_final_nodes]

    def _final_k(self, q: str) -> int:
        if wants_definition(q):  # short, crisp
            return min(8, CFG.max_final_nodes)
        if "return all videos" in q.lower():  # coverage query
            return CFG.max_final_nodes
        return min(10, CFG.max_final_nodes)

    def _qents(self, q: str) -> set[str]:
        ents = set(m.group(0).strip() for m in TICKER_RE.finditer(q))
        ents |= set(m.group(0).strip() for m in HANDLE_RE.finditer(q))
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", q):
            ce = canon_entity(tok)
            if ce: ents.add(ce)
        return {e.lower() for e in ents if not e.startswith("$")}

    def _maybe_early_abort_stage1(self, q: str, nodes: List[NodeWithScore]) -> Response | None:
        """Cheap pre-CE gate: low top score AND little entity overlap."""
        if not (CFG.enable_early_abort and nodes):
            return None
        top = max(float(n.score or 0.0) for n in nodes) if nodes else 0.0

        # unconditional cutoff
        if top < CFG.stage1_hard_min:
            log.info("qe[early-abort] q='%s' reason=stage1_hard_min top=%.5f<th=%.5f",
                     q, top, CFG.stage1_hard_min)
            return Response(CFG.abort_message, source_nodes=[])

        qents = self._qents(q)
        rel = 0
        for n in nodes:
            md = n.node.metadata or {}
            ents = {str(e).lower() for e in (md.get("entities") or [])}
            ents |= {str(e).lower() for e in (md.get("canonical_entities") or [])}
            if ents & qents:
                rel += 1
        if top < CFG.stage1_top_min and rel < CFG.stage1_min_relevant:
            log.info("qe[early-abort] q='%s' reason=stage1_low top=%.4f rel=%d", q, top, rel)
            return Response(CFG.abort_message, source_nodes=[])
        return None

    def _maybe_early_abort_post_ce(self, ce_norm: List[float]) -> Response | None:
        """Post-CE gate: after CE scoring but before synthesis."""
        if not CFG.enable_early_abort:
            return None
        mx = max(ce_norm) if ce_norm else 0.0
        if mx < CFG.ce_max_norm_min:
            log.info("qe[early-abort] reason=ce_low max_norm=%.4f", mx)
            return Response(CFG.abort_message, source_nodes=[])
        return None

    # -------- sync path --------
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        q = query_bundle.query_str
        trace: Dict[str, Any] = {
            "query": q,
            "config": cfg_snapshot(CFG),
            "models": model_snapshot(),
        }

        t_total = time_block()

        # 1) Retrieve (+ retriever snapshot)
        t_retrieve = time_block()
        with self.callback_manager.event(CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: q}) as revent:
            nodes: List[NodeWithScore] = self._retriever.retrieve(query_bundle)

            early = self._maybe_early_abort_stage1(q, nodes)
            if early:
                return early

            if hasattr(self._retriever, "debug_snapshot"):
                trace.update(self._retriever.debug_snapshot())

            # 2) Optional CE rerank (entity/metadata-aware)
            if self._ce and self._ce.enabled and nodes:
                packs = self._ce_pack(nodes)
                trace["pre_ce"] = [{"segment_id": sid, "score_stage1": sc} for (sid, _t, sc) in packs]

                metas: Dict[str, Dict[str, Any]] = {}
                for n in nodes:
                    md = n.node.metadata or {}
                    sid = md.get("segment_id") or md.get("id") or n.node.node_id
                    metas[sid] = {
                        "entities": md.get("entities") or [],
                        "canonical_entities": md.get("canonical_entities") or [],
                        "node_type": md.get("node_type") or md.get("document_type"),
                        "router_boost": md.get("router_boost"),
                        "is_explainer": md.get("is_explainer"),
                    }

                t_ce = time_block()
                rescored = self._ce.rerank_with_meta(q, packs, metas)
                trace["ce_ms"] = t_ce()
                trace["ce_scores"] = [{"segment_id": sid, "score_ce": sc} for (sid, _t, sc) in rescored]
                nodes = self._reinject_scores(nodes, rescored)

                # normalize CE -> [0,1] + percentile/absolute keep
                sig = type(self)._sigmoid
                pct = type(self)._percentile_cut
                ce_norm = [sig(float(n.score or 0.0)) for n in nodes]

                early = self._maybe_early_abort_post_ce(ce_norm)
                if early:
                    return early

                pcut = pct(ce_norm, CFG.ce_keep_percentile)
                kept = [n for n, s in zip(nodes, ce_norm) if (s >= pcut) or (s >= CFG.ce_abs_min)]
                if not kept:
                    kept = nodes[: CFG.ce_min_keep]
                elif len(kept) < CFG.ce_min_keep:
                    extra = [n for n in nodes if n not in kept][: (CFG.ce_min_keep - len(kept))]
                    kept.extend(extra)
                nodes = kept

                trace["ce_keep_policy"] = {
                    "percentile": CFG.ce_keep_percentile,
                    "abs_min": CFG.ce_abs_min,
                    "min_keep": CFG.ce_min_keep,
                    "pcut": pcut,
                    "kept_after_ce": len(nodes),
                }
            else:
                trace["ce_skipped"] = True

            # 3) Clip width before stitching
            nodes = nodes[: CFG.topk_post_rerank]

            # 4) Postprocess
            nodes = self._entity_canonicalizer._postprocess_nodes(nodes)
            nodes = self._speaker_propagator._postprocess_nodes(nodes)  # propagate speaker metadata

            # 5) Stitch adjacent clips
            nodes = self._stitch_adjacent(nodes)

            # 6) Adaptive cap, then annotate speakers on the FINAL set
            nodes = nodes[: self._final_k(q)]
            nodes = self._annotate_speakers(nodes)

            revent.on_end(payload={EventPayload.NODES: nodes})
        trace["retrieve_ms"] = t_retrieve()

        # Final kept (for sources list)
        trace["final_kept"] = self._final_sources_view(nodes)

        # No results fast-path
        if not nodes:
            trace["final_text"] = "No results found."
            trace["total_ms"] = t_total()
            log.debug(pretty(trace))
            log.info("qe[done] query='%s' -> no results (%.2f ms)", q, trace["total_ms"])
            return Response("No results found.", source_nodes=[])

        # 7) Synthesize + clean (includes final entity normalization)
        t_synth = time_block()
        resp = self._synthesize_clean(query_bundle, nodes)
        trace["synthesize_ms"] = t_synth()
        trace["final_text"] = str(resp)

        # 8) Total timing + emit logs
        trace["total_ms"] = t_total()
        log.debug(pretty(trace))
        log.info(
            "qe[summary] q='%s' kept=%d synth=%.2f ms total=%.2f ms",
            q, len(nodes), trace["synthesize_ms"], trace["total_ms"],
        )
        return resp

    # -------- async path --------
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        q = query_bundle.query_str
        trace: Dict[str, Any] = {"query": q}

        nodes: List[NodeWithScore] = await asyncio.to_thread(self._retriever.retrieve, query_bundle)

        early = self._maybe_early_abort_stage1(q, nodes)
        if early:
            return early

        if hasattr(self._retriever, "debug_snapshot"):
            trace.update(self._retriever.debug_snapshot())

        if self._ce and self._ce.enabled and nodes:
            packs = self._ce_pack(nodes)
            trace["pre_ce"] = [{"segment_id": sid, "score_stage1": sc} for (sid, _t, sc) in packs]
            rescored = await asyncio.to_thread(self._ce.rerank, q, packs)

            ce_norm = [self._sigmoid(float(sc)) for (_sid, _t, sc) in rescored]
            early = self._maybe_early_abort_post_ce(ce_norm)
            if early:
                return early

            trace["ce_scores"] = [{"segment_id": sid, "score_ce": sc} for (sid, _t, sc) in rescored]
            nodes = self._reinject_scores(nodes, rescored)

            ce_norm = [self._sigmoid(float(sc)) for (_sid, _t, sc) in rescored]
            early = self._maybe_early_abort_post_ce(ce_norm)
            if early:
                return early

        nodes = nodes[: CFG.topk_post_rerank]

        # Postprocess, stitch, cap, annotate (same order as sync)
        nodes = self._entity_canonicalizer._postprocess_nodes(nodes)
        nodes = self._speaker_propagator._postprocess_nodes(nodes)
        nodes = self._stitch_adjacent(nodes)
        nodes = nodes[: self._final_k(q)]
        nodes = self._annotate_speakers(nodes)

        trace["final_kept"] = self._final_sources_view(nodes)

        if not nodes:
            trace["final_text"] = "No results found."
            log.debug(pretty(trace))
            log.info("qe[done] query='%s' -> no results", q)
            return Response("No results found.", source_nodes=[])

        synth = getattr(self._core, "_response_synthesizer", None)
        asynth = getattr(synth, "asynthesize", None)
        if callable(asynth):
            raw = await asynth(query=query_bundle, nodes=nodes)
        else:
            raw = await asyncio.to_thread(synth.synthesize, query=query_bundle, nodes=nodes)

        try:
            cleaned = clean_model_refs(str(raw))
            cleaned = normalize_text_entities(cleaned)
            if hasattr(raw, "response"):
                raw.response = cleaned
                resp = raw
            else:
                resp = Response(cleaned, source_nodes=getattr(raw, "source_nodes", nodes))
        except Exception:
            resp = raw

        trace["final_text"] = str(resp)
        log.debug(pretty(trace))
        log.info("qe[summary] q='%s' kept=%d (async)", q, len(nodes))
        return resp