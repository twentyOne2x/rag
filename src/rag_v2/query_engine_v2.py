# File: src/rag_v2/query_engine_v2.py
from __future__ import annotations
import asyncio
from typing import List, Tuple, Any, Dict

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import CFG
from .rerankers.cross_encoder import CEReranker
from .logging_utils import (
    setup_logger,
    pretty,
    node_brief,
    clean_model_refs,
    time_block,
    cfg_snapshot,
    model_snapshot,
)

log = setup_logger("rag_v2.qe")


class ParentChildQueryEngineV2(BaseQueryEngine):
    """
    Query engine that:
      1) Uses a parent/child retriever,
      2) Optionally reranks with a cross-encoder,
      3) Synthesizes with LlamaIndex's core response synthesizer.

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

    def _synthesize_clean(self, query_bundle: QueryBundle, nodes: List[NodeWithScore]) -> Response:
        raw = self._core._response_synthesizer.synthesize(query=query_bundle, nodes=nodes)
        try:
            cleaned_text = clean_model_refs(str(raw))
            if hasattr(raw, "response"):
                raw.response = cleaned_text
                return raw
            return Response(cleaned_text, source_nodes=getattr(raw, "source_nodes", nodes))
        except Exception:
            return raw

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
        with self.callback_manager.event(
            CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: q}
        ) as revent:
            nodes: List[NodeWithScore] = self._retriever.retrieve(query_bundle)

            if hasattr(self._retriever, "debug_snapshot"):
                trace.update(self._retriever.debug_snapshot())

            # 2) Optional CE rerank
            if self._ce and self._ce.enabled and nodes:
                packs = self._ce_pack(nodes)  # [(segment_id, text, stage1_score)]
                trace["pre_ce"] = [{"segment_id": sid, "score_stage1": sc} for (sid, _t, sc) in packs]
                t_ce = time_block()
                rescored = self._ce.rerank(q, packs)  # [(sid, text, ce_score)]
                trace["ce_ms"] = t_ce()
                trace["ce_scores"] = [{"segment_id": sid, "score_ce": sc} for (sid, _t, sc) in rescored]
                nodes = self._reinject_scores(nodes, rescored)
            else:
                trace["ce_skipped"] = True

            # 3) Keep top-K post-rerank
            nodes = nodes[: CFG.topk_post_rerank]
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

        # 4) Synthesize + clean
        t_synth = time_block()
        resp = self._synthesize_clean(query_bundle, nodes)
        trace["synthesize_ms"] = t_synth()
        trace["final_text"] = str(resp)

        # 5) Total timing + emit logs
        trace["total_ms"] = t_total()
        log.debug(pretty(trace))

        # Concise human log like v1:
        log.info(
            "qe[summary] q='%s' kept=%d synth=%.2f ms total=%.2f ms",
            q,
            len(nodes),
            trace["synthesize_ms"],
            trace["total_ms"],
        )
        return resp

    # -------- async path --------
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        q = query_bundle.query_str
        trace: Dict[str, Any] = {"query": q}

        nodes: List[NodeWithScore] = await asyncio.to_thread(self._retriever.retrieve, query_bundle)
        if hasattr(self._retriever, "debug_snapshot"):
            trace.update(self._retriever.debug_snapshot())

        if self._ce and self._ce.enabled and nodes:
            packs = self._ce_pack(nodes)
            trace["pre_ce"] = [{"segment_id": sid, "score_stage1": sc} for (sid, _t, sc) in packs]
            rescored = await asyncio.to_thread(self._ce.rerank, q, packs)
            trace["ce_scores"] = [{"segment_id": sid, "score_ce": sc} for (sid, _t, sc) in rescored]
            nodes = self._reinject_scores(nodes, rescored)
        else:
            trace["ce_skipped"] = True

        nodes = nodes[: CFG.topk_post_rerank]
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
