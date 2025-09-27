from __future__ import annotations
import asyncio
from typing import List, Tuple

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import CFG
from .rerankers.cross_encoder import CEReranker


class ParentChildQueryEngineV2(BaseQueryEngine):
    """
    Query engine that:
      1) Uses a parent/child retriever,
      2) Optionally reranks with a cross-encoder,
      3) Synthesizes with LlamaIndex's core response synthesizer.

    Implements both sync _query() and async _aquery() to satisfy BaseQueryEngine.
    """

    def __init__(self, retriever, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self._retriever = retriever
        self._core = RetrieverQueryEngine.from_args(retriever=self._retriever)
        self._ce = CEReranker(model_name=CFG.ce_model, batch_size=CFG.ce_batch_size) if CFG.enable_ce else None

    # ---- PromptMixin hook (required in newer LlamaIndex versions) ----
    def _get_prompt_modules(self):
        # No custom prompts/modules; return empty dict to satisfy PromptMixin.
        return {}

    # ---- helpers ----
    def _ce_pack(self, nodes: List[NodeWithScore]) -> List[Tuple[str, str, float]]:
        items = []
        for n in nodes:
            md = n.node.metadata or {}
            sid = md.get("segment_id") or md.get("id") or n.node.node_id
            items.append((sid, n.node.get_content(), float(n.score or 0.0)))
        return items

    def _reinject_scores(self, nodes: List[NodeWithScore], rescored: List[Tuple[str, str, float]]):
        score_by_sid = {sid: sc for sid, _, sc in rescored}
        for n in nodes:
            md = n.node.metadata or {}
            sid = md.get("segment_id") or md.get("id") or n.node.node_id
            if sid in score_by_sid:
                n.score = score_by_sid[sid]
        nodes.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return nodes

    # ---- sync path ----
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as revent:
            nodes: List[NodeWithScore] = self._retriever.retrieve(query_bundle)

            # CE rerank (optional)
            if self._ce and self._ce.enabled and nodes:
                packs = self._ce_pack(nodes)
                rescored = self._ce.rerank(query_bundle.query_str, packs)
                nodes = self._reinject_scores(nodes, rescored)

            # keep top-K post-rerank
            nodes = nodes[:CFG.topk_post_rerank]
            revent.on_end(payload={EventPayload.NODES: nodes})

        if not nodes:
            return Response("No results found.", source_nodes=[])

        # synthesize using core synthesizer (same as your v1 custom QE)
        return self._core._response_synthesizer.synthesize(query=query_bundle, nodes=nodes)

    # ---- async path ----
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # Run blocking bits in a thread to keep this truly async-safe.
        # Retrieval (likely sync):
        nodes: List[NodeWithScore] = await asyncio.to_thread(self._retriever.retrieve, query_bundle)

        # Optional CE rerank (sync API):
        if self._ce and self._ce.enabled and nodes:
            packs = self._ce_pack(nodes)
            rescored = await asyncio.to_thread(self._ce.rerank, query_bundle.query_str, packs)
            nodes = self._reinject_scores(nodes, rescored)

        nodes = nodes[:CFG.topk_post_rerank]

        if not nodes:
            return Response("No results found.", source_nodes=[])

        # Use the async synthesizer if available
        synth = getattr(self._core, "_response_synthesizer", None)
        asynth = getattr(synth, "asynthesize", None)
        if callable(asynth):
            return await asynth(query=query_bundle, nodes=nodes)

        # Fallback to sync synth if async isn't available in this version
        return await asyncio.to_thread(synth.synthesize, query=query_bundle, nodes=nodes)
