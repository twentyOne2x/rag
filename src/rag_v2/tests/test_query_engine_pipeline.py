from __future__ import annotations

from typing import Any, List, Tuple

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_v2.query_engine_v2 import ParentChildQueryEngineV2
from rag_v2.rerankers.cross_encoder import CEReranker


class _HappyPathRetriever:
    """Retriever stub that always returns a few well-formed nodes."""

    def __init__(self) -> None:
        self.base = self
        self.similarity_top_k = 3
        self._last_nodes: List[NodeWithScore] = []

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes: List[NodeWithScore] = []
        for idx in range(3):
            meta = {
                "segment_id": f"seg-{idx}",
                "start_hms": f"00:00:{idx:02d}",
                "end_hms": f"00:00:{idx+1:02d}",
                "parent_id": f"parent-{idx}",
                "document_type": "youtube_video",
                "title": f"title-{idx}",
                "channel_name": "@stub",
                "entities": ["Cupsey"],
            }
            node = TextNode(text=f"clip-{idx}", id_=f"node-{idx}", metadata=meta)
            nodes.append(NodeWithScore(node=node, score=0.9 - idx * 0.1))
        self._last_nodes = nodes
        return nodes

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "entity_gate": {
                "required": [],
                "applied": False,
                "kept": len(self._last_nodes),
                "dropped": 0,
            }
        }


class _StubCrossEncoder:
    enabled = True
    batch_size = 8
    model_name = "stub/ce"

    def rerank_with_meta(
        self,
        query: str,
        packs: List[Tuple[str, str, float]],
        metas: dict[str, dict[str, Any]],
    ) -> List[Tuple[str, str, float]]:
        # Return the same ordering but tweak the scores to mimic CE work.
        return [(sid, text, score + 0.05) for sid, text, score in packs]


def test_query_pipeline_keeps_final_nodes_with_cross_encoder() -> None:
    engine = ParentChildQueryEngineV2(_HappyPathRetriever())
    engine._ce = _StubCrossEncoder()  # type: ignore[attr-defined]

    resp = engine.query(QueryBundle("who is cupsey?"))

    trace = engine.get_last_trace()
    assert trace.get("final_kept"), "expected final_kept to contain sources"

    summary = engine.get_last_progress_summary() or {}
    assert (summary.get("metadata") or {}).get("final_node_count", 0) > 0

    assert "No results found" not in str(resp)


def test_query_pipeline_falls_back_when_ce_fails(monkeypatch) -> None:
    engine = ParentChildQueryEngineV2(_HappyPathRetriever())
    ce = CEReranker(model_name="stub/failure")
    engine._ce = ce  # type: ignore[attr-defined]

    def boom(*args, **kwargs):
        raise RuntimeError("predict exploded")

    monkeypatch.setattr(ce, "_ensure_model", lambda: True)
    ce.model = type("M", (), {"predict": boom})()  # type: ignore

    resp = engine.query(QueryBundle("who is cupsey?"))

    trace = engine.get_last_trace()
    assert trace.get("final_kept")
    summary = engine.get_last_progress_summary() or {}
    assert (summary.get("metadata") or {}).get("final_node_count", 0) > 0
    assert "No results found" not in str(resp)


class _AcronymRetriever:
    """Retriever stub with conflicting acronym definitions in transcript text."""

    def __init__(self) -> None:
        self.base = self
        self.similarity_top_k = 2
        self._last_nodes: List[NodeWithScore] = []

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        rows = [
            (
                "seg-simd-cpu",
                (
                    "Data parallelism, or Single Instruction Multiple Data or SIMD for short, "
                    "is used in modern CPUs."
                ),
                "Breakpoint HPC",
                "00:05:10",
                "00:05:20",
                0.7,
            ),
            (
                "seg-simd-solana",
                (
                    "This SIMD proposal is in the Solana Improvement Documents repository. "
                    "Check the SIMD issue details there."
                ),
                "Solana Changelog",
                "00:01:00",
                "00:01:10",
                0.68,
            ),
        ]
        nodes: List[NodeWithScore] = []
        for sid, text, title, start_hms, end_hms, score in rows:
            meta = {
                "segment_id": sid,
                "start_hms": start_hms,
                "end_hms": end_hms,
                "parent_id": sid,
                "document_type": "youtube_video",
                "title": title,
                "channel_name": "@stub",
                "url": "https://example.com/watch?v=abc123xyz00",
            }
            node = TextNode(text=text, id_=sid, metadata=meta)
            nodes.append(NodeWithScore(node=node, score=score))
        self._last_nodes = nodes
        return nodes

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "entity_gate": {
                "required": ["simd"],
                "applied": True,
                "kept": len(self._last_nodes),
                "dropped": 0,
            }
        }


def test_query_pipeline_disambiguates_ambiguous_acronym(monkeypatch) -> None:
    engine = ParentChildQueryEngineV2(_AcronymRetriever())
    engine._ce = None  # type: ignore[attr-defined]

    synth = engine._core._response_synthesizer

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("LLM synthesis should be skipped for ambiguous acronym disambiguation")

    monkeypatch.setattr(synth, "synthesize", should_not_run)

    resp = engine.query(QueryBundle("What does SIMD mean?"))
    text = str(resp)
    assert "ambiguous" in text.lower()
    assert "single instruction" in text.lower()
    assert "solana improvement documents" in text.lower()
