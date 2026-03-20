from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from rag_v2.retriever.parent_child_retriever import ParentChildRetrieverV2


class _NoopBaseRetriever:
    def retrieve(self, *_args, **_kwargs):
        return []


def _build_node(text: str, entities: list[str] | None = None) -> NodeWithScore:
    meta = {
        "segment_id": "seg-1",
        "document_type": "youtube_video",
        "entities": entities or [],
        "canonical_entities": [],
    }
    node = TextNode(text=text, id_="node-1", metadata=meta)
    return NodeWithScore(node=node, score=0.42)


def test_entity_gate_keeps_node_via_text_match_when_metadata_misses() -> None:
    retriever = ParentChildRetrieverV2(_NoopBaseRetriever())
    nodes = [_build_node("This week we discuss SIMD proposals in Solana governance.")]

    kept, debug = retriever._entity_gate_nodes(nodes, {"simd"})

    assert len(kept) == 1
    assert debug.get("kept_via_text") == 1
    assert debug.get("dropped") == 0


def test_query_entities_ignores_enriched_prompt_tail() -> None:
    retriever = ParentChildRetrieverV2(_NoopBaseRetriever())
    query = (
        "What does SIMD mean?\n\n"
        "Answer thoroughly and use URL?t=START_SECONDSs with citations."
    )

    _, _, canonical = retriever._query_entities(query)

    assert "simd" in canonical
    assert "start_seconds" not in canonical
