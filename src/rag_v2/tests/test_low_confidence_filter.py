from __future__ import annotations

from typing import Any, List

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_v2.query_engine_v2 import ParentChildQueryEngineV2


class _StubRetriever:
    """Minimal retriever stub; only pieces touched in tests are defined."""

    def __init__(self) -> None:
        self.base = self
        self.similarity_top_k = None

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return []


def _build_node(score: float = 0.0) -> NodeWithScore:
    node = TextNode(text="stub", id_="node-1")
    return NodeWithScore(node=node, score=score)


def test_low_confidence_filter_drops_nodes_below_threshold() -> None:
    engine = ParentChildQueryEngineV2(_StubRetriever())
    nodes = [_build_node(score=0.0)]
    trace: dict[str, Any] = {}

    kept = engine._apply_low_confidence_filter(nodes, threshold=0.5, trace=trace)

    assert kept == []
    assert "low_confidence" in trace
    dropped = trace["low_confidence"].get("dropped") or []
    assert len(dropped) == 1
    # Confidence is attached to metadata for downstream inspection
    assert nodes[0].node.metadata["confidence_norm"] == 0.0


def test_low_confidence_filter_no_threshold_keeps_nodes() -> None:
    engine = ParentChildQueryEngineV2(_StubRetriever())
    nodes = [_build_node(score=0.42)]
    trace: dict[str, Any] = {}

    kept = engine._apply_low_confidence_filter(nodes, threshold=0.0, trace=trace)

    assert kept == nodes
    assert "low_confidence" not in trace
