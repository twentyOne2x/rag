from __future__ import annotations

import pytest

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import MetadataFilters

from rag_v2.retriever.parent_child_retriever import ParentChildRetrieverV2
from rag_v2.tenancy import TenantAuthorizationBackendError


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


def test_authorized_channel_filter_never_falls_back_to_unfiltered_retrieval() -> None:
    class HostileBaseRetriever:
        def __init__(self) -> None:
            self.bare_calls = 0

        def retrieve(self, *_args, **kwargs):
            if kwargs:
                raise TypeError("filtered retrieval unsupported")
            self.bare_calls += 1
            return [_build_node("secret from an unauthorized channel")]

    base = HostileBaseRetriever()
    retriever = ParentChildRetrieverV2(base)

    with pytest.raises(TenantAuthorizationBackendError):
        retriever._base_retrieve_with_filter(
            QueryBundle("private channel"),
            {"channel_id": {"$in": ["allowed-channel"]}},
        )

    assert base.bare_calls == 0


def test_authorized_channel_filter_uses_request_owned_typed_retriever() -> None:
    class RequestOwnedRetriever:
        def retrieve(self, _query_bundle):
            return [_build_node("allowed channel")]

    captured: list[MetadataFilters] = []

    def factory(filters: MetadataFilters):
        captured.append(filters)
        return RequestOwnedRetriever()

    retriever = ParentChildRetrieverV2(_NoopBaseRetriever(), factory)
    filters, debug = retriever._build_channel_filter(
        {
            "include_ids": ["allowed-channel"],
            "include_names": ["Allowed"],
            "exclude_ids": ["blocked-channel"],
        }
    )

    nodes = retriever._base_retrieve_with_filter(QueryBundle("allowed"), filters)

    assert len(nodes) == 1
    assert captured == [filters]
    assert debug["applied"] is True
    assert debug["expr"]["condition"] == "and"
