from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import QueryBundle, TextNode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.rag_v2.retriever.parent_child_retriever import ParentChildRetrieverV2

qdrant_adapter = __import__(
    "llama_index.vector_stores.qdrant", fromlist=["QdrantVectorStore"]
)
QdrantVectorStore = qdrant_adapter.QdrantVectorStore


def _retriever() -> ParentChildRetrieverV2:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="tenant-filter-test",
        vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE),
    )
    store = QdrantVectorStore(client=client, collection_name="tenant-filter-test")
    nodes = []
    for index, (channel_id, channel_name, entities) in enumerate(
        (
            ("tenant-a-channel", "Tenant A", ["allowed-entity"]),
            ("tenant-b-channel", "Tenant B", ["secret-entity"]),
        )
    ):
        node = TextNode(
            text=f"content-{channel_id}",
            id_=f"00000000-0000-4000-8000-{index:012d}",
            metadata={
                "channel_id": channel_id,
                "channel_name": channel_name,
                "entities": entities,
                "canonical_entities": entities,
            },
        )
        node.embedding = [0.5, 0.5, 0.5, 0.5]
        nodes.append(node)
    store.add(nodes)

    vector_index = VectorStoreIndex.from_vector_store(
        store,
        embed_model=MockEmbedding(embed_dim=4),
    )

    def filtered_factory(filters):
        return vector_index.as_retriever(similarity_top_k=10, filters=filters)

    return ParentChildRetrieverV2(
        vector_index.as_retriever(similarity_top_k=10),
        filtered_retriever_factory=filtered_factory,
    )


def _channel_ids(nodes) -> set[str]:
    return {str(node.node.metadata["channel_id"]) for node in nodes}


def test_qdrant_request_owned_filters_isolate_concurrent_tenants() -> None:
    retriever = _retriever()
    tenant_a, _ = retriever._build_channel_filter({"include_ids": ["tenant-a-channel"]})
    tenant_b, _ = retriever._build_channel_filter({"include_ids": ["tenant-b-channel"]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            retriever._base_retrieve_with_filter,
            QueryBundle("content"),
            tenant_a,
        )
        future_b = pool.submit(
            retriever._base_retrieve_with_filter,
            QueryBundle("content"),
            tenant_b,
        )

    assert _channel_ids(future_a.result()) == {"tenant-a-channel"}
    assert _channel_ids(future_b.result()) == {"tenant-b-channel"}


def test_qdrant_entity_second_pass_keeps_tenant_filter() -> None:
    retriever = _retriever()
    tenant_a, _ = retriever._build_channel_filter({"include_ids": ["tenant-a-channel"]})

    allowed, _ = retriever._entity_filtered_retrieve(
        QueryBundle("allowed entity"),
        {"allowed-entity"},
        tenant_a,
    )
    blocked, _ = retriever._entity_filtered_retrieve(
        QueryBundle("secret entity"),
        {"secret-entity"},
        tenant_a,
    )

    assert _channel_ids(allowed) == {"tenant-a-channel"}
    assert blocked == []
