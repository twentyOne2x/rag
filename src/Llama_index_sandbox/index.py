# src/Llama_index_sandbox/index.py

import logging
import os
from typing import Optional

# Core LlamaIndex (no legacy ServiceContext in normal flow)
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

# Pinecone client
from pinecone import Pinecone, ServerlessSpec

# Project imports
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_vector_store_index import (
    CustomVectorStoreIndex,
)
from src.Llama_index_sandbox.utils.utils import (
    timeit,
    load_vector_store_from_pinecone_database,
    load_vector_store_from_pinecone_database_legacy,
)

# Optional: legacy ServiceContext (only used when legacy retriever paths are hit)
try:
    from llama_index.legacy import ServiceContext  # type: ignore
except Exception:
    ServiceContext = None  # type: ignore


def _make_service_context(llm: Optional[object] = None):
    """
    Create a ServiceContext that enforces a 3072-dim OpenAI embedder.
    This is critical for any legacy retriever/query paths which *ignore*
    Settings.embed_model and otherwise default to 1536-dim.

    Safe to pass into core constructors as well (core will ignore it where irrelevant).
    """
    if ServiceContext is None:
        return None
    embed = OpenAIEmbedding(model="text-embedding-3-large", dimensions=3072)
    return ServiceContext.from_defaults(embed_model=embed, llm=llm)


@timeit
def initialise_vector_store(
    embedding_model_vector_dimension: int,
    vector_space_distance_metric: str = "cosine",
) -> PineconeVectorStore:
    """
    Create (or recreate) a Pinecone index and wrap it in a core PineconeVectorStore.
    """
    index_name = os.environ.get("PINECONE_INDEX_NAME", "icmfyi-v2")
    api_key = os.environ["PINECONE_API_KEY"]

    pc = Pinecone(api_key=api_key)

    # Delete if exists, then (re)create
    existing = {idx["name"] for idx in pc.list_indexes().get("indexes", [])}
    if index_name in existing:
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=embedding_model_vector_dimension,
        metric=vector_space_distance_metric,
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


@timeit
def load_nodes_into_vector_store_create_index(
    nodes,
    embedding_model_vector_dimension: int,
    vector_space_distance_metric: str,
) -> VectorStoreIndex:
    """
    Add nodes to Pinecone and return a core VectorStoreIndex for fast querying.
    We also provide a ServiceContext with a 3072-dim embedder so downstream
    query-time components are consistent if legacy code paths are used.
    """
    vector_store = initialise_vector_store(
        embedding_model_vector_dimension=embedding_model_vector_dimension,
        vector_space_distance_metric=vector_space_distance_metric,
    )

    # Add the data
    vector_store.add(nodes)

    # Build a matching service context (LLM taken from Settings if present)
    try:
        from llama_index.core import Settings
        llm = getattr(Settings, "llm", None)
    except Exception:
        llm = None
    service_context = _make_service_context(llm=llm)

    # Construct index; passing service_context is harmless on core, required on legacy
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,  # ensures 3072-dim embedder on legacy paths
    )
    return index


# src/Llama_index_sandbox/index.py
import logging, os
from llama_index.vector_stores.pinecone import PineconeVectorStore

def _detect_ns_attr(vs):
    if hasattr(vs, "_namespace"):
        return "_namespace"
    if hasattr(vs, "namespace"):
        return "namespace"
    return None

def _get_ns(vs):
    attr = _detect_ns_attr(vs)
    return getattr(vs, attr) if attr else None

def _get_index_name(vs):
    # Try multiple shapes across LI versions
    for attr in ("_index", "index", "_pinecone_index", "pinecone_index"):
        idx = getattr(vs, attr, None)
        if idx is None:
            continue
        for name_attr in ("_name", "name"):
            val = getattr(idx, name_attr, None)
            if val:
                return val
    return None

def _force_pinecone_namespace(vs, ns: str) -> None:
    """Force namespace on a PineconeVectorStore across LI versions + log it."""
    try:
        idx_name = _get_index_name(vs)
        ns_attr  = _detect_ns_attr(vs)
        ns_before = _get_ns(vs)

        logging.info(
            "[pinecone] reader pre: index=%r ns_attr=%r ns_before=%r env{INDEX=%r, NAMESPACE=%r}",
            idx_name, ns_attr, ns_before,
            os.getenv("PINECONE_INDEX_NAME"), os.getenv("PINECONE_NAMESPACE"),
        )

        if ns_attr:
            setattr(vs, ns_attr, ns)
            ns_after = _get_ns(vs)
            logging.info(
                "[pinecone] reader post: index=%r ns_attr=%r ns_after=%r (target=%r)",
                idx_name, ns_attr, ns_after, ns,
            )
        else:
            logging.warning("[pinecone] reader: no namespace attribute found on store; cannot set %r", ns)
    except Exception as e:
        logging.warning("[pinecone] could not enforce namespace %r: %s", ns, e)


@timeit
def load_index_from_disk() -> CustomVectorStoreIndex:
    """
    Load an existing Pinecone vector store and wrap it as a CustomVectorStoreIndex.
    """
    # Build a service context once, using the globally-configured LLM if present
    try:
        from llama_index.core import Settings
        llm = getattr(Settings, "llm", None)
    except Exception:
        llm = None
    service_context = _make_service_context(llm=llm)

    # Default namespace: 'videos' (overridable via env)
    target_ns = os.getenv("PINECONE_NAMESPACE", "videos")
    target_idx = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    logging.info("[pinecone] loader env: INDEX=%r NAMESPACE=%r (default ns='videos')", target_idx, target_ns)

    # Primary (core) loader
    try:
        vector_store = load_vector_store_from_pinecone_database()
        logging.info("[pinecone] loader path=core store_cls=%s", type(vector_store).__name__)
        if isinstance(vector_store, PineconeVectorStore):
            _force_pinecone_namespace(vector_store, target_ns)

        index = CustomVectorStoreIndex.from_vector_store(
            vector_store,
            service_context=service_context,
        )
        logging.info("Successfully loaded index from Pinecone (core store).")
        return index

    except Exception as e_core:
        logging.error(f"Core vector store load failed: {e_core}")

        # Fallback to legacy loader if needed (kept for migration period)
        try:
            vector_store_legacy = load_vector_store_from_pinecone_database_legacy()
            logging.info("[pinecone] loader path=legacy store_cls=%s", type(vector_store_legacy).__name__)
            if isinstance(vector_store_legacy, PineconeVectorStore):
                _force_pinecone_namespace(vector_store_legacy, target_ns)

            index = CustomVectorStoreIndex.from_vector_store(
                vector_store_legacy,
                service_context=service_context,
            )
            logging.info("Successfully loaded index from Pinecone (legacy store).")
            return index

        except Exception as e_legacy:
            logging.error(f"Legacy vector store load failed: {e_legacy}")

            # Last-ditch: construct a plain VectorStoreIndex if Custom fails
            try:
                vector_store = load_vector_store_from_pinecone_database()
                logging.info("[pinecone] loader path=plain store_cls=%s", type(vector_store).__name__)
                if isinstance(vector_store, PineconeVectorStore):
                    _force_pinecone_namespace(vector_store, target_ns)

                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    service_context=service_context,
                )
                logging.info("Loaded plain VectorStoreIndex from Pinecone (core store).")
                return index
            except Exception as e_plain:
                logging.error(f"load_index_from_disk ERROR: {e_plain}")
                raise
