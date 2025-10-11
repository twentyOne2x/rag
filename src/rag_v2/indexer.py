"""Pinecone index helpers for rag_v2.

This module centralises all Pinecone index lifecycle logic so we no longer
depend on the old `Llama_index_sandbox` package.  It supports both loading an
existing index for inference and (optionally) rebuilding an index from a set of
nodes when running ingestion jobs.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Iterable, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Default configuration -----------------------------------------------------

DEFAULT_INDEX_NAME = "icmfyi-v2"
DEFAULT_NAMESPACE = "videos"
DEFAULT_DIMENSION = 3072
DEFAULT_METRIC = "cosine"


def timeit(func):
    """Log how long a function call took (mirrors the old decorator)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.time() - start
            logging.info("%s completed in %.2fs", func.__qualname__, elapsed)

    return wrapper


def _pinecone_client() -> Pinecone:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY env var is required")
    return Pinecone(api_key=api_key)


def _index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)


def _namespace() -> str:
    return os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)


def _service_context():
    """Ensure legacy code paths use a 3072-dim embedder."""

    try:
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        return Settings.llm and Settings
    except Exception:
        return None


@timeit
def ensure_index(
    dimension: int = DEFAULT_DIMENSION,
    metric: str = DEFAULT_METRIC,
    *,
    delete_existing: bool = False,
) -> PineconeVectorStore:
    """Create (or re-create) the Pinecone index backing rag_v2."""

    client = _pinecone_client()
    index_name = _index_name()

    if delete_existing:
        logging.warning("Deleting existing Pinecone index %s", index_name)
        client.delete_index(index_name)

    existing = {item["name"] for item in client.list_indexes().get("indexes", [])}
    if index_name not in existing:
        logging.info("Creating Pinecone index %s (dim=%d metric=%s)", index_name, dimension, metric)
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

    vector_store = PineconeVectorStore(pinecone_index=client.Index(index_name), namespace=_namespace())
    return vector_store


def _force_namespace(store: PineconeVectorStore, namespace: str) -> None:
    if hasattr(store, "set_namespace"):
        store.set_namespace(namespace)
    elif hasattr(store, "namespace"):
        setattr(store, "namespace", namespace)
    elif hasattr(store, "_namespace"):
        setattr(store, "_namespace", namespace)


@timeit
def load_vector_store(*, namespace: Optional[str] = None) -> PineconeVectorStore:
    """Attach to the configured Pinecone index and return a vector store."""

    client = _pinecone_client()
    index_name = _index_name()
    store = PineconeVectorStore(pinecone_index=client.Index(index_name), namespace=namespace or _namespace())
    _force_namespace(store, namespace or _namespace())
    return store


@timeit
def load_index(namespace: Optional[str] = None) -> VectorStoreIndex:
    """Load the Pinecone-backed VectorStoreIndex for inference."""

    store = load_vector_store(namespace=namespace)

    try:
        llm = getattr(Settings, "llm", None)
    except Exception:
        llm = None

    service_context = None
    if llm is not None:
        try:
            embed_model = OpenAIEmbedding(model="text-embedding-3-large")
            service_context = Settings.from_defaults(llm=llm, embed_model=embed_model)
        except Exception:
            service_context = None

    return VectorStoreIndex.from_vector_store(store, service_context=service_context)


@timeit
def rebuild_index(
    nodes: Iterable,
    *,
    dimension: int = DEFAULT_DIMENSION,
    metric: str = DEFAULT_METRIC,
    namespace: Optional[str] = None,
) -> VectorStoreIndex:
    """Replace the Pinecone index contents with ``nodes`` and return it."""

    store = ensure_index(dimension=dimension, metric=metric, delete_existing=True)
    if namespace:
        _force_namespace(store, namespace)
    store.add(nodes)

    return VectorStoreIndex.from_vector_store(store)


__all__ = [
    "ensure_index",
    "load_vector_store",
    "load_index",
    "rebuild_index",
]
