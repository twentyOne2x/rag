"""Vector index helpers for rag_v2 (Qdrant + Pinecone)."""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    from llama_index.vector_stores.pinecone import PineconeVectorStore
except Exception:  # pragma: no cover - optional dependency for pinecone mode
    PineconeVectorStore = None  # type: ignore

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:  # pragma: no cover - optional dependency for pinecone mode
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore

try:
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except Exception:  # pragma: no cover - optional dependency for qdrant mode
    QdrantVectorStore = None  # type: ignore
    QdrantClient = None  # type: ignore
    qm = None  # type: ignore

# Load local dotenv when not running on Cloud Run ---------------------------------

if not os.environ.get("K_SERVICE"):
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        for extra in (".env", ".env.local"):
            path = Path.cwd() / extra
            if path.exists():
                load_dotenv(path, override=False)
    except Exception:
        # dotenv is optional; ignore if unavailable
        pass

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


def _backend() -> str:
    return (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()


def _pinecone_client() -> Any:
    if Pinecone is None:
        raise RuntimeError("Pinecone dependencies are not installed")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY env var is required")
    return Pinecone(api_key=api_key)


def _index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)


def _namespace() -> str:
    return os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)


def _qdrant_collection_name(namespace: str) -> str:
    index_name = _index_name()
    template = os.getenv("QDRANT_COLLECTION_TEMPLATE", "{index}__{namespace}")
    return template.format(index=index_name, namespace=namespace)


def _qdrant_client() -> Any:
    if QdrantClient is None:
        raise RuntimeError("Qdrant dependencies are not installed")
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


def _ensure_qdrant_collection(client: Any, collection_name: str, dimension: int) -> None:
    exists = False
    try:
        exists = bool(client.collection_exists(collection_name=collection_name))
    except Exception:
        names = {item.name for item in client.get_collections().collections}
        exists = collection_name in names

    if exists:
        return

    if qm is None:
        raise RuntimeError("qdrant-client models are unavailable")

    logging.info("Creating Qdrant collection %s (dim=%d)", collection_name, dimension)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
    )


@timeit
def ensure_index(
    dimension: int = DEFAULT_DIMENSION,
    metric: str = DEFAULT_METRIC,
    *,
    delete_existing: bool = False,
) -> Any:
    """Create (or re-create) the configured vector index backing rag_v2."""

    backend = _backend()

    if backend == "qdrant":
        client = _qdrant_client()
        collection_name = _qdrant_collection_name(_namespace())
        if delete_existing:
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception:
                pass
        _ensure_qdrant_collection(client, collection_name, dimension)
        if QdrantVectorStore is None:
            raise RuntimeError("Qdrant vector store adapter is not installed")
        return QdrantVectorStore(client=client, collection_name=collection_name)

    client = _pinecone_client()
    index_name = _index_name()

    if delete_existing:
        logging.warning("Deleting existing Pinecone index %s", index_name)
        client.delete_index(index_name)

    existing = {item["name"] for item in client.list_indexes().get("indexes", [])}
    if index_name not in existing:
        logging.info("Creating Pinecone index %s (dim=%d metric=%s)", index_name, dimension, metric)
        if ServerlessSpec is None:
            raise RuntimeError("Pinecone ServerlessSpec is unavailable")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

    if PineconeVectorStore is None:
        raise RuntimeError("Pinecone vector store adapter is not installed")

    vector_store = PineconeVectorStore(pinecone_index=client.Index(index_name), namespace=_namespace())
    return vector_store


def _force_namespace(store: Any, namespace: str) -> None:
    if hasattr(store, "set_namespace"):
        store.set_namespace(namespace)
    elif hasattr(store, "namespace"):
        setattr(store, "namespace", namespace)
    elif hasattr(store, "_namespace"):
        setattr(store, "_namespace", namespace)


@timeit
def load_vector_store(*, namespace: Optional[str] = None) -> Any:
    """Attach to the configured index and return a vector store."""

    backend = _backend()
    ns = namespace or _namespace()

    if backend == "qdrant":
        client = _qdrant_client()
        collection_name = _qdrant_collection_name(ns)
        _ensure_qdrant_collection(client, collection_name, DEFAULT_DIMENSION)
        if QdrantVectorStore is None:
            raise RuntimeError("Qdrant vector store adapter is not installed")
        return QdrantVectorStore(client=client, collection_name=collection_name)

    client = _pinecone_client()
    index_name = _index_name()
    if PineconeVectorStore is None:
        raise RuntimeError("Pinecone vector store adapter is not installed")
    store = PineconeVectorStore(pinecone_index=client.Index(index_name), namespace=ns)
    _force_namespace(store, ns)
    return store


@timeit
def load_index(namespace: Optional[str] = None) -> VectorStoreIndex:
    """Load the configured VectorStoreIndex for inference."""

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
    """Replace vector index contents with ``nodes`` and return it."""

    store = ensure_index(dimension=dimension, metric=metric, delete_existing=True)
    if namespace and _backend() == "pinecone":
        _force_namespace(store, namespace)
    store.add(nodes)

    return VectorStoreIndex.from_vector_store(store)


__all__ = [
    "ensure_index",
    "load_vector_store",
    "load_index",
    "rebuild_index",
]
