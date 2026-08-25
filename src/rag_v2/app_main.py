from __future__ import annotations

import os
import sys
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Make imports work whether run as "python src/rag_v2/app_main.py" or "python -m src.rag_v2.app_main" ---
try:
    from .retriever.parent_child_retriever import ParentChildRetrieverV2  # type: ignore
    from .query_engine_v2 import ParentChildQueryEngineV2  # type: ignore
except ImportError:
    _here = Path(__file__).resolve()
    _src = _here.parents[1]  # .../src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from rag_v2.retriever.parent_child_retriever import ParentChildRetrieverV2  # type: ignore
    from rag_v2.query_engine_v2 import ParentChildQueryEngineV2  # type: ignore

from .instrumentation import AppDiagnostics, ProgressRecorder
from .settings import config_value
from .indexer import load_index


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _configure_models() -> dict[str, object]:
    """Configure and prove the canonical LLM + embedding contract."""
    production = (
        os.getenv("ICMFYI_PRODUCTION") == "1"
        or os.getenv("ICMFYI_ENV", "").lower() == "production"
    )
    llm_provider = (os.getenv("RAG_LLM_PROVIDER", "openai") or "openai").strip().lower()
    llm_model = os.getenv(
        "RAG_LLM_MODEL",
        os.getenv(
            "INFERENCE_MODEL", config_value("models.llm_primary", default="gpt-4o-mini")
        ),
    )
    if llm_provider == "openai":
        if production and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for the configured production RAG LLM"
            )
        Settings.llm = OpenAI(model=llm_model)
    elif llm_provider == "mock" and not production:
        Settings.llm = MockLLM()
    else:
        raise RuntimeError(f"unsupported RAG_LLM_PROVIDER: {llm_provider}")

    embed_provider = (
        (os.getenv("EMBED_PROVIDER", "sentence-transformers") or "").strip().lower()
    )
    embed_model_name = os.getenv(
        "EMBED_MODEL",
        config_value("models.embedding_primary", default="Qwen/Qwen3-Embedding-0.6B"),
    )
    embed_model_revision = os.getenv("EMBED_MODEL_REVISION", "").strip()
    embed_dimension = _positive_int_env("EMBED_DIM", 1024)
    if embed_provider == "sentence-transformers":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        if production and not embed_model_revision:
            raise RuntimeError(
                "EMBED_MODEL_REVISION is required for sentence-transformers in production"
            )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device=os.getenv("RAG_EMBED_DEVICE", "cpu"),
            normalize=True,
            trust_remote_code=False,
            **({"revision": embed_model_revision} if embed_model_revision else {}),
        )
    elif embed_provider == "openai":
        if production and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for the configured production embedding provider"
            )
        Settings.embed_model = OpenAIEmbedding(
            model=embed_model_name,
            dimensions=embed_dimension,
        )
    else:
        raise RuntimeError(f"unsupported EMBED_PROVIDER: {embed_provider}")

    probe = Settings.embed_model.get_query_embedding("ICMFYI embedding dimension probe")
    if len(probe) != embed_dimension:
        raise RuntimeError(
            f"embedding dimension mismatch: model returned {len(probe)}, expected {embed_dimension}"
        )
    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embed_provider": embed_provider,
        "embed_model": embed_model_name,
        "embed_model_revision": embed_model_revision or None,
        "embed_dimension": embed_dimension,
    }


def _load_index_from_vector_store():
    """Attach to the configured vector store via rag_v2.indexer."""
    backend = (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()
    index_name = os.environ.get(
        "PINECONE_INDEX_NAME",
        config_value("pinecone.index_name", default="icmfyi-v2"),
    )
    namespace = os.environ.get(
        "PINECONE_NAMESPACE",
        config_value("pinecone.namespace", default="videos"),
    )
    os.environ.setdefault("PINECONE_INDEX_NAME", index_name)
    os.environ.setdefault("VECTOR_STORE", backend)
    index = load_index(namespace=namespace)
    os.environ.setdefault("PINECONE_NAMESPACE", namespace)
    return index


def bootstrap_query_engine_v2(
    similarity_top_k: int = 50, profiler: ProgressRecorder | None = None
):
    """
    Bootstraps the Parent/Child query engine with the configured vector index.
    Works regardless of how this file is executed.
    """
    profiler = profiler or ProgressRecorder(scope="startup")

    with profiler.step("configure_models", "Configure LLM + embeddings"):
        model_contract = _configure_models()
        profiler.metadata["model_contract"] = model_contract

    # Attach to vector index (inherits Settings.embed_model for query embeddings)
    with profiler.step("load_index", "Load vector index") as step:
        index = _load_index_from_vector_store()
        idx_name = os.getenv(
            "PINECONE_INDEX_NAME",
            config_value("pinecone.index_name", default="icmfyi-v2"),
        )
        namespace = os.getenv(
            "PINECONE_NAMESPACE",
            config_value("pinecone.namespace", default="videos"),
        )
        os.environ["PINECONE_INDEX_NAME"] = idx_name
        os.environ["PINECONE_NAMESPACE"] = namespace
        if step is not None:
            step.metadata.update(
                {
                    "similarity_top_k": similarity_top_k,
                    "vector_store": os.getenv("VECTOR_STORE", "pinecone"),
                    "index_name": idx_name,
                    "namespace": namespace,
                }
            )

    # Build base retriever, then wrap with ParentChildRetrieverV2
    with profiler.step("build_retriever", "Construct retriever stack") as step:
        base_retriever = index.as_retriever(
            similarity_top_k=similarity_top_k, verbose=False
        )

        def filtered_retriever(filters):
            # LlamaIndex binds metadata filters when constructing a retriever. Creating a
            # request-owned retriever avoids mutating the pooled engine's shared base
            # retriever and prevents concurrent tenant scopes from crossing.
            return index.as_retriever(
                similarity_top_k=similarity_top_k,
                verbose=False,
                filters=filters,
            )

        pc_retriever = ParentChildRetrieverV2(
            base_retriever,
            filtered_retriever_factory=filtered_retriever,
        )
        if step is not None:
            step.metadata["stage1_top_k"] = similarity_top_k

    # Pass through callback_manager when present
    with profiler.step("build_query_engine", "Initialize query engine") as step:
        qe = ParentChildQueryEngineV2(
            retriever=pc_retriever,
            callback_manager=getattr(index, "callback_manager", None),
        )
        if step is not None:
            step.metadata["ce_enabled"] = bool(getattr(qe, "_ce", None))

    profiler.metadata["similarity_top_k"] = similarity_top_k
    startup_profile = profiler.summary()
    qe.startup_profile = startup_profile
    AppDiagnostics.record_startup(startup_profile)
    return qe


if __name__ == "__main__":
    """
    Run from IDE directly. You can pass multiple questions as CLI args, e.g.:

        python src/rag_v2/app_main.py "what is a DAT on Solana?" \
            "return all videos about DATs and Kyle Samani"

    If no args are provided, a small default list is used.
    """
    qe = bootstrap_query_engine_v2()
    from llama_index.core.schema import QueryBundle

    questions = sys.argv[1:] or [
        "what is a DAT on Solana?",
        "return all videos about DATs and Kyle Samani",
        "show me all clips where Kyle Samani details how DATs will be deployed in DeFi",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n=== Q{i}: {q}\n")
        resp = qe.query(QueryBundle(q))
        print(resp)
