from __future__ import annotations

import os
import sys
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding  # <-- ensure 3072D

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

def _configure_models() -> None:
    """Configure the LLM + embedder used for inference."""
    llm_model = config_value("models.llm_primary", default="gpt-4o-mini")
    embed_model_name = config_value("models.embedding_primary", default="text-embedding-3-large")
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", llm_model))
    Settings.embed_model = OpenAIEmbedding(model=embed_model_name)


def _load_index_from_pinecone():
    """Attach to the existing Pinecone index via rag_v2.indexer."""
    index_name = os.environ.get(
        "PINECONE_INDEX_NAME",
        config_value("pinecone.index_name", default="icmfyi-v2"),
    )
    namespace = os.environ.get(
        "PINECONE_NAMESPACE",
        config_value("pinecone.namespace", default="videos"),
    )
    os.environ.setdefault("PINECONE_INDEX_NAME", index_name)
    index = load_index(namespace=namespace)
    os.environ.setdefault("PINECONE_NAMESPACE", namespace)
    return index


def bootstrap_query_engine_v2(similarity_top_k: int = 50, profiler: ProgressRecorder | None = None):
    """
    Bootstraps the Parent/Child query engine with your Pinecone-backed index.
    Works regardless of how this file is executed.
    """
    profiler = profiler or ProgressRecorder(scope="startup")

    with profiler.step("configure_models", "Configure LLM + embeddings"):
        _configure_models()

    # Attach to Pinecone (inherits Settings.embed_model for query embeddings)
    with profiler.step("load_index", "Load vector index from Pinecone") as step:
        index = _load_index_from_pinecone()
        idx_name = os.getenv("PINECONE_INDEX_NAME", config_value("pinecone.index_name", default="icmfyi-v2"))
        namespace = os.getenv("PINECONE_NAMESPACE", config_value("pinecone.namespace", default="videos"))
        os.environ["PINECONE_INDEX_NAME"] = idx_name
        os.environ["PINECONE_NAMESPACE"] = namespace
        step.metadata.update({
            "similarity_top_k": similarity_top_k,
            "index_name": idx_name,
            "namespace": namespace,
        })

    # Build base retriever, then wrap with ParentChildRetrieverV2
    with profiler.step("build_retriever", "Construct retriever stack") as step:
        base_retriever = index.as_retriever(similarity_top_k=similarity_top_k, verbose=False)
        pc_retriever = ParentChildRetrieverV2(base_retriever)
        step.metadata["stage1_top_k"] = similarity_top_k

    # Pass through callback_manager when present
    with profiler.step("build_query_engine", "Initialize query engine") as step:
        qe = ParentChildQueryEngineV2(
            retriever=pc_retriever,
            callback_manager=getattr(index, "callback_manager", None),
        )
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
