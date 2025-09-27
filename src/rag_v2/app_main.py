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

# Reuse your index loader (support both import styles)
try:
    from src.Llama_index_sandbox.index import load_index_from_disk  # type: ignore
except Exception:
    from Llama_index_sandbox.index import load_index_from_disk  # type: ignore


def _configure_models() -> None:
    """
    Configure both the LLM and the *embedding* model.
    IMPORTANT: Embedder must match your Pinecone index dimension (3072).
    """
    # LLM
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))

    # Embeddings (set to 3072D)
    embed_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    # NOTE: text-embedding-3-large returns 3072-dim vectors by default.
    Settings.embed_model = OpenAIEmbedding(model=embed_model_name)


def bootstrap_query_engine_v2(similarity_top_k: int = 50):
    """
    Bootstraps the Parent/Child query engine with your Pinecone-backed index.
    Works regardless of how this file is executed.
    """
    _configure_models()

    # Load index (will inherit Settings.embed_model for query embeddings)
    index = load_index_from_disk()

    # Build base retriever, then wrap with ParentChildRetrieverV2
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k, verbose=False)
    pc_retriever = ParentChildRetrieverV2(base_retriever)

    # Pass through callback_manager when present
    qe = ParentChildQueryEngineV2(
        retriever=pc_retriever,
        callback_manager=getattr(index, "callback_manager", None),
    )
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
