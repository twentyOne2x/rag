from __future__ import annotations

import os
import sys
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# --- Make imports work whether run as "python src/rag_v2/app_main.py" or "python -m src.rag_v2.app_main" ---

# Try relative (module execution). If that fails, add "<repo>/src" to sys.path and import absolutely.
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


def bootstrap_query_engine_v2(similarity_top_k: int = 50):
    """
    Bootstraps the Parent/Child query engine with your Pinecone-backed index.
    Works regardless of how this file is executed.
    """
    # Configure LLM (inherits your global Settings elsewhere too)
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))

    # Load index
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
    # Simple smoke test when running directly in your IDE
    qe = bootstrap_query_engine_v2()
    from llama_index.core.schema import QueryBundle

    resp = qe.query(QueryBundle("what is a DAT on Solana?"))
    print(resp)
