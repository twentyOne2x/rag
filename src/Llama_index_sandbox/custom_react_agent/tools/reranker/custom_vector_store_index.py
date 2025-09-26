# src/Llama_index_sandbox/custom_react_agent/tools/reranker/custom_vector_store_index.py

from __future__ import annotations

from typing import Any

# ✅ Core-only import
from llama_index.core import VectorStoreIndex

from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import (
    CustomQueryEngine,
)


class CustomVectorStoreIndex(VectorStoreIndex):
    """
    Core-native VectorStoreIndex subclass that returns our CustomQueryEngine.
    We build a core retriever directly and pass it to CustomQueryEngine.
    """

    def as_query_engine(self, **kwargs: Any) -> CustomQueryEngine:
        # Build a core retriever (lets caller pass similarity_top_k, filters, etc. via kwargs)
        retriever = super().as_retriever(**kwargs)

        # Try to pass a callback_manager if present on the index
        callback_manager = getattr(self, "callback_manager", None)

        return CustomQueryEngine(
            retriever=retriever,
            callback_manager=callback_manager,
        )
