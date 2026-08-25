from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.rag_v2 import indexer


class ExistingCollectionClient:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.create_calls = 0

    def collection_exists(self, *, collection_name: str) -> bool:
        assert collection_name == "icmfyi-v2__videos"
        return True

    def get_collection(self, *, collection_name: str):
        assert collection_name == "icmfyi-v2__videos"
        vectors = SimpleNamespace(size=self.dimension)
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
        )

    def create_collection(self, **_kwargs) -> None:
        self.create_calls += 1


def test_existing_qdrant_collection_must_match_embedding_dimension() -> None:
    client = ExistingCollectionClient(3072)

    with pytest.raises(RuntimeError, match="does not match EMBED_DIM 1024"):
        indexer._ensure_qdrant_collection(client, "icmfyi-v2__videos", 1024)

    assert client.create_calls == 0


def test_matching_qdrant_collection_is_reused_without_mutation() -> None:
    client = ExistingCollectionClient(1024)

    indexer._ensure_qdrant_collection(client, "icmfyi-v2__videos", 1024)

    assert client.create_calls == 0


def test_embedding_dimension_contract_rejects_invalid_values() -> None:
    with patch.dict(os.environ, {"EMBED_DIM": "1024"}, clear=False):
        assert indexer._embedding_dimension() == 1024
    with patch.dict(os.environ, {"EMBED_DIM": "0"}, clear=False):
        with pytest.raises(RuntimeError, match="positive integer"):
            indexer._embedding_dimension()
    with patch.dict(os.environ, {"EMBED_DIM": "not-a-number"}, clear=False):
        with pytest.raises(RuntimeError, match="positive integer"):
            indexer._embedding_dimension()
