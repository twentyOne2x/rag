from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from llama_index.core.embeddings import MockEmbedding

from src.rag_v2 import app_main


@pytest.fixture(autouse=True)
def restore_global_llama_settings():
    previous_llm = app_main.Settings._llm
    previous_embed_model = app_main.Settings._embed_model
    try:
        yield
    finally:
        app_main.Settings._llm = previous_llm
        app_main.Settings._embed_model = previous_embed_model


def _fake_huggingface_module(dimension: int) -> ModuleType:
    module = ModuleType("llama_index.embeddings.huggingface")

    class FakeHuggingFaceEmbedding(MockEmbedding):
        def __init__(self, **kwargs) -> None:
            super().__init__(embed_dim=dimension)
            object.__setattr__(self, "_test_kwargs", kwargs)

    module.HuggingFaceEmbedding = FakeHuggingFaceEmbedding
    return module


def test_local_embedding_model_is_revision_pinned_and_dimension_proved() -> None:
    revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    env = {
        "ICMFYI_PRODUCTION": "0",
        "RAG_LLM_PROVIDER": "mock",
        "EMBED_PROVIDER": "sentence-transformers",
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBED_MODEL_REVISION": revision,
        "EMBED_DIM": "1024",
        "RAG_EMBED_DEVICE": "cpu",
    }
    fake_module = _fake_huggingface_module(1024)

    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict(sys.modules, {"llama_index.embeddings.huggingface": fake_module}),
    ):
        contract = app_main._configure_models()

    assert contract["embed_dimension"] == 1024
    assert contract["embed_model_revision"] == revision
    assert app_main.Settings.embed_model._test_kwargs["revision"] == revision
    assert app_main.Settings.embed_model._test_kwargs["normalize"] is True
    assert app_main.Settings.embed_model._test_kwargs["trust_remote_code"] is False


def test_production_local_embedding_requires_exact_revision() -> None:
    env = {
        "ICMFYI_PRODUCTION": "1",
        "RAG_LLM_PROVIDER": "openai",
        "RAG_LLM_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "test-key",
        "EMBED_PROVIDER": "sentence-transformers",
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBED_DIM": "1024",
    }
    fake_module = _fake_huggingface_module(1024)

    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict(sys.modules, {"llama_index.embeddings.huggingface": fake_module}),
        pytest.raises(RuntimeError, match="EMBED_MODEL_REVISION"),
    ):
        app_main._configure_models()


def test_embedding_probe_rejects_wrong_dimension() -> None:
    env = {
        "ICMFYI_PRODUCTION": "0",
        "RAG_LLM_PROVIDER": "mock",
        "EMBED_PROVIDER": "sentence-transformers",
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBED_MODEL_REVISION": "revision",
        "EMBED_DIM": "1024",
    }
    fake_module = _fake_huggingface_module(384)

    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict(sys.modules, {"llama_index.embeddings.huggingface": fake_module}),
        pytest.raises(RuntimeError, match="model returned 384, expected 1024"),
    ):
        app_main._configure_models()
