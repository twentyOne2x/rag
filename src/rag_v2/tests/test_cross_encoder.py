from __future__ import annotations

from typing import List, Tuple

from rag_v2.rerankers.cross_encoder import CEReranker


def _pair() -> List[Tuple[str, str, float]]:
    return [("seg-1", "stub text", 0.42)]


def test_cross_encoder_disabled_when_model_init_fails(monkeypatch):
    import rag_v2.rerankers.cross_encoder as ce_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("init failed")

    monkeypatch.setattr(ce_mod, "CrossEncoder", boom, raising=False)
    ce = CEReranker(model_name="stub/fail")

    assert ce._ensure_model() is False
    assert ce.enabled is False
    assert ce.model is None


def test_cross_encoder_predict_failure_disables_and_returns_stage1(monkeypatch):
    ce = CEReranker(model_name="stub/fail-predict")
    monkeypatch.setattr(ce, "_ensure_model", lambda: True)

    class _Model:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("predict exploded")

    ce.model = _Model()  # type: ignore

    items = _pair()
    out = ce.rerank("stub query", items)

    assert out == items
    assert ce.enabled is False
    assert ce.model is None
