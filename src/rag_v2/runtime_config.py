from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Dict, Iterator

from .config import CFG, RetrievalConfig

_runtime_cfg: ContextVar[RetrievalConfig] = ContextVar("runtime_cfg", default=CFG)


def get_runtime_config() -> RetrievalConfig:
    """Return the runtime retrieval configuration for the current context."""
    return _runtime_cfg.get()


@contextmanager
def override_runtime_config(overrides: Dict[str, Any] | None) -> Iterator[RetrievalConfig]:
    """
    Temporarily override fields on the retrieval config for the current task.
    Only keys present on RetrievalConfig are applied; unknown keys are ignored.
    """
    if not overrides:
        yield get_runtime_config()
        return

    current = get_runtime_config()
    valid: Dict[str, Any] = {}
    for key, value in overrides.items():
        if hasattr(current, key):
            valid[key] = value
    if not valid:
        yield current
        return

    updated = replace(current, **valid)
    token = _runtime_cfg.set(updated)
    try:
        yield updated
    finally:
        _runtime_cfg.reset(token)
