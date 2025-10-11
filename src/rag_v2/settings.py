from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "rag_v2" / "config.yaml"
)


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the rag_v2 configuration from disk."""

    config_path = Path(path or os.getenv("RAG_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"rag_v2 config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"rag_v2 config at {config_path} must be a mapping")
    return data


def _lookup(config: dict[str, Any], path: Iterable[str]) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
        if cursor is None:
            return None
    return cursor


def config_value(path: str, *, cast: callable | None = None, default: Any = None, env: str | None = None) -> Any:
    """
    Return a configuration value at ``path`` (dot-separated). If an environment
    variable (``env`` or the upper-cased path) is present it overrides the file.
    The ``cast`` callable can coerce the value to a desired type.
    """

    env_key = env or path.replace(".", "_").upper()
    env_val = os.getenv(env_key)
    if env_val is not None:
        return cast(env_val) if cast else env_val

    cfg = load_config()
    parts = path.split(".")
    value = _lookup(cfg, parts)
    if value is None:
        return default
    return cast(value) if cast else value


def bool_cast(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def int_cast(value: Any) -> int:
    return int(value)


def float_cast(value: Any) -> float:
    return float(value)
