from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_SCOPE = "videos"

_CHANNEL_FILE = Path(__file__).resolve().parents[2] / "configs" / "rag_v2" / "channels.json"


def _config_path(scope: str) -> Path | None:
    if not _CHANNEL_FILE.exists():
        return None
    try:
        with _CHANNEL_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        namespaces = payload.get("namespaces") or {}
        if not isinstance(namespaces, dict):
            return None
        ns = namespaces.get(scope)
        if not isinstance(ns, dict):
            return None
        channels = ns.get("channels") or []
        return channels
    except Exception:
        return None


def _normalise_entries(raw: Iterable) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for item in raw:
        if isinstance(item, str):
            cleaned.append({"name": item})
        elif isinstance(item, dict):
            name = (item.get("name") if isinstance(item.get("name"), str) else None)
            if not name:
                continue
            entry: Dict[str, object] = {"name": name}
            count = item.get("count")
            if isinstance(count, (int, float)):
                entry["count"] = int(count)
            cleaned.append(entry)
    return cleaned


@lru_cache(maxsize=4)
def channel_catalog(scope: str = DEFAULT_SCOPE) -> List[Dict[str, object]]:
    """Return a list of channel dictionaries for the provided scope."""

    path = _config_path(scope)
    if path is None:
        return []
    catalog = _normalise_entries(path)
    catalog.sort(key=lambda entry: str(entry.get("name", "")).lower())
    return catalog


@lru_cache(maxsize=4)
def channel_names(scope: str = DEFAULT_SCOPE) -> List[str]:
    return [entry["name"] for entry in channel_catalog(scope) if entry.get("name")]
