from __future__ import annotations
import os
import logging
import re
from typing import Any, Dict, Iterable

from .pinecone_client import get_pinecone_index

log = logging.getLogger("rag_v2.parent_resolver")
log.setLevel(logging.INFO)

# simple in-process cache (PID-scoped)
_CACHE: Dict[str, Dict[str, str]] = {}

def _ns_default() -> str:
    # default to 'videos' if unset
    return os.getenv("PINECONE_NAMESPACE", "videos")

# YYYY-MM-DD anywhere in the title
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

def _infer_date_from_title(meta: Dict[str, Any]) -> str | None:
    title = (meta.get("title") or meta.get("parent_title") or "")[:200]
    m = _DATE_RE.search(title)
    return m.group(1) if m else None

import re
_DATE_ID_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]{11}_")
def _clean_title(raw: str | None) -> str:
    if not raw: return raw or ""
    return _DATE_ID_PREFIX_RE.sub("", raw).strip()


def fetch_parent_meta(parent_ids: Iterable[str], namespace: str | None = None) -> Dict[str, Dict[str, str]]:
    """
    Fetch parent rows (by id) from Pinecone and return a minimal dict per id.
    Robust to different FetchResponse shapes and preserves a simple in-proc cache.
    """
    ids = [str(pid) for pid in parent_ids if pid]
    if not ids:
        return {}

    # Serve from cache when possible
    missing = [pid for pid in ids if pid not in _CACHE]
    if not missing:
        return {pid: _CACHE.get(pid, {}) for pid in ids}

    ns = namespace or _ns_default()
    vectors: Dict[str, Any] = {}

    try:
        idx = get_pinecone_index()
        res = idx.fetch(ids=missing, namespace=ns)

        # Handle SDK object, dict, or simple attr
        if hasattr(res, "to_dict"):
            vectors = (res.to_dict() or {}).get("vectors", {}) or {}
        elif isinstance(res, dict):
            vectors = (res or {}).get("vectors", {}) or {}
        else:
            vectors = getattr(res, "vectors", {}) or {}

    except Exception as e:
        log.warning("[parents] fetch failed ns=%r ids=%d err=%s", ns, len(missing), e)
        vectors = {}

    found = 0
    for pid, row in vectors.items():
        # row can be a dict or an SDK record with .metadata
        meta = None
        if hasattr(row, "metadata"):
            meta = getattr(row, "metadata", None)
        if meta is None and isinstance(row, dict):
            meta = row.get("metadata")
        meta = meta or {}

        # Choose the best available date; fall back to parsing from title.
        date = meta.get("published_at") or meta.get("published_date") or _infer_date_from_title(meta)

        # map to the few fields children/UI need
        mapped = {
            "parent_title": _clean_title(meta.get("title")),
            "parent_channel_name": meta.get("channel_name"),
            "parent_published_at": date,
            "parent_published_date": date,
            "parent_url": (str(meta.get("url")) if meta.get("url") is not None else None),
        }
        _CACHE[str(pid)] = mapped
        found += 1

    missed = [pid for pid in missing if pid not in vectors]
    if found or missing:
        log.info(
            "[parents] ns=%r requested=%d found=%d missed=%d cached_total=%d",
            ns, len(missing), found, len(missed), len(_CACHE)
        )

    # Always return entries for all requested ids (cached or empty)
    return {pid: _CACHE.get(pid, {}) for pid in ids}
