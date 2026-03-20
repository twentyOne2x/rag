from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable

from .pinecone_client import get_pinecone_index

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - optional dependency when pinecone backend is used
    QdrantClient = None  # type: ignore

try:
    from qdrant_client.http import models as qm
except Exception:  # pragma: no cover
    qm = None  # type: ignore

log = logging.getLogger("rag_v2.parent_resolver")
log.setLevel(logging.INFO)

# simple in-process cache (PID-scoped)
_CACHE: Dict[str, Dict[str, str]] = {}


def _backend() -> str:
    return (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()


def _ns_default() -> str:
    # default to 'videos' if unset
    return os.getenv("PINECONE_NAMESPACE", "videos")


def _qdrant_collection_name(namespace: str) -> str:
    template = os.getenv("QDRANT_COLLECTION_TEMPLATE", "{index}__{namespace}")
    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    return template.format(index=index_name, namespace=namespace)


# YYYY-MM-DD anywhere in the title
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _infer_date_from_title(meta: Dict[str, Any]) -> str | None:
    title = (meta.get("title") or meta.get("parent_title") or "")[:200]
    m = _DATE_RE.search(title)
    return m.group(1) if m else None


_DATE_ID_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]{11}_")


def _clean_title(raw: str | None) -> str:
    if not raw:
        return raw or ""
    return _DATE_ID_PREFIX_RE.sub("", raw).strip()


def _map_parent_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    date = meta.get("published_at") or meta.get("published_date") or _infer_date_from_title(meta)
    duration_s = meta.get("duration_s") or meta.get("duration")
    speaker_primary = meta.get("speaker_primary") or meta.get("parent_speaker_primary")
    speaker_names_raw = meta.get("speaker_names") or meta.get("parent_speaker_names")
    speaker_names = None
    if isinstance(speaker_names_raw, list):
        speaker_names = [str(x).strip() for x in speaker_names_raw if x and str(x).strip()]
    return {
        "parent_title": _clean_title(meta.get("title")),
        "parent_channel_name": meta.get("channel_name"),
        "parent_channel_id": meta.get("channel_id"),
        "parent_published_at": date,
        "parent_published_date": date,
        "parent_url": (str(meta.get("url")) if meta.get("url") is not None else None),
        "parent_duration_s": float(duration_s) if duration_s is not None else None,
        "parent_speaker_primary": (str(speaker_primary).strip() if speaker_primary else None),
        "parent_speaker_names": speaker_names,
        # surface parent summary when present so downstream can use it
        "parent_topic_summary": meta.get("topic_summary") or meta.get("parent_topic_summary"),
    }


def _fetch_parents_pinecone(ids: Iterable[str], namespace: str) -> Dict[str, Dict[str, Any]]:
    vectors: Dict[str, Any] = {}
    idx = get_pinecone_index()
    res = idx.fetch(ids=list(ids), namespace=namespace)

    # Handle SDK object, dict, or simple attr
    if hasattr(res, "to_dict"):
        vectors = (res.to_dict() or {}).get("vectors", {}) or {}
    elif isinstance(res, dict):
        vectors = (res or {}).get("vectors", {}) or {}
    else:
        vectors = getattr(res, "vectors", {}) or {}

    out: Dict[str, Dict[str, Any]] = {}
    for pid, row in vectors.items():
        meta = None
        if hasattr(row, "metadata"):
            meta = getattr(row, "metadata", None)
        if meta is None and isinstance(row, dict):
            meta = row.get("metadata")
        out[str(pid)] = dict(meta or {})
    return out


def _fetch_parents_qdrant(ids: Iterable[str], namespace: str) -> Dict[str, Dict[str, Any]]:
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is not installed")
    if qm is None:
        raise RuntimeError("qdrant-client http models are unavailable")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    wanted = [str(i) for i in ids if i]
    if not wanted:
        return {}

    # Parent points in Qdrant use UUID point IDs; the external parent_id (YouTube id)
    # is stored in payload. Fetch by payload match instead of by point ID.
    flt = qm.Filter(
        must=[
            qm.FieldCondition(
                key="parent_id",
                match=qm.MatchAny(any=wanted),
            ),
            # Ensure we only match parent nodes (child nodes also carry parent_id).
            qm.FieldCondition(
                key="node_type",
                match=qm.MatchValue(value="parent"),
            ),
        ]
    )

    points: list[Any] = []
    offset = None
    while True:
        res = client.scroll(
            collection_name=_qdrant_collection_name(namespace),
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=min(256, max(1, len(wanted) - len(points))),
            offset=offset,
        )
        batch, offset = res
        points.extend(batch or [])
        if not offset or len(points) >= len(wanted):
            break

    out: Dict[str, Dict[str, Any]] = {}
    for row in points:
        payload = dict(getattr(row, "payload", None) or {})
        pid = payload.get("parent_id")
        if not pid:
            continue
        out[str(pid)] = payload
    return out


def fetch_parent_meta(parent_ids: Iterable[str], namespace: str | None = None) -> Dict[str, Dict[str, str]]:
    """
    Fetch parent rows (by id) from active vector store and return a minimal dict per id.
    Robust to different SDK response shapes and preserves a simple in-proc cache.
    """
    ids = [str(pid) for pid in parent_ids if pid]
    if not ids:
        return {}

    # Serve from cache when possible
    missing = [pid for pid in ids if pid not in _CACHE]
    if not missing:
        return {pid: _CACHE.get(pid, {}) for pid in ids}

    ns = namespace or _ns_default()
    rows: Dict[str, Dict[str, Any]] = {}

    try:
        if _backend() == "qdrant":
            rows = _fetch_parents_qdrant(missing, ns)
        else:
            rows = _fetch_parents_pinecone(missing, ns)
    except Exception as e:
        log.warning("[parents] fetch failed ns=%r ids=%d err=%s", ns, len(missing), e)
        rows = {}

    found = 0
    for pid, meta in rows.items():
        _CACHE[str(pid)] = _map_parent_meta(meta)
        found += 1

    missed = [pid for pid in missing if pid not in rows]
    if found or missing:
        log.info(
            "[parents] backend=%s ns=%r requested=%d found=%d missed=%d cached_total=%d",
            _backend(),
            ns,
            len(missing),
            found,
            len(missed),
            len(_CACHE),
        )

    # Always return entries for all requested ids (cached or empty)
    return {pid: _CACHE.get(pid, {}) for pid in ids}
