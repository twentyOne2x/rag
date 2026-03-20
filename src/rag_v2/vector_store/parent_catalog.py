from __future__ import annotations

import logging
import os
import re
import time
import base64
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple, FrozenSet

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.rag_v2.utils.youtube_metadata import enrich_rows_inplace


log = logging.getLogger("rag_v2.parent_catalog")


_TOKEN_RE = re.compile(r"[A-Za-z0-9@#_.-]+")
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Small, pragmatic stopword set. Keep it minimal so finance/proper nouns survive.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "latest",
    "me",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "vs",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if t and t not in _STOPWORDS and len(t) > 1]


def _normalize_date(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # common: YYYYMMDD (yt-dlp upload_date)
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    # ISO: keep date prefix
    if "T" in s:
        s = s.split("T", 1)[0]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return s[:10] if len(s) >= 10 else s


def _coerce_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _qdrant_collection_name(namespace: str) -> str:
    template = os.getenv("QDRANT_COLLECTION_TEMPLATE", "{index}__{namespace}")
    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    return template.format(index=index_name, namespace=namespace)


@dataclass(frozen=True)
class CatalogRow:
    video_id: str
    title: str
    description: Optional[str]
    channel_name: Optional[str]
    channel_id: Optional[str]
    published_at: Optional[str]
    duration_s: Optional[float]
    url: Optional[str]
    thumbnail_url: Optional[str]
    source: Optional[str] = None
    document_type: Optional[str] = None
    topic_summary: Optional[str] = None
    router_tags: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    canonical_entities: Optional[List[str]] = None
    speaker_names: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    router_boost: Optional[float] = None

    # Precomputed tokens for fast catalog search (built at load time).
    blob_tokens: FrozenSet[str] = frozenset()

    @property
    def search_blob(self) -> str:
        parts: List[str] = [self.title or ""]
        if self.description:
            parts.append(self.description)
        if self.channel_name:
            parts.append(self.channel_name)
        if self.topic_summary:
            parts.append(self.topic_summary)
        if self.router_tags:
            parts.extend([str(t) for t in self.router_tags if t])
        if self.aliases:
            parts.extend([str(a) for a in self.aliases if a])
        if self.canonical_entities:
            parts.extend([str(a) for a in self.canonical_entities if a])
        if self.speaker_names:
            parts.extend([str(a) for a in self.speaker_names if a])
        if self.entities:
            parts.extend([str(a) for a in self.entities if a])
        return " ".join(parts)


# PID-local cache: collection_name -> (loaded_ts, rows)
_CACHE: Dict[str, Tuple[float, List[CatalogRow]]] = {}


def _cache_ttl_s() -> int:
    try:
        return max(30, int(os.getenv("CATALOG_CACHE_TTL_S", "600")))
    except Exception:
        return 600


def _qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


def load_parent_catalog(
    *,
    namespace: str,
    force_refresh: bool = False,
) -> List[CatalogRow]:
    """
    Load all parent payload rows for the given namespace (Qdrant collection).

    This is intentionally simple: scan parent points (node_type=parent) and cache
    in-process with a TTL so repeated catalog queries are fast.
    """
    backend = (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()
    if backend != "qdrant":
        raise RuntimeError("catalog search is only implemented for VECTOR_STORE=qdrant")

    namespace_norm = (namespace or "").strip() or os.getenv("PINECONE_NAMESPACE", "videos")
    collection = _qdrant_collection_name(namespace_norm)
    now = time.time()
    ttl = float(_cache_ttl_s())

    cached = _CACHE.get(collection)
    if not force_refresh and cached is not None:
        loaded_ts, rows = cached
        if now - float(loaded_ts) < ttl:
            return rows

    client = _qdrant_client()
    rows: List[CatalogRow] = []
    offset = None
    scanned = 0

    flt = qm.Filter(
        must=[
            qm.FieldCondition(key="node_type", match=qm.MatchValue(value="parent")),
        ]
    )

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            scanned += 1
            payload = dict(getattr(p, "payload", None) or {})
            vid = (
                str(payload.get("parent_id") or payload.get("video_id") or payload.get("id") or "").strip()
            )
            if not vid:
                continue
            title = str(payload.get("title") or payload.get("parent_title") or vid).strip()
            channel_name = (payload.get("channel_name") or payload.get("parent_channel_name") or None)
            channel_id = (payload.get("channel_id") or payload.get("parent_channel_id") or None)
            published_at = _normalize_date(
                payload.get("published_at")
                or payload.get("published_date")
                or payload.get("date")
                or payload.get("parent_published_at")
                or payload.get("parent_published_date")
            )
            duration_s = _coerce_float(payload.get("duration_s") or payload.get("duration"))
            url = str(payload.get("url") or payload.get("parent_url") or "").strip() or None
            if not url and _YOUTUBE_ID_RE.fullmatch(vid):
                url = f"https://www.youtube.com/watch?v={vid}"
            thumb = str(payload.get("thumbnail_url") or payload.get("parent_thumbnail_url") or "").strip() or None
            if not thumb and _YOUTUBE_ID_RE.fullmatch(vid):
                thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            description = payload.get("description") or payload.get("parent_description") or None
            source = payload.get("source") or payload.get("parent_source") or None
            document_type = payload.get("document_type") or payload.get("parent_document_type") or None
            topic_summary = payload.get("topic_summary") or payload.get("parent_topic_summary") or None
            router_tags_raw = payload.get("router_tags") or payload.get("parent_router_tags")
            router_tags = None
            if isinstance(router_tags_raw, list):
                router_tags = [str(a) for a in router_tags_raw if a]
            aliases_raw = payload.get("aliases")
            aliases = None
            if isinstance(aliases_raw, list):
                aliases = [str(a) for a in aliases_raw if a]
            canonical_raw = payload.get("canonical_entities")
            canonical_entities = None
            if isinstance(canonical_raw, list):
                canonical_entities = [str(a) for a in canonical_raw if a]
            speaker_names_raw = payload.get("speaker_names") or payload.get("parent_speaker_names")
            speaker_names = None
            if isinstance(speaker_names_raw, list):
                speaker_names = [str(a) for a in speaker_names_raw if a]
            entities_raw = payload.get("entities") or payload.get("parent_entities")
            entities = None
            if isinstance(entities_raw, list):
                entities = [str(a) for a in entities_raw if a]
            router_boost = _coerce_float(payload.get("router_boost") or payload.get("parent_router_boost"))

            row = CatalogRow(
                video_id=vid,
                title=title,
                description=str(description).strip() if description else None,
                channel_name=str(channel_name).strip() if channel_name else None,
                channel_id=str(channel_id).strip() if channel_id else None,
                published_at=published_at,
                duration_s=duration_s,
                url=url,
                thumbnail_url=thumb,
                source=str(source).strip() if source else None,
                document_type=str(document_type).strip() if document_type else None,
                topic_summary=str(topic_summary).strip() if topic_summary else None,
                router_tags=router_tags,
                aliases=aliases,
                canonical_entities=canonical_entities,
                speaker_names=speaker_names,
                entities=entities,
                router_boost=router_boost,
            )
            row = replace(row, blob_tokens=frozenset(_tokenize(row.search_blob)))
            rows.append(row)
        if offset is None:
            break

    _CACHE[collection] = (now, rows)
    log.info("[catalog] loaded namespace=%s collection=%s parents=%d scanned=%d", namespace_norm, collection, len(rows), scanned)
    return rows


def search_parent_catalog(
    *,
    query: str,
    namespace: str,
    limit: int = 20,
    channel_filter: Optional[Dict[str, List[str]]] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    rows = load_parent_catalog(namespace=namespace, force_refresh=force_refresh)
    q_tokens = _tokenize(q)
    if not q_tokens:
        return []

    include_ids = set((channel_filter or {}).get("include_ids") or [])
    exclude_ids = set((channel_filter or {}).get("exclude_ids") or [])
    include_names = {s.lower() for s in (channel_filter or {}).get("include_names") or []}
    exclude_names = {s.lower() for s in (channel_filter or {}).get("exclude_names") or []}

    q_phrase = " ".join(q_tokens)

    scored: List[Tuple[float, CatalogRow]] = []
    for row in rows:
        if row.channel_id and row.channel_id in exclude_ids:
            continue
        if include_ids and (not row.channel_id or row.channel_id not in include_ids):
            continue

        if row.channel_name:
            cname = row.channel_name.lower()
            if cname in exclude_names:
                continue
            if include_names and cname not in include_names:
                continue

        blob = row.search_blob.lower()
        blob_tokens = row.blob_tokens
        overlap = sum(1 for t in q_tokens if t in blob_tokens)
        if overlap <= 0:
            continue

        # Core score: fraction of query terms matched.
        score = overlap / float(max(1, len(q_tokens)))
        # Small boosts for strong signals.
        title_lower = (row.title or "").lower()
        if q_phrase and q_phrase in title_lower:
            score += 0.35
        if q_phrase and q_phrase in blob:
            score += 0.10
        if any(t in title_lower for t in q_tokens):
            score += 0.05
        if row.router_boost:
            try:
                score *= max(0.90, min(1.25, 1.0 + 0.05 * (float(row.router_boost) - 1.0)))
            except Exception:
                pass
        scored.append((score, row))

    scored.sort(key=lambda it: (it[0], it[1].published_at or ""), reverse=True)
    out: List[Dict[str, Any]] = []
    for score, row in scored[: max(1, min(int(limit), 50))]:
        out.append(
            {
                "video_id": row.video_id,
                "parent_id": row.video_id,
                "title": row.title,
                "channel_name": row.channel_name,
                "channel_id": row.channel_id,
                "published_at": row.published_at,
                "duration_s": row.duration_s,
                "url": row.url,
                "thumbnail_url": row.thumbnail_url,
                "source": row.source,
                "document_type": row.document_type,
                "score": round(float(score), 4),
            }
        )
    # Best-effort YouTube API enrichment for missing published_at/duration/thumbnail/channel.
    # Keeps catalog UX useful even when older ingests didn't persist these fields.
    enrich_rows_inplace(out)
    return out


def _cursor_key(published_at: Optional[str], video_id: str) -> Tuple[str, str]:
    # published_at is already normalized to "YYYY-MM-DD" in CatalogRow when present.
    # Keep missing dates sortable and stable (push to the end).
    return (published_at or "0000-00-00", video_id or "")


def encode_recent_cursor(*, published_at: Optional[str], video_id: str) -> str:
    payload = {"published_at": published_at or None, "video_id": video_id}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_recent_cursor(raw: str) -> Optional[Tuple[str, str]]:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Restore padding.
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    try:
        decoded = base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")
        obj = json.loads(decoded)
        if not isinstance(obj, dict):
            return None
        pub = _normalize_date(obj.get("published_at")) or None
        vid = str(obj.get("video_id") or "").strip()
        if not vid:
            return None
        return _cursor_key(pub, vid)
    except Exception:
        return None


def list_recent_parent_catalog(
    *,
    namespace: str,
    limit: int = 50,
    since: Optional[str] = None,
    cursor: Optional[str] = None,
    channel_filter: Optional[Dict[str, List[str]]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return the most recent parent rows (metadata-first) with a stable cursor.

    This intentionally reuses the in-process cached parent catalog loader and
    performs sorting/pagination in memory. For single-tenant appliances this is
    usually fast enough and far more robust than bespoke Qdrant sorting logic.
    """
    hard_cap = 200
    lim = max(1, min(int(limit or 0), hard_cap))

    rows = load_parent_catalog(namespace=namespace, force_refresh=force_refresh)
    include_ids = set((channel_filter or {}).get("include_ids") or [])
    exclude_ids = set((channel_filter or {}).get("exclude_ids") or [])
    include_names = {s.lower() for s in (channel_filter or {}).get("include_names") or []}
    exclude_names = {s.lower() for s in (channel_filter or {}).get("exclude_names") or []}

    since_norm = _normalize_date(since) if since else None
    cursor_key = decode_recent_cursor(cursor or "") if cursor else None

    filtered: List[CatalogRow] = []
    for row in rows:
        if row.channel_id and row.channel_id in exclude_ids:
            continue
        if include_ids and (not row.channel_id or row.channel_id not in include_ids):
            continue

        if row.channel_name:
            cname = row.channel_name.lower()
            if cname in exclude_names:
                continue
            if include_names and cname not in include_names:
                continue

        if since_norm:
            pub = row.published_at or ""
            if not pub:
                continue
            if pub < since_norm:
                continue

        filtered.append(row)

    filtered.sort(key=lambda r: _cursor_key(r.published_at, r.video_id), reverse=True)

    out_rows: List[CatalogRow] = []
    for row in filtered:
        k = _cursor_key(row.published_at, row.video_id)
        if cursor_key and k >= cursor_key:
            # Still at/after the cursor (newer or same); keep skipping until we pass it.
            continue
        out_rows.append(row)
        if len(out_rows) >= lim:
            break

    exhausted = len(out_rows) < lim
    next_cursor = None
    if out_rows:
        last = out_rows[-1]
        next_cursor = encode_recent_cursor(published_at=last.published_at, video_id=last.video_id)

    results: List[Dict[str, Any]] = []
    for row in out_rows:
        results.append(
            {
                "video_id": row.video_id,
                "parent_id": row.video_id,
                "title": row.title,
                "description": row.description,
                "channel_name": row.channel_name,
                "channel_id": row.channel_id,
                "published_at": row.published_at,
                "duration_s": row.duration_s,
                "url": row.url,
                "thumbnail_url": row.thumbnail_url,
                "source": row.source,
                "document_type": row.document_type,
                "topic_summary": row.topic_summary,
                "router_tags": row.router_tags,
                "aliases": row.aliases,
                "canonical_entities": row.canonical_entities,
                "speaker_names": row.speaker_names,
                "entities": row.entities,
                "router_boost": row.router_boost,
            }
        )

    # Keep response robust for older ingests.
    enrich_rows_inplace(results)

    return {
        "results": results,
        "scanned": len(rows),
        "matched": len(filtered),
        "returned": len(results),
        "next_cursor": (next_cursor if not exhausted else None),
        "exhausted": exhausted,
        "since": since_norm,
    }
