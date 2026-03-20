from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

import requests


log = logging.getLogger("rag_v2.youtube_meta")

_YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
_DURATION_RE = re.compile(r"^PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?$")


def _api_key() -> str:
    return (os.getenv("YOUTUBE_API_KEY") or "").strip()


def _parse_duration_iso8601(raw: str) -> Optional[int]:
    s = (raw or "").strip()
    if not s:
        return None
    m = _DURATION_RE.match(s)
    if not m:
        return None
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m") or 0)
    seconds = int(m.group("s") or 0)
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def _pick_thumbnail(thumbnails: Any) -> Optional[str]:
    if not isinstance(thumbnails, dict):
        return None
    for key in ("maxres", "standard", "high", "medium", "default"):
        entry = thumbnails.get(key)
        if isinstance(entry, dict):
            url = (entry.get("url") or "").strip()
            if url:
                return url
    return None


@lru_cache(maxsize=4096)
def fetch_video_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort YouTube Data API enrichment for display metadata.

    Cached in-process to avoid repeat calls across queries.
    """
    vid = (video_id or "").strip()
    if not vid:
        return None
    key = _api_key()
    if not key:
        return None

    try:
        resp = requests.get(
            f"{_YOUTUBE_API_BASE}/videos",
            params={
                "part": "snippet,contentDetails",
                "id": vid,
                "maxResults": "1",
                "key": key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        items = payload.get("items") or []
        if not items:
            return None
        item = items[0] or {}
        snippet = item.get("snippet") or {}
        cd = item.get("contentDetails") or {}
        published_at = (snippet.get("publishedAt") or "").strip()
        if published_at and "T" in published_at:
            published_at = published_at.split("T", 1)[0]
        duration_s = _parse_duration_iso8601(str(cd.get("duration") or ""))
        thumb = _pick_thumbnail(snippet.get("thumbnails"))
        channel_title = (snippet.get("channelTitle") or "").strip() or None
        channel_id = (snippet.get("channelId") or "").strip() or None
        title = (snippet.get("title") or "").strip() or None
        return {
            "video_id": vid,
            "title": title,
            "channel_title": channel_title,
            "channel_id": channel_id,
            "published_at": published_at or None,
            "duration_s": float(duration_s) if duration_s is not None else None,
            "thumbnail_url": thumb,
        }
    except Exception as exc:  # pragma: no cover - network
        log.debug("[yt-meta] fetch failed video_id=%s err=%s", vid, exc)
        return None


def enrich_rows_inplace(rows: Iterable[Dict[str, Any]]) -> None:
    """
    Mutate a list of metadata rows in-place, filling missing fields (published_at,
    duration_s, thumbnail_url, channel) using YouTube API when configured.
    """
    for row in rows:
        try:
            vid = (row.get("video_id") or row.get("parent_id") or "").strip()
        except Exception:
            vid = ""
        if not vid:
            continue
        needs = (
            not row.get("published_at")
            or row.get("duration_s") is None
            or not row.get("thumbnail_url")
            or not row.get("channel_name")
            or not row.get("channel_id")
        )
        if not needs:
            continue
        meta = fetch_video_metadata(vid)
        if not meta:
            continue
        if not row.get("published_at"):
            row["published_at"] = meta.get("published_at")
        if row.get("duration_s") is None:
            row["duration_s"] = meta.get("duration_s")
        if not row.get("thumbnail_url"):
            row["thumbnail_url"] = meta.get("thumbnail_url")
        if not row.get("channel_name"):
            row["channel_name"] = meta.get("channel_title")
        if not row.get("channel_id"):
            row["channel_id"] = meta.get("channel_id")
        if not row.get("title"):
            row["title"] = meta.get("title")
