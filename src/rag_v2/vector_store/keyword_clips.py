from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .parent_resolver import fetch_parent_meta


def _backend() -> str:
    return (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()


def _qdrant_collection_name(namespace: str) -> str:
    template = os.getenv("QDRANT_COLLECTION_TEMPLATE", "{index}__{namespace}")
    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    return template.format(index=index_name, namespace=namespace)


def _qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_GENERIC_SPK_RE = re.compile(r"^(?:s\\d+|speaker\\s*\\d+|[a-z])$", re.IGNORECASE)


def _normalize_match_text(s: str) -> str:
    """
    Normalize text for keyword/phrase matching:
      - lower-case
      - replace punctuation/hyphens with spaces
      - collapse whitespace
    """
    if not s:
        return ""
    low = s.lower()
    low = _NON_ALNUM_RE.sub(" ", low)
    low = _WS_RE.sub(" ", low).strip()
    return low


def _to_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _hms_ms_to_seconds(hms: Any) -> Optional[float]:
    s = str(hms or "").strip()
    if not s or ":" not in s:
        return None
    parts = s.split(":")
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
    except Exception:
        return None
    if hh < 0 or mm < 0 or ss < 0:
        return None
    return float(hh * 3600 + mm * 60) + float(ss)


def _strip_time_param(url: str) -> str:
    if not url:
        return url
    # Simple safe strip to avoid stacking multiple time params.
    for sep in ("&t=", "?t="):
        if sep in url:
            return url.split(sep, 1)[0]
    return url


def _add_time_param(url: str, start_s: float) -> str:
    if not url:
        return url
    base = _strip_time_param(url)
    # Use integer seconds for stable URLs.
    sec = max(0, int(float(start_s)))
    join = "&" if "?" in base else "?"
    return f"{base}{join}t={sec}s"


def _match_preview(norm_text: str, pos: int, length: int, radius: int = 90) -> str:
    if not norm_text:
        return ""
    n = len(norm_text)
    if n <= 0:
        return ""
    a = max(0, min(n, int(pos) - radius))
    b = max(0, min(n, int(pos) + max(1, int(length)) + radius))
    snippet = norm_text[a:b].strip()
    if a > 0:
        snippet = "…" + snippet
    if b < n:
        snippet = snippet + "…"
    return snippet


def _first_match_info(raw_text: str, matcher: "KeywordMatcher") -> Optional[Dict[str, Any]]:
    raw = str(raw_text or "")
    norm = _normalize_match_text(raw)
    if not norm:
        return None

    pad = f" {norm} "
    # Prefer token matches when present.
    for tok in matcher.word_tokens:
        pat = f" {tok} "
        pos = pad.find(pat)
        if pos >= 0:
            # pos in pad includes leading space
            pos_norm = max(0, pos - 1)
            return {
                "kind": "token",
                "term_norm": tok,
                "pos": pos_norm,
                "denom": len(norm),
                "preview_norm": _match_preview(norm, pos_norm, len(tok)),
            }

    for phr in matcher.phrases_norm:
        pos = norm.find(phr)
        if pos >= 0:
            return {
                "kind": "phrase",
                "term_norm": phr,
                "pos": pos,
                "denom": len(norm),
                "preview_norm": _match_preview(norm, pos, len(phr)),
            }

    if matcher.phrases_compact:
        compact = norm.replace(" ", "")
        if compact:
            for phr in matcher.phrases_compact:
                pos = compact.find(phr)
                if pos >= 0:
                    return {
                        "kind": "phrase_compact",
                        "term_norm": phr,
                        "pos": pos,
                        "denom": len(compact),
                        "preview_norm": _match_preview(compact, pos, len(phr)),
                    }

    return None


def _looks_like_person_name(raw: str | None) -> bool:
    if not raw:
        return False
    s = str(raw).strip()
    if not s:
        return False
    if len(s) <= 3:
        return False
    if len(s) > 80:
        return False
    if _GENERIC_SPK_RE.fullmatch(s):
        return False
    if s.lower().startswith("speaker "):
        return False
    # Heuristic: names generally have a space; diarization labels usually don't.
    return " " in s


@dataclass(frozen=True)
class KeywordMatcher:
    phrases_norm: Tuple[str, ...]
    phrases_compact: Tuple[str, ...]
    word_tokens: Tuple[str, ...]

    def matches(self, text: str) -> bool:
        if not text:
            return False
        t = _normalize_match_text(text)
        if not t:
            return False

        # Whole-word token matches (preferred for single-token queries).
        if self.word_tokens:
            pad = f" {t} "
            for tok in self.word_tokens:
                if tok and f" {tok} " in pad:
                    return True

        # Phrase matches (space-normalized substring).
        for p in self.phrases_norm:
            if p and p in t:
                return True

        # Phrase matches with spaces removed. This catches common "token merge"
        # variants like "onchain" vs "on chain" (hyphen/punctuation differences).
        if self.phrases_compact:
            compact_text = t.replace(" ", "")
            if compact_text:
                for p in self.phrases_compact:
                    if p and p in compact_text:
                        return True

        return False


def build_keyword_matcher(
    *,
    query: str | None = None,
    phrases: Optional[Iterable[str]] = None,
) -> KeywordMatcher:
    ps: List[str] = []
    if phrases:
        ps.extend([str(p) for p in phrases if p is not None])
    if query:
        q = str(query).strip()
        if q:
            ps.append(q)

    # Normalize and keep unique in original order.
    seen = set()
    norm_phrases: List[str] = []
    for p in ps:
        n = _normalize_match_text(p)
        if not n:
            continue
        if n in seen:
            continue
        seen.add(n)
        norm_phrases.append(n)

    word_tokens: List[str] = []
    phrase_norm: List[str] = []
    for p in norm_phrases:
        toks = p.split()
        if len(toks) <= 1:
            # Single tokens should be matched as whole words to avoid substring noise
            # (e.g., "credit" should not match "accreditation").
            word_tokens.append(toks[0] if toks else p)
        else:
            phrase_norm.append(p)

    # Preserve order, unique tokens
    seen_tok = set()
    dedup_tokens: List[str] = []
    for t in word_tokens:
        if not t:
            continue
        if t in seen_tok:
            continue
        seen_tok.add(t)
        dedup_tokens.append(t)

    phrases_compact = tuple(p.replace(" ", "") for p in phrase_norm if p)
    return KeywordMatcher(
        phrases_norm=tuple(phrase_norm),
        phrases_compact=phrases_compact,
        word_tokens=tuple(dedup_tokens),
    )


def scan_keyword_clips_qdrant(
    *,
    query: str,
    phrases: Optional[Iterable[str]] = None,
    namespace: str,
    limit: int,
    offset: Optional[str] = None,
    channel_filter: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Scan Qdrant child nodes for keyword/phrase matches.

    This intentionally uses scroll + client-side matching so it works even when
    no text index exists. For large collections you can swap to Qdrant full-text
    indexes later, but local scale is fine.
    """
    if _backend() != "qdrant":
        raise RuntimeError("keyword clip scan is only supported for VECTOR_STORE=qdrant")

    q = (query or "").strip()
    if not q:
        return {
            "ok": True,
            "namespace": namespace,
            "query": query,
            "scanned": 0,
            "matched": 0,
            "results": [],
            "next_offset": None,
            "exhausted": True,
        }

    lim = max(0, int(limit))
    matcher = build_keyword_matcher(query=q, phrases=phrases)

    client = _qdrant_client()
    collection = _qdrant_collection_name(namespace)

    flt = qm.Filter(
        must=[
            qm.FieldCondition(key="node_type", match=qm.MatchValue(value="child")),
        ]
    )

    scanned = 0
    results: List[Dict[str, Any]] = []
    next_offset: Optional[str] = offset
    exhausted = False

    # If the caller requested a channel filter, child payloads may not have channel
    # fields. We evaluate filtering on parent payload fields (resolver) and keep a
    # small cache to avoid fetching parent rows repeatedly.
    parent_meta_cache: Dict[str, Dict[str, Any]] = {}
    pending_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    pending_parent_ids: List[str] = []

    def parent_passes(meta: Dict[str, Any]) -> bool:
        if not channel_filter:
            return True
        cid = meta.get("parent_channel_id")
        cname = (meta.get("parent_channel_name") or "")
        cname_low = str(cname).lower() if cname else ""

        include_ids = set(channel_filter.get("include_ids") or [])
        exclude_ids = set(channel_filter.get("exclude_ids") or [])
        include_names = {str(x).lower() for x in (channel_filter.get("include_names") or []) if x}
        exclude_names = {str(x).lower() for x in (channel_filter.get("exclude_names") or []) if x}

        if cid and cid in exclude_ids:
            return False
        if include_ids and (not cid or cid not in include_ids):
            return False
        if cname_low and cname_low in exclude_names:
            return False
        if include_names and (not cname_low or cname_low not in include_names):
            return False
        return True

    def flush_pending() -> None:
        nonlocal results, parent_meta_cache, pending_by_parent, pending_parent_ids
        if not pending_parent_ids:
            return
        # Fetch parent meta in batch.
        fetched = fetch_parent_meta(pending_parent_ids, namespace=namespace) or {}
        for pid in pending_parent_ids:
            parent_meta_cache[pid] = dict(fetched.get(pid) or {})
        for pid in pending_parent_ids:
            meta = parent_meta_cache.get(pid) or {}
            recs = pending_by_parent.pop(pid, [])
            if not recs:
                continue
            if parent_passes(meta):
                results.extend(recs)
        pending_parent_ids = []

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )
        if not points:
            exhausted = True
            next_offset = None
            break

        stop_now = False
        resume_offset: Optional[str] = None
        for idx, p in enumerate(points):
            scanned += 1
            payload = dict(getattr(p, "payload", None) or {})
            txt = payload.get("text") or ""
            if not matcher.matches(str(txt)):
                continue

            start_s = _to_float(payload.get("start_s"))
            if start_s is None:
                start_s = _to_float(payload.get("start_seconds"))
            if start_s is None:
                start_s = _hms_ms_to_seconds(payload.get("start_hms"))

            end_s = _to_float(payload.get("end_s"))
            if end_s is None:
                end_s = _hms_ms_to_seconds(payload.get("end_hms"))
            if end_s is None and start_s is not None:
                end_s = start_s + 1.0

            # Keep the payload minimal so we don't blow response sizes.
            # Caller can request parent meta separately.
            rec = {
                "segment_id": payload.get("segment_id") or str(getattr(p, "id", "")),
                "parent_id": payload.get("parent_id") or payload.get("video_id"),
                "video_id": payload.get("video_id") or payload.get("parent_id"),
                "document_type": payload.get("document_type"),
                "start_s": start_s,
                "end_s": end_s,
                "start_hms": payload.get("start_hms"),
                "end_hms": payload.get("end_hms"),
                "clip_url": payload.get("clip_url") or payload.get("url"),
                "speaker": payload.get("speaker"),
                "text": (str(txt)[:4000] if txt is not None else None),
                # Parsed numeric times (for micro-clip generation).
                "seg_start_s": start_s,
                "seg_end_s": end_s,
            }

            if not channel_filter:
                results.append(rec)
            else:
                pid = str(rec.get("parent_id") or "")
                if not pid:
                    continue
                if pid in parent_meta_cache:
                    if parent_passes(parent_meta_cache.get(pid) or {}):
                        results.append(rec)
                else:
                    pending_by_parent.setdefault(pid, []).append(rec)
                    if pid not in pending_parent_ids:
                        pending_parent_ids.append(pid)
                    if len(pending_parent_ids) >= 128:
                        flush_pending()

            if lim > 0 and len(results) >= lim:
                flush_pending()
                results = results[:lim]
                # IMPORTANT: Qdrant scroll offsets are inclusive. If we stop mid-batch and
                # return the server's `next_offset` (which points past the whole batch),
                # we'd skip unprocessed points. Instead, resume from the next point ID.
                if idx + 1 < len(points):
                    resume_offset = str(points[idx + 1].id)
                else:
                    resume_offset = next_offset
                stop_now = True
                break

        if stop_now:
            next_offset = resume_offset
            exhausted = next_offset is None
            break
        if next_offset is None:
            exhausted = True
            break

    flush_pending()

    # Parent metadata enrichment.
    parent_ids = sorted({str(r.get("parent_id") or "") for r in results if r.get("parent_id")})
    parent_meta_raw: Dict[str, Dict[str, Any]] = dict(parent_meta_cache)
    missing_parent_ids = [pid for pid in parent_ids if pid not in parent_meta_raw]
    if missing_parent_ids:
        fetched = fetch_parent_meta(missing_parent_ids, namespace=namespace) or {}
        for pid in missing_parent_ids:
            parent_meta_raw[pid] = dict(fetched.get(pid) or {})

    # Attach key parent fields so UI/scripts can render without extra calls.
    for r in results:
        pid = str(r.get("parent_id") or "")
        pm = parent_meta_raw.get(pid) or {}
        r.update(pm)

        # Best-effort identity attribution: prefer real person names, else fall back to channel.
        speaker_name = None
        raw_spk = r.get("speaker")
        if _looks_like_person_name(str(raw_spk) if raw_spk is not None else None):
            speaker_name = str(raw_spk).strip()
        else:
            # Prefer parent-level resolved speaker names when available.
            names = r.get("parent_speaker_names")
            candidates = []
            if isinstance(names, list):
                candidates = [str(x).strip() for x in names if _looks_like_person_name(str(x))]
                if candidates:
                    r["speaker_candidates"] = candidates
            if candidates:
                speaker_name = candidates[0]
            if not speaker_name:
                primary = r.get("parent_speaker_primary")
                if _looks_like_person_name(str(primary) if primary is not None else None):
                    speaker_name = str(primary).strip() or None
        if speaker_name:
            r["speaker_name"] = speaker_name

        # Stable grouping key for "different people": name if available, else channel, else raw label.
        identity = speaker_name or r.get("parent_channel_name") or raw_spk
        if identity:
            r["speaker_identity"] = str(identity).strip()
            if speaker_name:
                r["speaker_identity_type"] = "speaker_name"
            elif r.get("parent_channel_name"):
                r["speaker_identity_type"] = "channel"
            else:
                r["speaker_identity_type"] = "speaker_label"

        # Match info + micro-clip time estimate. This is a best-effort heuristic
        # (we only have segment-level timings, not word-level timestamps).
        mi = _first_match_info(str(r.get("text") or ""), matcher)
        if mi:
            r["match_kind"] = mi.get("kind")
            r["match_term_norm"] = mi.get("term_norm")
            r["match_preview_norm"] = mi.get("preview_norm")
            seg_a = _to_float(r.get("seg_start_s"))
            seg_b = _to_float(r.get("seg_end_s"))
            if seg_a is not None and seg_b is not None and seg_b > seg_a:
                dur = float(seg_b - seg_a)
                mid = float(mi.get("pos") or 0) + (len(str(mi.get("term_norm") or "")) / 2.0)
                denom = float(mi.get("denom") or 1) or 1.0
                frac = max(0.0, min(1.0, mid / denom))
                est = float(seg_a + frac * dur)
                r["match_estimated_s"] = est
                micro_len = float(os.getenv("KEYWORD_MICRO_CLIP_LEN_S", "1.0") or 1.0)
                micro_len = max(0.25, min(10.0, micro_len))
                micro_start = max(0.0, est - 0.25)
                micro_end = micro_start + micro_len
                r["micro_start_s"] = round(micro_start, 3)
                r["micro_end_s"] = round(micro_end, 3)
                base_url = str(r.get("parent_url") or r.get("url") or r.get("clip_url") or "")
                if base_url:
                    r["micro_clip_url"] = _add_time_param(base_url, micro_start)

    unique_parents = len({str(r.get("parent_id") or "") for r in results if r.get("parent_id")})
    unique_speakers = len({str(r.get("speaker_identity") or "").strip() for r in results if r.get("speaker_identity")})

    return {
        "ok": True,
        "namespace": namespace,
        "query": q,
        "match_phrases_norm": list(matcher.phrases_norm),
        "match_tokens_norm": list(matcher.word_tokens),
        "scanned": scanned,
        "matched": len(results),
        "unique_parents": unique_parents,
        "unique_speakers": unique_speakers,
        "results": results,
        "next_offset": next_offset,
        "exhausted": exhausted,
    }
