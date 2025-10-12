# File: src/rag_v2/logging_utils.py
from __future__ import annotations
import json
import logging
import os
import re
from typing import Any, Dict, Iterable

# Try to read central config; fall back to env-only if import order changes.
try:
    from .config import CFG  # type: ignore
except Exception:
    CFG = None  # type: ignore

# --- add near the top of logging_utils.py ---
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def _is_video(meta: dict) -> bool:
    dt = (meta.get("document_type") or "").lower()
    return "channel_name" in meta or dt in ("youtube_video", "stream")


def _hms_to_seconds(hms: str | None) -> int:
    """Accept HH:MM:SS or HH:MM:SS.mmm; return -1 on failure."""
    if not hms:
        return -1
    try:
        # strip fractional part if present
        if "." in hms:
            hms = hms.split(".", 1)[0]
        h, m, s = [int(x) for x in hms.split(":")]
        return h * 3600 + m * 60 + s
    except Exception:
        return -1

def _add_time_param(url: str, seconds: int) -> str:
    """Append/update t=Ns on the URL. Works for youtube.com and youtu.be."""
    try:
        seconds = max(0, int(seconds))
        u = urlparse(url)
        q = parse_qs(u.query)
        q["t"] = [f"{seconds}s"]
        new_q = urlencode(q, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))
    except Exception:
        sep = "&" if "?" in (url or "") else "?"
        return f"{url}{sep}t={max(0, int(seconds))}s"


def is_debug_enabled() -> bool:
    """
    Hard-disabled: we never emit deep JSON traces.
    Keeping the function for compatibility, but it always returns False.
    """
    return False


def setup_logger(name: str = "rag_v2") -> logging.Logger:
    """
    Human-friendly console logs at INFO only.
    - No file handlers
    - No DEBUG output
    - No propagation to root (prevents duplicate logs)
    """
    logger = logging.getLogger(name)

    # Ensure we don't inherit/root-log anything
    logger.propagate = False

    # Force INFO level globally for this logger
    logger.setLevel(logging.INFO)

    # Remove any pre-existing handlers (including old file handlers)
    for h in list(logger.handlers):
        try:
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        except Exception:
            pass

    # Console (human friendly)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(sh)

    return logger


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    try:
        return dict(o)
    except Exception:
        return str(o)


def node_brief(nws) -> Dict[str, Any]:
    md = (getattr(nws, "node", None) and getattr(nws.node, "metadata", None)) or {}
    text = (getattr(nws, "node", None) and getattr(nws.node, "get_content", None) and nws.node.get_content()) or ""
    return {
        "segment_id": md.get("segment_id") or md.get("id") or getattr(nws.node, "node_id", None),
        "parent_id": md.get("parent_id") or md.get("video_id"),
        "document_type": md.get("document_type"),
        "score": float(getattr(nws, "score", 0.0) or 0.0),
        "published_at": md.get("published_at") or md.get("published_date"),
        "is_explainer": bool(md.get("is_explainer") or False),
        "router_boost": md.get("router_boost"),
        "entities": md.get("entities"),
        "speaker": md.get("speaker"),
        "chapter": md.get("chapter"),
        "start_hms": md.get("start_hms"),
        "end_hms": md.get("end_hms"),
        "clip_url": md.get("clip_url") or md.get("url"),
        "title": md.get("title"),
        "channel_name": md.get("channel_name"),
        "channel_id": md.get("channel_id"),
        "parent_channel_name": md.get("parent_channel_name"),
        "parent_channel_id": md.get("parent_channel_id"),
        "text_preview": (text[:280] + "…") if text and len(text) > 280 else text,
    }


_CLEAN_PATTERNS: Iterable[re.Pattern] = [
    re.compile(r"\b(based on (the )?(provided )?(context|documents|sources))\b[:,]?\s*", re.I),
    re.compile(r"\b(according to (the )?(context|documents|sources))\b[:,]?\s*", re.I),
    re.compile(r"\b(as (seen|given|mentioned) (above|in the context))\b[:,]?\s*", re.I),
    re.compile(r"\b(from the retrieved (snippets|nodes|docs?|content))\b[:,]?\s*", re.I),
    re.compile(r"\b(the model|assistant) (found|retrieved|saw)\b[:,]?\s*", re.I),
]


_DATE_ID_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]{11}_")  # already present above
def scrub_link_titles(markdown: str) -> str:
    # Replace [YYYY-MM-DD_<id>_Title](url) -> [Title](url)
    def _fix(m):
        text, url = m.group(1), m.group(2)
        cleaned = _DATE_ID_PREFIX_RE.sub("", text).strip()
        return f"[{cleaned}]({url})"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _fix, markdown)

def clean_model_refs(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _CLEAN_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = scrub_link_titles(out)     # <--- add this line
    return out.strip()


# --- tiny timing + snapshots ---
import time
from dataclasses import asdict, is_dataclass


def time_block():
    start = time.perf_counter()

    def end():
        return round((time.perf_counter() - start) * 1000, 2)  # ms

    return end


def cfg_snapshot(cfg) -> dict:
    # turn the CFG dataclass into a plain dict safely
    try:
        return asdict(cfg) if is_dataclass(cfg) else {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
    except Exception:
        return {}


def model_snapshot():
    try:
        from llama_index.core import Settings
        llm = getattr(Settings, "llm", None)
        emb = getattr(Settings, "embed_model", None)
        return {
            "llm_model": getattr(llm, "model", None) or getattr(llm, "model_name", None),
            "embed_model": getattr(emb, "model", None) or getattr(emb, "model_name", None),
        }
    except Exception:
        return {}


# --- v1-style "Fetched based on..." formatting --------------------------------
from collections import defaultdict
from typing import List, Tuple


def _get_meta_and_score(nws) -> Tuple[dict, float]:
    """Works for LI legacy/core NodeWithScore shapes."""
    score = getattr(nws, "score", None)
    if hasattr(nws, "node") and getattr(nws, "node") is not None:
        meta = getattr(nws.node, "metadata", {}) or {}
    else:
        meta = getattr(nws, "metadata", {}) or {}
    return meta, float(score) if score is not None else None


def _format_timestamp_range(start_hms: str | None, end_hms: str | None) -> str:
    """Format timestamp range for display, e.g., '00:12:34–00:15:22'"""
    if start_hms and end_hms:
        return f"{start_hms}–{end_hms}"
    elif start_hms:
        return start_hms
    return ""

def _excerpt_edges(txt: str, n_words: int = 12) -> str:
    """
    Return the full excerpt that was fed to the LLM for this node.
    This affects display only; it does not change model inputs.
    """
    return txt or ""

def _node_meta_and_score(node_like):
    score = getattr(node_like, "score", None)
    if hasattr(node_like, "node") and getattr(node_like, "node") is not None:
        meta = getattr(node_like.node, "metadata", {}) or {}
        text = node_like.node.get_content() if hasattr(node_like.node, "get_content") else ""
    else:
        meta = getattr(node_like, "metadata", {}) or {}
        text = getattr(node_like, "text", "") or ""
    return meta, score, text


# --- date + title cleaning helpers --------------------------------------------
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Matches leading "YYYY-MM-DD_<11charID>_" prefix
_DATE_ID_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]{11}_")

def _infer_date(row_meta: dict) -> str | None:
    title = row_meta.get("title") or row_meta.get("parent_title") or ""
    m = _DATE_RE.search(title)
    return m.group(1) if m else None

def _clean_title(raw_title: str) -> str:
    """Strip 'YYYY-MM-DD_<videoid>_' prefix from titles if present."""
    if not raw_title:
        return raw_title
    cleaned = _DATE_ID_PREFIX_RE.sub("", raw_title).strip()
    return cleaned or raw_title


def format_metadata(response, source_nodes) -> str:
    lines = []
    if source_nodes is None:
        source_nodes = getattr(response, "source_nodes", []) or []

    rows = []
    for nws in source_nodes:
        meta, score, node_text = _node_meta_and_score(nws)

        # title (clean) + channel
        raw_title = meta.get("title") or meta.get("parent_title") or "N/A"
        title = _clean_title(raw_title)
        channel = meta.get("channel_name") or meta.get("parent_channel_name") or "N/A"

        # date with fallbacks (incl. parse from title)
        date = (
            meta.get("published_at")
            or meta.get("parent_published_at")
            or meta.get("published_date")
            or _infer_date(meta)
            or "N/A"
        )

        # build exact-start link (no -5s offset)
        base_url = meta.get("clip_url") or meta.get("url") or ""
        start_hms = meta.get("start_hms")
        end_hms = meta.get("end_hms")
        link_url = base_url
        if _is_video(meta):
            start_s = _hms_to_seconds(start_hms)
            if start_s >= 0:
                link_url = _add_time_param(base_url, start_s)

        excerpt_range = _format_timestamp_range(start_hms, end_hms)
        excerpt_edges = _excerpt_edges(node_text or "")

        rows.append({
            "title": title,
            "speaker": meta.get("speaker"),
            "channel": channel,
            "channel_id": meta.get("channel_id") or meta.get("parent_channel_id"),
            "url": link_url or "N/A",
            "date": date,
            "score": float(score) if score is not None else 0.0,
            "excerpt_range": excerpt_range,
            "excerpt_edges": excerpt_edges,
            "is_video": _is_video(meta),
            "authors": meta.get("authors"),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)

    for r in rows:
        if r["is_video"]:
            # Markdown link + timestamp range right after the title
            head = f"[Title]: {r['title']}"
            if r["excerpt_range"]:
                head += f" ({r['excerpt_range']})"
            parts = [
                head,
                *( [f"[Speaker]: {r['speaker']}"] if r["speaker"] else [] ),
                f"[Channel]: {r['channel']}",
                *( [f"[Channel ID]: {r['channel_id']}"] if r["channel_id"] else [] ),
                f"[Date]: {r['date']}",
                f"[Score]: {r['score']:.4f}",
            ]
        else:
            formatted_authors = None
            if r["authors"]:
                try:
                    formatted_authors = ", ".join(str(r["authors"]).split(", "))
                except Exception:
                    formatted_authors = str(r["authors"])
            parts = [
                f"[Title]: {r['title']}",
                f"[Authors]: {formatted_authors or 'N/A'}",
                f"[Date]: {r['date']}",
                f"[Score]: {r['score']:.4f}",
            ]

        if r["excerpt_range"] and r["excerpt_edges"]:
            parts.append(f"[Excerpt]: {r['excerpt_edges']}")

        lines.append(", ".join(parts))

    return "\n".join(lines)


def append_sources_block(text: str, source_nodes: List) -> str:
    suffix = format_metadata(response=None, source_nodes=source_nodes)
    if not suffix:
        return text
    return f"{text}\n\n Fetched based on the following sources: \n{suffix}"
