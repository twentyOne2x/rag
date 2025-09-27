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
        "text_preview": (text[:280] + "…") if text and len(text) > 280 else text,
    }

_CLEAN_PATTERNS: Iterable[re.Pattern] = [
    re.compile(r"\b(based on (the )?(provided )?(context|documents|sources))\b[:,]?\s*", re.I),
    re.compile(r"\b(according to (the )?(context|documents|sources))\b[:,]?\s*", re.I),
    re.compile(r"\b(as (seen|given|mentioned) (above|in the context))\b[:,]?\s*", re.I),
    re.compile(r"\b(from the retrieved (snippets|nodes|docs?|content))\b[:,]?\s*", re.I),
    re.compile(r"\b(the model|assistant) (found|retrieved|saw)\b[:,]?\s*", re.I),
]

def clean_model_refs(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _CLEAN_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
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

def format_sources_v1_style(source_nodes: List) -> str:
    """
    Group by title and render lines exactly like your v1 examples:
    [Title]: ..., [Channel name]: ... / [Authors]: ..., [url]/[Link]: ..., [Release date]: ..., [Highest Score]: ...
    """
    by_title = defaultdict(lambda: {
        "pdf_link": "N/A",
        "url": "N/A",
        "release_date": "N/A",
        "channel_name": "N/A",
        "authors": None,
        "highest_score": None,
        "is_video": False,
        "chunks": 0,
    })

    for nws in source_nodes or []:
        meta, score = _get_meta_and_score(nws)
        title = meta.get("title") or "N/A"
        item = by_title[title]
        # detect video vs doc
        is_video = "channel_name" in meta or meta.get("document_type") in ("youtube_video", "stream")
        item["is_video"] = item["is_video"] or is_video

        # common fields
        item["url"] = meta.get("url") or meta.get("clip_url") or item["url"]
        item["pdf_link"] = meta.get("pdf_link") or item["pdf_link"]
        # keep either 'published_at' or 'published_date'
        item["release_date"] = meta.get("published_at") or meta.get("published_date") or item["release_date"]
        item["channel_name"] = meta.get("channel_name") or item["channel_name"]

        # authors for docs (optional)
        if not is_video and meta.get("authors"):
            # accept list or comma-separated string
            authors = meta["authors"]
            if isinstance(authors, (list, tuple)):
                item["authors"] = ", ".join(map(str, authors))
            else:
                item["authors"] = str(authors)

        # score / chunks
        item["chunks"] += 1
        try:
            if score is not None and (item["highest_score"] is None or score > item["highest_score"]):
                item["highest_score"] = score
        except Exception:
            pass

    # Sort primarily by highest_score (desc), secondarily by chunks (desc)
    def _sort_key(tup):
        meta = tup[1]
        hs = meta["highest_score"]
        return (hs if hs is not None else -1, meta["chunks"])

    lines = []
    for title, meta in sorted(by_title.items(), key=_sort_key, reverse=True):
        if meta["is_video"]:
            lines.append(
                f"[Title]: {title}, [Channel name]: {meta['channel_name']}, "
                f"[url]: {meta['url']}, [Release date]: {meta['release_date']}, "
                f"[Highest Score]: {meta['highest_score']}"
            )
        else:
            # Non-video (papers/posts) variant
            lines.append(
                f"[Title]: {title}, [Authors]: {meta['authors']}, "
                f"[Link]: {meta['pdf_link'] or meta['url']}, [Release date]: {meta['release_date']}, "
                f"[Highest Score]: {meta['highest_score']}"
            )
    return "\n".join(lines)

def append_sources_block(text: str, source_nodes: List) -> str:
    """Append the v1-looking sources section to the answer text."""
    suffix = format_sources_v1_style(source_nodes)
    if not suffix:
        return text
    # match your spacing exactly:
    return f"{text}\n\n Fetched based on the following sources: \n{suffix}"
