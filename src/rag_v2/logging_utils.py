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


def _format_timestamp_range(start_hms: str, end_hms: str) -> str:
    """Format timestamp range for display, e.g., '00:12:34–00:15:22'"""
    if start_hms and end_hms:
        return f"{start_hms}–{end_hms}"
    elif start_hms:
        return start_hms
    return ""


def format_sources_v1_style(source_nodes: List) -> str:
    """
    Group by (title, clip_url) to show individual segments with their timestamps.
    Format: [Title], [Timestamp], [Speaker], [Chapter], [Channel], [clip_url], [Date], [Score]
    """
    # Group by (title, clip_url) so each unique segment gets its own line
    segments = []

    for nws in source_nodes or []:
        meta, score = _get_meta_and_score(nws)

        title = meta.get("title") or "N/A"
        clip_url = meta.get("clip_url") or meta.get("url") or "N/A"
        start_hms = meta.get("start_hms") or ""
        end_hms = meta.get("end_hms") or ""
        speaker = meta.get("speaker") or ""
        chapter = meta.get("chapter") or ""
        channel_name = meta.get("channel_name") or "N/A"
        release_date = meta.get("published_at") or meta.get("published_date") or "N/A"

        is_video = "channel_name" in meta or meta.get("document_type") in ("youtube_video", "stream")

        timestamp_str = _format_timestamp_range(start_hms, end_hms)

        segments.append({
            "title": title,
            "timestamp": timestamp_str,
            "speaker": speaker,
            "chapter": chapter,
            "channel_name": channel_name,
            "clip_url": clip_url,
            "release_date": release_date,
            "score": score if score is not None else 0.0,
            "is_video": is_video,
            "authors": meta.get("authors") if not is_video else None,
            "pdf_link": meta.get("pdf_link") if not is_video else None,
        })

    # Sort by score descending
    segments.sort(key=lambda x: x["score"], reverse=True)

    lines = []
    for seg in segments:
        if seg["is_video"]:
            # Video format with timestamp
            parts = [f"[Title]: {seg['title']}"]

            if seg["timestamp"]:
                parts.append(f"[Timestamp]: {seg['timestamp']}")

            if seg["speaker"]:
                parts.append(f"[Speaker]: {seg['speaker']}")

            if seg["chapter"]:
                parts.append(f"[Chapter]: {seg['chapter']}")

            parts.extend([
                f"[Channel]: {seg['channel_name']}",
                f"[Clip URL]: {seg['clip_url']}",
                f"[Date]: {seg['release_date']}",
                f"[Score]: {seg['score']:.4f}"
            ])

            lines.append(", ".join(parts))
        else:
            # Non-video (papers/posts) format
            parts = [
                f"[Title]: {seg['title']}",
                f"[Authors]: {seg['authors'] or 'N/A'}",
                f"[Link]: {seg['pdf_link'] or seg['clip_url']}",
                f"[Date]: {seg['release_date']}",
                f"[Score]: {seg['score']:.4f}"
            ]
            lines.append(", ".join(parts))

    return "\n".join(lines)


def append_sources_block(text: str, source_nodes: List) -> str:
    """Append the timestamp-aware sources section to the answer text."""
    suffix = format_sources_v1_style(source_nodes)
    if not suffix:
        return text
    return f"{text}\n\n Fetched based on the following sources: \n{suffix}"