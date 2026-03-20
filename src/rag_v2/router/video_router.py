# --- DROP-IN: entity-aware routing score ---
from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Set

# re-use your existing patterns/helpers
ROUTER_RE = re.compile(
    r"^(what is|what does|who is|what['’]s|who['’]s|how does|why\b)",
    re.I,
)
TICKER_RE = re.compile(r"(?:^|\s)\$[A-Z0-9]{2,6}\b")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,30}\b")
ACRONYM_ONLY_RE = re.compile(r"^[A-Z][A-Z0-9]{1,15}$")
DEF_FUZZ_RE = re.compile(r"\b(define|meaning of|stands for)\b", re.I)

_STOP = {"the","a","an","to","of","and","for","in","on","with","about","vs","from","is","are","how","what","who","why"}

def wants_definition(query: str) -> bool:
    raw = (query or "").strip()
    q = ""
    for line in raw.splitlines():
        clean = line.strip()
        if clean:
            q = clean
            break
    if not q:
        q = raw
    q = re.sub(r"[?!.]+$", "", q).strip()
    if ROUTER_RE.search(q):
        return True
    if TICKER_RE.search(q) or HANDLE_RE.search(q):
        return True
    if ACRONYM_ONLY_RE.fullmatch(q):
        return True
    if DEF_FUZZ_RE.search(q):
        return True
    return False

def router_bias(parent: Dict[str, Any]) -> float:
    base = 1.0
    if parent.get("is_explainer"):
        base *= 1.25
    rb = parent.get("router_boost")
    if isinstance(rb, (int, float)) and rb > 0:
        base *= float(rb)
    return base

def _canon_key(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith("$"):
        return s.upper()
    if s.startswith("@"):
        return s.lower()
    return s.lower()

def _query_entities(query: str) -> Set[str]:
    ents = set(m.group(0).strip() for m in TICKER_RE.finditer(query))
    ents |= set(m.group(0).strip() for m in HANDLE_RE.finditer(query))
    # canon
    return {_canon_key(e) for e in ents}

def _tokens(s: str) -> Set[str]:
    return {w for w in re.findall(r"[A-Za-z0-9_]{3,}", (s or "").lower()) if w not in _STOP}

def score_parent_for_router(parent: Dict[str, Any], query: str) -> Tuple[float, Dict[str, Any]]:
    """Heuristic router score using entities/tags + description/summary. Returns (score, debug)."""
    qents = _query_entities(query)
    qtoks = _tokens(query)

    aliases = parent.get("aliases") or []
    cents   = parent.get("canonical_entities") or []
    tags    = parent.get("router_tags") or []
    desc    = parent.get("topic_summary") or parent.get("description") or parent.get("title") or ""

    route_keys = {_canon_key(x) for x in (aliases + cents)}
    tag_keys   = {str(t).lower() for t in tags}
    d_toks     = _tokens(desc)

    ent_overlap  = len(route_keys & qents)
    tag_overlap  = len(tag_keys & qtoks)
    desc_overlap = len(d_toks & qtoks)

    # weighted sum -> multiplicative router bias
    s = 0.0
    s += 3.0 * min(ent_overlap, 3)      # entities dominate
    s += 1.0 * min(tag_overlap, 3)      # tags help
    s += 0.5 * min(desc_overlap, 5)     # description/summary nudges
    if wants_definition(query) and parent.get("is_explainer"):
        s *= 1.10

    s *= max(0.25, router_bias(parent))  # keep bounded if weird

    dbg = {
        "qents": sorted(qents),
        "ent_overlap": ent_overlap,
        "tag_overlap": tag_overlap,
        "desc_overlap": desc_overlap,
        "router_bias": router_bias(parent),
        "is_explainer": bool(parent.get("is_explainer")),
        "raw_desc_used": bool(desc),
    }
    return float(s), dbg
