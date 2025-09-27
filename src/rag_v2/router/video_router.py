from __future__ import annotations
import re
from typing import Dict, Any

ROUTER_RE = re.compile(r"^(what is|who is|how does|why\\b)", re.I)
TICKER_RE = re.compile(r"(?:^|\\s)\\$[A-Z0-9]{2,6}\\b")
HANDLE_RE = re.compile(r"(?<!\\w)@[A-Za-z0-9_]{2,30}\\b")

def wants_definition(query: str) -> bool:
    q = query.strip()
    if ROUTER_RE.search(q):
        return True
    if TICKER_RE.search(q) or HANDLE_RE.search(q):
        return True
    return False

def router_bias(parent: Dict[str, Any]) -> float:
    base = 1.0
    if parent.get("is_explainer"):
        base *= 1.25
    rb = parent.get("router_boost")
    if isinstance(rb, (int,float)) and rb > 0:
        base *= float(rb) / 1.0
    return base
