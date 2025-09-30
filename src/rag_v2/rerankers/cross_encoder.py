# --- DROP-IN: CE rerank that also uses entities/flags from metadata ---
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib
import re

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # optional


def _wants_def(q: str) -> bool:
    from ..router.video_router import wants_definition
    return wants_definition(q)


def _norm_ent(s: str) -> str:
    s = s.strip()
    return s.upper() if s.startswith("$") else s.lower()


def _qents(q: str):
    from ..router.video_router import TICKER_RE, HANDLE_RE
    from ..postprocessors.entity_utils import canon_entity

    ents = set(m.group(0).strip() for m in TICKER_RE.finditer(q))
    ents |= set(m.group(0).strip() for m in HANDLE_RE.finditer(q))
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", q):
        ce = canon_entity(tok)
        if ce:
            ents.add(ce)
    return {_norm_ent(e) for e in ents}


class CEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", batch_size: int = 32):
        self.enabled = CrossEncoder is not None
        self.batch = batch_size
        self.model = CrossEncoder(model_name) if self.enabled else None
        self._cache: Dict[str, float] = {}

    def _h(self, q: str, sid: str) -> str:
        return hashlib.sha1(f"{q}::{sid}".encode("utf-8")).hexdigest()

    def rerank(self, query: str, items: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        if not self.enabled or not items:
            return items
        pairs, keys = [], []
        for sid, text, _ in items:
            k = self._h(query, sid)
            keys.append(k)
            if k in self._cache:
                continue
            pairs.append((query, text))
        if pairs:
            scores = self.model.predict(pairs, batch_size=self.batch)
            for k, sc in zip([k for k in keys if k not in self._cache], scores):
                self._cache[k] = float(sc)
        rescored = []
        for (sid, text, _old) in items:
            k = self._h(query, sid)
            ce = self._cache.get(k, _old)
            rescored.append((sid, text, float(ce)))
        rescored.sort(key=lambda x: x[2], reverse=True)
        return rescored

    def rerank_with_meta(
        self,
        query: str,
        items: List[Tuple[str, str, float]],
        metas: Dict[str, Dict[str, Any]],
        entity_gain: float = 0.12,
        explainer_bonus: float = 1.05,
    ) -> List[Tuple[str, str, float]]:
        """
        items: [(segment_id, text, stage1_score)]
        metas: {segment_id: {"entities":[...], "canonical_entities":[...], "node_type": "...",
                             "router_boost": float, "is_explainer": bool}}
        """
        if not self.enabled or not items:
            return items

        base = self.rerank(query, items)
        qents = _qents(query)
        wants_def = _wants_def(query)

        out: List[Tuple[str, str, float]] = []
        for sid, text, ce in base:
            md = metas.get(sid, {}) or {}
            ents = {_norm_ent(str(x)) for x in (md.get("entities") or [])}
            ents |= {_norm_ent(str(x)) for x in (md.get("canonical_entities") or [])}
            overlap = len(ents & qents)

            s = float(ce)
            s *= (1.0 + entity_gain * min(overlap, 3))
            if wants_def and (md.get("is_explainer") or md.get("node_type") == "summary"):
                s *= explainer_bonus

            rb = md.get("router_boost")
            if isinstance(rb, (int, float)) and rb > 0:
                s *= float(rb)

            out.append((sid, text, s))

        out.sort(key=lambda x: x[2], reverse=True)
        return out
