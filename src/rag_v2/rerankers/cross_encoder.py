# --- DROP-IN: CE rerank with recency + stale-phrase penalty -------------------
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib
import re
from datetime import datetime

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # optional

from ..config import CFG
from ..utils.scoring import recency_decay
from ..router.video_router import wants_definition  # reuse signal


# stale / outdated phrasing we want to downweight if the content is old
STALE_PHRASE_RE = re.compile(
    r"\b("
    r"under (active )?development|coming soon|plan to|planned|roadmap|"
    r"prototype|alpha\b|beta\b|preview|we will|announc(?:e|ing|ement)"
    r")\b",
    re.I,
)


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


def _age_days(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.utcnow() - dt).days
    except Exception:
        return None


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
                             "router_boost": float, "is_explainer": bool,
                             "published_at": "YYYY-MM-DD" | "published_date": ...}}
        """
        if not self.enabled or not items:
            return items

        base = self.rerank(query, items)
        qents = _qents(query)
        wants_def = wants_definition(query)

        out: List[Tuple[str, str, float]] = []
        for sid, text, ce in base:
            md = metas.get(sid, {}) or {}
            ents = {_norm_ent(str(x)) for x in (md.get("entities") or [])}
            ents |= {_norm_ent(str(x)) for x in (md.get("canonical_entities") or [])}
            overlap = len(ents & qents)

            s = float(ce)

            # ---- entity overlap bonus
            s *= (1.0 + entity_gain * min(overlap, 3))

            # ---- explainer/definition nudge
            if wants_def and (md.get("is_explainer") or md.get("node_type") == "summary"):
                s *= explainer_bonus

            # ---- router boost passthrough (if any)
            rb = md.get("router_boost")
            if isinstance(rb, (int, float)) and rb > 0:
                s *= float(rb)

            # ---- NEW: recency penalty directly in CE layer
            pub = md.get("published_at") or md.get("published_date")
            if pub:
                decay = recency_decay(pub, CFG.recency_half_life_days)  # 0..1
                # let callers tune how strong CE should weight recency
                s *= decay ** float(getattr(CFG, "ce_recency_weight", 1.25))

                # ---- NEW: stale-phrase penalty if OLD + wording sounds outdated
                age = _age_days(pub)
                if age is not None and age >= int(getattr(CFG, "stale_phrase_age_days", 365)):
                    if STALE_PHRASE_RE.search(text or ""):
                        s *= float(getattr(CFG, "stale_phrase_penalty_mult", 0.82))

            out.append((sid, text, s))

        out.sort(key=lambda x: x[2], reverse=True)
        return out
