from __future__ import annotations
from typing import List, Tuple
import hashlib

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # optional dependency

class CEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", batch_size: int = 32):
        self.enabled = CrossEncoder is not None
        self.batch = batch_size
        self.model = CrossEncoder(model_name) if self.enabled else None
        self._cache = {}

    def _h(self, q: str, sid: str) -> str:
        return hashlib.sha1(f"{q}::{sid}".encode("utf-8")).hexdigest()

    def rerank(self, query: str, items: List[Tuple[str, str, float]]) -> List[Tuple[str,str,float]]:
        """
        items: list of (segment_id, text, stage1_score)
        returns same list re-scored by CE (descending)
        """
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
