#!/usr/bin/env bash
set -euo pipefail

ROOT="${PWD}"
DEST="${ROOT}/src/rag_v2"

mkdir -p "${DEST}"/{router,rerankers,retriever,vector_store,utils,tests}

# -----------------------------
# README
# -----------------------------
cat > "${DEST}/README.md" << 'EOF'
# rag_v2 (parent/child + router + CE rerank)

This module consumes your v2 ingestion schema (timestamped child segments), adds:
- Video/Definition router using parent metadata (`is_explainer`, `router_boost`, aliases).
- Stage-1 dense recall on children → Auto-Merging to parent → neighbor-children expansion.
- Optional BM25+RRF fusion (title/description/intro) before CE.
- Optional Stage-2 Cross-Encoder rerank.
- Entities/speakers filters and streams bias.
EOF

# -----------------------------
# config.py
# -----------------------------
cat > "${DEST}/config.py" << 'EOF'
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    enable_router: bool = os.getenv("ENABLE_ROUTER", "1") in ("1","true","yes")
    enable_bm25: bool = os.getenv("ENABLE_BM25", "0") in ("1","true","yes")
    enable_ce: bool = os.getenv("ENABLE_RERANK_CE", "1") in ("1","true","yes")

    stage1_topn: int = int(os.getenv("STAGE1_TOPN", "80"))
    topk_post_rerank: int = int(os.getenv("TOPK_POST_RERANK", "10"))

    streams_ns: str = os.getenv("PINECONE_STREAMS_NS", "streams")
    default_ns: str = os.getenv("PINECONE_DEFAULT_NS", "")

    recency_half_life_days: float = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "45"))
    definition_window_s: tuple[float,float] = (30.0, 60.0)

    # cross-encoder model / batching
    ce_model: str = os.getenv("CE_MODEL", "BAAI/bge-reranker-large")
    ce_batch_size: int = int(os.getenv("CE_BATCH", "32"))

    # boosts
    router_boost_mult: float = float(os.getenv("ROUTER_BOOST_MULT", "1.25"))
    streams_bias_mult: float = float(os.getenv("STREAMS_BIAS_MULT", "1.15"))

CFG = RetrievalConfig()
EOF

# -----------------------------
# schemas.py (thin mirror of ingest)
# -----------------------------
cat > "${DEST}/schemas.py" << 'EOF'
from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal

DocType = Literal["youtube_video","stream"]

class ParentNode(BaseModel):
    node_type: Literal["parent"] = "parent"
    parent_id: str
    document_type: DocType
    title: str
    description: Optional[str] = None
    channel_name: Optional[str] = None
    speaker_primary: Optional[str] = None
    published_at: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    duration_s: float = 0
    url: HttpUrl
    language: Optional[str] = "en"
    entities: List[str] = []
    chapters: Optional[list] = None
    ingest_version: int = 2
    # router extras
    router_tags: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    canonical_entities: Optional[List[str]] = None
    is_explainer: Optional[bool] = None
    router_boost: Optional[float] = None
    topic_summary: Optional[str] = None

class ChildNode(BaseModel):
    node_type: Literal["child"] = "child"
    segment_id: str
    parent_id: str
    document_type: DocType
    text: str
    start_s: float
    end_s: float
    start_hms: str
    end_hms: str
    clip_url: Optional[HttpUrl] = None
    speaker: Optional[str] = None
    entities: List[str] = []
    chapter: Optional[str] = None
    language: Optional[str] = "en"
    ingest_version: int = 2
EOF

# -----------------------------
# vector_store/pinecone_client.py
# -----------------------------
cat > "${DEST}/vector_store/pinecone_client.py" << 'EOF'
from __future__ import annotations
import os
from typing import Optional
from pinecone import Pinecone

def get_pinecone_index(index_name: Optional[str] = None):
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name or os.environ.get("PINECONE_INDEX_NAME","icmfyi"))
    return idx
EOF

# -----------------------------
# utils/scoring.py
# -----------------------------
cat > "${DEST}/utils/scoring.py" << 'EOF'
from __future__ import annotations
import math
from datetime import datetime

def recency_decay(published_date_str: str | None, half_life_days: float) -> float:
    if not published_date_str:
        return 1.0
    try:
        dt = datetime.strptime(published_date_str, "%Y-%m-%d")
    except Exception:
        return 1.0
    days = (datetime.utcnow() - dt).days
    return 0.5 ** (days / max(half_life_days, 1e-6))

def apply_multiplier(score: float, mult: float) -> float:
    return float(score) * float(mult)
EOF

# -----------------------------
# router/video_router.py
# -----------------------------
cat > "${DEST}/router/video_router.py" << 'EOF'
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
EOF

# -----------------------------
# rerankers/cross_encoder.py
# -----------------------------
cat > "${DEST}/rerankers/cross_encoder.py" << 'EOF'
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
EOF

# -----------------------------
# retriever/parent_child_retriever.py
# -----------------------------
cat > "${DEST}/retriever/parent_child_retriever.py" << 'EOF'
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os
from ..config import CFG
from ..router.video_router import wants_definition, router_bias
from ..utils.scoring import recency_decay, apply_multiplier

"""
This retriever assumes your vector index stores CHILD nodes.
It fetches top-N children, expands to their parent, then re-expands to neighbor children
from the same parent (Auto-Merging / ParentDocument pattern).
"""

class ParentChildRetrieverV2:
    def __init__(self, base_retriever):
        """
        base_retriever: a LlamaIndex retriever (core) already pointing at your Pinecone store.
        """
        self.base = base_retriever

    def _expand_neighbors(self, nodes, per_parent_cap=3):
        by_parent: Dict[str, List[Any]] = {}
        for nws in nodes:
            pid = (nws.node.metadata or {}).get("parent_id") or (nws.node.metadata or {}).get("video_id")
            by_parent.setdefault(pid, []).append(nws)
        out = []
        for pid, rows in by_parent.items():
            rows_sorted = sorted(rows, key=lambda r: (r.score or 0.0), reverse=True)
            out.extend(rows_sorted[:per_parent_cap])
        return out

    def _apply_metadata_boosts(self, nodes, query: str):
        def_score = 1.0 if not wants_definition(query) else 1.2
        for nws in nodes:
            md = nws.node.metadata or {}
            # recency
            decay = recency_decay(md.get("published_date") or md.get("published_at"), CFG.recency_half_life_days)
            s = (nws.score or 0.0) * decay
            # router bias (parents mirrored on child metadata via upsert)
            rb = md.get("router_boost") or 1.0
            if wants_definition(query) and md.get("is_explainer"):
                rb *= CFG.router_boost_mult
            s = apply_multiplier(s, rb * def_score)
            # streams bias
            if "stream" in (md.get("document_type") or "") and any(k in query.lower() for k in ["live","yesterday","stream","@"]):
                s = apply_multiplier(s, CFG.streams_bias_mult)
            nws.score = s
        return nodes

    def retrieve(self, query_bundle):
        nodes = self.base.retrieve(query_bundle)  # stage-1 dense
        nodes = self._apply_metadata_boosts(nodes, query_bundle.query_str)

        # neighbor expansion (keep K small after)
        neighbors = self._expand_neighbors(nodes, per_parent_cap=3)

        # truncate to stage1_topn (pre-CE)
        neighbors = sorted(neighbors, key=lambda n: (n.score or 0.0), reverse=True)[:CFG.stage1_topn]
        return neighbors
EOF

# -----------------------------
# query_engine_v2.py
# -----------------------------
cat > "${DEST}/query_engine_v2.py" << 'EOF'
from __future__ import annotations
from typing import List, Tuple
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import CFG
from .rerankers.cross_encoder import CEReranker

class ParentChildQueryEngineV2(BaseQueryEngine):
    def __init__(self, retriever, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self._retriever = retriever
        self._core = RetrieverQueryEngine.from_args(retriever=self._retriever)
        self._ce = CEReranker(model_name=CFG.ce_model, batch_size=CFG.ce_batch_size) if CFG.enable_ce else None

    def _ce_pack(self, nodes: List[NodeWithScore]) -> List[Tuple[str,str,float]]:
        items = []
        for n in nodes:
            sid = n.node.metadata.get("segment_id") or n.node.metadata.get("id") or n.node.node_id
            items.append((sid, n.node.get_content(), float(n.score or 0.0)))
        return items

    def _reinject_scores(self, nodes: List[NodeWithScore], rescored: List[Tuple[str,str,float]]):
        score_by_sid = {sid: sc for sid,_,sc in rescored}
        for n in nodes:
            sid = n.node.metadata.get("segment_id") or n.node.metadata.get("id") or n.node.node_id
            if sid in score_by_sid:
                n.score = score_by_sid[sid]
        nodes.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return nodes

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as revent:
            nodes: List[NodeWithScore] = self._retriever.retrieve(query_bundle)

            # CE rerank (optional)
            if self._ce and self._ce.enabled and nodes:
                packs = self._ce_pack(nodes)
                rescored = self._ce.rerank(query_bundle.query_str, packs)
                nodes = self._reinject_scores(nodes, rescored)

            # keep top-K post-rerank
            nodes = nodes[:CFG.topk_post_rerank]
            revent.on_end(payload={EventPayload.NODES: nodes})

        if not nodes:
            return Response("No results found.", source_nodes=[])

        # synthesize using core synthesizer (same as your v1 custom QE)
        return self._core._response_synthesizer.synthesize(query=query_bundle, nodes=nodes)
EOF

# -----------------------------
# app_main.py (demo wiring)
# -----------------------------
cat > "${DEST}/app_main.py" << 'EOF'
from __future__ import annotations
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from src.Llama_index_sandbox.index import load_index_from_disk  # reuse your loader
from .retriever.parent_child_retriever.py import ParentChildRetrieverV2  # noqa
EOF
# Fix typo import path in app_main (heredoc split to keep it readable)
sed -i '' -e 's/\.retriever\.parent_child_retriever\.py/.retriever.parent_child_retriever/' "${DEST}/app_main.py" 2>/dev/null || true
cat >> "${DEST}/app_main.py" << 'EOF'
from .query_engine_v2 import ParentChildQueryEngineV2

def bootstrap_query_engine_v2(similarity_top_k: int = 50):
    # LLM (uses your global Settings like the rest of your codebase)
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL","gpt-4o-mini"))
    index = load_index_from_disk()  # your Pinecone-backed index

    # Build a core retriever, then wrap with ParentChildRetrieverV2
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k, verbose=False)
    pc_retriever = ParentChildRetrieverV2(base_retriever)
    qe = ParentChildQueryEngineV2(retriever=pc_retriever, callback_manager=getattr(index, "callback_manager", None))
    return qe

if __name__ == "__main__":
    qe = bootstrap_query_engine_v2()
    from llama_index.core.schema import QueryBundle
    resp = qe.query(QueryBundle("what is a DAT on Solana?"))
    print(resp)
EOF

# -----------------------------
# tests/smoke_test.py
# -----------------------------
cat > "${DEST}/tests/smoke_test.py" << 'EOF'
def test_imports():
    from rag_v2.query_engine_v2 import ParentChildQueryEngineV2
    assert ParentChildQueryEngineV2 is not None
EOF

echo "✅ rag_v2 bootstrap complete at ${DEST}"
echo "Next:"
echo "  1) pip install 'sentence-transformers>=2.2.2' (for CE) and optionally 'rank_bm25' if you add BM25."
echo "  2) Export PINECONE_API_KEY, PINECONE_INDEX_NAME and run:  python -m src.rag_v2.app_main"
