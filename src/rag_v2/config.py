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
