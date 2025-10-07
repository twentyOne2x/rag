from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional  # <-- add

@dataclass
class RetrievalConfig:
    # --------- DEBUG / LOGGING ---------
    # Hardcode default here; env var can override.
    rag_debug: bool = os.getenv("RAG_DEBUG", "false").lower() in ("1", "true", "yes", "on")
    # If set (path or filename), logs also go to a file in JSON (one trace per line).
    debug_log_path: Optional[str] = os.getenv("RAG_DEBUG_FILE") or None

    # --------- FEATURES ---------
    enable_router: bool = os.getenv("ENABLE_ROUTER", "1") in ("1","true","yes")
    enable_bm25: bool = os.getenv("ENABLE_BM25", "0") in ("1","true","yes")
    enable_ce: bool = os.getenv("ENABLE_RERANK_CE", "1") in ("1","true","yes")

    stage1_topn: int = int(os.getenv("STAGE1_TOPN", "240"))     # was 80
    topk_post_rerank: int = int(os.getenv("TOPK_POST_RERANK", "40"))

    # CE keep policy (relative + absolute)
    ce_keep_percentile: float = float(os.getenv("CE_KEEP_PERCENTILE", "0.85"))
    ce_min_keep: int = int(os.getenv("CE_MIN_KEEP", "20"))
    ce_abs_min: float = float(os.getenv("CE_ABS_MIN", "0.32"))  # after sigmoid

    # per-parent diversity + entity nudges
    max_segments_per_parent: int = int(os.getenv("MAX_SEGMENTS_PER_PARENT", "8"))
    entity_filter_boost: float = float(os.getenv("ENTITY_FILTER_BOOST", "1.5"))
    entity_overlap_gain: float = float(os.getenv("ENTITY_OVERLAP_GAIN", "0.2"))

    # stitching (merge adjacent child clips from same parent)
    stitch_gap_seconds: int = int(os.getenv("STITCH_GAP_SECONDS", "12"))
    stitch_target_tokens: int = int(os.getenv("STITCH_TARGET_TOKENS", "360"))
    stitch_max_merge: int = int(os.getenv("STITCH_MAX_MERGE", "3"))

    # output quoting
    quote_min_count: int = int(os.getenv("QUOTE_MIN_COUNT", "4"))
    max_final_nodes: int = int(os.getenv("MAX_FINAL_NODES", "12"))\

    streams_ns: str = os.getenv("PINECONE_STREAMS_NS", "streams")
    default_ns: str = os.getenv("PINECONE_DEFAULT_NS", "")

    recency_half_life_days: float = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "45"))
    definition_window_s: tuple[float,float] = (30.0, 60.0)

    # cross-encoder model / batching
    ce_model: str = os.getenv("CE_MODEL", "BAAI/bge-reranker-large")
    ce_batch_size: int = int(os.getenv("CE_BATCH", "32"))

    # --- NEW knobs for CE-layer recency/staleness ---
    ce_recency_weight: float = float(os.getenv("CE_RECENCY_WEIGHT", "1.25"))
    stale_phrase_penalty_mult: float = float(os.getenv("STALE_PHRASE_PENALTY_MULT", "0.82"))
    stale_phrase_age_days: int = int(os.getenv("STALE_PHRASE_AGE_DAYS", "365"))

    # --- Early-abort gates ---
    enable_early_abort: bool = os.getenv("ENABLE_EARLY_ABORT", "1") in ("1","true","yes")
    # Pre-CE gate (uses stage1/meta-boosted scores; tune to your store)
    stage1_top_min: float = float(os.getenv("STAGE1_TOP_MIN", "0.02"))
    stage1_hard_min: float = float(os.getenv("STAGE1_HARD_MIN", "0.05"))
    stage1_min_relevant: int = int(os.getenv("STAGE1_MIN_RELEVANT", "1"))
    # Post-CE gate (uses sigmoid-normalized CE scores; lower than ce_abs_min)
    ce_max_norm_min: float = float(os.getenv("CE_MAX_NORM_MIN", "0.28"))
    abort_message: str = os.getenv(
        "ABORT_MESSAGE",
        "I don’t know based on the sources I can search."
    )
    # NEW: require at least one entity hit for definition-style queries
    def_require_entity_hit: bool = os.getenv("DEF_REQUIRE_ENTITY_HIT", "1") in ("1", "true", "yes")

    # boosts
    router_boost_mult: float = float(os.getenv("ROUTER_BOOST_MULT", "1.25"))
    streams_bias_mult: float = float(os.getenv("STREAMS_BIAS_MULT", "1.15"))

CFG = RetrievalConfig()

# --- Entity canonicalization (add this) ---
ENT_CANON_MAP = {
    "solana": "Solana",
    "sol": "SOL",
    "soul": "SOL",
    "$sol": "SOL",
    "$soul": "SOL",
    "anza labs": "Anza",
    "firedancer": "Firedancer",
    "anza": "Anza",
    "anza labs": "Anza",
    "firedancer": "Firedancer",
    "frankendancer": "Firedancer",
    "alpenglow": "Alpenglow",
    "aster": "Aster",
    "Salana": "Solana",
    "Cupsy": "Cupsey",
    "cupsy":"Cupsey",
    "haster": "aster",
    "astro": "astro",
}
# Normalize keys to lowercase for lookups
ENT_CANON_MAP = {k.lower(): v for k, v in ENT_CANON_MAP.items()}
