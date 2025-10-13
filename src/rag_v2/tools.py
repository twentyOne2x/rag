# File: src/rag_v2/tools.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# Resilient imports (run as module or script)
try:
    from .app_main import bootstrap_query_engine_v2  # type: ignore
    from .logging_utils import clean_model_refs       # type: ignore
except ImportError:
    _here = Path(__file__).resolve()
    _src = _here.parents[1]  # .../src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from rag_v2.app_main import bootstrap_query_engine_v2  # type: ignore
    from rag_v2.logging_utils import clean_model_refs      # type: ignore

from .config import CFG  # after sys.path fix

_QE = None


def _get_qe():
    """Lazy-init the query engine once per process."""
    global _QE
    if _QE is None:
        Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))
        _QE = bootstrap_query_engine_v2()
    return _QE


def _looks_chinese(s: str) -> bool:
    """True if the string contains any CJK Unified Ideographs (basic block)."""
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def search_videos_and_clips(query: str, top_k: Optional[int] = None) -> str:
    """
    Runs the v2 query engine with a retrieval-oriented prompt:
      - ≥N short verbatim quotes with timestamp ranges
      - Inline markdown citation immediately after each quote to the exact clip start
      - Speaker attribution when available
      - Language of the answer mirrors the input (CN->CN, else EN)
    """
    qe = _get_qe()

    # Language + citation behavior hints (kept short, model-friendly)
    lang_hint = " 请用中文回答。" if _looks_chinese(query) else " Answer in English."
    cite_hint = (
        " Immediately after each quoted excerpt, add a markdown link to the exact clip start "
        "using the video's official title (omit any date/ID prefixes) as the link text, "
        "e.g., [Some Talk Title](URL?t=START_SECONDSs). "
    )

    enriched_q = (
        query
        + "\n\n"
        + "Answer thoroughly using multiple distinct passages. "
        f"Provide ≥{CFG.quote_min_count} citations; for each citation, quote 2–3 sentences (≈120–300 chars) verbatim, including 1 sentence of lead‑in and 1 of follow‑through when helpful, and include each clip's timestamp range in parentheses. "
        + cite_hint
        + "Prefer stitching adjacent clips from the same video when context helps. "
        "End with a concise takeaway. "
        "When quoting, attribute to the named speaker if metadata provides one "
        "(use `speaker` or infer from the video title); avoid phrases like “the speaker says”."
        + lang_hint
    )

    resp = qe.query(enriched_q)
    return clean_model_refs(str(resp))
