from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from .config import CFG

# --- resilient imports (run as module or script) ---
try:
    from .app_main import bootstrap_query_engine_v2  # type: ignore
    from .logging_utils import clean_model_refs  # type: ignore
except ImportError:
    _here = Path(__file__).resolve()
    _src = _here.parents[1]  # .../src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from rag_v2.app_main import bootstrap_query_engine_v2  # type: ignore
    from rag_v2.logging_utils import clean_model_refs  # type: ignore

_QE = None

def _get_qe():
    global _QE
    if _QE is None:
        # Make sure LLM is configured (agent will also set this, but safe here)
        Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))
        _QE = bootstrap_query_engine_v2()
    return _QE

def search_videos_and_clips(query: str, top_k: Optional[int] = None) -> str:
    qe = _get_qe()
    # light instruction to increase depth + quotes with timestamps
    enriched_q = (
        query
        + "\n\n"
        + "Answer thoroughly using multiple distinct passages. "
          f"Quote ≥{CFG.quote_min_count} short excerpts verbatim and include each clip's timestamp range in parentheses. "
          "Prefer stitching adjacent clips from the same video when context helps. "
          "End with a concise takeaway."
        + "When quoting, attribute to the named speaker if metadata provides one "
        + "(use `speaker` or infer from the video title); avoid phrases like “the speaker says”. "
    )
    resp = qe.query(enriched_q)
    return clean_model_refs(str(resp))

