from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

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
    """
    Agent tool: query the video/stream index and synthesize an answer
    with links to relevant clips/segments.
    """
    qe = _get_qe()
    resp = qe.query(query)
    return clean_model_refs(str(resp))
