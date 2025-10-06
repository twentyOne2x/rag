# File: src/rag_v2/agent_main.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- resilient imports (run as module or script) ---
try:
    from .tools import search_videos_and_clips  # type: ignore
except ImportError:
    _here = Path(__file__).resolve()
    _src = _here.parents[1]  # .../src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from rag_v2.tools import search_videos_and_clips  # type: ignore


def _configure_models() -> None:
    """Match your Pinecone 3072D index + pick LLM."""
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))
    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )


# Simple heuristic router: send to video search when query looks retrieval-ish
_VIDEO_HINTS = re.compile(
    r"\b(video|videos|clip|clips|stream|youtube|timestamp|episode|show me|return all|"
    r"what is|who is|how does|why|what['’]s|who['’]s)\b"
    r"|@[\w_]{2,30}\b"
    r"|DATs?\b"
    r"|Kyle\s+Samani\b"
    r"|firedancer\b|frankendancer\b|anza\b|alpenglow\b|aster\b",
    re.IGNORECASE,
)
def _should_use_video_tool(q: str) -> bool:
    return bool(_VIDEO_HINTS.search(q))


def _llm_answer(q: str) -> str:
    # deprecated; we no longer fall back to an LLM
    return "I don’t know."

class TinyV2Agent:
    """
    Minimal agent that works across LlamaIndex versions without agent framework deps.
    - Routes to your v2 video/clip tool when appropriate.
    - Otherwise calls the LLM directly.
    """

    def __init__(self, verbose: bool = True):
        _configure_models()
        self.verbose = verbose

    def chat(self, question: str) -> str:
        if self.verbose:
            print(f"[router] q='{question}'")
        if _should_use_video_tool(question):
            if self.verbose:
                print("[router] -> video tool")
            return search_videos_and_clips(question)
        if self.verbose:
            print("[router] -> no matching tool")
        # Explicitly avoid hallucinations
        return (
            "I don’t know based on the sources I can search. "
            "Try asking for videos/clips and include names, channels, or tickers."
        )


def build_agent_v2(verbose: bool = True) -> TinyV2Agent:
    return TinyV2Agent(verbose=verbose)


if __name__ == "__main__":
    agent = build_agent_v2(verbose=True)

    # Provide questions via CLI args; fall back to defaults
    questions = sys.argv[1:] or [
        # "return all videos about DATs and Kyle Samani",
        # "what is a DAT on Solana?",
        # "show me all clips where Kyle Samani details how DATs will be deployed in DeFi",
        "who's cupsey?",
        "what's aster?",
        # "what's alpenglow?",
        # "what's firedancer?",
        # "return all videos about firedancer",
        # "return all videos from Anza",
        # "how much money has been raised on solana DATs and where will it be deployed in defi?",
        # "what is the first and most recent mention of firedancer?"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n=== Q{i}: {q}\n")
        answer = agent.chat(q)
        print(answer)
