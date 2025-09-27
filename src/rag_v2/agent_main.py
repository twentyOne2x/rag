from __future__ import annotations

import os
import sys
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# --- resilient imports (run as module or script) ---
try:
    from .tools import search_videos_and_clips  # type: ignore
except ImportError:
    _here = Path(__file__).resolve()
    _src = _here.parents[1]  # .../src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from rag_v2.tools import search_videos_and_clips  # type: ignore


def build_agent_v2(verbose: bool = True) -> ReActAgent:
    """
    Agent that can decide when to call the v2 video/stream tool,
    or answer directly for broader questions.
    """
    # LLM + Embeddings (match 3072D Pinecone index)
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))
    Settings.embed_model = OpenAIEmbedding(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))

    video_tool = FunctionTool.from_defaults(
        fn=search_videos_and_clips,
        name="video_search",
        description=(
            "Use this to answer questions about videos, streams, and clips. "
            "Great for entity/topic queries (e.g., DATs, Kyle Samani) and clip retrieval."
        ),
    )

    agent = ReActAgent.from_tools(
        tools=[video_tool],
        llm=Settings.llm,
        verbose=verbose,
        max_iterations=6,
    )
    return agent


if __name__ == "__main__":
    agent = build_agent_v2(verbose=True)

    # Provide questions via CLI args; fall back to defaults
    questions = sys.argv[1:] or [
        "what is a DAT on Solana?",
        "return all videos about DATs and Kyle Samani",
        "show me all clips where Kyle Samani details how DATs will be deployed in DeFi",
        "who's cupsey?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n=== Q{i}: {q}\n")
        answer = agent.chat(q)
        print(answer)
