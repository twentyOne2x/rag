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
    r"|DATs?\b|Kyle\s+Samani\b|firedancer\b|frankendancer\b|anza\b|alpenglow\b|aster\b"
    r"|视频|影片|片段|直播|时间戳|集数|返回所有|来自|是什么|谁是|怎么|为何",
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
        # "In the video where Cooker talks to Threadguy about Aster, did Cooker call the fact that Threadguy will soon interview CZ, the CEO of Binance?",
        # "who's cupsey?",



        # "what's aster?",
        # "who's cookerflips?",
        # "what's alpenglow?",
        # "what's firedancer?",
        # "return all videos about firedancer",
        # "return all videos from Anza",
        # "how much money has been raised on solana DATs and where will it be deployed in defi?",
        # "what is the first and most recent mention of firedancer?"


        "什么是 Aster？",                               # What's Aster?
        "谁是 Cookerflips？",                          # Who is Cookerflips?
        "什么是 Alpenglow？",                          # What's Alpenglow?
        "什么是 Firedancer？",                         # What's Firedancer?
        "返回所有与 Firedancer 相关的视频",             # Return all videos about Firedancer
        "返回所有来自 Anza 的视频",                    # Return all videos from Anza
        "Solana 上的 DAT 是什么？",                    # What is a DAT on Solana?
        "把 Kyle Samani 讲解 DAT 部署到 DeFi 的片段都找出来",  # Show all clips where Kyle Samani details deploying DATs to DeFi
        "在 Cooker 和 Threadguy 谈到 Aster 的那期里，他是否预言 Threadguy 会采访 CZ？",  # In the episode where Cooker talks Aster, did he call that TG would interview CZ?
        "找出 Firedancer 的最早与最新一次提及",          # What are the first and most recent mentions of Firedancer?
        "请给出 Alpenglow 将最终确认时间降到 ~100ms 的依据与片段",  # Provide evidence/clips that Alpenglow brings finality to ~100ms
        "Anza 与 Firedancer 的性能差异有哪些？给出视频引用",     # What are the perf differences between Anza and Firedancer? Cite videos
        "列出最近一个月关于 Solana MEV/订单流 的要点与出处",     # Summarize last-month highlights on Solana MEV/orderflow with sources
        "返回所有关于 Pump.fun 与公平发行模因币 的视频",        # Return all videos about Pump.fun and fair-launch memecoins

    ]

    for i, q in enumerate(questions, 1):
        print(f"\n=== Q{i}: {q}\n")
        answer = agent.chat(q)
        print(answer)
