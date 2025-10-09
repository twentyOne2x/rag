# app.py
import os, asyncio
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from rag_v2.app_main import bootstrap_query_engine_v2  # your code
from rag_v2.logging_utils import clean_model_refs

# Allow multiple origins: "https://www.mev.fyi,https://icm.fyi,http://localhost:3000"
APP_ORIGINS = [o.strip() for o in os.getenv("APP_ORIGINS", os.getenv("APP_ORIGIN", "http://localhost:3000")).split(",") if o.strip()]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

qe = None  # lazily initialized once per container

@app.on_event("startup")
async def _startup():
    global qe
    Settings.llm = OpenAI(model=os.getenv("INFERENCE_MODEL", "gpt-4o-mini"))
    qe = bootstrap_query_engine_v2()  # builds retriever + query engine once

# ── Request/Response models ───────────────────────────────────────────────────

class ChatReq(BaseModel):
    message: str
    # Accept both names; frontend currently sends "chat_history"
    history: Optional[List[Any]] = None
    chat_history: Optional[List[Any]] = None
    # Router hints (optional)
    scope: Optional[str] = None            # "videos" | "streams" | "auto"
    definition: Optional[bool] = None      # definition/explainer intent
    # Future: user/session ids if you want per-user routing/ablation
    # user_id: Optional[str] = None

class ChatResp(BaseModel):
    response: str
    # Optional legacy fallback field the frontend can parse if present
    formatted_metadata: Optional[str] = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")

    # Normalize history
    history = req.chat_history if req.chat_history is not None else req.history

    # If your QE supports hints, pass them via metadata/context kwargs.
    # Keep this tolerant: if QE ignores, nothing breaks.
    qe_kwargs = {
        "router_scope": req.scope,          # your QE can read/use this hint
        "definition_mode": req.definition,  # your QE can bias retrieval
        "history": history,                 # if you thread history inside QE
    }

    try:
        loop = asyncio.get_running_loop()
        def _run():
            # Your QE should already format the final answer text and append
            # a trailing sources block exactly like the frontend expects:
            #
            # "...\n\nFetched based on the following sources:\n"
            # "[Title]: <title> (HH:MM:SS–HH:MM:SS), [Speaker]: X, [Channel]: Y, [Date]: YYYY-MM-DD, [Score]: 0.8123, [Excerpt]: …"
            #
            # Also embed inline markdown links in the body:
            #   [<title>](https://youtube.com/watch?v=...&t=123s)
            #
            resp_obj = qe.query(req.message, **{k: v for k, v in qe_kwargs.items() if v is not None})
            text = clean_model_refs(str(resp_obj))
            # If you later compute a pre-built metadata blob, attach here:
            # meta = getattr(resp_obj, "formatted_metadata", None)
            meta = None
            return text, meta

        answer_text, formatted_metadata = loop.run_in_executor(None, _run)
        answer_text, formatted_metadata = await answer_text, await formatted_metadata  # unpack futures
        return ChatResp(response=answer_text, formatted_metadata=formatted_metadata)
    except Exception as e:
        raise HTTPException(500, str(e))
