# app.py
import os, asyncio
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_v2.app_main import bootstrap_query_engine_v2  # your code
from src.rag_v2.logging_utils import clean_model_refs
from src.rag_v2.instrumentation import AppDiagnostics, ProgressRecorder

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
    profiler = ProgressRecorder(scope="startup")
    qe = bootstrap_query_engine_v2(profiler=profiler)  # builds retriever + query engine once

# ── Request/Response models ───────────────────────────────────────────────────

class ChannelFilter(BaseModel):
    include_ids: Optional[List[str]] = None
    exclude_ids: Optional[List[str]] = None
    include_names: Optional[List[str]] = None
    exclude_names: Optional[List[str]] = None


class ChatReq(BaseModel):
    message: str
    # Accept both names; frontend currently sends "chat_history"
    history: Optional[List[Any]] = None
    chat_history: Optional[List[Any]] = None
    # Router hints (optional)
    scope: Optional[str] = None            # "videos" | "streams" | "auto"
    definition: Optional[bool] = None      # definition/explainer intent
    channel_filter: Optional[ChannelFilter] = None
    # Future: user/session ids if you want per-user routing/ablation
    # user_id: Optional[str] = None

class ChatResp(BaseModel):
    response: str
    # Optional legacy fallback field the frontend can parse if present
    formatted_metadata: Optional[str] = None
    request_id: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None

@app.get("/healthz")
def healthz():
    startup = AppDiagnostics.get_startup_profile() or {}
    summary = {
        "total_ms": startup.get("total_ms"),
        "request_id": startup.get("request_id"),
    }
    return {"ok": True, "startup": summary}

@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe is None:
        raise HTTPException(503, "query engine not initialized")

    # Normalize history
    history = req.chat_history if req.chat_history is not None else req.history

    # If your QE supports hints, pass them via metadata/context kwargs.
    # Keep this tolerant: if QE ignores, nothing breaks.
    qe_kwargs = {
        "router_scope": req.scope,          # your QE can read/use this hint
        "definition_mode": req.definition,  # your QE can bias retrieval
        "history": history,                 # if you thread history inside QE
    }
    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if req.channel_filter:
        channel_filter_payload = req.channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None
    if channel_filter_payload:
        qe_kwargs["channel_filter"] = channel_filter_payload

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
            progress = ProgressRecorder(scope="rag_query")
            resp_obj = qe.query(
                req.message,
                progress=progress,
                **{k: v for k, v in qe_kwargs.items() if v is not None},
            )
            text = clean_model_refs(str(resp_obj))
            # Pull any pre-built metadata blob if the QE returns it.
            meta = getattr(resp_obj, "formatted_metadata", None)
            trace = qe.get_last_trace()
            return text, meta, trace

        fut = loop.run_in_executor(None, _run)
        answer_text, formatted_metadata, trace = await fut
        trace = trace or {}
        diagnostics_payload = {
            "request_id": trace.get("request_id"),
            "total_ms": trace.get("total_ms"),
            "timings": trace.get("timings"),
            "progress": trace.get("progress"),
            "progress_metadata": trace.get("progress_metadata"),
            "models": trace.get("models"),
            "final_kept": trace.get("final_kept"),
            "config": trace.get("config"),
            "early_abort": trace.get("early_abort"),
            "channel_filter": trace.get("channel_filter"),
        }
        diagnostics_payload = {k: v for k, v in diagnostics_payload.items() if v is not None}
        return ChatResp(
            response=answer_text,
            formatted_metadata=formatted_metadata,
            request_id=trace.get("request_id"),
            diagnostics=diagnostics_payload or None,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
