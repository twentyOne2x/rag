# app.py
import os, asyncio, json, re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag_v2.app_main import bootstrap_query_engine_v2  # your code
from src.rag_v2.logging_utils import clean_model_refs
from src.rag_v2.instrumentation import AppDiagnostics, ProgressRecorder
from src.rag_v2.channel_catalog import channel_catalog, channel_names
from src.rag_v2.settings import config_value
from src.rag_v2.config import CFG

# Allow multiple origins: "https://www.icm.fyi,https://app.icm.fyi,http://localhost:3000"
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


class ChannelInfo(BaseModel):
    name: str
    count: Optional[int] = None


class ChannelsResp(BaseModel):
    scope: str
    channels: List[ChannelInfo]
    default_selected: List[str]


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


@app.get("/channels", response_model=ChannelsResp)
def channels(scope: str = Query(default="videos", min_length=1)) -> ChannelsResp:
    normalized_scope = scope or "videos"
    catalog = [ChannelInfo(**entry) for entry in channel_catalog(normalized_scope)]
    defaults = [entry.name for entry in catalog]
    return ChannelsResp(scope=normalized_scope, channels=catalog, default_selected=defaults)

@app.get("/healthz")
def healthz():
    startup = AppDiagnostics.get_startup_profile() or {}
    summary = {
        "total_ms": startup.get("total_ms"),
        "request_id": startup.get("request_id"),
    }
    telemetry = AppDiagnostics.get_last_telemetry() or {}
    recent = [
        {
            "request_id": trace.get("request_id"),
            "total_ms": trace.get("total_ms"),
            "query": trace.get("query"),
        }
        for trace in AppDiagnostics.recent_traces()[-5:]
    ]
    return {"ok": True, "startup": summary, "telemetry": telemetry, "recent_queries": recent}

@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe is None:
        raise HTTPException(503, "query engine not initialized")

    try:
        loop = asyncio.get_running_loop()
        def _run():
            return _execute_query(
                message=req.message,
                history=req.chat_history if req.chat_history is not None else req.history,
                scope=req.scope,
                definition=req.definition,
                channel_filter=req.channel_filter,
                enforce_prompt=True,
            )

        fut = loop.run_in_executor(None, _run)
        answer_text, formatted_metadata, trace = await fut
        return _build_chat_response(answer_text, formatted_metadata, trace)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat/simple", response_model=ChatResp)
async def chat_simple(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe is None:
        raise HTTPException(503, "query engine not initialized")

    loop = asyncio.get_running_loop()

    def _run():
        return _execute_query(
            message=req.message,
            history=req.chat_history if req.chat_history is not None else req.history,
            scope=req.scope,
            definition=req.definition,
            channel_filter=req.channel_filter,
            enforce_prompt=False,
        )

    try:
        answer_text, formatted_metadata, trace = await loop.run_in_executor(None, _run)
        return _build_chat_response(answer_text, formatted_metadata, trace)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe is None:
        raise HTTPException(503, "query engine not initialized")

    history = req.chat_history if req.chat_history is not None else req.history

    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    finished_sentinel = object()

    def enqueue(payload: Dict[str, Any]) -> None:
        try:
            data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            asyncio.run_coroutine_threadsafe(queue.put(data), loop)
        except Exception:
            pass

    def progress_listener(event: Dict[str, Any]) -> None:
        enqueue({"type": "progress", "event": event})

    progress = ProgressRecorder(scope="rag_query", listener=progress_listener)

    def worker():
        try:
            text, formatted_metadata, trace = _execute_query(
                message=req.message,
                history=history,
                scope=req.scope,
                definition=req.definition,
                channel_filter=req.channel_filter,
                enforce_prompt=True,
                progress=progress,
            )
            enqueue(
                {
                    "type": "result",
                    "response": text,
                    "formatted_metadata": formatted_metadata,
                    "diagnostics": _build_diagnostics(trace),
                }
            )
        except Exception as exc:
            enqueue({"type": "error", "error": str(exc)})
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(finished_sentinel), loop)

    loop.run_in_executor(None, worker)

    async def event_stream():
        try:
            while True:
                item = await queue.get()
                if item is finished_sentinel:
                    break
                if item is not None:
                    yield item
        finally:
            # drain remaining items if any
            while not queue.empty():
                _ = queue.get_nowait()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _looks_chinese(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _enrich_query(message: str) -> str:
    lang_hint = " 请用中文回答。" if _looks_chinese(message) else " Answer in English."
    cite_hint = (
        " Immediately after each quoted excerpt, add a markdown link to the exact clip start "
        "using the video's official title (omit any date/ID prefixes) as the link text, "
        "e.g., [Some Talk Title](URL?t=START_SECONDSs). "
    )
    return (
        message
        + "\n\n"
        + "Answer thoroughly using multiple distinct passages. "
        f"Quote ≥{CFG.quote_min_count} short excerpts verbatim and include each clip's timestamp range in parentheses. "
        + cite_hint
        + "Prefer stitching adjacent clips from the same video when context helps. "
        "End with a concise takeaway. "
        "When quoting, attribute to the named speaker if metadata provides one "
        "(use `speaker` or infer from the video title); avoid phrases like “the speaker says”."
        + lang_hint
    )


def _execute_query(
    *,
    message: str,
    history: Optional[List[Any]],
    scope: Optional[str],
    definition: Optional[bool],
    channel_filter: Optional[ChannelFilter],
    enforce_prompt: bool,
    progress: Optional[ProgressRecorder] = None,
) -> tuple[str, Optional[str], Dict[str, Any]]:
    if qe is None:
        raise RuntimeError("Query engine not initialized")

    qe_kwargs: Dict[str, Any] = {
        "router_scope": scope,
        "definition_mode": definition,
        "history": history,
    }

    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if channel_filter:
        channel_filter_payload = channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None
    if channel_filter_payload:
        scope_for_filter = scope or config_value("pinecone.namespace", default="videos")
        include_names = channel_filter_payload.get("include_names")
        if include_names:
            known = set(channel_names(scope_for_filter))
            if known and known.issubset(set(include_names)):
                channel_filter_payload.pop("include_names", None)
        if channel_filter_payload:
            qe_kwargs["channel_filter"] = channel_filter_payload

    query_text = _enrich_query(message) if enforce_prompt else message
    recorder = progress or ProgressRecorder(scope="rag_query")
    resp_obj = qe.query(
        query_text,
        progress=recorder,
        **{k: v for k, v in qe_kwargs.items() if v is not None},
    )
    answer_text = clean_model_refs(str(resp_obj))
    formatted_metadata = getattr(resp_obj, "formatted_metadata", None)
    trace = qe.get_last_trace() or {}
    return answer_text, formatted_metadata, trace


def _build_diagnostics(trace: Dict[str, Any]) -> Dict[str, Any]:
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
        "telemetry_summary": trace.get("telemetry_summary"),
    }
    return {k: v for k, v in diagnostics_payload.items() if v is not None}


def _build_chat_response(answer_text: str, formatted_metadata: Optional[str], trace: Dict[str, Any]) -> ChatResp:
    diagnostics_payload = _build_diagnostics(trace)
    return ChatResp(
        response=answer_text,
        formatted_metadata=formatted_metadata,
        request_id=trace.get("request_id"),
        diagnostics=diagnostics_payload or None,
    )
