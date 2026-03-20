import os
import asyncio
import json
import queue
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag_v2.app_main import bootstrap_query_engine_v2  # your code
from src.rag_v2.logging_utils import clean_model_refs, setup_logger
from src.rag_v2.instrumentation import AppDiagnostics, ProgressRecorder
from src.rag_v2.channel_catalog import channel_catalog, channel_names
from src.rag_v2.settings import config_value, load_config
from src.rag_v2.config import CFG
from src.rag_v2.query_engine_v2 import ParentChildQueryEngineV2
from src.rag_v2.vector_store.parent_catalog import search_parent_catalog, list_recent_parent_catalog
from src.rag_v2.vector_store.keyword_clips import scan_keyword_clips_qdrant


log = setup_logger("rag_v2.app")

# Allow multiple origins: "https://www.icm.fyi,https://app.icm.fyi,http://localhost:3000"
APP_ORIGINS = [o.strip() for o in os.getenv("APP_ORIGINS", os.getenv("APP_ORIGIN", "http://localhost:3000")).split(",") if o.strip()]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class EngineUnavailableError(RuntimeError):
    """Raised when no query engine is available to service a request."""


class QueryEnginePool:
    """Thread-safe pool of ParentChildQueryEngineV2 instances."""

    def __init__(
        self,
        factory: Callable[[ProgressRecorder], ParentChildQueryEngineV2],
        *,
        size: int,
        acquire_timeout: float,
    ) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        if acquire_timeout <= 0:
            raise ValueError("acquire_timeout must be > 0")

        self._factory = factory
        self._size = size
        self._acquire_timeout = acquire_timeout
        self._pool: "queue.Queue[ParentChildQueryEngineV2]" = queue.Queue(maxsize=size)

        primary_profiler = ProgressRecorder(scope="startup")
        primary_engine = self._factory(primary_profiler)
        primary_profile = getattr(primary_engine, "startup_profile", None)
        self._pool.put(primary_engine)
        log.info("rag[app] initialized query engine #1/%d", size)

        for i in range(1, size):
            profiler = ProgressRecorder(scope="startup", enabled=False)
            engine = self._factory(profiler)
            if primary_profile is not None:
                engine.startup_profile = primary_profile
            self._pool.put(engine)
            log.info("rag[app] initialized query engine #%d/%d", i + 1, size)

        if primary_profile is not None:
            AppDiagnostics.record_startup(primary_profile)

    @property
    def size(self) -> int:
        return self._size

    @property
    def acquire_timeout(self) -> float:
        return self._acquire_timeout

    @contextmanager
    def acquire(self) -> Iterator[ParentChildQueryEngineV2]:
        try:
            engine = self._pool.get(timeout=self._acquire_timeout)
        except queue.Empty as exc:
            raise EngineUnavailableError(
                f"no query engine available within {self._acquire_timeout:.1f}s"
            ) from exc

        try:
            yield engine
        finally:
            self._pool.put(engine)


qe_pool: Optional[QueryEnginePool] = None  # lazily initialized once per container


def _int_from_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        log.warning("rag[app] invalid int for %s=%r; falling back to %d", name, raw, default)
        return default


def _float_from_env(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        log.warning("rag[app] invalid float for %s=%r; falling back to %.2f", name, raw, default)
        return default


def _baseline_mode_entry() -> Dict[str, Any]:
    """Default mode used when no configuration is provided."""
    return {
        "name": "quick",
        "label": "Quick Research",
        "description": "Balanced latency/cost retrieval used when no mode is specified.",
        "prompt": {"quote_min_count": CFG.quote_min_count},
        "retrieval": {},
        "aliases": ["quick", "default"],
    }


@lru_cache(maxsize=1)
def _mode_registry() -> tuple[str, Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Load the configured research modes once and build an alias map so requests
    can refer to them by canonical name or shorthand (e.g., 'quick', 'deep').
    """
    config = load_config()
    modes_section = config.get("modes") or {}

    canonical: Dict[str, Dict[str, Any]] = {}
    alias_map: Dict[str, str] = {}

    def register(entry: Dict[str, Any]) -> None:
        canon_name = entry["name"]
        canonical[canon_name] = {
            "name": canon_name,
            "label": entry.get("label"),
            "description": entry.get("description"),
            "prompt": dict(entry.get("prompt") or {}),
            "retrieval": dict(entry.get("retrieval") or {}),
            "aliases": [str(a).strip().lower() for a in entry.get("aliases") or []],
        }
        alias_map.setdefault(canon_name, canon_name)
        for alias in canonical[canon_name]["aliases"]:
            alias_map.setdefault(alias, canon_name)

    # Ensure at least one mode is always available.
    register(_baseline_mode_entry())

    for name, payload in modes_section.items():
        if name == "default":
            continue
        canon_name = str(name).strip().lower()
        if not canon_name:
            continue
        aliases = {canon_name}
        for alias in payload.get("aliases") or []:
            alias_str = str(alias).strip().lower()
            if alias_str:
                aliases.add(alias_str)
        entry = {
            "name": canon_name,
            "label": payload.get("label") or canon_name.replace("_", " ").title(),
            "description": payload.get("description"),
            "prompt": dict(payload.get("prompt") or {}),
            "retrieval": dict(payload.get("retrieval") or {}),
            "aliases": sorted(aliases),
        }
        register(entry)

    default_src = modes_section.get("default")
    env_override = os.getenv("RAG_DEFAULT_MODE")
    default_raw = env_override if env_override else default_src
    default_key = str(default_raw).strip().lower() if default_raw else "quick"
    if default_key not in canonical:
        default_key = "quick"

    # Allow callers to request "default" explicitly.
    alias_map.setdefault("default", default_key)
    return default_key, canonical, alias_map


def _coerce_override(key: str, value: Any) -> Any:
    """Cast override values to match the RetrievalConfig field types."""
    base = getattr(CFG, key, None)
    if isinstance(base, bool):
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(base, int) and not isinstance(base, bool):
        return int(value)
    if isinstance(base, float):
        return float(value)
    if base is None:
        return value
    try:
        return type(base)(value)
    except Exception:
        return value


def _sanitize_mode_overrides(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return only valid retrieval config overrides with coerced types."""
    sanitized: Dict[str, Any] = {}
    if not raw:
        return sanitized
    for key, value in raw.items():
        if value is None or not hasattr(CFG, key):
            continue
        try:
            sanitized[key] = _coerce_override(key, value)
        except Exception:
            continue
    return sanitized


def _resolve_research_mode(requested: Optional[str]) -> Dict[str, Any]:
    """Resolve the requested research mode (or fallback) and return settings."""
    default_key, canonical, alias_map = _mode_registry()
    candidate = (requested or "").strip().lower()
    fallback_used = False
    canonical_key = alias_map.get(candidate) if candidate else None
    if not canonical_key:
        fallback_used = bool(candidate)
        canonical_key = default_key
    entry = canonical.get(canonical_key) or canonical[default_key]
    prompt = dict(entry.get("prompt") or {})
    prompt.setdefault("quote_min_count", CFG.quote_min_count)
    retrieval = dict(entry.get("retrieval") or {})
    return {
        "name": entry["name"],
        "label": entry.get("label"),
        "description": entry.get("description"),
        "prompt": prompt,
        "retrieval": retrieval,
        "requested": requested,
        "fallback_used": fallback_used,
        "available_modes": sorted(canonical.keys()),
    }


@app.on_event("startup")
async def _startup():
    global qe_pool

    def _factory(profiler: ProgressRecorder) -> ParentChildQueryEngineV2:
        return bootstrap_query_engine_v2(profiler=profiler)

    pool_size = _int_from_env("RAG_ENGINE_POOL_SIZE", default=1)
    acquire_timeout = _float_from_env("RAG_ENGINE_ACQUIRE_TIMEOUT", default=30.0)
    qe_pool = QueryEnginePool(
        _factory,
        size=pool_size,
        acquire_timeout=acquire_timeout,
    )
    log.info(
        "rag[app] query engine pool ready size=%d acquire_timeout=%.1fs",
        qe_pool.size,
        qe_pool.acquire_timeout,
    )


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
    research_mode: Optional[str] = None    # "quick" | "deep" | alias
    mode: Optional[str] = None             # alias for research_mode
    # Future: user/session ids if you want per-user routing/ablation
    # user_id: Optional[str] = None


class ChatResp(BaseModel):
    response: str
    # Optional legacy fallback field the frontend can parse if present
    formatted_metadata: Optional[str] = None
    request_id: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class CatalogSearchReq(BaseModel):
    query: str
    namespace: Optional[str] = None
    limit: int = 20
    channel_filter: Optional[ChannelFilter] = None
    refresh: bool = False


class CatalogSearchResp(BaseModel):
    ok: bool
    namespace: str
    query: str
    results: List[Dict[str, Any]]


class CatalogRecentReq(BaseModel):
    namespace: Optional[str] = None
    limit: int = 50
    since: Optional[str] = None  # YYYY-MM-DD (or ISO; date prefix used)
    cursor: Optional[str] = None
    channel_filter: Optional[ChannelFilter] = None
    refresh: bool = False


class CatalogRecentResp(BaseModel):
    ok: bool
    namespace: str
    since: Optional[str] = None
    scanned: int
    matched: int
    returned: int
    next_cursor: Optional[str] = None
    exhausted: bool
    results: List[Dict[str, Any]]


class KeywordClipsReq(BaseModel):
    query: str
    phrases: Optional[List[str]] = None
    namespace: Optional[str] = None
    limit: int = 200
    offset: Optional[str] = None
    channel_filter: Optional[ChannelFilter] = None


class KeywordClipsResp(BaseModel):
    ok: bool
    namespace: str
    query: str
    match_phrases_norm: List[str]
    match_tokens_norm: List[str]
    scanned: int
    matched: int
    unique_parents: int
    unique_speakers: int
    next_offset: Optional[str] = None
    exhausted: bool
    results: List[Dict[str, Any]]


def _requested_mode(req: ChatReq) -> Optional[str]:
    return req.research_mode or req.mode


@app.get("/channels", response_model=ChannelsResp)
def channels(scope: str = Query(default="videos", min_length=1)) -> ChannelsResp:
    normalized_scope = scope or "videos"
    catalog = [ChannelInfo(**entry) for entry in channel_catalog(normalized_scope)]
    defaults = [entry.name for entry in catalog]
    return ChannelsResp(scope=normalized_scope, channels=catalog, default_selected=defaults)


@app.post("/catalog/search", response_model=CatalogSearchResp)
def catalog_search(req: CatalogSearchReq) -> CatalogSearchResp:
    namespace = (req.namespace or "").strip() or config_value("pinecone.namespace", default="videos")
    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if req.channel_filter:
        channel_filter_payload = req.channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None
    results = search_parent_catalog(
        query=req.query,
        namespace=namespace,
        limit=req.limit,
        channel_filter=channel_filter_payload,
        force_refresh=bool(req.refresh),
    )
    return CatalogSearchResp(ok=True, namespace=namespace, query=req.query, results=results)


@app.post("/catalog/recent", response_model=CatalogRecentResp)
def catalog_recent(req: CatalogRecentReq) -> CatalogRecentResp:
    namespace = (req.namespace or "").strip() or config_value("pinecone.namespace", default="videos")
    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if req.channel_filter:
        channel_filter_payload = req.channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None

    res = list_recent_parent_catalog(
        namespace=namespace,
        limit=req.limit,
        since=req.since,
        cursor=req.cursor,
        channel_filter=channel_filter_payload,
        force_refresh=bool(req.refresh),
    )
    return CatalogRecentResp(
        ok=True,
        namespace=namespace,
        since=res.get("since"),
        scanned=int(res.get("scanned") or 0),
        matched=int(res.get("matched") or 0),
        returned=int(res.get("returned") or 0),
        next_cursor=res.get("next_cursor"),
        exhausted=bool(res.get("exhausted")),
        results=list(res.get("results") or []),
    )


@app.post("/clips/keyword", response_model=KeywordClipsResp)
def clips_keyword(req: KeywordClipsReq) -> KeywordClipsResp:
    namespace = (req.namespace or "").strip() or config_value("pinecone.namespace", default="videos")

    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if req.channel_filter:
        channel_filter_payload = req.channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None

    limit = int(req.limit or 0)
    if limit <= 0:
        raise HTTPException(400, "limit must be > 0 (use the CLI script for no-limit dumps)")
    if limit > 2000:
        raise HTTPException(400, "limit too large; cap at 2000 (page using offset)")

    try:
        res = scan_keyword_clips_qdrant(
            query=req.query,
            phrases=req.phrases,
            namespace=namespace,
            limit=limit,
            offset=req.offset,
            channel_filter=channel_filter_payload,
        )
        return KeywordClipsResp(**res)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/healthz")
def healthz():
    startup = AppDiagnostics.get_startup_profile() or {}
    summary = {
        "total_ms": startup.get("total_ms"),
        "request_id": startup.get("request_id"),
    }
    telemetry = AppDiagnostics.get_last_telemetry() or {}
    telemetry_histogram = AppDiagnostics.telemetry_snapshot()
    recent = [
        {
            "request_id": trace.get("request_id"),
            "total_ms": trace.get("total_ms"),
            "query": trace.get("query"),
        }
        for trace in AppDiagnostics.recent_traces()[-5:]
    ]
    return {
        "ok": True,
        "startup": summary,
        "telemetry": telemetry,
        "telemetry_histogram": telemetry_histogram,
        "recent_queries": recent,
    }


_CATALOG_QUERY_RE = re.compile(
    r"^\s*(?:all|list|show|find|give|what(?:'s| is)|which)?\s*(?:the\s+)?(?:videos|episodes|talks)\s+(?:about|on|regarding|mentioning)\s+(?P<topic>.+?)\s*$",
    re.IGNORECASE,
)


def _format_duration_s(duration_s: Optional[float]) -> Optional[str]:
    if duration_s is None:
        return None
    try:
        seconds = int(round(float(duration_s)))
    except Exception:
        return None
    if seconds <= 0:
        return None
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _maybe_catalog_answer(
    *,
    message: str,
    namespace: str,
    channel_filter: Optional[Dict[str, List[str]]],
    limit: int = 20,
) -> Optional[tuple[str, List[Dict[str, Any]]]]:
    """
    Lightweight metadata-only listing, for queries like:
      - "all videos about Anza"
      - "list videos mentioning firedancer"

    This deliberately does not run an LLM. It returns a ranked list of matching
    videos based on parent metadata (title/topic/entities/channel).
    """
    m = _CATALOG_QUERY_RE.match(message or "")
    if not m:
        return None

    topic = (m.group("topic") or "").strip()
    if not topic:
        return None

    results = search_parent_catalog(
        query=topic,
        namespace=namespace,
        limit=limit,
        channel_filter=channel_filter,
    )
    if not results:
        return (
            f"No matching video metadata found for: {topic}\n\n"
            "Try a broader term, or ask a content question so I can search inside transcripts.",
            [],
        )

    lines: List[str] = [f"Top {min(len(results), limit)} videos about: {topic}"]
    for idx, row in enumerate(results[:limit], start=1):
        title = str(row.get("title") or row.get("video_id") or "").strip()
        channel = str(row.get("channel_name") or "").strip()
        published_at = str(row.get("published_at") or "").strip()
        published_at = published_at.split("T", 1)[0] if "T" in published_at else published_at
        duration = _format_duration_s(row.get("duration_s"))
        url = str(row.get("url") or "").strip()
        meta_bits = [b for b in [channel or None, published_at or None, duration or None] if b]
        meta = f" ({' | '.join(meta_bits)})" if meta_bits else ""
        if url:
            lines.append(f"{idx}. {title}{meta}\n{url}")
        else:
            lines.append(f"{idx}. {title}{meta}")
    return "\n".join(lines), results


@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe_pool is None:
        raise HTTPException(503, "query engine not initialized")

    requested_mode = _requested_mode(req)
    namespace = config_value("pinecone.namespace", default="videos")
    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if req.channel_filter:
        channel_filter_payload = req.channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None

    catalog = _maybe_catalog_answer(
        message=req.message,
        namespace=namespace,
        channel_filter=channel_filter_payload,
        limit=20,
    )
    if catalog is not None:
        answer_text, results = catalog
        # Include raw catalog rows in diagnostics so the frontend can render them later.
        trace = {"request_id": f"catalog:{int(time.time() * 1000)}", "catalog_results": results}
        return _build_chat_response(answer_text, None, trace)

    try:
        loop = asyncio.get_running_loop()

        def _run():
            return _execute_query(
                message=req.message,
                history=req.chat_history if req.chat_history is not None else req.history,
                scope=req.scope,
                definition=req.definition,
                channel_filter=req.channel_filter,
                research_mode=requested_mode,
                enforce_prompt=True,
            )

        fut = loop.run_in_executor(None, _run)
        answer_text, formatted_metadata, trace = await fut
        return _build_chat_response(answer_text, formatted_metadata, trace)
    except EngineUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat/simple", response_model=ChatResp)
async def chat_simple(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe_pool is None:
        raise HTTPException(503, "query engine not initialized")

    loop = asyncio.get_running_loop()
    requested_mode = _requested_mode(req)

    def _run():
        return _execute_query(
            message=req.message,
            history=req.chat_history if req.chat_history is not None else req.history,
            scope=req.scope,
            definition=req.definition,
            channel_filter=req.channel_filter,
            research_mode=requested_mode,
            enforce_prompt=False,
        )

    try:
        answer_text, formatted_metadata, trace = await loop.run_in_executor(None, _run)
        return _build_chat_response(answer_text, formatted_metadata, trace)
    except EngineUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    if not req.message:
        raise HTTPException(400, "message is required")
    if qe_pool is None:
        raise HTTPException(503, "query engine not initialized")

    requested_mode = _requested_mode(req)
    history = req.chat_history if req.chat_history is not None else req.history

    queue_out: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    finished_sentinel = object()

    def enqueue(payload: Dict[str, Any]) -> None:
        try:
            data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            asyncio.run_coroutine_threadsafe(queue_out.put(data), loop)
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
                research_mode=requested_mode,
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
        except EngineUnavailableError as exc:
            enqueue({"type": "error", "error": str(exc)})
        except Exception as exc:
            enqueue({"type": "error", "error": str(exc)})
        finally:
            asyncio.run_coroutine_threadsafe(queue_out.put(finished_sentinel), loop)

    loop.run_in_executor(None, worker)

    async def event_stream():
        try:
            while True:
                item = await queue_out.get()
                if item is finished_sentinel:
                    break
                if item is not None:
                    yield item
        finally:
            # drain remaining items if any
            while not queue_out.empty():
                _ = queue_out.get_nowait()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _looks_chinese(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _enrich_query(message: str, quote_min_count: int, mode_name: str) -> str:
    lang_hint = " 请用中文回答。" if _looks_chinese(message) else " Answer in English."
    cite_hint = (
        " Immediately after each quoted excerpt, add a markdown link to the exact clip start "
        "using the video's official title (omit any date/ID prefixes) as the link text, "
        "e.g., [Some Talk Title](URL?t=START_SECONDSs). "
    )
    recency_hint = (
        " Include the clip's publish month/year (or exact date when available). "
        "If a cited clip is older than roughly two weeks relative to today, call out its age and note any implications for present-day relevance."
    )
    confidence_hint = (
        " Write with confident, declarative language when the cited evidence aligns; prefer “X is…” over hedged phrasing like “X appears to be…”. "
        "Use hedging only when sources conflict or explicitly signal uncertainty."
    )
    deep_structure_hint = ""
    if mode_name == "deep":
        deep_structure_hint = (
            " Organise the response into an `Executive Summary` section (2–4 bullet points that synthesise the findings without inline citations), "
            "followed by a `Detailed Evidence` section that groups related findings under short subheadings and anchors every bullet or paragraph with citations, "
            "and conclude with a `Key Takeaways` paragraph that summarises the overall insight and, if helpful, references the single most decisive citation."
        )
    score_hint = (
        " If the top candidate sources all score below ~1% similarity, state that the system could not find strong matches instead of synthesising a weak answer."
    )
    grounding_hint = (
        " Use only retrieved transcript evidence for definitions. "
        "If sources present multiple meanings for a term/acronym, explicitly say it is ambiguous and list each meaning instead of choosing one as definitive."
    )
    return (
        message
        + "\n\n"
        + "Answer thoroughly using multiple distinct passages. "
        f"Provide ≥{quote_min_count} citations; for each citation, quote 2–3 sentences (≈120–300 chars) verbatim, including 1 sentence of lead‑in and 1 of follow‑through when helpful, and include each clip's timestamp range in parentheses. "
        + cite_hint
        + recency_hint
        + confidence_hint
        + deep_structure_hint
        + score_hint
        + grounding_hint
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
    research_mode: Optional[str],
    enforce_prompt: bool,
    progress: Optional[ProgressRecorder] = None,
) -> tuple[str, Optional[str], Dict[str, Any]]:
    if qe_pool is None:
        raise RuntimeError("Query engine pool not initialized")

    qe_kwargs: Dict[str, Any] = {
        "router_scope": scope,
        "definition_mode": definition,
        "history": history,
    }
    mode_details = _resolve_research_mode(research_mode)
    raw_quote = mode_details["prompt"].get("quote_min_count", CFG.quote_min_count)
    try:
        quote_min_count = max(1, int(raw_quote))
    except Exception:
        quote_min_count = CFG.quote_min_count
    retrieval_overrides = _sanitize_mode_overrides(mode_details.get("retrieval"))
    if retrieval_overrides:
        qe_kwargs["mode_overrides"] = retrieval_overrides
    qe_kwargs["research_mode"] = mode_details["name"]

    channel_filter_payload: Optional[Dict[str, List[str]]] = None
    if channel_filter:
        channel_filter_payload = channel_filter.dict(exclude_none=True)
        if not channel_filter_payload:
            channel_filter_payload = None
    if channel_filter_payload:
        scope_for_filter = scope or config_value("pinecone.namespace", default="videos")
        # Treat the configured channel catalog as an allowlist. Even when callers select "all",
        # we still want a filter to avoid pulling in unrelated channels that happen to exist in
        # the vector store.
        known = set(channel_names(scope_for_filter))
        include_names = channel_filter_payload.get("include_names")
        if include_names is not None and known:
            # Preserve caller ordering while ensuring we don't accept arbitrary names.
            channel_filter_payload["include_names"] = [n for n in include_names if n in known]
        qe_kwargs["channel_filter"] = channel_filter_payload

    query_text = (
        _enrich_query(message, quote_min_count, mode_details["name"])
        if enforce_prompt
        else message
    )
    recorder = progress or ProgressRecorder(scope="rag_query")
    recorder.metadata["research_mode"] = mode_details["name"]
    if mode_details.get("label"):
        recorder.metadata["research_mode_label"] = mode_details["label"]
    if mode_details.get("fallback_used"):
        recorder.metadata["mode_fallback"] = True
    if retrieval_overrides:
        recorder.metadata["mode_overrides"] = retrieval_overrides

    with qe_pool.acquire() as engine:
        resp_obj = engine.query(
            query_text,
            progress=recorder,
            **{k: v for k, v in qe_kwargs.items() if v is not None},
        )
        answer_text = clean_model_refs(str(resp_obj))
        formatted_metadata = getattr(resp_obj, "formatted_metadata", None)
        trace = engine.get_last_trace() or {}

    mode_summary = {
        "name": mode_details["name"],
        "label": mode_details.get("label"),
        "description": mode_details.get("description"),
        "requested": mode_details.get("requested"),
        "fallback_used": mode_details.get("fallback_used"),
        "available_modes": mode_details.get("available_modes"),
        "prompt": {"quote_min_count": quote_min_count},
        "retrieval_overrides": retrieval_overrides or None,
    }
    trace.setdefault("research_mode", mode_details["name"])
    trace.setdefault("mode_config", {k: v for k, v in mode_summary.items() if v})
    if mode_details.get("label"):
        trace.setdefault("mode_label", mode_details["label"])
    if mode_details.get("requested"):
        trace.setdefault("mode_requested", mode_details["requested"])

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
        "catalog_results": trace.get("catalog_results"),
        "definition_disambiguation": trace.get("definition_disambiguation"),
        "config": trace.get("config"),
        "early_abort": trace.get("early_abort"),
        "channel_filter": trace.get("channel_filter"),
        "research_mode": trace.get("research_mode"),
        "mode_config": trace.get("mode_config"),
        "mode_label": trace.get("mode_label"),
        "mode_requested": trace.get("mode_requested"),
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
