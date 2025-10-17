# File: src/rag_v2/query_engine_v2.py
from __future__ import annotations
import asyncio
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import time

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import CFG
from .runtime_config import get_runtime_config, override_runtime_config
from .postprocessors.speaker_propagator import SpeakerPropagator
from .rerankers.cross_encoder import CEReranker
from .postprocessors.entity_canonicalizer import EntityCanonicalizer  # NEW
from .postprocessors.entity_utils import normalize_text_entities  # NEW
from .router.video_router import TICKER_RE, HANDLE_RE
from .postprocessors.entity_utils import canon_entity
from .logging_utils import (
    setup_logger,
    pretty,
    node_brief,
    clean_model_refs,
    cfg_snapshot,
    model_snapshot,
    append_sources_block,
)
from .router.video_router import wants_definition
from .instrumentation import AppDiagnostics, ProgressRecorder, ProgressEvent
from .telemetry import TelemetryCollector, JsonlTelemetryWriter
from .settings import config_value

log = setup_logger("rag_v2.qe")


class ParentChildQueryEngineV2(BaseQueryEngine):
    """
    Query engine that:
      1) Uses a parent/child retriever,
      2) Optionally reranks with a cross-encoder,
      3) Applies entity canonicalization to fix text errors,  # NEW
      4) Synthesizes with LlamaIndex's core response synthesizer.

    Emits a single JSON 'trace' per query at DEBUG (file sink if configured) and
    concise INFO logs for humans.
    """

    def __init__(self, retriever, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self._retriever = retriever
        self._core = RetrieverQueryEngine.from_args(retriever=self._retriever)
        self._ce = (
            CEReranker(model_name=CFG.ce_model, batch_size=CFG.ce_batch_size)
            if CFG.enable_ce
            else None
        )
        # NEW: Entity canonicalizer to fix "Soul" -> "SOL" etc.
        self._entity_canonicalizer = EntityCanonicalizer()
        self._speaker_propagator = SpeakerPropagator()
        self.startup_profile: Optional[Dict[str, Any]] = None
        self._last_trace: Dict[str, Any] = {}
        self._last_progress_summary: Optional[Dict[str, Any]] = None
        self._active_progress: Optional[ProgressRecorder] = None
        self._last_abort_details: Optional[Dict[str, Any]] = None
        self._active_channel_filter: Optional[Dict[str, Any]] = None
        self._telemetry_enabled: bool = os.getenv("RAG_ENABLE_TELEMETRY", "1") in ("1", "true", "yes")
        self._telemetry_writer: Optional[JsonlTelemetryWriter] = self._init_telemetry_writer()
        self._pending_hints: Dict[str, Any] = {}

    @staticmethod
    def _normalize_channel_filter(channel_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[str]]]:
        if not channel_filter:
            return None
        allowed_keys = ("include_ids", "exclude_ids", "include_names", "exclude_names")
        normalized: Dict[str, List[str]] = {}
        for key in allowed_keys:
            vals = channel_filter.get(key)
            if not vals:
                continue
            clean_vals = []
            for v in vals:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    clean_vals.append(s)
            if clean_vals:
                normalized[key] = clean_vals
        return normalized or None

    def _init_telemetry_writer(self) -> Optional[JsonlTelemetryWriter]:
        path_str = os.getenv("RAG_TELEMETRY_PATH")
        if not path_str:
            return None
        try:
            return JsonlTelemetryWriter(Path(path_str))
        except Exception as exc:
            log.warning("telemetry[writer] failed to initialise path=%s err=%s", path_str, exc)
            return None

    # Required for some LI versions
    def _get_prompt_modules(self):
        return {}

    # -------- helpers --------
    def _ce_pack(self, nodes: List[NodeWithScore]) -> List[Tuple[str, str, float]]:
        items: List[Tuple[str, str, float]] = []
        for n in nodes:
            md = n.node.metadata or {}
            sid = md.get("segment_id") or md.get("id") or n.node.node_id
            items.append((sid, n.node.get_content(), float(n.score or 0.0)))
        return items

    def _reinject_scores(
            self, nodes: List[NodeWithScore], rescored: List[Tuple[str, str, float]]
    ) -> List[NodeWithScore]:
        score_by_sid = {sid: sc for sid, _, sc in rescored}
        for n in nodes:
            md = n.node.metadata or {}
            sid = md.get("segment_id") or md.get("id") or n.node.node_id
            if sid in score_by_sid:
                n.score = score_by_sid[sid]
        nodes.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return nodes

    def _final_sources_view(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in nodes:
            view = node_brief(n)
            md = n.node.metadata or {}
            start_hms = md.get("start_hms")
            start_seconds = self._hms_to_seconds(start_hms) if start_hms else None
            video_id = md.get("parent_id") or md.get("video_id")
            view.update(
                {
                    "url": md.get("clip_url") or md.get("url"),
                    "clip_url": md.get("clip_url") or md.get("url"),
                    "title": md.get("title"),
                    "channel_name": md.get("channel_name"),
                    "channel_id": md.get("channel_id"),
                    "parent_id": video_id,
                    "video_id": video_id,
                    "start_hms": start_hms,
                    "end_hms": md.get("end_hms"),
                    "start_seconds": start_seconds,
                }
            )
            out.append(view)
        return out
    @staticmethod
    def _source_snapshot(nodes: List[NodeWithScore], limit: int = 5) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in nodes[:limit]:
            md = n.node.metadata or {}
            title = md.get("title") or md.get("parent_title")
            if not title:
                continue
            out.append(
                {
                    "title": title,
                    "channel": md.get("channel_name") or md.get("parent_channel_name"),
                    "score": round(float(n.score or 0.0), 4),
                    "segment_id": md.get("segment_id") or md.get("id"),
                }
            )
        return out

    def _annotate_speakers(self, nodes):
        out = []
        for n in nodes:
            md = n.node.metadata or {}
            spk = md.get("speaker")
            if spk:
                hms = " - ".join(filter(None, [md.get("start_hms"), md.get("end_hms")]))
                # mutate text once per final node
                prefixed = f"[{spk}{' | ' + hms if hms else ''}] " + n.node.get_content()
                n.node.text = normalize_text_entities(prefixed)
            out.append(n)
        return out

    @staticmethod
    def _normalize_node_entities(nodes):
        if not nodes:
            return nodes
        for item in nodes:
            node = getattr(item, "node", None) or item
            if node is None:
                continue
            text = None
            if hasattr(node, "get_content"):
                try:
                    text = node.get_content()
                except Exception:
                    text = getattr(node, "text", None)
            else:
                text = getattr(node, "text", None)
            if not text:
                continue
            cleaned = normalize_text_entities(text)
            if hasattr(node, "text"):
                node.text = cleaned
            elif hasattr(node, "set_content"):
                try:
                    node.set_content(cleaned)
                except Exception:
                    pass
        return nodes

    def _synthesize_clean(self, query_bundle: QueryBundle, nodes: List[NodeWithScore]) -> Response:
        raw = self._core._response_synthesizer.synthesize(query=query_bundle, nodes=nodes)
        try:
            cleaned_text = clean_model_refs(str(raw))
            # NEW: Apply final entity normalization to the answer
            cleaned_text = normalize_text_entities(cleaned_text)
            self._normalize_node_entities(nodes)

            # prefer whatever the synthesizer returned for source_nodes; else fall back to our `nodes`
            src_nodes = getattr(raw, "source_nodes", None) or nodes
            self._normalize_node_entities(src_nodes)
            final_text = append_sources_block(cleaned_text, src_nodes)

            if hasattr(raw, "response"):
                # mutate in-place to preserve Response extras/metadata and sources
                raw.response = final_text
                # make sure source_nodes are present
                if not getattr(raw, "source_nodes", None):
                    raw.source_nodes = nodes
                return raw

            return Response(final_text, source_nodes=src_nodes)
        except Exception:
            # last-resort: return raw as-is
            return raw

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except Exception:
            return float(x)

    @staticmethod
    def _percentile_cut(scores, p: float) -> float:
        if not scores:
            return float("inf")
        s = sorted(scores)
        idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return float(s[idx])

    @staticmethod
    def _rough_token_count(txt: str) -> int:
        # ~4 chars/token heuristic
        return max(1, len(txt) // 4)

    @staticmethod
    def _ensure_metadata(node: NodeWithScore) -> Dict[str, Any]:
        md = getattr(node.node, "metadata", None)
        if not isinstance(md, dict):
            md = {}
            node.node.metadata = md
        return md

    def _tag_stage1_scores(self, nodes: List[NodeWithScore]) -> None:
        for item in nodes or []:
            md = self._ensure_metadata(item)
            if "score_stage1" not in md:
                try:
                    md["score_stage1"] = float(item.score or 0.0)
                except Exception:
                    md["score_stage1"] = 0.0

    def _tag_ce_scores(self, nodes: List[NodeWithScore]) -> None:
        for item in nodes or []:
            md = self._ensure_metadata(item)
            try:
                raw = float(item.score or 0.0)
            except Exception:
                raw = 0.0
            md["score_ce"] = raw
            md["score_ce_norm"] = self._sigmoid(raw)

    def _node_confidence(self, node: NodeWithScore) -> float:
        md = self._ensure_metadata(node)
        if "score_ce_norm" in md:
            try:
                return max(0.0, float(md["score_ce_norm"]))
            except Exception:
                pass
        if "score_stage1" in md:
            try:
                return max(0.0, float(md["score_stage1"]))
            except Exception:
                pass
        try:
            return max(0.0, float(node.score or 0.0))
        except Exception:
            return 0.0

    def _apply_low_confidence_filter(
        self,
        nodes: List[NodeWithScore],
        threshold: float,
        trace: Optional[Dict[str, Any]] = None,
        progress: Optional[ProgressRecorder] = None,
        step: Optional[ProgressEvent] = None,
    ) -> List[NodeWithScore]:
        if not nodes or threshold <= 0:
            return nodes

        kept: List[NodeWithScore] = []
        dropped_info: List[Dict[str, Any]] = []
        for item in nodes:
            confidence = self._node_confidence(item)
            md = self._ensure_metadata(item)
            md["confidence_norm"] = confidence
            if confidence >= threshold:
                kept.append(item)
            else:
                sid = md.get("segment_id") or md.get("id") or item.node.node_id
                dropped_info.append(
                    {
                        "segment_id": sid,
                        "confidence": round(confidence, 4),
                        "title": md.get("title") or md.get("parent_title"),
                        "threshold": threshold,
                    }
                )

        if dropped_info:
            low_conf = trace.setdefault("low_confidence", {})
            existing = low_conf.get("dropped") or []
            existing.extend(dropped_info)
            low_conf["dropped"] = existing
            low_conf["threshold"] = threshold
            low_conf["kept_after_filter"] = len(kept)
            if progress is not None:
                progress.metadata["low_confidence_dropped"] = progress.metadata.get("low_confidence_dropped", 0) + len(dropped_info)
            if step is not None:
                step.metadata["dropped_low_confidence"] = len(dropped_info)
                step.metadata["threshold"] = threshold

        if not kept and trace is not None:
            low_conf = trace.setdefault("low_confidence", {})
            low_conf["exhausted"] = True

        return kept

    @staticmethod
    def _hms_to_seconds(hms: str) -> int:
        if not hms: return -1
        try:
            h, m, s = [int(x) for x in hms.split(":")]
            return h * 3600 + m * 60 + s
        except Exception:
            return -1

    def _stitch_adjacent(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        groups: Dict[str, List[NodeWithScore]] = {}
        for n in nodes:
            md = n.node.metadata or {}
            pid = str(md.get("parent_id") or md.get("video_id"))
            groups.setdefault(pid, []).append(n)

        out: List[NodeWithScore] = []
        cfg = get_runtime_config()
        gap_s = cfg.stitch_gap_seconds
        target_tokens = cfg.stitch_target_tokens
        max_merge = cfg.stitch_max_merge

        for pid, rows in groups.items():
            def _sec(node, key):
                return type(self)._hms_to_seconds((node.node.metadata or {}).get(key) or "")

            rows.sort(key=lambda r: _sec(r, "start_hms"))
            i = 0
            while i < len(rows):
                chunk = [rows[i]]
                i += 1
                # greedily merge forward while gaps small and under token budget
                while i < len(rows) and len(chunk) < max_merge:
                    prev = chunk[-1]
                    cur = rows[i]
                    end_prev = _sec(prev, "end_hms")
                    start_cur = _sec(cur, "start_hms")
                    if end_prev >= 0 and start_cur >= 0 and (start_cur - end_prev) <= gap_s:
                        merged_text = " ".join(c.node.get_content() for c in (chunk + [cur]))
                        if type(self)._rough_token_count(merged_text) <= target_tokens:
                            chunk.append(cur)
                            i += 1
                            continue
                    break

                if len(chunk) == 1:
                    out.append(chunk[0])
                else:
                    base = chunk[0]
                    md = base.node.metadata or {}
                    md["start_hms"] = (chunk[0].node.metadata or {}).get("start_hms")
                    md["end_hms"] = (chunk[-1].node.metadata or {}).get("end_hms")
                    base.node.text = "\n".join(c.node.get_content() for c in chunk)
                    base.score = max(float(c.score or 0.0) for c in chunk)
                    out.append(base)

        out.sort(key=lambda n: (n.score or 0.0), reverse=True)
        return out[: cfg.max_final_nodes]

    def _final_k(self, q: str) -> int:
        cfg = get_runtime_config()
        if wants_definition(q):  # short, crisp
            return min(8, cfg.max_final_nodes)
        if "return all videos" in q.lower():  # coverage query
            return cfg.max_final_nodes
        return min(10, cfg.max_final_nodes)

    def _qents(self, q: str) -> set[str]:
        ents = set(m.group(0).strip() for m in TICKER_RE.finditer(q))
        ents |= set(m.group(0).strip() for m in HANDLE_RE.finditer(q))
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", q):
            ce = canon_entity(tok)
            if ce: ents.add(ce)
        return {e.lower() for e in ents if not e.startswith("$")}

    def _maybe_early_abort_stage1(self, q: str, nodes: List[NodeWithScore]) -> Response | None:
        """Cheap pre-CE gate: low top score AND little entity overlap."""
        cfg = get_runtime_config()
        if not (cfg.enable_early_abort and nodes):
            return None
        top = max(float(n.score or 0.0) for n in nodes) if nodes else 0.0

        # unconditional cutoff
        if top < cfg.stage1_hard_min:
            self._last_abort_details = {
                "stage": "retrieve",
                "reason": "stage1_hard_min",
                "top_score": top,
                "threshold": cfg.stage1_hard_min,
            }
            log.info("qe[early-abort] q='%s' reason=stage1_hard_min top=%.5f<th=%.5f",
                     q, top, cfg.stage1_hard_min)
            return Response(cfg.abort_message, source_nodes=[])

        qents = self._qents(q)
        rel = 0
        for n in nodes:
            md = n.node.metadata or {}
            ents = {str(e).lower() for e in (md.get("entities") or [])}
            ents |= {str(e).lower() for e in (md.get("canonical_entities") or [])}
            if ents & qents:
                rel += 1
        if top < cfg.stage1_top_min and rel < cfg.stage1_min_relevant:
            self._last_abort_details = {
                "stage": "retrieve",
                "reason": "stage1_low",
                "top_score": top,
                "relevant_hits": rel,
                "threshold": cfg.stage1_top_min,
                "min_relevant": cfg.stage1_min_relevant,
            }
            log.info("qe[early-abort] q='%s' reason=stage1_low top=%.4f rel=%d", q, top, rel)
            return Response(cfg.abort_message, source_nodes=[])
        return None

    def _maybe_early_abort_post_ce(self, ce_norm: List[float]) -> Response | None:
        """Post-CE gate: after CE scoring but before synthesis."""
        cfg = get_runtime_config()
        if not cfg.enable_early_abort:
            return None
        mx = max(ce_norm) if ce_norm else 0.0
        if mx < cfg.ce_max_norm_min:
            self._last_abort_details = {
                "stage": "rerank_cross_encoder",
                "reason": "ce_low",
                "max_norm": mx,
                "threshold": cfg.ce_max_norm_min,
            }
            log.info("qe[early-abort] reason=ce_low max_norm=%.4f", mx)
            return Response(cfg.abort_message, source_nodes=[])
        return None

    # -------- sync path --------
    def query(self, query: Any, progress: Optional[ProgressRecorder] = None, **kwargs) -> RESPONSE_TYPE:
        """
        Override to accept an optional ProgressRecorder.
        """
        channel_filter_raw = kwargs.pop("channel_filter", None)
        channel_filter = self._normalize_channel_filter(channel_filter_raw)
        history_hint = kwargs.pop("history", None)
        router_scope_hint = kwargs.pop("router_scope", None)
        definition_hint = kwargs.pop("definition_mode", None)
        mode_hint = kwargs.pop("research_mode", None)
        mode_overrides = kwargs.pop("mode_overrides", None)
        extra_kwargs = dict(kwargs)
        if extra_kwargs:
            log.debug("qe[query] dropping unsupported kwargs: %s", sorted(extra_kwargs.keys()))

        recorder = progress or ProgressRecorder(scope="rag_query")
        prev = self._active_progress
        prev_filter = self._active_channel_filter
        self._active_progress = recorder
        self._active_channel_filter = channel_filter
        prev_hints = self._pending_hints
        self._pending_hints = {
            "history": history_hint,
            "router_scope": router_scope_hint,
            "definition_mode": definition_hint,
            "research_mode": mode_hint,
        }

        if channel_filter and hasattr(self._retriever, "set_channel_filter"):
            try:
                self._retriever.set_channel_filter(channel_filter)
            except Exception:
                pass
        elif hasattr(self._retriever, "set_channel_filter"):
            try:
                self._retriever.set_channel_filter(None)
            except Exception:
                pass

        recorder.metadata["channel_filter"] = channel_filter

        try:
            with override_runtime_config(mode_overrides):
                return super().query(query)
        finally:
            self._active_progress = prev
            self._active_channel_filter = prev_filter
            self._pending_hints = prev_hints
            if hasattr(self._retriever, "set_channel_filter"):
                try:
                    self._retriever.set_channel_filter(prev_filter)
                except Exception:
                    pass

    def get_last_trace(self) -> Dict[str, Any]:
        return dict(self._last_trace)

    def get_last_progress_summary(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_progress_summary) if self._last_progress_summary else None

    def _finalize_trace(self, trace: Dict[str, Any], progress: ProgressRecorder) -> Dict[str, Any]:
        summary = progress.summary()
        trace["request_id"] = summary["request_id"]
        trace["total_ms"] = summary["total_ms"]
        trace["timings"] = summary["timings"]
        trace["progress"] = summary["progress"]
        trace["progress_metadata"] = summary["metadata"]
        telemetry_summary = self._collect_telemetry(summary, trace) if self._telemetry_enabled else None
        if telemetry_summary:
            trace["telemetry_summary"] = telemetry_summary
            AppDiagnostics.record_telemetry(telemetry_summary)
        self._last_trace = trace
        self._last_progress_summary = summary
        AppDiagnostics.record_query(trace)
        return summary

    def _collect_telemetry(self, progress_summary: Dict[str, Any], trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        events = progress_summary.get("progress", []) or []
        if not events:
            return None

        collector = TelemetryCollector(
            service_name=os.getenv("RAG_SERVICE_NAME", "rag-v2"),
            environment=os.getenv(
                "RAG_ENV",
                config_value("environment", default="dev"),
            ),
            writer=self._telemetry_writer,
        )

        for ev in events:
            metadata = dict(ev.get("metadata") or {})
            metadata.update({
                "status": ev.get("status"),
                "label": ev.get("label"),
            })
            duration = ev.get("duration_ms")
            if duration is None:
                duration = 0.0
            collector.record_stage(ev.get("name", "unknown"), float(duration), metadata=metadata)

        summary = collector.summary()
        summary["request_id"] = progress_summary.get("request_id")
        summary["query"] = trace.get("query")
        summary["channel_filter"] = trace.get("channel_filter")
        summary["total_ms"] = progress_summary.get("total_ms")
        return summary

    def _query(self, query_bundle: QueryBundle, **kwargs) -> RESPONSE_TYPE:
        cfg_runtime = get_runtime_config()
        q = query_bundle.query_str
        progress = self._active_progress or ProgressRecorder(scope="rag_query")
        progress.metadata.setdefault("query", q)

        hints = dict(getattr(self, "_pending_hints", {}) or {})

        required_entities: Set[str] = set()
        retriever_query_entities = getattr(self._retriever, "_query_entities", None)
        if callable(retriever_query_entities):
            try:
                _, _, canonical = retriever_query_entities(q)
                if canonical:
                    required_entities = set(canonical)
            except Exception:
                required_entities = set()
        set_entity_requirements = getattr(self._retriever, "set_entity_requirements", None)
        if callable(set_entity_requirements):
            try:
                set_entity_requirements(required_entities if required_entities else None)
            except Exception:
                pass
        if required_entities:
            progress.metadata["required_entities"] = sorted(required_entities)

        cfg_runtime = get_runtime_config()

        # Optional hints the caller may provide (history, router scope, definition mode, etc.).
        history = hints.get("history")
        router_scope = hints.get("router_scope")
        definition_mode = hints.get("definition_mode")
        research_mode = hints.get("research_mode")
        mode_overrides_hint = hints.get("mode_overrides") or {}

        if history is not None:
            progress.metadata["history_len"] = len(history) if hasattr(history, "__len__") else 1
        if router_scope:
            progress.metadata["router_scope"] = router_scope
        if definition_mode is not None:
            progress.metadata["definition_mode"] = bool(definition_mode)
        if research_mode:
            progress.metadata["research_mode"] = research_mode
        if mode_overrides_hint:
            progress.metadata["mode_overrides"] = mode_overrides_hint

        trace: Dict[str, Any] = {
            "query": q,
            "config": cfg_snapshot(cfg_runtime),
            "models": model_snapshot(),
        }
        trace["request_id"] = progress.request_id
        trace["channel_filter"] = self._active_channel_filter
        trace["router_scope"] = router_scope
        trace["definition_mode"] = definition_mode
        trace["history_present"] = bool(history)
        trace["required_entities"] = sorted(required_entities)
        if research_mode:
            trace["research_mode"] = research_mode
        if mode_overrides_hint:
            trace["mode_overrides"] = mode_overrides_hint

        self._last_abort_details = None
        nodes: List[NodeWithScore] = []
        final_nodes: List[NodeWithScore] = []
        early: Optional[Response] = None

        with self.callback_manager.event(CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: q}) as revent:
            with progress.step("retrieve", "Finding likely sources (vector retrieval)") as retrieve_step:
                nodes = self._retriever.retrieve(query_bundle)
                self._tag_stage1_scores(nodes)
                retrieve_step.metadata.update({
                    "initial_candidates": len(nodes),
                    "similarity_top_k": getattr(
                        getattr(self._retriever, "base", None),
                        "similarity_top_k",
                        None,
                    ),
                })
                if nodes:
                    scores = [float(n.score or 0.0) for n in nodes]
                    retrieve_step.metadata["score_min"] = min(scores)
                    retrieve_step.metadata["score_max"] = max(scores)
                    retrieve_step.metadata["top_sources"] = self._source_snapshot(nodes)

            trace["stage1_count"] = len(nodes)

            if hasattr(self._retriever, "debug_snapshot"):
                trace.update(self._retriever.debug_snapshot())
            gate_info = trace.get("entity_gate")
            if gate_info:
                progress.metadata["entity_gate"] = gate_info
                if gate_info.get("applied") and gate_info.get("kept", 0) == 0:
                    required = gate_info.get("required") or trace.get("required_entities") or []
                    self._last_abort_details = {
                        "stage": "retrieve",
                        "reason": "entity_gate",
                        "required_entities": required,
                    }
                    progress.add_event(
                        "rerank_cross_encoder",
                        status="skipped",
                        label="Re-scoring sources (cross-encoder rerank)",
                        metadata={"reason": "entity_gate", "required_entities": required},
                    )
                    progress.add_event(
                        "review_docs",
                        status="skipped",
                        label="Cleaning and enriching notes (post-processing pipeline)",
                        metadata={"reason": "entity_gate"},
                    )
                    progress.add_event(
                        "stitch",
                        status="skipped",
                        label="Merging adjacent clips (temporal stitching)",
                        metadata={"reason": "entity_gate"},
                    )
                    progress.add_event(
                        "synthesize",
                        status="skipped",
                        label="Writing final answer (LLM synthesis)",
                        metadata={"reason": "entity_gate"},
                    )
                    progress.add_event(
                        "validate_answer",
                        status="not_implemented",
                        label="Final validation step (post-answer checks)",
                    )
                    trace["final_kept"] = []
                    trace["final_text"] = cfg_runtime.abort_message
                    summary = self._finalize_trace(trace, progress)
                    log.info(
                        "qe[entity-gate] [%s] q='%s' required=%s -> no matches",
                        summary["request_id"],
                        q,
                        required,
                    )
                    return Response(cfg_runtime.abort_message, source_nodes=[])

            max_post_boost = max((float(n.score or 0.0) for n in nodes), default=0.0)
            trace["post_boost_max_score"] = max_post_boost
            progress.metadata["post_boost_max_score"] = max_post_boost
            if max_post_boost < cfg_runtime.post_boost_hard_min:
                self._last_abort_details = {
                    "stage": "retrieve",
                    "reason": "post_boost_low",
                    "max_score": max_post_boost,
                    "threshold": cfg_runtime.post_boost_hard_min,
                }
                progress.add_event(
                    "rerank_cross_encoder",
                    status="skipped",
                    label="Re-scoring sources (cross-encoder rerank)",
                    metadata={"reason": "post_boost_low", "max_score": max_post_boost},
                )
                progress.add_event(
                    "review_docs",
                    status="skipped",
                    label="Cleaning and enriching notes (post-processing pipeline)",
                    metadata={"reason": "post_boost_low"},
                )
                progress.add_event(
                    "stitch",
                    status="skipped",
                    label="Merging adjacent clips (temporal stitching)",
                    metadata={"reason": "post_boost_low"},
                )
                progress.add_event(
                    "synthesize",
                    status="skipped",
                    label="Writing final answer (LLM synthesis)",
                    metadata={"reason": "post_boost_low"},
                )
                progress.add_event(
                    "validate_answer",
                    status="not_implemented",
                    label="Final validation step (post-answer checks)",
                )
                trace["final_kept"] = []
                trace["final_text"] = cfg_runtime.abort_message
                summary = self._finalize_trace(trace, progress)
                log.info(
                    "qe[post-boost-low] [%s] q='%s' max=%.4f<th=%.4f",
                    summary["request_id"],
                    q,
                    max_post_boost,
                    cfg_runtime.post_boost_hard_min,
                )
                return Response(cfg_runtime.abort_message, source_nodes=[])

            early = self._maybe_early_abort_stage1(q, nodes)
            if early:
                if self._last_abort_details:
                    trace["early_abort"] = self._last_abort_details
                progress.add_event(
                    "rerank_cross_encoder",
                    status="skipped",
                    label="Re-scoring sources (cross-encoder rerank)",
                    metadata={"reason": "stage1_abort"},
                )
                final_nodes = []
            else:
                if self._ce and self._ce.enabled and nodes:
                    with progress.step("rerank_cross_encoder", "Re-scoring sources (cross-encoder rerank)") as ce_step:
                        ce_step.metadata.update({
                            "model": getattr(self._ce, "model_name", "unknown"),
                            "batch_size": getattr(self._ce, "batch_size", None),
                        })
                        packs = self._ce_pack(nodes)
                        trace["pre_ce"] = [{"segment_id": sid, "score_stage1": sc} for (sid, _t, sc) in packs]

                        metas: Dict[str, Dict[str, Any]] = {}
                        for n in nodes:
                            md = n.node.metadata or {}
                            sid = md.get("segment_id") or md.get("id") or n.node.node_id
                            metas[sid] = {
                                "entities": md.get("entities") or [],
                                "canonical_entities": md.get("canonical_entities") or [],
                                "speaker": md.get("speaker"),
                                "channel_name": md.get("channel_name"),
                                "parent_channel_name": md.get("parent_channel_name"),
                                "parent_topic_summary": md.get("parent_topic_summary"),
                                "node_type": md.get("node_type") or md.get("document_type"),
                                "router_boost": md.get("router_boost"),
                                "is_explainer": md.get("is_explainer"),
                            }

                        total = len(packs)
                        batch_size = int(getattr(self._ce, "batch_size", 32) or 32)
                        rescored_chunks: List[Tuple[str, str, float]] = []
                        last_tick = 0.0
                        tick_min_ms = 0.25  # seconds
                        for i in range(0, total, batch_size):
                            chunk = packs[i : i + batch_size]
                            rescored_part = self._ce.rerank_with_meta(q, chunk, metas)
                            rescored_chunks.extend(rescored_part)
                            now = time.perf_counter()
                            if now - last_tick >= tick_min_ms:
                                processed = min(i + len(chunk), total)
                                label_suffix = f"{processed}/{total}" if total else f"{processed}"
                                progress.add_event(
                                    "rerank_cross_encoder",
                                    status="in_progress",
                                    label=f"Re-scoring sources (cross-encoder rerank) [{label_suffix}]",
                                    metadata={
                                        "processed": processed,
                                        "total": total,
                                        "percent": round(processed / max(1, total), 4),
                                    },
                                )
                                last_tick = now

                        rescored = rescored_chunks
                        trace["ce_scores"] = [{"segment_id": sid, "score_ce": sc} for (sid, _t, sc) in rescored]
                        if required_entities:
                            weight = float(getattr(cfg_runtime, "entity_overlap_weight", 0.4))
                            blended: List[Tuple[str, str, float]] = []
                            blend_debug: List[Dict[str, Any]] = []
                            for sid, text, sc in rescored:
                                meta = metas.get(sid) or {}
                                ents = set()
                                if hasattr(self._retriever, "_canonical_entities_from_metadata"):
                                    ents = self._retriever._canonical_entities_from_metadata(meta)
                                overlap = (len(required_entities & ents) / len(required_entities)) if required_entities else 0.0
                                blended_score = (1.0 - weight) * float(sc) + weight * overlap
                                blended.append((sid, text, blended_score))
                                blend_debug.append({
                                    "segment_id": sid,
                                    "ce_raw": float(sc),
                                    "overlap": round(overlap, 4),
                                    "blended": round(blended_score, 4),
                                })
                            rescored = blended
                            trace["ce_entity_blend"] = blend_debug
                            trace["entity_overlap_weight"] = weight

                        alpha = float(getattr(cfg_runtime, "summary_rerank_alpha_def", 0.2)) if wants_definition(q) else float(getattr(cfg_runtime, "summary_rerank_alpha_default", 0.05))
                        if alpha > 0:
                            q_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", q.lower()))
                            blended2: List[Tuple[str, str, float]] = []
                            blend_dbg: List[Dict[str, Any]] = []
                            for sid, text, sc in rescored:
                                meta = metas.get(sid) or {}
                                summ = (meta.get("parent_topic_summary") or "").lower()
                                s_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", summ)) if summ else set()
                                overlap = (len(q_tokens & s_tokens) / max(1, len(q_tokens))) if (q_tokens and s_tokens) else 0.0
                                blended_score = (1.0 - alpha) * float(sc) + alpha * overlap
                                blended2.append((sid, text, blended_score))
                                blend_dbg.append({
                                    "segment_id": sid,
                                    "ce_in": float(sc),
                                    "summary_overlap": round(overlap, 4),
                                    "alpha": alpha,
                                    "blended": round(blended_score, 4),
                                })
                            rescored = blended2
                            trace["ce_summary_blend"] = blend_dbg

                        nodes = self._reinject_scores(nodes, rescored)

                        ce_norm = [self._sigmoid(float(n.score or 0.0)) for n in nodes]
                        early = self._maybe_early_abort_post_ce(ce_norm)
                        if early:
                            ce_step.metadata["abort"] = self._last_abort_details
                            if self._last_abort_details:
                                trace["early_abort"] = self._last_abort_details
                        else:
                            pcut = self._percentile_cut(ce_norm, cfg_runtime.ce_keep_percentile)
                            kept = [n for n, s in zip(nodes, ce_norm) if (s >= pcut) or (s >= cfg_runtime.ce_abs_min)]
                            if not kept:
                                kept = nodes[: cfg_runtime.ce_min_keep]
                            elif len(kept) < cfg_runtime.ce_min_keep:
                                extra = [n for n in nodes if n not in kept][: (cfg_runtime.ce_min_keep - len(kept))]
                                kept.extend(extra)
                            nodes = kept
                            self._tag_ce_scores(nodes)
                            trace["ce_keep_policy"] = {
                                "percentile": cfg_runtime.ce_keep_percentile,
                                "abs_min": cfg_runtime.ce_abs_min,
                                "min_keep": cfg_runtime.ce_min_keep,
                                "pcut": pcut,
                                "kept_after_ce": len(nodes),
                            }
                            ce_step.metadata.update({
                                "kept_after_ce": len(nodes),
                                "pcut": pcut,
                                "kept_sources": self._source_snapshot(nodes),
                            })

                    trace["ce_ms"] = ce_step.duration_ms
                else:
                    progress.add_event(
                        "rerank_cross_encoder",
                        status="skipped",
                        label="Re-scoring sources (cross-encoder rerank)",
                        metadata={
                            "enabled": bool(self._ce and self._ce.enabled),
                            "reason": "no_candidates" if not nodes else "disabled",
                        },
                    )
                    trace["ce_skipped"] = True

                    if not early:
                        nodes = nodes[: cfg_runtime.topk_post_rerank]

                        with progress.step("review_docs", "Cleaning and enriching notes (post-processing pipeline)") as review_step:
                            review_step.metadata["input_count"] = len(nodes)
                            nodes = self._entity_canonicalizer._postprocess_nodes(nodes)
                            nodes = self._speaker_propagator._postprocess_nodes(nodes)
                            review_step.metadata["output_count"] = len(nodes)
                            review_step.metadata["sample_sources"] = self._source_snapshot(nodes)

                        with progress.step("stitch", "Merging adjacent clips (temporal stitching)") as stitch_step:
                            stitch_step.metadata["input_count"] = len(nodes)
                            nodes = self._stitch_adjacent(nodes)
                            nodes = nodes[: self._final_k(q)]
                            nodes = self._annotate_speakers(nodes)
                            stitch_step.metadata["output_count"] = len(nodes)
                            stitch_step.metadata["final_candidates"] = self._source_snapshot(nodes)

                        final_nodes = nodes
                    else:
                        final_nodes = []

            revent.on_end(payload={EventPayload.NODES: final_nodes})

        nodes = final_nodes
        progress.metadata["final_node_count"] = len(nodes)

        # capture durations recorded so far
        retrieve_step_duration = progress.timings().get("retrieve")
        if retrieve_step_duration is not None:
            trace["retrieve_ms"] = retrieve_step_duration

        if early:
            progress.add_event(
                "review_docs",
                status="skipped",
                label="Cleaning and enriching notes (post-processing pipeline)",
                metadata={"reason": "early_abort"},
            )
            progress.add_event(
                "stitch",
                status="skipped",
                label="Merging adjacent clips (temporal stitching)",
                metadata={"reason": "early_abort"},
            )
            progress.add_event(
                "synthesize",
                status="skipped",
                label="Writing final answer (LLM synthesis)",
                metadata={"reason": "early_abort"},
            )
            progress.add_event(
                "validate_answer",
                status="not_implemented",
                label="Final validation step (post-answer checks)",
            )
            trace["final_kept"] = []
            trace["final_text"] = str(early)
            summary = self._finalize_trace(trace, progress)
            log.debug(pretty(trace))
            log.info(
                "qe[early-abort] [%s] q='%s' reason=%s total=%.2f ms",
                summary["request_id"],
                q,
                (self._last_abort_details or {}).get("reason"),
                summary["total_ms"],
            )
            return early

        if not nodes:
            progress.add_event(
                "synthesize",
                status="skipped",
                label="Writing final answer (LLM synthesis)",
                metadata={"reason": "no_results"},
            )
            progress.add_event(
                "validate_answer",
                status="not_implemented",
                label="Final validation step (post-answer checks)",
            )
            trace["final_kept"] = []
            trace["final_text"] = "No results found."
            summary = self._finalize_trace(trace, progress)
            log.debug(pretty(trace))
            log.info(
                "qe[done] [%s] query='%s' -> no results (%.2f ms)",
                summary["request_id"],
                q,
                summary["total_ms"],
            )
            return Response("No results found.", source_nodes=[])

        threshold = float(getattr(cfg_runtime, "min_final_score", 0.0) or 0.0)
        if threshold > 0:
            nodes = self._apply_low_confidence_filter(nodes, threshold, trace, progress)
            if not nodes:
                progress.add_event(
                    "synthesize",
                    status="skipped",
                    label="Writing final answer (LLM synthesis)",
                    metadata={"reason": "low_confidence"},
                )
                progress.add_event(
                    "validate_answer",
                    status="not_implemented",
                    label="Final validation step (post-answer checks)",
                )
                trace["final_kept"] = []
                trace["final_text"] = cfg_runtime.abort_message
                summary = self._finalize_trace(trace, progress)
                log.info(
                    "qe[low-confidence] [%s] q='%s' threshold=%.4f -> insufficient evidence",
                    summary["request_id"],
                    q,
                    threshold,
                )
                return Response(cfg_runtime.abort_message, source_nodes=[])

        trace["final_kept"] = self._final_sources_view(nodes)

        # Optionally prepend parent summaries as additional context
        if getattr(cfg_runtime, "enable_parent_summary_context", True):
            summaries = []
            seen = set()
            max_chars = int(getattr(cfg_runtime, "summary_max_len_chars", 600))
            cap_tokens = int(getattr(cfg_runtime, "summary_context_token_cap", 800))
            used_tokens = 0
            for n in nodes:
                md = n.node.metadata or {}
                title = md.get("title") or md.get("parent_title")
                summ = md.get("parent_topic_summary")
                if not summ or not title:
                    continue
                key = (title, summ)
                if key in seen:
                    continue
                seen.add(key)
                text = f"• {title}: {str(summ)[:max_chars]}"
                t = max(1, len(text) // 4)
                if used_tokens + t > cap_tokens:
                    break
                summaries.append(text)
                used_tokens += t
            if summaries and nodes:
                preface = "Parent summaries:\n" + "\n".join(summaries) + "\n\n"
                nodes[0].node.text = preface + nodes[0].node.get_content()
                progress.metadata["parent_summaries_included"] = len(summaries)

        with progress.step("synthesize", "Writing final answer (LLM synthesis)") as synth_step:
            resp = self._synthesize_clean(query_bundle, nodes)
            synth_step.metadata["llm_model"] = trace["models"].get("llm_model")
            synth_step.metadata["tokens_estimate"] = self._rough_token_count(str(resp))
            synth_step.metadata["final_sources"] = self._source_snapshot(nodes, limit=10)
        trace["synthesize_ms"] = synth_step.duration_ms

        progress.add_event(
            "validate_answer",
            status="not_implemented",
            label="Final validation step (post-answer checks)",
        )

        trace["final_text"] = str(resp)
        summary = self._finalize_trace(trace, progress)
        log.debug(pretty(trace))
        log.info(
            "qe[summary] [%s] q='%s' kept=%d synth=%.2f ms total=%.2f ms",
            summary["request_id"],
            q,
            len(nodes),
            trace["synthesize_ms"],
            summary["total_ms"],
        )
        return resp

    # -------- async path --------
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        q = query_bundle.query_str
        cfg_runtime = get_runtime_config()
        trace: Dict[str, Any] = {"query": q}

        nodes: List[NodeWithScore] = await asyncio.to_thread(self._retriever.retrieve, query_bundle)
        self._tag_stage1_scores(nodes)

        early = self._maybe_early_abort_stage1(q, nodes)
        if early:
            return early

        if hasattr(self._retriever, "debug_snapshot"):
            trace.update(self._retriever.debug_snapshot())

        if self._ce and self._ce.enabled and nodes:
            packs = self._ce_pack(nodes)
            trace["pre_ce"] = [{"segment_id": sid, "score_stage1": sc} for (sid, _t, sc) in packs]
            rescored = await asyncio.to_thread(self._ce.rerank, q, packs)

            ce_norm = [self._sigmoid(float(sc)) for (_sid, _t, sc) in rescored]
            early = self._maybe_early_abort_post_ce(ce_norm)
            if early:
                return early

            trace["ce_scores"] = [{"segment_id": sid, "score_ce": sc} for (sid, _t, sc) in rescored]
            nodes = self._reinject_scores(nodes, rescored)

            ce_norm = [self._sigmoid(float(sc)) for (_sid, _t, sc) in rescored]
            early = self._maybe_early_abort_post_ce(ce_norm)
            if early:
                return early
            self._tag_ce_scores(nodes)

        nodes = nodes[: cfg_runtime.topk_post_rerank]

        # Postprocess, stitch, cap, annotate (same order as sync)
        nodes = self._entity_canonicalizer._postprocess_nodes(nodes)
        nodes = self._speaker_propagator._postprocess_nodes(nodes)
        nodes = self._stitch_adjacent(nodes)
        nodes = nodes[: self._final_k(q)]
        nodes = self._annotate_speakers(nodes)

        if not nodes:
            trace["final_text"] = "No results found."
            log.debug(pretty(trace))
            log.info("qe[done] query='%s' -> no results", q)
            return Response("No results found.", source_nodes=[])

        threshold = float(getattr(cfg_runtime, "min_final_score", 0.0) or 0.0)
        if threshold > 0:
            nodes = self._apply_low_confidence_filter(nodes, threshold, trace, progress=None)
            if not nodes:
                trace["final_text"] = cfg_runtime.abort_message
                log.info("qe[low-confidence] query='%s' threshold=%.4f -> insufficient evidence", q, threshold)
                return Response(cfg_runtime.abort_message, source_nodes=[])

        trace["final_kept"] = self._final_sources_view(nodes)

        synth = getattr(self._core, "_response_synthesizer", None)
        asynth = getattr(synth, "asynthesize", None)
        if callable(asynth):
            raw = await asynth(query=query_bundle, nodes=nodes)
        else:
            raw = await asyncio.to_thread(synth.synthesize, query=query_bundle, nodes=nodes)

        try:
            cleaned = clean_model_refs(str(raw))
            cleaned = normalize_text_entities(cleaned)
            self._normalize_node_entities(nodes)
            source_nodes = getattr(raw, "source_nodes", None) or nodes
            self._normalize_node_entities(source_nodes)
            if hasattr(raw, "response"):
                raw.response = cleaned
                if not getattr(raw, "source_nodes", None):
                    raw.source_nodes = source_nodes
                resp = raw
            else:
                resp = Response(cleaned, source_nodes=source_nodes)
        except Exception:
            resp = raw

        trace["final_text"] = str(resp)
        log.debug(pretty(trace))
        log.info("qe[summary] q='%s' kept=%d (async)", q, len(nodes))
        return resp
