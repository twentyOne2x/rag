from __future__ import annotations

import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    """Return an ISO8601 UTC timestamp with millisecond precision."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class ProgressEvent:
    name: str
    label: str
    status: str = "pending"  # pending | in_progress | completed | failed | skipped | not_implemented
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class _ProgressStep:
    """Context manager returned by ProgressRecorder.step()."""

    def __init__(self, recorder: "ProgressRecorder", event: ProgressEvent):
        self._recorder = recorder
        self.event = event
        self._start_monotonic: Optional[float] = None

    def __enter__(self) -> ProgressEvent:
        self.event.status = "in_progress"
        self.event.started_at = _now_iso()
        self._start_monotonic = time.perf_counter()
        return self.event

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.event.ended_at = _now_iso()
        if self._start_monotonic is not None:
            self.event.duration_ms = round((time.perf_counter() - self._start_monotonic) * 1000, 2)
            self._recorder._register_timing(self.event.name, self.event.duration_ms)

        if exc_type is not None:
            self.event.status = "failed"
            self.event.metadata.setdefault("error", str(exc))
            # don't swallow the exception
            return False

        if self.event.status not in ("skipped", "not_implemented"):
            self.event.status = "completed"
        return False


class ProgressRecorder:
    """
    Lightweight helper that tracks ordered pipeline steps with durations.
    Designed to be thread-confined (one recorder per request).
    """

    def __init__(self, scope: str = "request", request_id: Optional[str] = None, enabled: bool = True):
        self.scope = scope
        self.request_id = request_id or str(uuid.uuid4())
        self.enabled = enabled
        self.started_at = _now_iso()
        self._start_monotonic = time.perf_counter()
        self.events: List[ProgressEvent] = []
        self.metadata: Dict[str, Any] = {}
        self._timings: Dict[str, float] = {}

    def step(self, name: str, label: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a timed step.
        Example:

            with recorder.step("retrieve", "Fetching docs") as ev:
                ...
                ev.metadata["doc_count"] = len(nodes)
        """
        if not self.enabled:
            return nullcontext()
        event = ProgressEvent(name=name, label=label or name, metadata=dict(metadata or {}))
        self.events.append(event)
        return _ProgressStep(self, event)

    def add_event(
        self,
        name: str,
        status: str,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Append a non-timed event (e.g., skipped/not implemented)."""
        if not self.enabled:
            return
        ev = ProgressEvent(
            name=name,
            label=label or name,
            status=status,
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        if duration_ms is not None:
            self._register_timing(name, duration_ms)
        self.events.append(ev)

    def _register_timing(self, name: str, duration_ms: float) -> None:
        self._timings[name] = duration_ms

    def timings(self) -> Dict[str, float]:
        return dict(self._timings)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start_monotonic) * 1000, 2)

    def summary(self) -> Dict[str, Any]:
        progress = [asdict(ev) for ev in self.events]
        return {
            "request_id": self.request_id,
            "scope": self.scope,
            "started_at": self.started_at,
            "total_ms": self.total_ms(),
            "progress": progress,
            "timings": self.timings(),
            "metadata": dict(self.metadata),
        }


class AppDiagnostics:
    """
    Simple in-memory store for the latest startup profile and query trace.
    Useful for debugging and exposing to API responses.
    """

    startup_profile: Optional[Dict[str, Any]] = None
    last_query_trace: Optional[Dict[str, Any]] = None

    @classmethod
    def record_startup(cls, profile: Dict[str, Any]) -> None:
        cls.startup_profile = profile

    @classmethod
    def record_query(cls, trace: Dict[str, Any]) -> None:
        cls.last_query_trace = trace

    @classmethod
    def get_startup_profile(cls) -> Optional[Dict[str, Any]]:
        return cls.startup_profile

    @classmethod
    def get_last_query_trace(cls) -> Optional[Dict[str, Any]]:
        return cls.last_query_trace
