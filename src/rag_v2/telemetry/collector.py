from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

from .writer import JsonlTelemetryWriter


def _default_env() -> str:
    return os.getenv("RAG_ENV", os.getenv("ENVIRONMENT", "dev"))


@dataclass
class TelemetryEvent:
    """Single stage timing record."""

    stage: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class TelemetryCollector:
    """
    Collects timing events per request and aggregates summary statistics.
    Thread-safe so it can be reused across concurrent requests if needed.
    """

    def __init__(
        self,
        service_name: str = "rag-v2",
        environment: str | None = None,
        writer: Optional[JsonlTelemetryWriter] = None,
    ):
        self.service_name = service_name
        self.environment = environment or _default_env()
        self.events: List[TelemetryEvent] = []
        self._lock = threading.Lock()
        self._writer = writer

    def record_event(self, event: TelemetryEvent) -> None:
        with self._lock:
            self.events.append(event)
        if self._writer:
            self._writer.write_event(event)

    def record_stage(self, stage: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.record_event(TelemetryEvent(stage=stage, duration_ms=duration_ms, metadata=metadata or {}))

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            events_copy = list(self.events)

        counts: Dict[str, int] = {}
        totals: Dict[str, float] = {}
        maximums: Dict[str, float] = {}

        for evt in events_copy:
            counts[evt.stage] = counts.get(evt.stage, 0) + 1
            totals[evt.stage] = totals.get(evt.stage, 0.0) + float(evt.duration_ms)
            maximums[evt.stage] = max(maximums.get(evt.stage, 0.0), float(evt.duration_ms))

        summary = {
            "service_name": self.service_name,
            "environment": self.environment,
            "counts": counts,
            "totals_ms": totals,
            "max_ms": maximums,
            "event_count": len(events_copy),
        }

        if self._writer:
            self._writer.write_summary(summary)

        return summary

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self.events)

