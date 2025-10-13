from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict


class TelemetryHistogram:
    """
    Thread-safe rolling aggregate of stage durations across requests.
    Stores count, totals, min/max, and standard deviation per stage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._request_count = 0
        self._last_updated: float | None = None

    def update(self, summary: Dict[str, Any]) -> None:
        """
        Merge a TelemetryCollector.summary() payload into the aggregate view.
        """
        if not summary:
            return

        counts = dict(summary.get("counts") or {})
        totals = dict(summary.get("totals_ms") or {})
        maximums = dict(summary.get("max_ms") or {})
        minimums = dict(summary.get("min_ms") or {})
        sum_squares = dict(summary.get("sum_squares") or {})

        now = time.time()

        with self._lock:
            for stage, count in counts.items():
                if not count:
                    continue
                stat = self._stats.setdefault(
                    stage,
                    {
                        "count": 0,
                        "total_ms": 0.0,
                        "max_ms": 0.0,
                        "min_ms": None,
                        "sum_squares": 0.0,
                    },
                )
                stat["count"] += int(count)
                stat["total_ms"] += float(totals.get(stage, 0.0))
                stat["max_ms"] = max(stat["max_ms"], float(maximums.get(stage, 0.0)))

                if stage in minimums:
                    stage_min = float(minimums.get(stage, 0.0))
                    if stat["min_ms"] is None or stage_min < stat["min_ms"]:
                        stat["min_ms"] = stage_min

                stat["sum_squares"] += float(sum_squares.get(stage, 0.0))

            self._request_count += 1
            self._last_updated = now

    def snapshot(self) -> Dict[str, Any]:
        """
        Return the current aggregate statistics including per-stage averages and stdev.
        """
        with self._lock:
            stages = {}
            for stage, stat in self._stats.items():
                count = stat["count"]
                if count <= 0:
                    continue
                total = float(stat["total_ms"])
                avg = total / count
                variance = max((stat["sum_squares"] / count) - (avg * avg), 0.0)
                stages[stage] = {
                    "count": stat["count"],
                    "avg_ms": round(avg, 2),
                    "total_ms": round(total, 2),
                    "max_ms": round(stat["max_ms"], 2),
                    "min_ms": round(stat["min_ms"], 2) if stat["min_ms"] is not None else None,
                    "std_ms": round(math.sqrt(variance), 2),
                }

            return {
                "requests": self._request_count,
                "last_updated": self._last_updated,
                "stages": stages,
            }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._request_count = 0
            self._last_updated = None
