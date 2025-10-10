from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_v2.telemetry.collector import TelemetryCollector, TelemetryEvent
from rag_v2.telemetry.cache import DiagnosticsCache
from rag_v2.telemetry.writer import JsonlTelemetryWriter


class TelemetryTests(unittest.TestCase):
    def test_collector_aggregates_stage_metrics(self) -> None:
        collector = TelemetryCollector(service_name="unit-test", environment="test")

        collector.record_event(TelemetryEvent(stage="retrieve", duration_ms=100.0, metadata={"k": "v"}))
        collector.record_event(TelemetryEvent(stage="retrieve", duration_ms=50.0))
        collector.record_event(TelemetryEvent(stage="synthesize", duration_ms=200.0))

        summary = collector.summary()

        self.assertEqual(summary["service_name"], "unit-test")
        self.assertEqual(summary["environment"], "test")
        self.assertEqual(summary["counts"]["retrieve"], 2)
        self.assertEqual(summary["counts"]["synthesize"], 1)
        self.assertAlmostEqual(summary["totals_ms"]["retrieve"], 150.0)
        self.assertAlmostEqual(summary["max_ms"]["retrieve"], 100.0)
        self.assertAlmostEqual(summary["max_ms"]["synthesize"], 200.0)
        self.assertEqual(collector.events[0].metadata["k"], "v")

    def test_writer_persists_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics.jsonl"
            writer = JsonlTelemetryWriter(output_path=output)

            event = TelemetryEvent(stage="retrieve", duration_ms=123.4, metadata={"foo": "bar"})
            writer.write_event(event)
            writer.write_summary({"counts": {"retrieve": 1}})

            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            second = json.loads(lines[1])

            self.assertEqual(first["stage"], "retrieve")
            self.assertAlmostEqual(first["duration_ms"], 123.4)
            self.assertEqual(second["counts"]["retrieve"], 1)

    def test_diagnostics_cache_keeps_last_n_entries(self) -> None:
        cache = DiagnosticsCache(capacity=3)

        for i in range(5):
            cache.add({"request_id": f"req-{i}", "total_ms": i})

        items = cache.snapshot()
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["request_id"], "req-2")
        self.assertEqual(items[-1]["request_id"], "req-4")

        cache.clear()
        self.assertEqual(cache.snapshot(), [])


if __name__ == "__main__":
    unittest.main()
