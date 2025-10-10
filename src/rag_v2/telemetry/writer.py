from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .collector import TelemetryEvent


class JsonlTelemetryWriter:
    """
    Simple JSONL writer used to persist telemetry events and summaries.
    Safe for concurrent writes via append with buffering disabled.
    """

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_line(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_event(self, event: "TelemetryEvent") -> None:
        payload = event.to_dict()
        self._write_line(payload)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        payload = {"summary": True, **summary}
        self._write_line(payload)
