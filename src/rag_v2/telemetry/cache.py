from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, Dict, List


class DiagnosticsCache:
    """
    Fixed-size ring buffer of recent diagnostics traces.
    Thread-safe and suitable for in-process debugging.
    """

    def __init__(self, capacity: int = 100):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: Deque[Dict] = deque(maxlen=capacity)
        self._lock = Lock()

    def add(self, trace: Dict) -> None:
        with self._lock:
            self._items.append(trace)

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
