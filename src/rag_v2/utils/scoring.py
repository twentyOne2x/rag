from __future__ import annotations
import math
from datetime import datetime

def recency_decay(published_date_str: str | None, half_life_days: float) -> float:
    if not published_date_str:
        return 1.0
    try:
        dt = datetime.strptime(published_date_str, "%Y-%m-%d")
    except Exception:
        return 1.0
    days = (datetime.utcnow() - dt).days
    return 0.5 ** (days / max(half_life_days, 1e-6))

def apply_multiplier(score: float, mult: float) -> float:
    return float(score) * float(mult)
