"""Temps : tout est en UTC, int64, millisecondes (spec 4.3)."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

_DURATION_RE = re.compile(r"^(\d+)([smhd])$")
_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_duration(s: str) -> int:
    """'5m' -> 300."""
    m = _DURATION_RE.match(s)
    if not m:
        raise ValueError(f"durée invalide : {s!r}")
    return int(m.group(1)) * _UNIT_S[m.group(2)]


def iso_to_ms(s: str | None) -> int | None:
    """ISO 8601 (avec 'Z' ou '+00') -> ms UTC."""
    if not s:
        return None
    s = s.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    elif re.search(r"[+-]\d\d$", s):
        s += ":00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def utc_hour(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H")


def floor_to(ts: int, step: int) -> int:
    return ts - ts % step
