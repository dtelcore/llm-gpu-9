"""Ops containment for the autotrainer daemon.

Stops sequential retry burn: poison keys after a crashed child, persist
consumed/poison across daemon restarts, back off, and cap trains per hour.
Does not import model.gpt.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from training.cabinet_index import normalize_question

HOUR_S = 3600.0


def harvest_key(row: dict) -> str:
    return normalize_question(
        str(row.get("canonical_question") or row.get("typed_question") or "")
    )


def harvest_keys(rows: Iterable[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = harvest_key(row)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def block_id(keys: Iterable[str]) -> str:
    blob = "|".join(sorted({k for k in keys if k}))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class DaemonState:
    byte_offset: int = 0
    file_ino: int | None = None
    file_dev: int | None = None
    consumed: set[str] = field(default_factory=set)
    poison: set[str] = field(default_factory=set)
    poison_reasons: dict[str, str] = field(default_factory=dict)
    recent_trains: list[float] = field(default_factory=list)
    backoff_until: float = 0.0
    consecutive_failures: int = 0
    gate_retries: dict[str, int] = field(default_factory=dict)

    def skip_keys(self) -> set[str]:
        return set(self.consumed) | set(self.poison)


def load_state(path: Path) -> DaemonState:
    if not path.is_file():
        return DaemonState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DaemonState()
    if not isinstance(data, dict):
        return DaemonState()
    return DaemonState(
        byte_offset=int(data.get("byte_offset") or 0),
        file_ino=data.get("file_ino"),
        file_dev=data.get("file_dev"),
        consumed=set(data.get("consumed") or []),
        poison=set(data.get("poison") or []),
        poison_reasons=dict(data.get("poison_reasons") or {}),
        recent_trains=[float(x) for x in (data.get("recent_trains") or [])],
        backoff_until=float(data.get("backoff_until") or 0),
        consecutive_failures=int(data.get("consecutive_failures") or 0),
        gate_retries={str(k): int(v) for k, v in (data.get("gate_retries") or {}).items()},
    )


def save_state(path: Path, state: DaemonState, *, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    prune_recent_trains(state, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "byte_offset": int(state.byte_offset),
        "file_ino": state.file_ino,
        "file_dev": state.file_dev,
        "consumed": sorted(state.consumed),
        "poison": sorted(state.poison),
        "poison_reasons": state.poison_reasons,
        "recent_trains": state.recent_trains,
        "backoff_until": state.backoff_until,
        "consecutive_failures": state.consecutive_failures,
        "gate_retries": state.gate_retries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_log_cursor(path: Path, state: DaemonState) -> int:
    """Bind offset to this inode. Truncate clamps; it does not replay from 0."""
    if not path.is_file():
        state.file_ino = None
        state.file_dev = None
        state.byte_offset = 0
        return 0
    info = path.stat()
    ino, dev = int(info.st_ino), int(info.st_dev)
    rotated = (
        state.file_ino is not None
        and state.file_dev is not None
        and (ino != int(state.file_ino) or dev != int(state.file_dev))
    )
    if rotated:
        state.byte_offset = 0
    elif info.st_size < int(state.byte_offset):
        state.byte_offset = int(info.st_size)
    state.file_ino = ino
    state.file_dev = dev
    return int(state.byte_offset)


def prune_recent_trains(state: DaemonState, now: float) -> None:
    cutoff = now - HOUR_S
    state.recent_trains = [ts for ts in state.recent_trains if ts >= cutoff]


def refuse_train_reason(
    state: DaemonState,
    *,
    now: Optional[float] = None,
    max_trains_per_hour: int = 2,
) -> Optional[str]:
    now = time.time() if now is None else now
    if now < float(state.backoff_until):
        return "backoff"
    prune_recent_trains(state, now)
    if int(max_trains_per_hour) >= 0 and len(state.recent_trains) >= int(max_trains_per_hour):
        return "max_trains_per_hour"
    return None


def apply_backoff(
    state: DaemonState,
    *,
    now: Optional[float] = None,
    base_s: float = 300.0,
    max_s: float = 3600.0,
) -> float:
    now = time.time() if now is None else now
    state.consecutive_failures = int(state.consecutive_failures) + 1
    delay = min(float(max_s), float(base_s) * (2 ** (state.consecutive_failures - 1)))
    state.backoff_until = now + delay
    return delay


def mark_success(state: DaemonState) -> None:
    state.consecutive_failures = 0
    state.backoff_until = 0.0


def note_train_started(state: DaemonState, *, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    prune_recent_trains(state, now)
    state.recent_trains.append(now)


def _poison_keys(state: DaemonState, keys: Iterable[str], reason: str) -> None:
    for key in keys:
        if not key:
            continue
        state.poison.add(key)
        state.poison_reasons[key] = reason


def on_promoted(state: DaemonState, keys: Iterable[str], scanned_offset: int) -> None:
    for key in keys:
        if key:
            state.consumed.add(key)
            state.poison.discard(key)
            state.poison_reasons.pop(key, None)
    state.byte_offset = int(scanned_offset)
    state.gate_retries.pop(block_id(keys), None)
    mark_success(state)


def on_crash(
    state: DaemonState,
    keys: Iterable[str],
    scanned_offset: int,
    *,
    reason: str,
    now: Optional[float] = None,
    base_s: float = 300.0,
    max_s: float = 3600.0,
) -> None:
    """Quarantine this harvest block and advance so it is not relaunched."""
    _poison_keys(state, keys, reason)
    state.byte_offset = int(scanned_offset)
    state.gate_retries.pop(block_id(keys), None)
    apply_backoff(state, now=now, base_s=base_s, max_s=max_s)


def on_gate_fail(
    state: DaemonState,
    keys: Iterable[str],
    scanned_offset: int,
    *,
    reason: str,
    max_retries: int = 2,
    now: Optional[float] = None,
    base_s: float = 300.0,
    max_s: float = 3600.0,
) -> str:
    """Keep the rows for another cycle; poison and advance after max_retries."""
    bid = block_id(keys)
    state.gate_retries[bid] = int(state.gate_retries.get(bid) or 0) + 1
    apply_backoff(state, now=now, base_s=base_s, max_s=max_s)
    if state.gate_retries[bid] >= int(max_retries):
        _poison_keys(state, keys, f"gate:{reason}")
        state.byte_offset = int(scanned_offset)
        state.gate_retries.pop(bid, None)
        return "gate_retry_exhausted"
    return "keep"
