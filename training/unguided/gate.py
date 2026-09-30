"""Promotion gate: fail closed unless anchors hold and the harvest queue has no new swaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from training.cabinet_index import normalize_question
from training.unguided.harvest import ANCHOR_QUESTIONS


@dataclass
class GateResult:
    passed: bool
    reason: str
    missing_anchors: list[str]


def gate_eval(
    summary: Optional[Mapping],
    *,
    anchors: Optional[Iterable[str]] = None,
) -> GateResult:
    """Daemon-side gate. Reads the child's JSON summary; does not load Metal."""
    wanted = [normalize_question(q) for q in (anchors or ANCHOR_QUESTIONS)]
    wanted = [q for q in wanted if q]
    if not summary:
        return GateResult(False, "missing_eval_summary", wanted)

    hits = summary.get("anchor_hits")
    if not isinstance(hits, Mapping):
        return GateResult(False, "missing_anchor_hits", wanted)

    missing = []
    for key in wanted:
        if not bool(hits.get(key)):
            missing.append(key)
    if missing:
        return GateResult(False, "anchor_miss", missing)

    swaps = int(summary.get("harvested_entity_swaps") or 0)
    if swaps > 0:
        return GateResult(False, f"harvested_entity_swaps={swaps}", [])

    harvested_exact = summary.get("harvested_exact")
    if harvested_exact is False:
        return GateResult(False, "harvested_exact_false", [])
    if harvested_exact is not None:
        try:
            if float(harvested_exact) < 1.0:
                return GateResult(False, f"harvested_exact={harvested_exact}", [])
        except (TypeError, ValueError):
            return GateResult(False, "harvested_exact_invalid", [])

    return GateResult(True, "ok", [])
