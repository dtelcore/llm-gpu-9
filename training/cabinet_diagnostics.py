"""Structured cabinet turn diagnostics and generation-target comparison.

Live traffic records observed route facts. ROUTER_FALSE_MISS / ROUTER_FALSE_HIT
are only assigned when a labeled intent is supplied (eval runs).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

from training.cabinet_index import CabinetFact, CabinetIndex, normalize_question

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")

INVENTOR_FAMILY = "inventor"
CAPITAL_FAMILY = "capital"

_INVENTOR_MARKERS = (
    "who invented ",
    "who is credited with inventing ",
    "what did ",
)
_CAPITAL_MARKERS = (
    "what is the capital of ",
    "where is ",
    "what country is ",
    "'s capital",
)


def normalize_answer(text: str) -> str:
    s = " ".join((text or "").split()).strip()
    s = s.casefold()
    s = _PUNCT.sub(" ", s)
    s = _SPACE.sub(" ", s).strip()
    return s


def answers_match(expected: str, generated: str) -> bool:
    left = normalize_answer(expected)
    right = normalize_answer(generated)
    return bool(left) and left == right


def relation_family(user: str) -> str:
    key = normalize_question(user)
    if any(key.startswith(m) or key.endswith(m.rstrip()) for m in _INVENTOR_MARKERS):
        if key.startswith("what did ") and key.endswith(" invent"):
            return INVENTOR_FAMILY
        if not key.startswith("what did "):
            return INVENTOR_FAMILY
    if any(m in key for m in _CAPITAL_MARKERS):
        return CAPITAL_FAMILY
    return ""


def classify_generation(
    expected: str,
    generated: str,
    index: Optional[CabinetIndex],
    fact: Optional[CabinetFact],
) -> str:
    """Per-turn class: MATCH, BINDING_ENTITY_SWAP, or TARGET_MISMATCH."""
    if answers_match(expected, generated):
        return "MATCH"
    family = relation_family(fact.user) if fact is not None else ""
    gen_norm = normalize_answer(generated)
    if family and index is not None and gen_norm:
        for other in index.unique_facts():
            if other.source != "trained":
                continue
            if fact is not None and other.key == fact.key:
                continue
            if relation_family(other.user) != family:
                continue
            if normalize_answer(other.assistant) == gen_norm:
                return "BINDING_ENTITY_SWAP"
    return "TARGET_MISMATCH"


def classify_labeled_route(
    *,
    intent: str,
    kind: str,
    fact: Optional[CabinetFact],
) -> str:
    """Eval-only labels. Live traffic must not guess these."""
    want = (intent or "").strip().casefold()
    if want in ("cabinet", "generate", "trained"):
        if kind == "cabinet" and fact is not None and fact.source == "trained":
            return "ROUTE_OK"
        return "ROUTER_FALSE_MISS"
    if want in ("miss", "search", "search_or_miss", "calc"):
        if kind == "cabinet":
            return "ROUTER_FALSE_HIT"
        if want == "search_or_miss" and kind in ("search", "miss"):
            return "ROUTE_OK"
        if want == kind:
            return "ROUTE_OK"
        return "ROUTER_FALSE_MISS" if want != "miss" else "ROUTE_OK"
    return "UNLABELED"


def new_event_id(now: Optional[datetime] = None) -> str:
    ts = now or datetime.now(timezone.utc)
    return ts.strftime("evt_%Y%m%d_%H%M%S_%f")


def append_jsonl(path: Union[str, Path], record: dict) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_turn_record(
    *,
    raw_query: str,
    decision,
    generated: str,
    expected: str,
    checkpoint: str,
    classification: str,
    detail: str,
    event_id: Optional[str] = None,
) -> dict:
    now = datetime.now(timezone.utc)
    fact = getattr(decision, "fact", None)
    return {
        "event_id": event_id or new_event_id(now),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": checkpoint,
        "input": {
            "raw_query": raw_query,
            "normalized_query": getattr(decision, "normalized", "") or normalize_question(raw_query),
            "canonical_question": getattr(decision, "canonical", "") or (fact.user if fact else ""),
            "expected_target": expected,
        },
        "router_telemetry": {
            "selected_route": getattr(decision, "kind", ""),
            "matched_cabinet_key": fact.key if fact is not None else "",
            "key_match_type": getattr(decision, "match_type", "none") or "none",
            "cabinet_hit": fact is not None,
            "source": getattr(decision, "source", "") or (fact.source if fact else ""),
            "cabinet_returned_value": expected,
        },
        "generation": {
            "text": generated,
            "detail": detail,
        },
        "diagnostic_classification": classification,
    }


def token_rank_from_logits(logits_row, expected_id: int, k: int = 5) -> dict:
    """Rank/probability of an expected id in a 1-D logit vector (host NumPy)."""
    import numpy as np

    row = np.asarray(logits_row, dtype=np.float32).reshape(-1)
    if row.size == 0 or expected_id < 0 or expected_id >= row.size:
        return {"rank": None, "prob": 0.0, "logit": None, "top_k": []}
    shifted = row - float(row.max())
    exp = np.exp(shifted)
    probs = exp / float(exp.sum())
    order = np.argsort(probs)[::-1]
    rank = int(np.where(order == int(expected_id))[0][0]) + 1
    top = []
    for idx in order[: max(1, int(k))]:
        top.append(
            {
                "id": int(idx),
                "logit": float(row[int(idx)]),
                "prob": float(probs[int(idx)]),
            }
        )
    return {
        "rank": rank,
        "prob": float(probs[int(expected_id)]),
        "logit": float(row[int(expected_id)]),
        "top_k": top,
    }


def score_teacher_forced(logits, prompt_len: int, expected_ids: Iterable[int], k: int = 5) -> list:
    """Score expected answer tokens using teacher-forced next-token logits.

    ``logits[t]`` predicts the token at position ``t+1``. The first expected
    token is predicted at index ``prompt_len - 1``.
    """
    import numpy as np

    arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[0]
    rows = []
    ids = [int(i) for i in expected_ids]
    for offset, tid in enumerate(ids):
        pos = int(prompt_len) - 1 + offset
        if pos < 0 or pos >= arr.shape[0]:
            rows.append({"token_id": tid, "rank": None, "prob": 0.0, "top_k": []})
            continue
        rec = token_rank_from_logits(arr[pos], tid, k=k)
        rec["token_id"] = tid
        rows.append(rec)
    return rows
