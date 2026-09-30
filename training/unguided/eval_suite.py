"""Eval suite for unguided runs: val loss, router fixture, optional teacher-forced ranks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paths import DATA_DIR

logger = logging.getLogger("llm_gpu.unguided")

DEFAULT_FIXTURE = DATA_DIR / "cabinet_binding_eval.jsonl"

DEFAULT_ANCHORS = (
    "What is the capital of France?",
    "Where is Paris?",
    "What did William Cubitt invent?",
    "Who invented penal treadmill?",
    "What is sequential layer streaming?",
    "How do I disable layer streaming?",
    "How do I force layer streaming?",
)


@dataclass
class EvalResults:
    val_loss: float | None
    val_ppl: float | None
    cabinet_exact_match: float | None
    cabinet_n: int = 0
    router_precision: float | None = None
    router_recall: float | None = None
    nan_detected: bool = False
    anchor_hits: dict[str, bool] = field(default_factory=dict)
    harvested_entity_swaps: int = 0
    harvested_exact: float | None = None


VAL_ONLY_PROBE_MODES = frozenset({"english", "inject", "tinystories"})


def probe_mode(policy: dict | None) -> str:
    raw = str((policy or {}).get("probe_mode") or "cabinet").strip().lower()
    return raw if raw in {"cabinet", "english", "inject", "tinystories"} else "cabinet"


def run_eval_suite(session: Any, policy: dict) -> EvalResults:
    """Three lightweight evals. Never launches a second Metal process.

    ``probe_mode=english`` / ``inject`` / ``tinystories`` is val CE / ppl
    only — no cabinet index, no France/Paris teacher-forced anchors, no
    router fixture.
    """
    from training.eval import evaluate_val_loss, perplexity_from_loss

    val_loss = None
    val_ppl = None
    nan = False
    try:
        val_loss, val_ppl = evaluate_val_loss(
            session.model,
            session.val_dataset,
            seed=int(getattr(session.args, "seed", 42) or 42),
        )
        if val_loss is not None and not math.isfinite(val_loss):
            nan = True
            val_ppl = None
        elif val_loss is not None and val_ppl is None:
            val_ppl = perplexity_from_loss(val_loss)
    except Exception as exc:
        logger.warning("evaluate_val_loss failed: %s", exc)
        val_loss = None

    if probe_mode(policy) in VAL_ONLY_PROBE_MODES:
        return EvalResults(
            val_loss=val_loss,
            val_ppl=val_ppl,
            cabinet_exact_match=None,
            nan_detected=nan,
        )

    from training.cabinet_index import load_cabinet, normalize_question
    from tools.eval_cabinet_bindings import eval_router, eval_teacher_forced, load_fixture

    fixture = Path((policy.get("eval_fixture") or DEFAULT_FIXTURE))
    dataset_path = Path(session.dataset_path) if session.dataset_path else DATA_DIR / "chat_facts_v7.jsonl"
    index = load_cabinet(dataset_path) if dataset_path.is_file() else load_cabinet(DATA_DIR / "chat_facts_v7.jsonl")

    cabinet_exact = None
    cabinet_n = 0
    router_prec = None
    router_rec = None
    if fixture.is_file():
        try:
            rows = load_fixture(fixture)
            report = eval_router(index, rows)
            cabinet_n = int(report.get("n") or 0)
            router_prec = float(report.get("router_precision"))
            router_rec = float(report.get("router_recall"))
        except Exception as exc:
            logger.warning("router fixture eval skipped: %s", exc)

    anchors = list(policy.get("gate_anchors") or DEFAULT_ANCHORS)
    harvested = list(policy.get("harvested_questions") or [])
    anchor_hits: dict[str, bool] = {}
    exact_hits = 0
    exact_n = 0
    harvested_swaps = 0
    harvested_ok = 0
    harvested_n = 0

    def _score(question: str) -> bool | None:
        fact = index.lookup(question, source="trained") or index.lookup(question)
        if fact is None:
            return None
        rec = eval_teacher_forced(session.model, session.tokenizer, fact)
        return bool(rec.get("exact"))

    try:
        for question in anchors:
            ok = _score(question)
            key = normalize_question(question)
            if ok is None:
                anchor_hits[key] = False
                continue
            anchor_hits[key] = ok
            exact_n += 1
            exact_hits += int(ok)
        for row in harvested:
            question = str(row.get("canonical_question") or row.get("typed_question") or "")
            if not question:
                continue
            ok = _score(question)
            if ok is None:
                continue
            harvested_n += 1
            harvested_ok += int(ok)
            classification = str(row.get("classification") or "")
            if classification == "BINDING_ENTITY_SWAP" and not ok:
                harvested_swaps += 1
        if exact_n:
            cabinet_exact = exact_hits / float(exact_n)
    except Exception as exc:
        logger.warning("teacher-forced eval skipped: %s", exc)

    harvested_exact = (harvested_ok / float(harvested_n)) if harvested_n else None

    return EvalResults(
        val_loss=val_loss,
        val_ppl=val_ppl,
        cabinet_exact_match=cabinet_exact,
        cabinet_n=cabinet_n or exact_n,
        router_precision=router_prec,
        router_recall=router_rec,
        nan_detected=nan,
        anchor_hits=anchor_hits,
        harvested_entity_swaps=harvested_swaps,
        harvested_exact=harvested_exact,
    )
