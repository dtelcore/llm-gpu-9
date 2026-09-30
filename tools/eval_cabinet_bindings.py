"""Offline cabinet routing + optional teacher-forced binding probe.

Router metrics use a labeled JSONL fixture. Model scoring is opt-in via
``--checkpoint`` and does not run during normal App traffic.

Does not assign cosine/entropy pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import DATA_DIR, OUTPUT_ROOT
from training.cabinet_diagnostics import classify_labeled_route, score_teacher_forced
from training.cabinet_index import CabinetIndex, load_cabinet, merge_cabinet, normalize_question
from training.router import alias_learned_topics, alias_trained_topics, route

DEFAULT_FIXTURE = DATA_DIR / "cabinet_binding_eval.jsonl"
DEFAULT_FACTS = DATA_DIR / "chat_facts_v7.jsonl"


def load_fixture(path: Path) -> List[dict]:
    rows = []
    if not path.is_file():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and rec.get("query"):
            rows.append(rec)
    return rows


def eval_router(index: CabinetIndex, rows: Sequence[dict]) -> dict:
    labeled = 0
    ok = 0
    false_miss = 0
    false_hit = 0
    details = []
    for rec in rows:
        query = str(rec.get("query") or "")
        intent = str(rec.get("intent") or "")
        decision = route(query, index, search_enabled=False)
        label = classify_labeled_route(intent=intent, kind=decision.kind, fact=decision.fact)
        labeled += 1
        if label == "ROUTE_OK":
            ok += 1
        elif label == "ROUTER_FALSE_MISS":
            false_miss += 1
        elif label == "ROUTER_FALSE_HIT":
            false_hit += 1
        want_canon = normalize_question(str(rec.get("canonical") or ""))
        got_canon = normalize_question(decision.canonical or (decision.fact.user if decision.fact else ""))
        if want_canon and decision.kind == "cabinet" and want_canon != got_canon:
            false_miss += 1
            ok = max(0, ok - 1)
            label = "ROUTER_FALSE_MISS"
        details.append(
            {
                "query": query,
                "intent": intent,
                "kind": decision.kind,
                "match_type": decision.match_type,
                "canonical": decision.canonical,
                "label": label,
            }
        )
    precision = 1.0
    recall = 1.0
    cabinet_pred = [d for d in details if d["kind"] == "cabinet"]
    cabinet_true = [d for d in details if d["intent"] in ("cabinet", "generate", "trained")]
    if cabinet_pred:
        precision = sum(1 for d in cabinet_pred if d["label"] == "ROUTE_OK") / len(cabinet_pred)
    if cabinet_true:
        recall = sum(1 for d in cabinet_true if d["label"] == "ROUTE_OK") / len(cabinet_true)
    return {
        "n": labeled,
        "route_ok": ok,
        "false_miss": false_miss,
        "false_hit": false_hit,
        "router_precision": precision,
        "router_recall": recall,
        "details": details,
    }


def eval_teacher_forced(model, tokenizer, fact, *, top_k: int = 5) -> dict:
    prompt_ids = tokenizer.encode(fact.generate_prompt)
    expected_ids = tokenizer.encode(fact.assistant)
    if not prompt_ids or not expected_ids:
        return {"exact": False, "mean_rank": None, "tokens": []}
    import numpy as np

    window = list(prompt_ids) + list(expected_ids)
    max_len = int(getattr(getattr(model, "config", None), "max_len", 0) or 0)
    if max_len:
        window = window[:max_len]
    logits, _cache = model.forward(np.asarray(window, dtype=np.int64))
    scored = score_teacher_forced(logits, len(prompt_ids), expected_ids[: max(0, len(window) - len(prompt_ids))], k=top_k)
    ranks = [row["rank"] for row in scored if row.get("rank") is not None]
    return {
        "exact": bool(ranks) and all(r == 1 for r in ranks),
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "tokens": scored,
        "user": fact.user,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cabinet routing and optional token ranks")
    parser.add_argument("--facts", type=str, default=str(DEFAULT_FACTS))
    parser.add_argument("--learned", type=str, default=str(OUTPUT_ROOT / "cabinet_learned.jsonl"))
    parser.add_argument("--fixture", type=str, default=str(DEFAULT_FIXTURE))
    parser.add_argument("--checkpoint", type=str, default="", help="Optional; teacher-forced ranks (Metal)")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    facts = Path(args.facts)
    if facts.is_file():
        index = load_cabinet(facts)
        alias_trained_topics(index)
        merge_cabinet(index, Path(args.learned), source="learned")
        alias_learned_topics(index)
    else:
        index = CabinetIndex()
        alias_trained_topics(index)
        alias_learned_topics(index)
    rows = load_fixture(Path(args.fixture))
    report = eval_router(index, rows)
    print(
        f"router n={report['n']} ok={report['route_ok']} "
        f"false_miss={report['false_miss']} false_hit={report['false_hit']} "
        f"precision={report['router_precision']:.3f} recall={report['router_recall']:.3f}"
    )
    if str(args.checkpoint).strip():
        from training.checkpoint import load_checkpoint
        from model.gpt import GPTModel

        gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
        model = GPTModel(gpt_config, params)
        trained = [f for f in index.unique_facts() if f.source == "trained"]
        hits = 0
        n = 0
        for fact in trained[: min(len(trained), 32)]:
            rec = eval_teacher_forced(model, tokenizer, fact, top_k=int(args.top_k))
            n += 1
            hits += int(bool(rec.get("exact")))
        print(f"teacher_forced exact@{n}={hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
