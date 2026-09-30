#!/usr/bin/env python3
"""Host-only v10 mix: native User:/Assistant: only, modest repeats, wiki slice.

v10_4 failed because seven shared wrappers (``It is known that`` / ``A short
note on``) taught a global template and a random slot. This builder writes
canonical Q/A plus a trailing `` User:`` train suffix so generate can stop.
Wiki ``data/train.txt`` lines live in the same JSONL as prose records.

Usage:
  python tools/build_chat_facts_v10.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_english_phase2_mix import (
    interleave_stride,
    load_prose_lines,
    max_identical_count,
    subsample_prose,
    unique_pairs_from_jsonl,
)
from tools.make_chat_trainset import qa_record, wrap_native
from training.cabinet_index import normalize_question
from training.chat_format import CHAT_STOP_STRINGS

DEFAULT_V9 = ROOT / "data" / "chat_facts_v9.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "train.txt"
DEFAULT_OUT = ROOT / "data" / "chat_facts_v10.jsonl"
DEFAULT_FACT_FRACTION = 0.45
MIN_FACT_FRACTION = 0.40
MAX_FACT_FRACTION = 0.50
DEFAULT_CORE_REPEAT = 40
DEFAULT_TAIL_REPEAT = 10
RECITATION_REPEAT_REFUSE = 300
TRAIN_STOP_SUFFIX = CHAT_STOP_STRINGS[0]
WRAPPER_MARKERS = (
    "it is known that ",
    "a short note on ",
    "regarding ",
    "in plain terms, ",
    "students remembering ",
    "if you ask ",
    "question: ",
)

PROBE_CORE: Tuple[str, ...] = (
    "What is the capital of France?",
    "What is France's capital?",
    "Where is Paris?",
    "What country is Paris in?",
    "What is the capital of Japan?",
    "What is Japan's capital?",
    "Where is Tokyo?",
    "What country is Tokyo in?",
    "Which organ system does the kidney belong to?",
    "Which organ system does the heart belong to?",
    "Is 1 a prime number?",
    "Is 2 a prime number?",
    "Is 3 a prime number?",
    "Is 4 a prime number?",
    "Is 17 a prime number?",
    "Is 42 a prime number?",
)

# Pins, not "maybe". v9 generate swapped L'Hopital→Cramer (same named-after
# template) and dropped "not" on the prime-1 gold. Keep split geo Assistants.
GOLD_FIXES: Dict[str, Tuple[str, str]] = {
    "Who is L'Hopital's rule named after?": (
        "Who is L'Hopital's rule named after?",
        "Guillaume de l'Hopital is the namesake of L'Hopital's rule.",
    ),
    "Who is Cramer's rule named after?": (
        "Who is Cramer's rule named after?",
        "Cramer's rule takes its name from Gabriel Cramer.",
    ),
    "Is 1 a prime number?": (
        "Is 1 a prime number?",
        "No, 1 is not a prime number.",
    ),
    "Is 17 a prime number?": (
        "Is 17 a prime number?",
        "Yes, 17 is a prime number.",
    ),
    "Is 42 a prime number?": (
        "Is 42 a prime number?",
        "No, 42 is a composite number.",
    ),
}

GEO_SPLIT_PINS: Tuple[Tuple[str, str], ...] = (
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is France's capital?", "The capital of France is Paris."),
    ("Where is Paris?", "Paris is a city in France."),
    ("What country is Paris in?", "Paris is in France."),
    ("What is the capital of Belgium?", "The capital of Belgium is Brussels."),
    ("Where is Brussels?", "Brussels is a city in Belgium."),
)


def apply_gold_fixes(pairs: Dict[str, str]) -> Dict[str, str]:
    """Replace pinned Assistants (and User spelling) by normalized key."""
    by_norm = {normalize_question(user): user for user in pairs}
    out = dict(pairs)
    for query, (user, assistant) in GOLD_FIXES.items():
        old_user = by_norm.get(normalize_question(query))
        if old_user is not None and old_user != user:
            out.pop(old_user, None)
        out[user] = assistant
        by_norm[normalize_question(user)] = user
    return out


def assert_gold_pins(pairs: Dict[str, str]) -> None:
    by_norm = {normalize_question(user): assistant for user, assistant in pairs.items()}
    for user, assistant in GOLD_FIXES.values():
        got = by_norm.get(normalize_question(user))
        if got != assistant:
            raise ValueError(f"Gold pin failed for {user!r}: expected {assistant!r}, got {got!r}")
    for user, assistant in GEO_SPLIT_PINS:
        got = by_norm.get(normalize_question(user))
        if got != assistant:
            raise ValueError(f"Geo split pin failed for {user!r}: expected {assistant!r}, got {got!r}")
    france = by_norm.get(normalize_question("What is the capital of France?"))
    belgium = by_norm.get(normalize_question("What is the capital of Belgium?"))
    if france == belgium:
        raise ValueError("France and Belgium must keep split Assistants")


def canonical_line(user: str, assistant: str) -> str:
    return wrap_native(user, assistant)


def train_line(user: str, assistant: str) -> str:
    """Native chat document plus the generate stop marker (`` User:``)."""
    return canonical_line(user, assistant).rstrip() + TRAIN_STOP_SUFFIX


def core_norms(core: Sequence[str] = PROBE_CORE) -> set[str]:
    return {normalize_question(user) for user in core}


def repeat_for_user(
    user: str,
    *,
    core: Sequence[str] = PROBE_CORE,
    core_repeat: int = DEFAULT_CORE_REPEAT,
    tail_repeat: int = DEFAULT_TAIL_REPEAT,
) -> int:
    if normalize_question(user) in core_norms(core):
        return int(core_repeat)
    return int(tail_repeat)


def native_fact_lines(
    pairs: Dict[str, str],
    *,
    core_repeat: int = DEFAULT_CORE_REPEAT,
    tail_repeat: int = DEFAULT_TAIL_REPEAT,
) -> List[str]:
    core_n = int(core_repeat)
    tail_n = int(tail_repeat)
    if core_n < 1 or tail_n < 1:
        raise ValueError("core_repeat and tail_repeat must be >= 1")
    if max(core_n, tail_n) >= RECITATION_REPEAT_REFUSE:
        raise ValueError(
            f"Refuse recitation pack: repeat {max(core_n, tail_n)} >= {RECITATION_REPEAT_REFUSE}"
        )
    lines: List[str] = []
    for user, assistant in pairs.items():
        line = canonical_line(user, assistant)
        low = line.casefold()
        if any(marker in low for marker in WRAPPER_MARKERS):
            raise ValueError(f"Wrapper frame leaked into native mix: {line!r}")
        n = repeat_for_user(user, core_repeat=core_n, tail_repeat=tail_n)
        lines.extend([line] * n)
    cap = max(core_n, tail_n)
    dup = max_identical_count(lines)
    if dup > cap:
        raise ValueError(
            f"Fact lines repeated {dup}×, above core/tail cap {cap}. Refuse recitation pack."
        )
    return lines


def record_for_line(line: str, canonical: Dict[str, Tuple[str, str]]) -> dict:
    hit = canonical.get(line)
    if hit is not None:
        rec = qa_record(hit[0], hit[1])
        rec["text"] = train_line(hit[0], hit[1])
        return rec
    return {"kind": "prose", "text": line}


def build_records(
    *,
    pairs: Dict[str, str],
    prose_lines: Sequence[str],
    fact_fraction: float = DEFAULT_FACT_FRACTION,
    seed: int = 42,
    core_repeat: int = DEFAULT_CORE_REPEAT,
    tail_repeat: int = DEFAULT_TAIL_REPEAT,
) -> Tuple[List[dict], List[str], List[str]]:
    frac = float(fact_fraction)
    if not (MIN_FACT_FRACTION <= frac <= MAX_FACT_FRACTION):
        raise ValueError(
            f"fact_fraction must be in [{MIN_FACT_FRACTION}, {MAX_FACT_FRACTION}], got {frac}"
        )
    facts = native_fact_lines(pairs, core_repeat=core_repeat, tail_repeat=tail_repeat)
    canonical = {canonical_line(user, assistant): (user, assistant) for user, assistant in pairs.items()}
    n_prose = max(1, int(round(len(facts) * (1.0 - frac) / frac)))
    anchor = subsample_prose(list(prose_lines), n_prose, seed=int(seed))
    rng = __import__("random").Random(int(seed))
    shuffled = list(facts)
    rng.shuffle(shuffled)
    mixed_lines = interleave_stride(anchor, shuffled)
    records = [record_for_line(line, canonical) for line in mixed_lines]
    return records, facts, mixed_lines


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chat_facts_v10.jsonl (host-only)")
    parser.add_argument("--v9", type=str, default=str(DEFAULT_V9), help="v9 cabinet JSONL")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN), help="Wiki prose anchor")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fact-fraction", type=float, default=DEFAULT_FACT_FRACTION)
    parser.add_argument("--core-repeat", type=int, default=DEFAULT_CORE_REPEAT)
    parser.add_argument("--tail-repeat", type=int, default=DEFAULT_TAIL_REPEAT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    v9_path = Path(args.v9)
    train_path = Path(args.train)
    out_path = Path(args.output)
    if not v9_path.is_file():
        print(f"ERROR: v9 cabinet not found: {v9_path}", file=sys.stderr)
        return 2
    if not train_path.is_file():
        print(f"ERROR: prose file not found: {train_path}", file=sys.stderr)
        return 2

    print(f"Loading unique facts from {v9_path}...")
    pairs = apply_gold_fixes(unique_pairs_from_jsonl(v9_path))
    assert_gold_pins(pairs)
    print(f"Streaming wiki from {train_path}...")
    prose_lines = load_prose_lines(train_path)
    records, facts, mixed_lines = build_records(
        pairs=pairs,
        prose_lines=prose_lines,
        fact_fraction=float(args.fact_fraction),
        seed=int(args.seed),
        core_repeat=int(args.core_repeat),
        tail_repeat=int(args.tail_repeat),
    )
    n_gold = sum(1 for rec in records if rec.get("query"))
    n_frame = sum(1 for rec in records if rec.get("kind") == "frame")
    n_prose = sum(1 for rec in records if rec.get("kind") == "prose")
    n_stop = sum(
        1
        for rec in records
        if rec.get("query") and str(rec.get("text") or "").endswith(TRAIN_STOP_SUFFIX)
    )
    frac = (len(facts) / len(mixed_lines)) if mixed_lines else 0.0
    if n_frame:
        raise ValueError(f"Wrapper/frame records are forbidden in v10 native mix (got {n_frame})")
    if n_stop != n_gold:
        raise ValueError(f"Every gold record needs a trailing {TRAIN_STOP_SUFFIX!r} train line ({n_stop}/{n_gold})")
    if not (MIN_FACT_FRACTION <= frac <= MAX_FACT_FRACTION):
        raise ValueError(f"Built fact fraction {frac:.3f} is outside 40–50%")
    write_jsonl(out_path, records)
    print("")
    print("--- Native mix stats ---")
    print(f"Unique Facts Loaded: {len(pairs)}")
    print(f"Native chat copies: {n_gold}")
    print(f"Wrapper/frame records: {n_frame}")
    print(f"Prose Anchor Lines: {n_prose}")
    print(f"Fact line fraction: {frac:.3f}")
    print(f"Total Combined Output Lines: {len(records)}")
    print(f"Maximum identical native lines: {max_identical_count(facts)}")
    print(f"Maximum identical lines in mix: {max_identical_count(mixed_lines)}")
    print(f"Dataset Mix written to {out_path} successfully.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
