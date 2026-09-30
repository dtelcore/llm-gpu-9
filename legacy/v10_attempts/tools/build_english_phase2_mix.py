#!/usr/bin/env python3
"""Host-only Phase 2 mix: unique v9 facts, distinct frames, interleaved into train.txt.

Does not touch Metal. Zero identical fact-frame duplication (no 300× loops).

Usage:
  python tools/build_english_phase2_mix.py
  python tools/build_english_phase2_mix.py --dense
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.cabinet_index import _pairs_from_jsonl, normalize_question
from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX
from training.unguided.prober import family

DEFAULT_V9 = ROOT / "data" / "chat_facts_v9.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "train.txt"
DEFAULT_OUT = ROOT / "data" / "english_phase2_mix.txt"
DEFAULT_OUT_DENSE = ROOT / "data" / "english_phase2b_mix.txt"

TARGET_FAMILIES = frozenset({
    "capital_of", "possess_capital", "country_of", "where_is", "organ_system", "prime",
})
PHASE2B_MUST = (
    "What is the capital of France?",
    "What is France's capital?",
    "Where is Paris?",
    "What country is Paris in?",
    "What is the capital of Japan?",
    "What is Japan's capital?",
    "Where is Tokyo?",
    "What country is Tokyo in?",
    "What is the capital of Germany?",
    "Where is Berlin?",
    "Which organ system does the ureter belong to?",
    "Which organ system does the kidney belong to?",
    "Which organ system does the urethra belong to?",
    "Which organ system does the heart belong to?",
    "Which organ system does the brain belong to?",
    "Which organ system does the liver belong to?",
    "Which organ system does the lung belong to?",
    "Which organ system does the spleen belong to?",
    "Is 17 a prime number?",
    "Is 42 a prime number?",
    "Is 1 a prime number?",
)


def unique_pairs_from_jsonl(path: Path) -> Dict[str, str]:
    """Last assistant wins per user. v9 is 300×; this collapses to unique keys."""
    pairs: Dict[str, str] = {}
    for user, assistant in _pairs_from_jsonl(path):
        user = " ".join((user or "").split())
        assistant = " ".join((assistant or "").split())
        if user and assistant:
            pairs[user] = assistant
    return pairs


def _bare(text: str, prefix: str) -> str:
    s = " ".join((text or "").split())
    if s.casefold().startswith(prefix.casefold()):
        s = s[len(prefix) :].strip()
    return s


def generate_frames(user: str, assistant: str, *, dense: bool = False) -> List[str]:
    """Distinct linguistic mappings. dict.fromkeys drops accidental collisions."""
    user = " ".join((user or "").split())
    assistant = " ".join((assistant or "").split())
    clean_u = _bare(user, USER_PREFIX).replace("?", "").strip()
    clean_a = _bare(assistant, ASSISTANT_PREFIX).strip()
    frames = [
        f"{USER_PREFIX}{user} {ASSISTANT_PREFIX}{assistant}".rstrip(),
        f"It is known that {clean_u} is resolved by {clean_a}.",
        f"Regarding {clean_u}, the corresponding detail is {clean_a}.",
        f"Question: {clean_u}? Response: {clean_a}.",
    ]
    if dense:
        frames.extend([
            f"If you ask {clean_u}, the answer is {clean_a}.",
            f"In plain terms, {clean_u} maps to {clean_a}.",
            f"A short note on {clean_u}: {clean_a}",
            f"Students remembering {clean_u} recall {clean_a}.",
        ])
    return list(dict.fromkeys(frame for frame in frames if frame.strip()))


def select_target_pairs(
    pairs: Dict[str, str],
    *,
    must: Sequence[str] = PHASE2B_MUST,
    max_facts: int = 60,
    max_per_family: int = 15,
) -> Dict[str, str]:
    """Keep probe geo/organ/prime keys; drop the rest of v9."""
    by_norm = {normalize_question(user): (user, assistant) for user, assistant in pairs.items()}
    out: Dict[str, str] = {}
    counts: Counter[str] = Counter()
    for query in must:
        hit = by_norm.get(normalize_question(query))
        if hit is None:
            continue
        user, assistant = hit
        if user in out:
            continue
        out[user] = assistant
        counts[family(user)] += 1
    for user in sorted(pairs):
        if len(out) >= int(max_facts):
            break
        if user in out:
            continue
        fam = family(user)
        if fam not in TARGET_FAMILIES:
            continue
        if counts[fam] >= int(max_per_family):
            continue
        out[user] = pairs[user]
        counts[fam] += 1
    return out


def subsample_prose(prose_lines: Sequence[str], n: int, *, seed: int = 42) -> List[str]:
    """Strided sample so the wiki anchor stays diverse without the full 1.5M lines."""
    lines = list(prose_lines)
    n = max(0, int(n))
    if n <= 0:
        return []
    if n >= len(lines):
        return lines
    stride = max(1, len(lines) // n)
    picked = [lines[i] for i in range(0, len(lines), stride)][:n]
    if len(picked) < n:
        seen = set(picked)
        extra = [line for line in lines if line not in seen]
        rng = __import__("random").Random(int(seed))
        rng.shuffle(extra)
        picked.extend(extra[: n - len(picked)])
    return picked[:n]


def all_fact_frames(pairs: Dict[str, str], *, dense: bool = False) -> List[str]:
    frames: List[str] = []
    for user, assistant in pairs.items():
        frames.extend(generate_frames(user, assistant, dense=dense))
    return frames


def max_identical_count(lines: Sequence[str]) -> int:
    if not lines:
        return 0
    return max(Counter(lines).values())


def interleave_stride(prose_lines: Sequence[str], fact_frames: Sequence[str]) -> List[str]:
    """Insert one fact frame on a uniform stride so facts are not a prefix clump."""
    total_prose = len(prose_lines)
    total_facts = len(fact_frames)
    if total_facts == 0:
        return list(prose_lines)
    stride = max(1, total_prose // total_facts) if total_prose else 1
    mixed: List[str] = []
    fact_idx = 0
    for i, line in enumerate(prose_lines):
        mixed.append(line)
        if i % stride == 0 and fact_idx < total_facts:
            mixed.append(fact_frames[fact_idx])
            fact_idx += 1
    while fact_idx < total_facts:
        mixed.append(fact_frames[fact_idx])
        fact_idx += 1
    return mixed


def load_prose_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def write_mix(path: Path, lines: Iterable[str]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
            n += 1
    return n


def build_mix(
    *,
    pairs: Dict[str, str],
    prose_lines: Sequence[str],
    seed: int = 42,
    dense: bool = False,
    fact_fraction: float | None = None,
) -> Tuple[List[str], List[str]]:
    frames = all_fact_frames(pairs, dense=dense)
    dup = max_identical_count(frames)
    if dup > 1:
        raise ValueError(
            f"Fact frames are not unique (max identical count={dup}). "
            "Refuse to write a recitation pack."
        )
    anchor = list(prose_lines)
    if fact_fraction is not None:
        frac = float(fact_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"fact_fraction must be in (0, 1), got {frac}")
        n_prose = max(1, int(round(len(frames) * (1.0 - frac) / frac)))
        anchor = subsample_prose(anchor, n_prose, seed=int(seed))
    rng = __import__("random").Random(int(seed))
    shuffled = list(frames)
    rng.shuffle(shuffled)
    mixed = interleave_stride(anchor, shuffled)
    return mixed, frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 2 English inject mix (host-only)")
    parser.add_argument("--v9", type=str, default=str(DEFAULT_V9), help="v9 cabinet JSONL")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN), help="Phase 1 prose anchor")
    parser.add_argument("--output", type=str, default=None, help="Interleaved mix path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Phase 2b: geo/organ/prime only, 8 frames each, subsample prose to --fact-fraction",
    )
    parser.add_argument("--fact-fraction", type=float, default=None, dest="fact_fraction")
    parser.add_argument("--max-facts", type=int, default=60, dest="max_facts")
    parser.add_argument("--max-per-family", type=int, default=15, dest="max_per_family")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    v9_path = Path(args.v9)
    train_path = Path(args.train)
    dense = bool(args.dense)
    out_path = Path(args.output) if args.output else (DEFAULT_OUT_DENSE if dense else DEFAULT_OUT)
    if not v9_path.is_file():
        print(f"ERROR: v9 cabinet not found: {v9_path}", file=sys.stderr)
        return 2
    if not train_path.is_file():
        print(f"ERROR: prose file not found: {train_path}", file=sys.stderr)
        return 2

    print(f"Loading unique facts from {v9_path}...")
    pairs = unique_pairs_from_jsonl(v9_path)
    if dense:
        pairs = select_target_pairs(
            pairs,
            max_facts=int(args.max_facts),
            max_per_family=int(args.max_per_family),
        )
        print(f"Phase 2b target keys: {len(pairs)}")
    print(f"Streaming and interleaving with {train_path}...")
    prose_lines = load_prose_lines(train_path)
    fact_fraction = args.fact_fraction
    if dense and fact_fraction is None:
        fact_fraction = 0.15
    mixed, frames = build_mix(
        pairs=pairs,
        prose_lines=prose_lines,
        seed=int(args.seed),
        dense=dense,
        fact_fraction=fact_fraction,
    )
    write_mix(out_path, mixed)
    fact_dup = max_identical_count(frames)
    mix_dup = max_identical_count(mixed)
    n_prose = len(mixed) - len(frames)
    frac = (len(frames) / len(mixed)) if mixed else 0.0
    print("")
    print("--- Uniqueness & Dataset Mix Stats ---")
    print(f"Unique Facts Loaded: {len(pairs)}")
    print(f"Total Fact Frames Generated: {len(frames)}")
    print(f"Prose Anchor Lines: {n_prose}")
    print(f"Fact Frame Fraction: {frac:.3f}")
    print(f"Total Combined Output Lines: {len(mixed)}")
    print(f"Maximum Identical Fact-Frame Lines: {fact_dup}")
    print(f"Maximum Identical Lines In Mix: {mix_dup}")
    print(f"Dataset Mix written to {out_path} successfully.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
