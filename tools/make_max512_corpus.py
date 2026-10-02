"""Build the gpu9_max512 mixed corpus: stories + chat + tool-format demos.

Output: data/gpu9_max512.txt (one document per line, the train.py combiner format).

Mix (deterministic, seed 42):
  ~70%  raw TinyStories lines (all 4 train shards, narrative English)
  ~22%  chat-wrapped stories  (User: <request> Assistant: <story>)
   ~8%  tool-format demos     (User: <q> Assistant: Tool: ... | Result: ... | Answer: ...)

Tool demos use plain text only (no special tokens) so any BPE vocab covers
them, and they match the inference loop in tools/assistant_tools.py exactly:

    User: What is 17 + 4? Assistant: Tool: calc | Query: 17 + 4 | Result: 21 | Answer: 17 + 4 is 21.
    User: What city is the capital of France? Assistant: Tool: search | Query: capital of France
        | Result: Paris is the capital of France. | Answer: The capital of France is Paris.

Usage:
    python tools/make_max512_corpus.py
    python tools/make_max512_corpus.py --output data/gpu9_max512.txt --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_story_chat import wrap_story  # noqa: E402
from training.chat_format import format_conversation  # noqa: E402

SHARDS = [f"data/tinystories_packed/text/train-0000{i}.txt" for i in range(4)]

CAPITALS = (
    ("France", "Paris"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Germany", "Berlin"),
    ("England", "London"),
    ("Japan", "Tokyo"),
    ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Egypt", "Cairo"),
    ("India", "New Delhi"),
    ("Mexico", "Mexico City"),
)

NAMES = ("Lily", "Tom", "Lucy", "Jack", "Mia", "Ben", "Anna", "Sam")

CALCS = (
    ("17 + 4", "21"),
    ("9 + 6", "15"),
    ("12 - 5", "7"),
    ("20 - 8", "12"),
    ("3 * 4", "12"),
    ("5 * 6", "30"),
    ("20 / 4", "5"),
    ("2 + 3 + 4", "9"),
    ("10 - 3 + 2", "9"),
    ("7 + 8", "15"),
)


def tool_demo_calc(expr: str, value: str) -> str:
    q = f"What is {expr}?"
    body = (
        f"Tool: calc | Query: {expr} | Result: {value} "
        f"| Answer: {expr} is {value}."
    )
    return format_conversation([], pending_user=q, open_assistant=False) + " " + body


def tool_demo_search(question: str, query: str, result: str, answer: str) -> str:
    body = (
        f"Tool: search | Query: {query} | Result: {result} | Answer: {answer}"
    )
    return format_conversation([], pending_user=question, open_assistant=False) + " " + body


def tool_demo_direct(question: str, answer: str) -> str:
    return format_conversation(
        [], pending_user=question, open_assistant=False,
    ) + " " + format_conversation([("assistant", answer)], open_assistant=False)


def build_tool_demos(rng: random.Random, n: int) -> List[str]:
    """Small deterministic set of tool-format + direct-answer QA lines."""
    demos: List[str] = []
    for expr, value in CALCS:
        demos.append(tool_demo_calc(expr, value))
    for country, city in CAPITALS:
        q = f"What city is the capital of {country}?"
        demos.append(tool_demo_search(
            q, f"capital of {country}",
            f"{city} is the capital of {country}.",
            f"The capital of {country} is {city}.",
        ))
    demos.append(tool_demo_direct("Who are you?", "I am a helpful story assistant."))
    demos.append(tool_demo_direct(
        "What is the capital of Atlantis?",
        "Atlantis is a myth. It has no capital city.",
    ))
    for name in NAMES:
        demos.append(tool_demo_direct(
            f"Tell me a short story about {name}.",
            f"Once upon a time there was a little child named {name} who loved stories.",
        ))
    # Repeat to reach n (keeps the fraction stable without inventing facts).
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(demos[i % len(demos)])
        i += 1
    rng.shuffle(out)
    return out


def load_shard_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = " ".join(raw.split())
            if line:
                lines.append(line)
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the gpu9_max512 mixed corpus")
    parser.add_argument("--output", type=str, default=str(ROOT / "data" / "gpu9_max512.txt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-frac", type=float, default=0.22)
    parser.add_argument("--tool-docs", type=int, default=4000,
                        help="Number of tool-format demo docs to append (~8% only if corpus is small; "
                             "capped to 8% of story docs, whichever is smaller)")
    parser.add_argument("--max-story-docs", type=int, default=None,
                        help="Optional cap on raw story docs (for smoke tests)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    stories: List[str] = []
    for rel in SHARDS:
        path = ROOT / rel
        if not path.is_file():
            print(f"Missing shard: {path}", file=sys.stderr)
            return 1
        stories.extend(load_shard_lines(path))
    if args.max_story_docs is not None:
        stories = stories[: max(0, int(args.max_story_docs))]
    print(f"Loaded {len(stories):,} raw story docs from {len(SHARDS)} shards")

    n_chat = int(len(stories) * args.chat_frac)
    idx = list(range(len(stories)))
    rng.shuffle(idx)
    chat_docs = [wrap_story(stories[i], template_index=i) for i in idx[:n_chat]]
    raw_docs = [stories[i] for i in idx[n_chat:]]

    n_tool = min(int(args.tool_docs), int(len(stories) * 0.08))
    tool_docs = build_tool_demos(rng, n_tool)

    docs = raw_docs + chat_docs + tool_docs
    rng.shuffle(docs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in docs:
            handle.write(doc.rstrip() + "\n")
    print(f"Wrote {len(docs):,} docs ({len(raw_docs):,} raw + {len(chat_docs):,} chat "
          f"+ {len(tool_docs):,} tool) -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
