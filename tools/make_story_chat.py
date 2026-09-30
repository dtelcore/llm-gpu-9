"""
tools/make_story_chat.py

Wrap TinyStories (one document per line) as simple User/Assistant chat lines
for mixed chat_5m training. Output stays one document per line so the existing
data/ combiner works.

Usage:
    python tools/make_story_chat.py --input data/tiny_stories.txt --output data/story_chat.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.chat_format import format_conversation

_WORD_RE = re.compile(r"[A-Za-z]+")

_TOPICS = (
    "cat", "dog", "fox", "bear", "bird", "mouse", "frog", "fish", "horse",
    "pig", "duck", "rabbit", "lion", "wolf", "hen", "cow", "sheep", "boy",
    "girl", "king", "queen", "dragon", "tree", "house", "friend", "family",
)

_TEMPLATES = (
    "Tell me a story about {topic}.",
    "What happened next?",
    "Make it about a {topic}.",
    "Tell me a short story.",
    "Can you tell me a story?",
)


def _topic_from_story(story: str) -> str:
    words = [w.lower() for w in _WORD_RE.findall(story)]
    for topic in _TOPICS:
        if topic in words:
            return topic
    for word in words:
        if len(word) >= 4:
            return word
    return "a friend"


def wrap_story(story: str, template_index: int = 0) -> str:
    """One training document: User request + Assistant story."""
    topic = _topic_from_story(story)
    question = _TEMPLATES[template_index % len(_TEMPLATES)].format(topic=topic)
    return format_conversation(
        [],
        pending_user=question,
        open_assistant=False,
    ) + " " + format_conversation(
        [("assistant", story)],
        open_assistant=False,
    )


def make_story_chat_lines(
    stories: Sequence[str],
    *,
    keep_raw: float = 0.2,
    seed: int = 42,
) -> List[str]:
    """Convert story lines to chat documents; keep a raw narrative fraction."""
    import random

    keep_raw = min(1.0, max(0.0, float(keep_raw)))
    rng = random.Random(seed)
    out: List[str] = []
    for i, raw in enumerate(stories):
        story = " ".join(raw.split())
        if not story:
            continue
        if rng.random() < keep_raw:
            out.append(story)
            continue
        out.append(wrap_story(story, template_index=i))
    return out


def load_story_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def write_lines(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line.rstrip() + "\n")
            count += 1
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap TinyStories lines as User/Assistant chat documents",
    )
    parser.add_argument(
        "--input", type=str, default=str(ROOT / "data" / "tiny_stories.txt"),
        help="Source stories (one document per line)",
    )
    parser.add_argument(
        "--output", type=str, default=str(ROOT / "data" / "story_chat.txt"),
        help="Destination chat corpus (one document per line)",
    )
    parser.add_argument(
        "--keep-raw", type=float, default=0.2,
        help="Fraction of original story lines kept unchanged (default 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Optional cap on input stories (after skipping blanks)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"Input not found: {src}", file=sys.stderr)
        print("Put TinyStories under data/tiny_stories.txt (one story per line).", file=sys.stderr)
        return 1
    stories = load_story_lines(src)
    if args.max_docs is not None:
        stories = stories[: max(0, int(args.max_docs))]
    if not stories:
        print(f"No documents in {src}", file=sys.stderr)
        return 1
    lines = make_story_chat_lines(stories, keep_raw=args.keep_raw, seed=args.seed)
    n = write_lines(Path(args.output), lines)
    print(f"Wrote {n:,} documents to {args.output} (from {len(stories):,} stories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
