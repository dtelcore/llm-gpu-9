"""
tools/make_chat_trainset.py

Turn one-fact-per-line prose (wiki ``data/train.txt``) into chat documents.

Training lines (default ``native``) match ``training/chat_format.py``:

    User: Tell me about the Erie Canal. Assistant: The Erie Canal stretched 363 miles ...

    JSONL (``--jsonl``) uses the query{user} / response{assistant} schema:

    {"query": {"user": "..."}, "response": {"assistant": "..."}}

``--markers braces`` writes the same pair as a single text line:

    query{user} ... response{assistant} ...

Questions use a named entity or short title from the fact. Lines with no
recoverable topic are skipped (no more "What is Soon we dropped into a living?").

Prefer ``native`` for a fine-tune of run8+16: interactive.py and quality
probes look for ``User:`` / ``Assistant:``.

Usage:
    python tools/make_chat_trainset.py
    python tools/make_chat_trainset.py --input data/train.txt --output data/chat_train.txt --jsonl data/chat_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.chat_format import ASSISTANT_ROLE, USER_ROLE, format_conversation

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z][A-Za-z'.-]*|[A-Za-z]\.[A-Za-z](?:\.[A-Za-z])*\.?")
_LEAD_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_HEADING_SPLIT = re.compile(r"\s*[\(:]")

# Sentence-initial function words — not topics even when capitalized.
_SKIP_LEAD = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "there", "here",
    "it", "its", "we", "our", "i", "you", "he", "she", "they", "their",
    "them", "his", "her", "my", "your",
    "in", "on", "at", "by", "for", "from", "with", "to", "of", "as",
    "after", "before", "when", "while", "if", "although", "however",
    "also", "then", "soon", "sooner", "later", "once", "since", "because", "during", "until",
    "all", "every", "each", "some", "any", "no", "not", "one", "two",
    "how", "what", "why", "where", "who", "which", "whose",
    "and", "or", "but", "yet", "so", "nor",
    "many", "most", "more", "other", "another", "both", "few", "several",
    "various", "such", "same", "own", "different",
    "according", "additionally", "instead", "moreover", "nevertheless",
    "therefore", "thus", "still", "even", "just", "also",
    "keep", "see", "note", "please", "click", "visit", "using", "used",
    "make", "making", "take", "taking", "get", "getting",
    "below", "above", "following", "including", "located",
    "did", "does", "do", "should", "would", "could", "can", "may",
})

_FUNCTION = _SKIP_LEAD | frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "will", "shall", "must", "might",
})

_NP_CONNECTORS = frozenset({"of", "the", "and", "for", "in", "de", "van", "von", "da", "di"})

_AUX_OR_LIGHT = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "can", "could", "will", "would", "may", "might", "must", "should",
    "shall",
})

_MONTHS = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})

_WEAK_LEAD = frozenset({
    "new", "old", "first", "second", "last", "next", "early", "late",
    "small", "large", "great", "recent", "current", "further", "additional",
    "main", "major", "minor", "good", "best", "better", "worst",
})

_SPAN_BREAK = re.compile(r"[,;|/]|:\s|\d")

_BAD_TOPIC_PREFIXES = (
    "in addition",
    "for example",
    "in particular",
    "in general",
    "in fact",
    "in order",
    "as well",
    "such as",
    "sooner rather",
    "rather than",
    "on the other",
    "according to",
    "as a result",
    "in this",
    "in that",
)

_QUESTION_TEMPLATES = (
    "Tell me about {topic}.",
    "What is {topic}?",
    "Tell me about {topic}.",
    "Can you explain {topic}?",
    "What do we know about {topic}?",
    "Tell me about {topic}.",
)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _words(text: str) -> List[str]:
    return [word for word, _break in _iter_tokens(text)]


def _iter_tokens(text: str) -> List[Tuple[str, bool]]:
    """Word tokens plus a flag that the previous gap breaks a name span."""
    out: List[Tuple[str, bool]] = []
    last = 0
    for match in _WORD_RE.finditer(text):
        word = match.group().rstrip(".,;:!?")
        if not word:
            last = match.end()
            continue
        gap = text[last:match.start()]
        brk = last > 0 and bool(_SPAN_BREAK.search(gap))
        out.append((word, brk))
        last = match.end()
    return out


def _looks_verbal(word: str) -> bool:
    low = word.lower().rstrip(".")
    if low in _AUX_OR_LIGHT:
        return True
    if len(low) >= 6 and (low.endswith("ing") or low.endswith("ed")):
        return True
    if len(low) >= 6 and low.endswith(("ate", "ize", "ise", "ify")):
        return True
    return False


def _is_acronym(word: str) -> bool:
    bare = word.rstrip(".")
    if len(bare) >= 2 and bare.isupper() and all(c.isalpha() or c == "." for c in word):
        return True
    if re.fullmatch(r"[A-Z](?:\.[A-Z])+\.?", word):
        return True
    return False


def _is_proper(word: str, *, sentence_initial: bool, next_proper: bool) -> bool:
    if _is_acronym(word):
        return True
    if not word or not word[0].isupper():
        return False
    if word.lower() in _MONTHS and not next_proper:
        return False
    if sentence_initial and word.lower() in _SKIP_LEAD:
        return False
    if sentence_initial and not next_proper and not _is_acronym(word):
        # "Soon", "Advances", "Children" — not a name unless a name continues.
        return False
    return True


def _proper_spans(tokens: Sequence[Tuple[str, bool]]) -> List[Tuple[int, int]]:
    """Inclusive-exclusive [start, end) spans of proper-noun runs."""
    words = [word for word, _brk in tokens]
    breaks = [brk for _word, brk in tokens]
    n = len(words)
    proper = [False] * n
    for i, word in enumerate(words):
        nxt = False
        if i + 1 < n and not breaks[i + 1]:
            nxt_word = words[i + 1]
            nxt = bool(
                _is_acronym(nxt_word)
                or (nxt_word[:1].isupper() and nxt_word.lower() not in _SKIP_LEAD)
                or nxt_word.lower() in _NP_CONNECTORS
            )
        proper[i] = _is_proper(word, sentence_initial=(i == 0), next_proper=nxt)
        if i == 0 and word.lower() in {"the", "a", "an"}:
            proper[i] = False

    spans: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not proper[i]:
            i += 1
            continue
        j = i + 1
        while j < n:
            if breaks[j]:
                break
            if proper[j]:
                j += 1
                continue
            if (
                words[j].lower() in _NP_CONNECTORS
                and j + 1 < n
                and not breaks[j + 1]
                and proper[j + 1]
            ):
                j += 2
                continue
            break
        spans.append((i, j))
        i = j
    return spans


def _extend_name_noun(words: Sequence[str], start: int, end: int) -> Tuple[int, int]:
    """Turn 'Canadian' / 'French' into 'Canadian heritage' / 'French government'."""
    if end != start + 1 or end >= len(words):
        return start, end
    nxt = words[end]
    low = nxt.lower()
    if nxt[:1].isupper():
        return start, end
    if low in _FUNCTION or low in _NP_CONNECTORS or _looks_verbal(nxt):
        return start, end
    if low.endswith("s") and len(low) >= 5:
        return start, end
    return start, end + 1


def _format_topic(words: Sequence[str], start: int, end: int) -> Optional[str]:
    if start < 0 or end <= start:
        return None
    piece = list(words[start:end])
    if start > 0 and words[start - 1].lower() in {"the", "a", "an"}:
        piece = [words[start - 1]] + piece
    topic = " ".join(piece).rstrip(" ,;:-")
    topic = re.sub(r"\s+", " ", topic).strip()
    if len(topic) < 2 or len(topic) > 80:
        return None
    core = _LEAD_ARTICLE.sub("", topic).strip()
    if not core or core.lower() in _SKIP_LEAD:
        return None
    if core.lower() in _MONTHS:
        return None
    low = core.lower()
    if any(low == prefix or low.startswith(prefix + " ") for prefix in _BAD_TOPIC_PREFIXES):
        return None
    return topic


def _heading_topic(cleaned: str) -> Optional[str]:
    chunk = _HEADING_SPLIT.split(cleaned, maxsplit=1)[0].strip()
    if chunk == cleaned:
        if ":" in cleaned:
            chunk = cleaned.split(":", 1)[0].strip()
        else:
            return None
    words = _words(chunk)
    if not (2 <= len(words) <= 8):
        return None
    lead = words[0].lower()
    if lead in _SKIP_LEAD:
        return None
    if any(w.lower() in _AUX_OR_LIGHT for w in words):
        return None
    return _format_topic(words, 0, len(words))


def _definition_subject(cleaned: str) -> Optional[str]:
    match = re.match(
        r"^(?P<subj>.{2,60}?)\s+(?:is|are|was|were)\s+"
        r"(?:a|an|the|one|considered|known|called|used|among|part)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    words = _words(match.group("subj"))
    if not (1 <= len(words) <= 6):
        return None
    lead = words[0].lower()
    if lead in _SKIP_LEAD and lead not in {"the", "a", "an"}:
        return None
    start = 1 if lead in {"the", "a", "an"} and len(words) > 1 else 0
    return _format_topic(words, start, len(words))


def _leading_noun_phrase(tokens: Sequence[Tuple[str, bool]]) -> Optional[str]:
    if not tokens:
        return None
    words = [word for word, _brk in tokens]
    breaks = [brk for _word, brk in tokens]
    i = 0
    if words[0].lower() in {"the", "a", "an"}:
        i = 1
    while i < len(words) and words[i].lower() in _WEAK_LEAD:
        i += 1
    if i >= len(words) or words[i].lower() in _SKIP_LEAD:
        return None
    start = i
    taken: List[int] = []
    while i < len(words) and (i - start) < 4:
        if taken and breaks[i]:
            break
        low = words[i].lower()
        if low in _AUX_OR_LIGHT or _looks_verbal(words[i]):
            break
        if (
            taken
            and words[i][0].islower()
            and low not in _NP_CONNECTORS
            and low.endswith("s")
        ):
            break
        if low in _NP_CONNECTORS:
            if not taken or i + 1 >= len(words):
                break
            nxt = words[i + 1]
            if nxt.lower() in _FUNCTION or _looks_verbal(nxt):
                break
            taken.append(i)
            i += 1
            continue
        if low in _FUNCTION:
            break
        taken.append(i)
        i += 1
    if not taken:
        return None
    # Drop trailing connectors ("Advances in").
    end = taken[-1] + 1
    while end > start and words[end - 1].lower() in _NP_CONNECTORS:
        end -= 1
    if end - start < 1:
        return None
    if end - start == 1 and (len(words[start]) < 4 or not words[start][0].isupper()):
        return None
    return _format_topic(words, start, end)


def topic_from_fact(text: str) -> Optional[str]:
    """Named entity or short title from a fact line; None if nothing usable."""
    cleaned = _clean(text)
    if not cleaned:
        return None

    heading = _heading_topic(cleaned)
    if heading:
        return heading

    tokens = _iter_tokens(cleaned)
    words = [word for word, _brk in tokens]
    spans = _proper_spans(tokens)
    if spans:
        s, e = spans[0]
        strong = (e - s) >= 2 or _is_acronym(words[s]) or s > 0
        if strong:
            s, e = _extend_name_noun(words, s, e)
            topic = _format_topic(words, s, e)
            if topic:
                return topic
    multi = [(s, e) for s, e in spans if (e - s) >= 2]
    if multi:
        s, e = min(multi, key=lambda se: se[0])
        topic = _format_topic(words, s, e)
        if topic:
            return topic

    defined = _definition_subject(cleaned)
    if defined:
        return defined

    leading = _leading_noun_phrase(tokens)
    if leading:
        return leading

    for s, e in spans:
        if e - s != 1:
            continue
        if words[s].lower() in _MONTHS:
            continue
        topic = _format_topic(words, s, e)
        if topic:
            return topic
    return None


def make_user_query(fact: str, template_index: int = 0) -> Optional[str]:
    topic = topic_from_fact(fact)
    if not topic:
        return None
    return _QUESTION_TEMPLATES[template_index % len(_QUESTION_TEMPLATES)].format(topic=topic)


def wrap_native(user: str, assistant: str, *, system: Optional[str] = None) -> str:
    """One training document: User query + Assistant response."""
    return format_conversation(
        [(USER_ROLE, user), (ASSISTANT_ROLE, assistant)],
        system=system,
        open_assistant=False,
    )


def wrap_braces(user: str, assistant: str, *, system: Optional[str] = None) -> str:
    """Literal query{user} / response{assistant} markers on one line."""
    sys_part = ""
    if system:
        sys_part = " ".join(system.split()) + " "
    u = " ".join(user.split())
    a = " ".join(assistant.split())
    return f"{sys_part}query{{user}} {u} response{{assistant}} {a}".strip()


def qa_record(user: str, assistant: str) -> dict:
    return {"query": {"user": user}, "response": {"assistant": assistant}}


def iter_source_lines(path: Path, *, min_chars: int, max_docs: Optional[int]) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = " ".join(raw.split())
            if len(line) < min_chars:
                continue
            yield line
            n += 1
            if max_docs is not None and n >= int(max_docs):
                return


def wrap_line(
    fact: str,
    index: int,
    *,
    markers: str,
    system: Optional[str],
) -> Optional[Tuple[str, str, str]]:
    user = make_user_query(fact, template_index=index)
    if user is None:
        return None
    assistant = fact
    if markers == "braces":
        text = wrap_braces(user, assistant, system=system)
    else:
        text = wrap_native(user, assistant, system=system)
    return text, user, assistant


def write_corpus(
    facts: Iterable[str],
    out_path: Path,
    *,
    markers: str = "native",
    system: Optional[str] = None,
    jsonl_path: Optional[Path] = None,
    keep_raw: float = 0.0,
    seed: int = 42,
) -> Tuple[int, int, int]:
    import random

    keep_raw = min(1.0, max(0.0, float(keep_raw)))
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_handle = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = open(jsonl_path, "w", encoding="utf-8")
    n_out = 0
    n_chat = 0
    n_skip = 0
    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            for i, fact in enumerate(facts):
                if keep_raw > 0.0 and rng.random() < keep_raw:
                    handle.write(fact + "\n")
                    n_out += 1
                    continue
                wrapped = wrap_line(
                    fact, i, markers=markers, system=system,
                )
                if wrapped is None:
                    n_skip += 1
                    continue
                text, user, assistant = wrapped
                handle.write(text + "\n")
                n_out += 1
                n_chat += 1
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(qa_record(user, assistant), ensure_ascii=False) + "\n")
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
    return n_out, n_chat, n_skip


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a User/Assistant (query/response) chat train set from fact lines",
    )
    parser.add_argument(
        "--input", type=str, default=str(ROOT / "data" / "train.txt"),
        help="Source facts (one document per line). Default: data/train.txt",
    )
    parser.add_argument(
        "--output", type=str, default=str(ROOT / "data" / "chat_train.txt"),
        help="Destination chat corpus (one document per line)",
    )
    parser.add_argument(
        "--jsonl", type=str, default=None,
        help="Optional JSONL with query.user / response.assistant objects",
    )
    parser.add_argument(
        "--markers", choices=("native", "braces"), default="native",
        help="native = User:/Assistant: (train this). braces = query{user} ... response{assistant} ...",
    )
    parser.add_argument(
        "--system", type=str, default="",
        help="Optional system prefix on each native/braces line",
    )
    parser.add_argument(
        "--min-chars", type=int, default=40,
        help="Skip source lines shorter than this (default 40)",
    )
    parser.add_argument(
        "--keep-raw", type=float, default=0.0,
        help="Fraction of source lines kept as plain prose (default 0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Optional cap on kept source lines",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"Input not found: {src}", file=sys.stderr)
        return 1
    system = args.system.strip() or None
    facts = iter_source_lines(src, min_chars=int(args.min_chars), max_docs=args.max_docs)
    jsonl = Path(args.jsonl) if args.jsonl else None
    n_out, n_chat, n_skip = write_corpus(
        facts,
        Path(args.output),
        markers=args.markers,
        system=system,
        jsonl_path=jsonl,
        keep_raw=args.keep_raw,
        seed=args.seed,
    )
    extra = f", jsonl={args.jsonl}" if jsonl else ""
    print(
        f"Wrote {n_out:,} documents ({n_chat:,} chat, {n_skip:,} skipped no-topic) "
        f"to {args.output} markers={args.markers}{extra}"
    )
    if args.markers == "braces":
        print(
            "Note: braces lines are not the live train format. "
            "Use --markers native to fine-tune run8+16.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
