"""Python router: cabinet generate, then calc, then Wikipedia, then miss.

Never loads a second checkpoint. Search snippets are the answer; they are
not fed back into the GPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

from tools.calc import try_calc
from training.cabinet_index import (
    CabinetFact,
    CabinetIndex,
    normalize_question,
    remember,
)
from training.chat_format import is_chat_model_name

SearchFn = Callable[[str], Optional[str]]

MISS_HINT = (
    "That is not in the local fact cabinet. For a short encyclopedia summary, "
    "ask a What/Who/Tell-me-about question or a short topic (Wikipedia). For "
    "open English, start a story checkpoint: python interactive.py "
    "--checkpoint <story> --no-router"
)

RELATED_MISS_HINT = (
    "Not a trained follow-up. Ask one of these, or :search."
)

_FACT_PREFIXES = (
    "what is",
    "what are",
    "what's",
    "whats",
    "who is",
    "who was",
    "who were",
    "who's",
    "where is",
    "where was",
    "when is",
    "when was",
    "tell me about",
    "how many",
    "how much",
)

# Strip these before retrying calc so "what is 1 + 1" is arithmetic, not Wikipedia.
_CALC_WRAPPERS = (
    "what is the value of",
    "what is",
    "what's",
    "whats",
    "calculate",
    "compute",
)
_CALC_OPS = set("+-*/^%")

# Bare noun phrases may search; story openings must not.
_STORY_PREFIXES = (
    "once upon",
    "once there",
    "there once",
    "let me tell",
    "write a",
    "write me",
    "continue the",
    "continue this",
)
_MAX_TOPIC_WORDS = 8


@dataclass(frozen=True)
class RouteDecision:
    kind: str  # cabinet | calc | search | miss
    text: str
    fact: Optional[CabinetFact] = None
    detail: str = ""
    related: Tuple[str, ...] = ()
    normalized: str = ""
    canonical: str = ""
    match_type: str = "none"
    source: str = ""


def router_enabled(explicit: Optional[bool], model_name: str) -> bool:
    """Chat checkpoints default to the router; story checkpoints do not."""
    if explicit is not None:
        return bool(explicit)
    return is_chat_model_name(model_name)


def _bare_query(text: str) -> str:
    s = " ".join((text or "").split())
    if s.casefold().startswith("user:"):
        s = s.split(":", 1)[1].strip()
        s = " ".join(s.split())
    return s


def _calc_remainder(text: str) -> Optional[str]:
    s = _bare_query(text)
    low = s.casefold()
    for prefix in sorted(_CALC_WRAPPERS, key=len, reverse=True):
        if low.startswith(prefix):
            rest = s[len(prefix):].strip(" :?")
            rest = " ".join(rest.split())
            return rest or None
    return None


def try_calc_query(text: str) -> Optional[str]:
    """Calc on the raw text, or on a 'what is …' remainder that is an expression."""
    hit = try_calc(text)
    if hit is not None:
        return hit
    rest = _calc_remainder(text)
    if not rest:
        return None
    if not any(c.isdigit() for c in rest):
        return None
    if not any(c in _CALC_OPS for c in rest):
        return None
    return try_calc(rest)


def looks_like_fact_question(text: str) -> bool:
    s = _bare_query(text).casefold()
    if not s:
        return False
    if any(s.startswith(p) for p in _FACT_PREFIXES):
        return True
    if any(s.startswith(p) for p in _STORY_PREFIXES):
        return False
    if s.startswith(("python ", ":", "/")):
        return False
    words = s.split()
    # "2+2" is one token and is calc; "once upon a time" is a story prefix.
    return 2 <= len(words) <= _MAX_TOPIC_WORDS


def search_topic(text: str) -> str:
    """Noun/topic for Wikipedia: drop 'what is' / 'tell me about' / 'a' / '?'."""
    s = _bare_query(text)
    if s.endswith("?"):
        s = s[:-1].rstrip()
        s = " ".join(s.split())
    s = s.rstrip(".")
    s = " ".join(s.split())
    low = s.casefold()
    for prefix in sorted(_FACT_PREFIXES, key=len, reverse=True):
        if low.startswith(prefix):
            s = s[len(prefix):].strip(" :?.")
            s = " ".join(s.split())
            break
    parts = s.split()
    if parts and parts[0].casefold() in ("a", "an", "the"):
        s = " ".join(parts[1:])
    return s


def remember_search_hit(
    index: Optional[CabinetIndex],
    path,
    typed: str,
    extract: str,
) -> Optional[CabinetFact]:
    """Persist one JSONL line for the typed question; topic is a memory alias only."""
    if index is None or not extract:
        return None
    fact = remember(index, path, typed, extract)
    if fact is None:
        return None
    topic = search_topic(typed)
    if topic:
        index.add_alias(topic, fact)
    return fact


def alias_learned_topics(index: CabinetIndex) -> None:
    """Rebuild topic aliases after loading learned JSONL (not written to disk)."""
    for fact in list(index.unique_facts()):
        if fact.source != "learned":
            continue
        topic = search_topic(fact.user)
        if topic:
            index.add_alias(topic, fact)


def alias_short_topics(index: CabinetIndex) -> None:
    """Drop a leading ``sequential`` so layer streaming hits the trained topic."""
    for fact in list(index.unique_facts()):
        if fact.source != "trained":
            continue
        topic = search_topic(fact.user)
        if not topic:
            continue
        parts = topic.split()
        if len(parts) >= 2 and parts[0].casefold() == "sequential":
            index.add_alias(" ".join(parts[1:]), fact)


def alias_trained_topics(index: CabinetIndex) -> None:
    """Topic aliases for trained rows (what is neonics → Tell me about Neonics.)."""
    for fact in list(index.unique_facts()):
        if fact.source != "trained":
            continue
        topic = search_topic(fact.user)
        if topic:
            index.add_alias(topic, fact)
    alias_short_topics(index)
    index.load_file_aliases()


def _cabinet_decision(
    fact: CabinetFact,
    index: Optional[CabinetIndex],
    *,
    match_type: str,
    normalized: str = "",
) -> RouteDecision:
    learned = fact.source == "learned"
    related: Tuple[str, ...] = ()
    if index is not None and not learned:
        related = tuple(
            index.related_prompts(index.entities_of(fact), exclude_key=fact.key)
        )
    return RouteDecision(
        kind="cabinet",
        text=fact.assistant if learned else fact.generate_prompt,
        fact=fact,
        detail="replay" if learned else "generate",
        related=related,
        normalized=normalized,
        canonical=fact.user,
        match_type=match_type,
        source=fact.source,
    )


def _related_tuple(index: Optional[CabinetIndex], entities: Iterable[str]) -> Tuple[str, ...]:
    if index is None:
        return ()
    return tuple(index.related_prompts(entities))


def route(
    text: str,
    index: Optional[CabinetIndex],
    *,
    search_enabled: bool = True,
    search_fn: Optional[SearchFn] = None,
    last_entities: Optional[Iterable[str]] = None,
) -> RouteDecision:
    """Cabinet first, then calc, then related miss, then Wikipedia, then polite miss."""
    raw = text or ""
    normalized = normalize_question(raw)
    session_ents = frozenset(e for e in (last_entities or ()) if e)

    if index is not None:
        trained = index.resolve_trained(raw)
        if trained is None:
            topic = search_topic(raw)
            if topic:
                trained = index.resolve_trained(topic)
        if trained is not None:
            fact, match_type = trained
            return _cabinet_decision(
                fact, index, match_type=match_type, normalized=normalized,
            )

        learned = index.resolve_learned(raw)
        if learned is None:
            topic = search_topic(raw)
            if topic:
                learned = index.resolve_learned(topic)
        if learned is not None:
            fact, match_type = learned
            return _cabinet_decision(
                fact, index, match_type=match_type, normalized=normalized,
            )

        hits = index.related_template_hits(raw, session_ents)
        if len(hits) == 1:
            return _cabinet_decision(
                hits[0], index, match_type="trained_article", normalized=normalized,
            )
        if len(hits) > 1:
            return RouteDecision(
                kind="miss",
                text=RELATED_MISS_HINT,
                detail="related_ambiguous",
                related=tuple(f.user for f in hits[:5]),
                normalized=normalized,
            )

    calc = try_calc_query(raw)
    if calc is not None:
        return RouteDecision(kind="calc", text=calc, detail="calc", normalized=normalized)

    if session_ents:
        mentioned = index.entities_mentioned(raw) if index is not None else frozenset()
        related = _related_tuple(index, session_ents | mentioned)
        return RouteDecision(
            kind="miss",
            text=RELATED_MISS_HINT,
            detail="related_miss",
            related=related,
            normalized=normalized,
        )

    if search_enabled and looks_like_fact_question(raw):
        topic = search_topic(raw)
        fn = search_fn
        extract = fn(topic) if fn is not None and topic else None
        if extract:
            return RouteDecision(
                kind="search", text=extract, detail="wikipedia", normalized=normalized,
            )
        return RouteDecision(
            kind="miss", text=MISS_HINT, detail="search_failed", normalized=normalized,
        )

    return RouteDecision(kind="miss", text=MISS_HINT, detail="no_route", normalized=normalized)
